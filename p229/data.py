"""Daten aus den Excel-Dateien laden und zu einer Tabelle `d` zusammenführen.

Argus-Excel (Add-in): Blatt "Tabelle1", Zeile 2 = Beschreibung, ab Zeile 3 Tageswerte.
Marktdaten-Excel: je Blatt Datum + Wert (Brent, WTI, Heizöl/NYMEX, Wechselkurs).
"""
import warnings

import numpy as np
import pandas as pd

from .config import EXO, SERIES

_REGION = [("southwest", "suedwest"), ("southeast", "suedost"), ("rhine-main", "rheinmain"), ("south", "sued")]


def _serie_name(desc):
    """Beschreibungstext der Argus-Spalte -> Serienname (oder None, z. B. 'all regions')."""
    t = str(desc).lower()
    if "ice nwe month 1" in t:
        return "ice_gasoil"
    if "all regions" in t:
        return None
    produkt = "diesel" if "diesel en 590" in t else "heizoel" if "heating oil" in t else None
    if produkt is None:
        return None
    for key, name in _REGION:
        if key in t:
            return f"{produkt}_{name}"
    return None


def load_argus_excel(path, sheet="Tabelle1"):
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    head = raw.iloc[1]                                   # Zeile 2: Beschreibungen
    body = raw.iloc[2:].copy()
    cols = {0: "date"}
    for j in range(1, raw.shape[1]):
        n = _serie_name(head.iloc[j])
        if n:
            cols[j] = n
    df = body[list(cols)].rename(columns=cols)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    fehlt = [c for c in SERIES + ["ice_gasoil"] if c not in df.columns]
    if fehlt:
        raise ValueError(f"Argus-Spalten nicht erkannt: {fehlt}")
    return df.drop_duplicates("date", keep="last").reset_index(drop=True)


def inspect_excel(path, n=6):
    """Struktur jeder Tabelle ausgeben (Blattnamen, Größe, erste Zeilen) – zum Prüfen vor dem Laden."""
    xl = pd.ExcelFile(path)
    for s in xl.sheet_names:
        df = xl.parse(s, header=None, nrows=n + 4)
        print(f"--- Blatt '{s}' (erste Zeilen) ---")
        print(df.head(n + 4).to_string(max_colwidth=40))


def load_market_sheet(path, sheet, name, date_col=None, value_col=None, header_row=None):
    """Ein Marktdaten-Blatt als Serie `name`.

    Ohne Angabe wird das Blatt automatisch gelesen: erste Spalte mit Datumswerten, danach die
    erste Spalte mit überwiegend Zahlen. Bei Zweifel date_col/value_col (0-basiert) und
    header_row angeben, nachdem `inspect_excel` die Struktur gezeigt hat.
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dt = raw.apply(lambda s: pd.to_datetime(s, errors="coerce"))
    if date_col is None:
        is_date = [(dt[c].notna() & pd.to_numeric(raw[c], errors="coerce").isna()).mean() for c in raw.columns]
        date_col = int(np.argmax(is_date))
        if max(is_date) < 0.5:
            raise ValueError(f"Blatt '{sheet}': keine Datumsspalte gefunden")
    if value_col is None:
        num = [(pd.to_numeric(raw[c], errors="coerce").notna().mean() if c != date_col else -1) for c in raw.columns]
        value_col = int(np.argmax(num))
    out = pd.DataFrame({"date": dt[date_col], name: pd.to_numeric(raw[value_col], errors="coerce")})
    out = out.dropna(subset=["date", name])
    out = out[out["date"] >= "1990-01-01"].drop_duplicates("date", keep="last").sort_values("date")
    return out.reset_index(drop=True)


def build_dataset(argus, markt, max_ffill=5):
    """Argus-Handelstage als Gerüst, Marktdaten per Datum dazu, Lücken bis `max_ffill` Tage vorwärts füllen.

    Rückgabe: (d, bericht). Das Vorwärtsfüllen nutzt nur vergangene Werte (kein Look-ahead).
    """
    d = argus.copy()
    for m in markt:
        d = d.merge(m, on="date", how="left")
    cols = SERIES + EXO
    fehlt = [c for c in cols if c not in d.columns]
    if fehlt:
        raise ValueError(f"Spalten fehlen: {fehlt}")
    vorher = d[cols].isna().sum()
    d[cols] = d[cols].ffill(limit=max_ffill)
    d = d.dropna(subset=cols).reset_index(drop=True)
    bericht = pd.DataFrame({"fehlend_vor_ffill": vorher, "fehlend_nach_ffill": d[cols].isna().sum()})
    return d, bericht
