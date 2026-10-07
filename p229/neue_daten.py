"""Neue, bisher ungesehene Tage: Daten laden, mit der Historie verbinden, Signale erzeugen."""
import glob
import os

import numpy as np
import pandas as pd

from .beobachten import NAMEN, signal
from .config import SERIES
from .data import load_argus_excel
from .features import build_features
from .walk_forward import walk_forward

SPALTEN = SERIES + ["ice_gasoil"]


def suche_neueste_api_csv(ordner):
    """Neueste Datei argus_api_*.csv im Ordner (nach Namen sortiert) oder None."""
    treffer = sorted(glob.glob(os.path.join(ordner, "argus_api_*.csv")))
    return treffer[-1] if treffer else None


def lade_neue_argus(pfad):
    """CSV (Spalte date + 8 Serien + ice_gasoil) oder Argus-Excel im Add-in-Format."""
    if str(pfad).lower().endswith(".csv"):
        df = pd.read_csv(pfad)
        datum = next((c for c in df.columns if c.lower() in ("date", "datum", "publicationdate")), df.columns[0])
        df["date"] = pd.to_datetime(df[datum].astype(str).str[:10])
        fehlt = [c for c in SPALTEN if c not in df.columns]
        if fehlt:
            raise ValueError(f"In der Datei fehlen Spalten: {fehlt}")
        df = df[["date"] + SPALTEN].copy()
        for c in SPALTEN:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return load_argus_excel(pfad)


def kombiniere_argus(alt, neu):
    """Historie + neue Tage. Bei Überlappung gelten die alten (geprüften) Werte; Abweichungen werden berichtet."""
    gemeinsam = alt["date"][alt["date"].isin(neu["date"])]
    a = alt.set_index("date").loc[gemeinsam, SPALTEN]
    n = neu.set_index("date").loc[gemeinsam, SPALTEN]
    abw = (a - n).abs().max().max() if len(gemeinsam) else np.nan
    zusatz = neu[~neu["date"].isin(alt["date"])]
    out = pd.concat([alt, zusatz]).sort_values("date").reset_index(drop=True)
    bericht = {"Tage_alt": len(alt), "Tage_ueberlappung": len(gemeinsam), "max_Abweichung_Ueberlappung": float(abw) if abw == abw else np.nan,
               "Tage_neu_dazu": len(zusatz), "erster_neuer_Tag": zusatz["date"].min() if len(zusatz) else None,
               "letzter_Tag": out["date"].max()}
    return out, bericht


def signale_neue_tage(d, H, start, ende, delta, retrain_every=5):
    """Vorhersage und Signal für jeden Tag von start bis ende, auch für Tage, deren Ergebnis noch nicht feststeht."""
    F = build_features(d, H)
    res = walk_forward(d, H, F, (start, ende), entry_lag=0, feat_lag=0, retrain_every=retrain_every, keep_unlabeled=True)
    res["Signal"] = [NAMEN[int(s)] for s in signal(res["proba"].values, delta)]
    res["Ergebnis_bekannt"] = res["y_true"].notna()
    res.insert(0, "Serie", H)
    return res, F
