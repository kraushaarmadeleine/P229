# Databricks notebook source
# MAGIC %md
# MAGIC # db_11: XGBoost-Testlauf mit allen Kennzahlen
# MAGIC
# MAGIC **Frage:** Wie gut sagt XGBoost aus den Excel-Daten die 3-Tage-Richtung der 8 Preisreihen voraus, und wie sehen Trefferquote, Precision und Recall genau aus?
# MAGIC
# MAGIC **Was dieses Notebook macht**
# MAGIC 1. Zeigt genau, **welche Daten** aus welcher Datei, welchem Blatt und welcher Spalte genommen wurden (Zelle 2).
# MAGIC 2. Prüft, dass die Merkmale nicht in die Zukunft schauen (Zelle 3).
# MAGIC 3. Läuft XGBoost (feste Einstellungen wie in db_08, kein Tuning) im Test-Zeitraum 2024-11-15 bis 2026-01-20 (Zelle 4).
# MAGIC 4. Zeigt alle Pflichtkennzahlen (Zelle 5 bis 7): Trefferquote mit Konfidenzintervall, Konfusionsmatrix, Precision, Recall, Brier-Skill, Vergleich mit einfachen Regeln, Kosten gegenüber „immer kaufen“.
# MAGIC
# MAGIC **Wie man die Begriffe liest**
# MAGIC - **Signal „Kaufen“** = Modell sagt: Preis in 3 Tagen höher. **Signal „Warten“** = Modell sagt: nicht höher.
# MAGIC - **Precision** = von allen „Kaufen“-Signalen: Wie oft stieg der Preis wirklich? (Wie zuverlässig ist ein Kaufsignal?)
# MAGIC - **Recall** = von allen Tagen mit steigendem Preis: Wie viele hat das Modell als „Kaufen“ erkannt?
# MAGIC - **Basisrate** = Anteil der Tage, an denen der Preis wirklich stieg. Eine Precision nur knapp über der Basisrate heißt: Das Signal hilft kaum.
# MAGIC
# MAGIC Dauer: ca. 15 Minuten für alle 8 Serien, ca. 2 Minuten pro Serie. Zwischenstände werden gesichert.

# COMMAND ----------
# MAGIC %pip install xgboost scikit-learn openpyxl

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# Zelle 1: Pfade, Code, Daten laden.
import os, sys, time
import numpy as np, pandas as pd

BASE_PATH = "/Volumes/dev_workspace/p229/p229_files"
OUT = f"{BASE_PATH}/output"
ARGUS_XLSX = f"{BASE_PATH}/Heizoel und Diesel bis 2016 und ICE Daten.xlsx"
MARKT_XLSX = f"{BASE_PATH}/P229 - Daten von Brent, WTI, Heizöl, Wechselkurs.xlsx"

sys.path.insert(0, "/tmp/p229_code")
from p229.config import SERIES, EXO, DEV, TEST, SCEN, LAST_ARGUS
from p229.data import load_argus_excel, load_market_sheet, build_dataset, inspect_excel
from p229.features import build_features
from p229.leakage import leak_test
from p229.walk_forward import walk_forward
from p229.evaluate import evaluate_run
from p229.stats import holm

argus, argus_info = load_argus_excel(ARGUS_XLSX, return_info=True)
BLAETTER = [("Brent Rohöl", "brent"), ("WIT Rohöl", "wti"),
            ("Heizöl", "nymex_heating_oil"), ("USD_EUR Wechselkurs", "usd_eur")]
markt, markt_info = [], []
for blatt, name in BLAETTER:
    m, i = load_market_sheet(MARKT_XLSX, blatt, name, return_info=True)
    markt.append(m); markt_info.append(i)
d_all, fuell_bericht = build_dataset(argus, markt)
d = d_all[d_all["date"] <= LAST_ARGUS].reset_index(drop=True)
print("Tabelle d:", d.shape, d["date"].min().date(), "bis", d["date"].max().date())

# COMMAND ----------
# Zelle 2: DATENPROTOKOLL. Welche Daten genau? Bitte prüfen, ob die Spalten und Zeiträume zu deinem Verständnis passen.
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)
print("=== Argus-Excel:", os.path.basename(ARGUS_XLSX), "(Blatt 'Tabelle1') ===")
print(argus_info.drop(columns="Beschreibung").to_string(index=False))
print("\nBeschreibung der ersten Spalte (Beispiel):", argus_info["Beschreibung"].iloc[0])
print("\n=== Marktdaten-Excel:", os.path.basename(MARKT_XLSX), "===")
print(pd.DataFrame(markt_info).to_string(index=False))
print("\n=== Fehlende Werte vor und nach dem Vorwärtsfüllen (max. 5 Tage) ===")
print(fuell_bericht.to_string())
print("\n=== Rohe Kopfzeilen der Marktdaten-Excel (erste 4 Zeilen je Blatt, zeigt die Original-Reihennamen) ===")
inspect_excel(MARKT_XLSX, n=0)
print("\nGenutzte Variablen: 8 Preisreihen (Argus-Index, Euro/100 l) + ICE Gasoil (USD/t), Brent, WTI, NYMEX Heating Oil, USD/EUR")
print("Nicht genutzt: Argus 'all regions', low/high-Werte, OMR-Übersicht 2017, Platts, Nachrichten, Rheinfracht")

# COMMAND ----------
# Zelle 3: Leakage-Test und Auswahl. Für einen Probelauf (ca. 2 Minuten) SERIEN_LAUF = ["heizoel_sued"] eintragen.
for H in SERIES:
    leak_test(d, H)
print("Leakage-Test bestanden für", len(SERIES), "Serien")

SERIEN_LAUF = list(SERIES)
SZENARIEN = ["S0 (Kauf zu Preis t)"]          # optional zusätzlich: "S1 (Kauf zu Preis t+1)", "S2 (Daten nur bis t-1)"
FENSTER = {"Test": TEST}                       # optional zusätzlich: "Entwicklung": DEV
print("Serien:", SERIEN_LAUF, "| Szenarien:", SZENARIEN, "| Fenster:", list(FENSTER))

# COMMAND ----------
# Zelle 4: XGBoost-Lauf. Jeder Lauf wird als CSV gesichert und beim Neustart wiederverwendet.
rows, RES = [], {}
t0 = time.time()
for H in SERIEN_LAUF:
    F = build_features(d, H)
    for szen in SZENARIEN:
        el, fl = SCEN[szen]
        for fenster, win in FENSTER.items():
            pfad = f"{OUT}/db11_proba_{H}_{fenster}_{szen[:2]}.csv"
            if os.path.exists(pfad):
                res = pd.read_csv(pfad, parse_dates=["date"])
            else:
                res = walk_forward(d, H, F, win, entry_lag=el, feat_lag=fl)
                res.to_csv(pfad, index=False)
            RES[(H, fenster, szen)] = res
            r = evaluate_run(d, H, F, res, el, fl)
            r.update({"Fenster": fenster, "Szenario": szen[:2],
                      "Basisrate": res["y_true"].mean(),
                      "Anteil_Kaufen_Signal": float((res["proba"] >= 0.5).mean())})
            rows.append(r)
    print(f"{H} fertig ({time.time() - t0:.0f}s)")
ALL11 = pd.DataFrame(rows)
ALL11.to_csv(f"{OUT}/db11_ergebnisse.csv", index=False)

# COMMAND ----------
# Zelle 5: Trefferquote. Holm-Korrektur über alle Zeilen (Serien x Szenarien). Die 8 Serien sind NICHT unabhängig.
T = ALL11.copy()
T["p_holm"] = holm(T["p_nicht_ueberl"].values)
print(T[["Szenario", "Serie", "n", "DA", "Wilson_lo", "Wilson_hi", "Boot_lo", "Boot_hi", "p_nicht_ueberl", "p_holm",
         "Mehrheit", "DA_NYMEX_Regel", "DA_Momentum", "Brier_Skill"]].round(3).to_string(index=False))
print("\nMittel Trefferquote je Szenario:")
print(T.groupby("Szenario")[["DA", "Mehrheit", "DA_NYMEX_Regel", "DA_Momentum"]].mean().round(3).to_string())

# COMMAND ----------
# Zelle 6: Precision, Recall und Konfusionsmatrix.
T["Lift_Precision"] = T["Precision"] - T["Basisrate"]          # wie viel besser als "irgendein Tag"
print(T[["Szenario", "Serie", "TP", "FP", "TN", "FN", "Precision", "Recall", "Basisrate",
         "Anteil_Kaufen_Signal", "Lift_Precision"]].round(3).to_string(index=False))
print("\nTP = Kaufen gesagt, Preis stieg | FP = Kaufen gesagt, Preis stieg nicht (zu früh gekauft)")
print("FN = Warten gesagt, Preis stieg (zu lange gewartet) | TN = Warten gesagt, Preis stieg nicht")
H0 = SERIEN_LAUF[0]; S0 = SZENARIEN[0]; Z = T[(T["Serie"] == H0) & (T["Szenario"] == S0[:2])].iloc[0]
print(f"\nKonfusionsmatrix {H0} ({S0}):")
print(pd.DataFrame([[int(Z.TP), int(Z.FN)], [int(Z.FP), int(Z.TN)]],
                   index=["Preis stieg", "Preis stieg nicht"], columns=["Modell: Kaufen", "Modell: Warten"]).T.to_string())

# COMMAND ----------
# Zelle 7: Kosten gegenüber "immer kaufen" (Euro pro 100 l und Tag; Ersparnis > 0 = Modell besser).
print(T[["Szenario", "Serie", "Regret_Modell", "Regret_immer_kaufen", "Ersparnis_vs_kaufen", "Ersp_lo", "Ersp_hi",
         "davon_zu_frueh_gekauft", "davon_zu_lange_gewartet"]].round(3).to_string(index=False))
print("\nSerien mit Boot_lo > 0.5:", int((T["Boot_lo"] > 0.5).sum()), "von", len(T),
      "| Serien mit Ersparnis-Untergrenze > 0:", int((T["Ersp_lo"] > 0).sum()), "von", len(T))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Interpretation (nach dem Lauf gemeinsam ausfüllen)
# MAGIC - Trefferquote und Konfidenzintervall: liegt die untere Grenze über 50 %?
# MAGIC - Precision gegenüber Basisrate: wie viel zuverlässiger ist ein Kaufsignal als ein beliebiger Tag?
# MAGIC - Recall: wie viele der echten Anstiege werden erkannt, und wie viele „Kaufen“-Signale sind Fehlalarme?
# MAGIC - Was zeigt das **nicht**? Gilt nur für das gewählte Szenario und diesen Zeitraum. Die 8 Serien sind abhängig. Kein neuer Zeitraum.
