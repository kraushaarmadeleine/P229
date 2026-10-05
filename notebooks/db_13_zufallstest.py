# Databricks notebook source
# MAGIC %md
# MAGIC # db_13: Zufallstest (Placebo-Test)
# MAGIC
# MAGIC **Frage:** Stammen die 60–65 % Trefferquote wirklich aus echtem Zusammenhang, oder steckt ein Fehler im Aufbau?
# MAGIC
# MAGIC **Idee:** Wir verschieben die Antworten („Preis stieg / stieg nicht“) um einige Hundert Tage gegenüber den Marktdaten. Damit ist jeder echte Zusammenhang zerstört, alles andere bleibt gleich (gleiche Daten, gleiches Modell, gleicher Ablauf, auch die zeitliche Struktur der Antworten bleibt erhalten).
# MAGIC
# MAGIC - **Erwartung bei sauberem Aufbau:** Trefferquote um 50 %. Dass einzelne Läufe zufällig bei 53–55 % liegen, ist normal.
# MAGIC - **Alarmzeichen:** Trefferquote deutlich über 55 % auch bei verschobenen Antworten. Dann sieht das Modell irgendwo Information, die es nicht haben dürfte.
# MAGIC
# MAGIC **Dauer:** Probelauf (2 Serien, 3 Verschiebungen plus echter Lauf) ca. 15 Minuten. Zwischenstände werden gesichert.
# MAGIC
# MAGIC Ablauf: Installation, dann Zelle 0 bis 4 der Reihe nach ausführen.

# COMMAND ----------
# MAGIC %pip install xgboost scikit-learn openpyxl

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# Zelle 1: Pfade, Code, Daten. (Zelle 0, die den Code anlegt, steht direkt über dieser Zelle.)
import os, sys, time
import numpy as np, pandas as pd

BASE_PATH = "/Volumes/dev_workspace/p229/p229_files"
OUT = f"{BASE_PATH}/output"
ARGUS_XLSX = f"{BASE_PATH}/Heizoel und Diesel bis 2016 und ICE Daten.xlsx"
MARKT_XLSX = f"{BASE_PATH}/P229 - Daten von Brent, WTI, Heizöl, Wechselkurs.xlsx"

sys.path.insert(0, "/tmp/p229_code")
from p229.config import SERIES, TEST, LAST_ARGUS
from p229.data import load_argus_excel, load_market_sheet, build_dataset
from p229.features import build_features
from p229.leakage import leak_test
from p229.walk_forward import walk_forward
from p229.evaluate import evaluate_run

argus = load_argus_excel(ARGUS_XLSX)
markt = [
    load_market_sheet(MARKT_XLSX, "Brent Rohöl", "brent"),
    load_market_sheet(MARKT_XLSX, "WIT Rohöl", "wti"),
    load_market_sheet(MARKT_XLSX, "Heizöl", "nymex_heating_oil"),
    load_market_sheet(MARKT_XLSX, "USD_EUR Wechselkurs", "usd_eur"),
]
d_all, _ = build_dataset(argus, markt)
d = d_all[d_all["date"] <= LAST_ARGUS].reset_index(drop=True)
for H in SERIES:
    leak_test(d, H)
print("Tabelle d:", d.shape, "| Leakage-Test bestanden")

# COMMAND ----------
# Zelle 2: Einstellungen. Für alle 8 Serien SERIEN_LAUF = list(SERIES) (dann ca. 1 Stunde).
SERIEN_LAUF = ["heizoel_sued", "diesel_sued"]
VERSCHIEBUNGEN = [90, 250, 500]        # Tage, um die die Antworten verschoben werden (0 = echt, läuft immer mit)
print("Serien:", SERIEN_LAUF, "| Verschiebungen:", VERSCHIEBUNGEN)

# COMMAND ----------
# Zelle 3: Läufe. Jeder Lauf wird als CSV gesichert und beim Neustart wiederverwendet.
rows = []
t0 = time.time()
for H in SERIEN_LAUF:
    F = build_features(d, H)
    for k in [0] + VERSCHIEBUNGEN:
        pfad = f"{OUT}/db13_proba_{H}_shift{k}.csv"
        if os.path.exists(pfad):
            res = pd.read_csv(pfad, parse_dates=["date"])
        else:
            res = walk_forward(d, H, F, TEST, entry_lag=0, feat_lag=0, label_shift=k)
            res.to_csv(pfad, index=False)
        r = evaluate_run(d, H, F, res, 0, 0)
        rows.append({"Serie": H, "Verschiebung": k, "Art": "echt" if k == 0 else "Placebo", "n": r["n"], "DA": r["DA"],
                     "Boot_lo": r["Boot_lo"], "Boot_hi": r["Boot_hi"], "p_nicht_ueberl": r["p_nicht_ueberl"],
                     "Basisrate": res["y_true"].mean()})
        print(f"{H} Verschiebung {k}: Trefferquote {r['DA']:.3f} ({time.time() - t0:.0f}s)")
Z = pd.DataFrame(rows)
Z.to_csv(f"{OUT}/db13_ergebnisse.csv", index=False)

# COMMAND ----------
# Zelle 4: Auswertung.
pd.set_option("display.width", 250)
print(Z.round(3).to_string(index=False))
echt = Z[Z["Art"] == "echt"]["DA"].mean()
pl = Z[Z["Art"] == "Placebo"]
print(f"\nEchtes Ziel:   mittlere Trefferquote {echt:.3f}")
print(f"Placebo-Läufe: mittlere Trefferquote {pl['DA'].mean():.3f} | niedrigste {pl['DA'].min():.3f} | höchste {pl['DA'].max():.3f}")
print(f"Placebo-Läufe mit Bootstrap-Untergrenze > 0.5: {int((pl['Boot_lo'] > 0.5).sum())} von {len(pl)}")
if pl["DA"].max() <= 0.57 and pl["DA"].mean() < 0.54:
    print("\nBefund: Mit zerstörtem Zusammenhang fällt die Trefferquote auf etwa 50 %. Das spricht gegen einen Fehler im Aufbau (z. B. Blick in die Zukunft).")
else:
    print("\nACHTUNG: Placebo-Läufe liegen deutlich über 50 %. Das kann ein Hinweis auf einen Fehler im Aufbau sein. Bitte Ergebnis an Claude schicken.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Was dieser Test zeigt und was nicht
# MAGIC - **Zeigt:** Das Modell erreicht die 60 % nur, wenn die echten Antworten dabei sind. Das spricht gegen versehentlichen Blick in die Zukunft oder einen Rechenfehler.
# MAGIC - **Zeigt nicht:** Dass das Modell auf neuen Tagen funktioniert oder dass der Einkäufer am selben Tag noch zum Argus-Preis kaufen kann.
# MAGIC - Mit nur 2 Serien und 3 Verschiebungen schwanken die Placebo-Werte zufällig um etwa 3 bis 5 Prozentpunkte.
