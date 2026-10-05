# Databricks notebook source
# MAGIC %md
# MAGIC # db_09: Neuaufbau aus den Excel-Dateien
# MAGIC
# MAGIC **Frage:** Kann XGBoost aus den Excel-Daten (Argus + Marktdaten) die 3-Tage-Richtung des Heizöl- und Dieselpreises besser vorhersagen als Zufall und als einfache Regeln?
# MAGIC
# MAGIC **Ablauf (Zelle für Zelle ausführen, nach jeder Zelle die Ausgabe lesen):**
# MAGIC 1. Setup und Pfade
# MAGIC 2. Struktur der Marktdaten-Excel ansehen
# MAGIC 3. Daten laden und prüfen
# MAGIC 4. Abgleich mit der Gold-Tabelle (zeigt, ob der Neuaufbau identisch ist)
# MAGIC 5. Leakage-Test
# MAGIC 6. Schnelltest: eine Serie
# MAGIC 7. Alle 8 Serien (ca. 15 Minuten)
# MAGIC 8. Auswertung mit Holm-Korrektur
# MAGIC
# MAGIC Der Code liegt im Repo-Ordner `p229/` und ist mit `pytest` getestet. Keine Zugangsdaten nötig.

# COMMAND ----------
# MAGIC %pip install xgboost scikit-learn openpyxl

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# Zelle 1: Pfade. REPO_PATH anpassen: Ordner, in dem p229/ liegt (Git-Ordner in Databricks).
import sys, time
import numpy as np, pandas as pd

REPO_PATH = "/Workspace/Users/<DEIN-USER>/P229"          # <-- anpassen
sys.path.insert(0, REPO_PATH)

BASE_PATH = "/Volumes/dev_workspace/p229/p229_files"
OUT = f"{BASE_PATH}/output"
ARGUS_XLSX = f"{BASE_PATH}/Heizoel und Diesel bis 2016 und ICE Daten.xlsx"
MARKT_XLSX = f"{BASE_PATH}/P229 - Daten von Brent, WTI, Heizöl, Wechselkurs.xlsx"

from p229.config import SERIES, EXO, DEV, TEST, SCEN, LAST_ARGUS
from p229.data import inspect_excel, load_argus_excel, load_market_sheet, build_dataset
from p229.features import build_features
from p229.leakage import leak_test
from p229.walk_forward import walk_forward
from p229.evaluate import evaluate_run
from p229.stats import holm
print("Import ok")

# COMMAND ----------
# Zelle 2: Struktur der Marktdaten-Excel ansehen (nur lesen, nichts ändern).
# Bitte die Ausgabe kopieren und mir schicken, falls Zelle 3 meckert.
inspect_excel(MARKT_XLSX)

# COMMAND ----------
# Zelle 3: Daten laden. Blattnamen wie im Projektdokument; falls sie anders heißen, hier anpassen.
argus = load_argus_excel(ARGUS_XLSX)
markt = [
    load_market_sheet(MARKT_XLSX, "Brent Rohöl", "brent"),
    load_market_sheet(MARKT_XLSX, "WIT Rohöl", "wti"),
    load_market_sheet(MARKT_XLSX, "Heizöl", "nymex_heating_oil"),          # ANNAHME: Blatt "Heizöl" = nymex_heating_oil
    load_market_sheet(MARKT_XLSX, "USD_EUR Wechselkurs", "usd_eur"),
]
for m in markt:
    print(m.columns[1], len(m), m["date"].min().date(), "bis", m["date"].max().date(),
          "| Min/Max:", round(m.iloc[:, 1].min(), 3), round(m.iloc[:, 1].max(), 3))
d_all, bericht = build_dataset(argus, markt)
print("\nFehlende Werte vor/nach Vorwärtsfüllen (max. 5 Tage):")
print(bericht.to_string())
d = d_all[d_all["date"] <= LAST_ARGUS].reset_index(drop=True)      # Argus-Phase wie in db_08
print("\nTabelle d:", d.shape, d["date"].min().date(), "bis", d["date"].max().date())

# COMMAND ----------
# Zelle 4: Abgleich mit der Gold-Tabelle. Erwartung: Abweichung 0 (oder erklärbar).
g = (spark.table("dev_workspace.p229.gold_df_features").select("date", *(SERIES + EXO)).toPandas()
     .assign(date=lambda x: pd.to_datetime(x["date"])).set_index("date").sort_index())
n = d.set_index("date")
gem = n.index.intersection(g.index)
print(f"Zeilen neu: {len(n)} | Gold: {len(g[g.index <= LAST_ARGUS])} | gemeinsame Tage: {len(gem)}")
print("Tage nur neu:", len(n.index.difference(g.index)), "| nur Gold:", len(g[g.index <= LAST_ARGUS].index.difference(n.index)))
diff = (n.loc[gem, SERIES + EXO] - g.loc[gem, SERIES + EXO]).abs()
print("\nGrößte Abweichung je Spalte:")
print(diff.max().round(5).to_string())
print("\nTage mit Abweichung > 0.001:")
print((diff > 0.001).sum().to_string())

# COMMAND ----------
# Zelle 5: Leakage-Test für alle Serien (Features dürfen die Zukunft nicht kennen).
for H in SERIES:
    leak_test(d, H)
print("Leakage-Test bestanden für", len(SERIES), "Serien")

# COMMAND ----------
# Zelle 6: Schnelltest, eine Serie, Szenario S0, Testfenster (ca. 1-2 Minuten).
H = "heizoel_sued"
F = build_features(d, H)
el, fl = SCEN["S0 (Kauf zu Preis t)"]
t0 = time.time()
res = walk_forward(d, H, F, TEST, entry_lag=el, feat_lag=fl)
r = evaluate_run(d, H, F, res, el, fl)
print(f"Dauer {time.time()-t0:.0f}s | n={r['n']}")
print(f"Trefferquote {r['DA']:.3f} | Wilson [{r['Wilson_lo']:.3f}, {r['Wilson_hi']:.3f}] | Bootstrap [{r['Boot_lo']:.3f}, {r['Boot_hi']:.3f}]")
print(f"p (nicht überlappend, max über 3 Offsets) {r['p_nicht_ueberl']:.3f} | Mehrheitsklasse {r['Mehrheit']:.3f}")
print(f"Baselines: NYMEX-Regel {r['DA_NYMEX_Regel']:.3f} | Momentum {r['DA_Momentum']:.3f}")
print(f"Konfusion: TP {r['TP']} FP {r['FP']} TN {r['TN']} FN {r['FN']} | Precision {r['Precision']:.3f} Recall {r['Recall']:.3f}")
print(f"Brier-Skill {r['Brier_Skill']:.3f}")
print(f"Ersparnis gegenüber 'immer kaufen' pro Tag {r['Ersparnis_vs_kaufen']:.3f} [{r['Ersp_lo']:.3f}, {r['Ersp_hi']:.3f}]")

# COMMAND ----------
# Zelle 7: Alle 8 Serien, S0, Test (ca. 15 Minuten). Ergebnisse werden pro Serie gesichert.
rows, RES = [], {}
t0 = time.time()
for H in SERIES:
    F = build_features(d, H)
    res = walk_forward(d, H, F, TEST, entry_lag=el, feat_lag=fl)
    RES[H] = res
    res.to_csv(f"{OUT}/db09_proba_{H}_Test_S0.csv", index=False)
    rows.append(evaluate_run(d, H, F, res, el, fl))
    print(f"{H} fertig ({time.time()-t0:.0f}s)")
ALL9 = pd.DataFrame(rows)
ALL9.to_csv(f"{OUT}/db09_ergebnisse_Test_S0.csv", index=False)

# COMMAND ----------
# Zelle 8: Auswertung. Holm über die 8 Serien. Die 8 Serien sind NICHT unabhängig.
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
T = ALL9.copy()
T["p_holm"] = holm(T["p_nicht_ueberl"].values)
T["p_holm_NYMEX"] = holm(T["p_McNemar_NYMEX"].values)
print(T[["Serie", "n", "DA", "Wilson_lo", "Boot_lo", "Boot_hi", "p_nicht_ueberl", "p_holm",
         "DA_NYMEX_Regel", "DA_Momentum", "p_holm_NYMEX", "Brier_Skill"]].round(3).to_string(index=False))
print("\nKosten gegenüber 'immer kaufen' (Ersparnis > 0 = Modell besser):")
print(T[["Serie", "Regret_Modell", "Regret_immer_kaufen", "Ersparnis_vs_kaufen", "Ersp_lo", "Ersp_hi"]].round(3).to_string(index=False))
print("\nSerien mit Bootstrap-Untergrenze > 0.5:", int((T["Boot_lo"] > 0.5).sum()), "von", len(T))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Interpretation (nach Lauf ausfüllen)
# MAGIC - Was zeigen die Ergebnisse? (Trefferquote, Bootstrap, Holm, Baselines)
# MAGIC - Was zeigen sie **nicht**? (Szenario S0 setzt voraus, dass zum Argus-Preis von t noch gekauft werden kann. Kein neuer Zeitraum. 8 Serien sind abhängig.)
# MAGIC - Empfehlung erst nach dem Holdout-Test (siehe `docs/PRAEREGISTRIERUNG_holdout_A.md`).
