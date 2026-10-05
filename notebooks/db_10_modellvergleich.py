# Databricks notebook source
# MAGIC %md
# MAGIC # db_10: Welches Modell ist am besten?
# MAGIC
# MAGIC **Frage:** Welches Modell sagt aus den Excel-Daten die 3-Tage-Richtung des Heizöl- und Dieselpreises am besten voraus, und ist das Ergebnis wirklich besser als eine einfache Regel?
# MAGIC
# MAGIC **Fairer Vergleich:** Alle Modelle bekommen dieselben Daten und dieselben Merkmale. Sie werden jeden Tag neu mit den dann bekannten Daten trainiert (kein Blick in die Zukunft). Dasselbe Szenario wie vorher (S0: Kauf zum Preis von Tag t).
# MAGIC
# MAGIC **Modelle:** XGBoost (bisher), Logistische Regression, Random Forest, Logistisch klein (nur 7 Merkmale), Mehrheitsklasse (Dummy: tippt immer die häufigere Richtung).
# MAGIC **Einfache Regeln zum Vergleich:** NYMEX-Regel (US-Preis gestiegen = Kaufen) und Momentum (gestern gestiegen = Kaufen).
# MAGIC
# MAGIC **Zwei Zeiträume:** Entwicklung (2023-04-21 bis 2024-11-01) und Test (2024-11-15 bis 2026-01-20). Die Rangfolge gilt nur dann als belastbar, wenn sie in **beiden** Zeiträumen ähnlich ist.
# MAGIC
# MAGIC **Ehrlicher Vorbehalt vorab:** Mit etwa 300 Tagen und überlappenden 3-Tage-Labels sind kleine Unterschiede zwischen Modellen Zufall. Ein klarer Sieger ist nur zu erwarten, wenn er deutlich vorne liegt.
# MAGIC
# MAGIC Ablauf: Installation, Zelle 0 bis 7 der Reihe nach ausführen. Zelle 4 dauert lang (bis zu 1 Stunde, der Random Forest ist der langsamste), sie speichert Zwischenstände und kann nach einem Abbruch einfach neu gestartet werden.

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
from p229.config import SERIES, EXO, DEV, TEST, LAST_ARGUS
from p229.data import load_argus_excel, load_market_sheet, build_dataset
from p229.features import build_features
from p229.leakage import leak_test
from p229.walk_forward import walk_forward
from p229.evaluate import evaluate_run
from p229.models import MODELLE, klein_features
from p229.stats import holm, mcnemar

argus = load_argus_excel(ARGUS_XLSX)
markt = [
    load_market_sheet(MARKT_XLSX, "Brent Rohöl", "brent"),
    load_market_sheet(MARKT_XLSX, "WIT Rohöl", "wti"),
    load_market_sheet(MARKT_XLSX, "Heizöl", "nymex_heating_oil"),
    load_market_sheet(MARKT_XLSX, "USD_EUR Wechselkurs", "usd_eur"),
]
d_all, _ = build_dataset(argus, markt)
d = d_all[d_all["date"] <= LAST_ARGUS].reset_index(drop=True)
print("Tabelle d:", d.shape, d["date"].min().date(), "bis", d["date"].max().date())
for H in SERIES:
    leak_test(d, H)
print("Leakage-Test bestanden")

# COMMAND ----------
# Zelle 2: Welche Serien und Modelle laufen? Für einen schnellen Probelauf (ca. 10 Minuten)
# hier zwei Serien eintragen, z. B. SERIEN_LAUF = ["heizoel_sued", "diesel_sued"].
SERIEN_LAUF = list(SERIES)
FENSTER = {"Entwicklung": DEV, "Test": TEST}
SZENARIO = (0, 0)          # S0: Kauf zu Preis t, Daten bis t
print("Serien:", SERIEN_LAUF)
print("Modelle:", list(MODELLE))

# COMMAND ----------
# Zelle 3: Hilfsfunktion. Jeder Lauf wird als CSV gesichert und beim Neustart wiederverwendet.
def lauf(modell, H, fenster):
    fabrik, modus = MODELLE[modell]
    F_voll = build_features(d, H)
    F_mod = F_voll if modus is None else F_voll[["date"] + klein_features(H)]
    sicher = modell.replace(" ", "_").replace("(", "").replace(")", "")
    pfad = f"{OUT}/db10_proba_{sicher}_{H}_{fenster}.csv"
    if os.path.exists(pfad):
        res = pd.read_csv(pfad, parse_dates=["date"])
    else:
        res = walk_forward(d, H, F_mod, FENSTER[fenster], entry_lag=SZENARIO[0], feat_lag=SZENARIO[1],
                           model_factory=fabrik)
        res.to_csv(pfad, index=False)
    return res, evaluate_run(d, H, F_voll, res, SZENARIO[0], SZENARIO[1])

# COMMAND ----------
# Zelle 4: Alle Modelle, alle Serien, beide Zeiträume (bis zu 1 Stunde für alles).
rows, RES = [], {}
t0 = time.time()
for modell in MODELLE:
    for H in SERIEN_LAUF:
        for fenster in FENSTER:
            res, r = lauf(modell, H, fenster)
            RES[(modell, H, fenster)] = res
            rows.append({"Modell": modell, "Fenster": fenster, **r})
    print(f"{modell} fertig ({time.time() - t0:.0f}s)")
ALL10 = pd.DataFrame(rows)
ALL10.to_csv(f"{OUT}/db10_ergebnisse.csv", index=False)

# COMMAND ----------
# Zelle 5: Rangliste je Zeitraum. Durchschnitt über die Serien (die Serien sind NICHT unabhängig).
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
regeln = (ALL10[ALL10["Modell"] == "XGBoost"].groupby("Fenster")[["DA_NYMEX_Regel", "DA_Momentum"]].mean())
for fenster in FENSTER:
    g = ALL10[ALL10["Fenster"] == fenster].groupby("Modell").agg(
        Trefferquote=("DA", "mean"), Boot_lo_Mittel=("Boot_lo", "mean"),
        Serien_Boot_lo_ueber_50=("Boot_lo", lambda x: int((x > 0.5).sum())),
        Brier_Skill=("Brier_Skill", "mean"), Ersparnis_pro_Tag=("Ersparnis_vs_kaufen", "mean"),
        Serien_Ersparnis_ueber_0=("Ersp_lo", lambda x: int((x > 0).sum())))
    g = g.sort_values("Trefferquote", ascending=False)
    print(f"\n=== {fenster}: Rangliste (Mittel über {len(SERIEN_LAUF)} Serien) ===")
    print(g.round(3).to_string())
    print("Einfache Regeln, gleiche Tage: NYMEX-Regel", round(regeln.loc[fenster, "DA_NYMEX_Regel"], 3),
          "| Momentum", round(regeln.loc[fenster, "DA_Momentum"], 3))

# COMMAND ----------
# Zelle 6: Ist ein Modell wirklich besser als XGBoost? Paarvergleich (McNemar) je Serie, Test-Zeitraum.
# Holm-Korrektur über alle Vergleiche. Erwartung: kaum etwas ist signifikant.
vgl = []
for modell in MODELLE:
    if modell == "XGBoost":
        continue
    for H in SERIEN_LAUF:
        a, b = RES[("XGBoost", H, "Test")], RES[(modell, H, "Test")]
        ok_x = ((a["proba"].values >= 0.5).astype(int) == a["y_true"].values)
        ok_m = ((b["proba"].values >= 0.5).astype(int) == b["y_true"].values)
        assert (a["date"].values == b["date"].values).all()
        vgl.append({"Modell": modell, "Serie": H, "DA_Modell": ok_m.mean(), "DA_XGBoost": ok_x.mean(),
                    "Differenz": ok_m.mean() - ok_x.mean(), "p_McNemar": mcnemar(ok_m, ok_x)})
V = pd.DataFrame(vgl)
V["p_holm"] = holm(V["p_McNemar"].values)
print(V.groupby("Modell").agg(Differenz_Mittel=("Differenz", "mean"), kleinster_p=("p_McNemar", "min"),
                              kleinster_p_holm=("p_holm", "min")).round(3).to_string())
print("\nVergleiche mit p_holm < 0.05:", int((V["p_holm"] < 0.05).sum()), "von", len(V))

# COMMAND ----------
# Zelle 7: Einzelwerte je Serie für das beste Modell nach Entwicklungszeitraum (Auswahl OHNE Test-Blick).
dev_rang = ALL10[ALL10["Fenster"] == "Entwicklung"].groupby("Modell")["DA"].mean().sort_values(ascending=False)
bestes = dev_rang.index[0]
print("Bestes Modell nach Entwicklungszeitraum:", bestes, round(dev_rang.iloc[0], 3))
print("\nSein Ergebnis im Test-Zeitraum (hier erst angesehen):")
print(ALL10[(ALL10["Modell"] == bestes) & (ALL10["Fenster"] == "Test")][
    ["Serie", "n", "DA", "Wilson_lo", "Boot_lo", "Boot_hi", "p_nicht_ueberl", "DA_NYMEX_Regel", "Brier_Skill",
     "Ersparnis_vs_kaufen", "Ersp_lo"]].round(3).to_string(index=False))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Interpretation (nach dem Lauf gemeinsam ausfüllen)
# MAGIC - Welches Modell liegt in **beiden** Zeiträumen vorn? Wenn die Reihenfolge wechselt, gibt es keinen klaren Sieger.
# MAGIC - Ist der Vorsprung gegenüber XGBoost und gegenüber der NYMEX-Regel größer als der Zufall? (Zelle 6, Holm)
# MAGIC - Was zeigt das Ergebnis **nicht**? Gilt nur für S0 (Kauf zum Preis von t), nur für diese Zeiträume, die 8 Serien sind abhängig.
# MAGIC - Auswahl des Modells erst nach dem Holdout-Test auf neuen Tagen.
