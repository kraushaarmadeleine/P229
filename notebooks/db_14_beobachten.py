# Databricks notebook source
# MAGIC %md
# MAGIC # db_14: Dritte Stufe „Beobachten“
# MAGIC
# MAGIC **Frage:** Wird das Signal besser, wenn das Modell bei unsicheren Tagen weder „Kaufen“ noch „Warten“ sagt, sondern „Beobachten“?
# MAGIC
# MAGIC **Regel (vorab festgelegt, XGBoost, Szenario S0):**
# MAGIC - **Kaufen**, wenn die Wahrscheinlichkeit p ≥ 0,5 + δ ist.
# MAGIC - **Warten**, wenn p < 0,5 − δ ist.
# MAGIC - **Beobachten** dazwischen (Modell ist unsicher, keine Entscheidung).
# MAGIC - δ = 0 ist das bisherige Signal (nie Beobachten).
# MAGIC
# MAGIC **Wie δ gewählt wird (ohne Test-Blick):** Wir legen fest, welchen Anteil der unsichersten Tage wir „Beobachten“ nennen (z. B. 30 %). Daraus wird δ **nur aus dem Entwicklungszeitraum** berechnet. Das Ergebnis wird danach **einmal** im Testzeitraum geprüft. Aus mehreren Anteilen wählt die Regel automatisch den mit der besten Trefferquote auf Entwicklung, wobei höchstens 40 % Beobachten erlaubt sind (sonst bleibt zu wenig übrig).
# MAGIC
# MAGIC **Kosten:** „Beobachten“ wird zweimal gerechnet: Variante A = wie Warten (später entscheiden), Variante B = wie Kaufen.
# MAGIC
# MAGIC **Voraussetzung:** Die Vorhersagen aus db_10 (`db10_proba_XGBoost_*.csv`) liegen im Output-Ordner. Fehlen sie, rechnet das Notebook sie neu (ca. 100 Sekunden je Lauf).
# MAGIC
# MAGIC Ablauf: Installation, dann Zelle 0 bis 6 der Reihe nach ausführen. Dauer: wenige Minuten.

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
from p229.config import SERIES, DEV, TEST, LAST_ARGUS
from p229.data import load_argus_excel, load_market_sheet, build_dataset
from p229.features import build_features
from p229.leakage import leak_test
from p229.walk_forward import walk_forward
from p229.beobachten import bewerte_dreistufig, signal, NAMEN

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
# Zelle 2: Einstellungen.
SERIEN_LAUF = list(SERIES)
FENSTER = {"Entwicklung": DEV, "Test": TEST}
Q_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]       # Anteil der unsichersten Tage, der "Beobachten" heißen soll
Q_ERLAUBT = [0.2, 0.3, 0.4]                   # aus diesen wählt die Regel (auf Entwicklung)
print("Serien:", len(SERIEN_LAUF), "| Anteile:", Q_GRID, "| erlaubt für die Auswahl:", Q_ERLAUBT)

# COMMAND ----------
# Zelle 3: Vorhersagen laden (aus db_10) oder neu rechnen.
PROBA = {}
for H in SERIEN_LAUF:
    for fenster, win in FENSTER.items():
        pfad = f"{OUT}/db10_proba_XGBoost_{H}_{fenster}.csv"
        if os.path.exists(pfad):
            PROBA[(H, fenster)] = pd.read_csv(pfad, parse_dates=["date"])
        else:
            print("Rechne neu:", H, fenster)
            res = walk_forward(d, H, build_features(d, H), win, entry_lag=0, feat_lag=0)
            res.to_csv(pfad, index=False)
            PROBA[(H, fenster)] = res
print("Geladen:", len(PROBA), "Läufe. n je Lauf:", sorted({len(v) for v in PROBA.values()}))
p_dev = np.concatenate([PROBA[(H, "Entwicklung")]["proba"].values for H in SERIEN_LAUF])
print("Wahrscheinlichkeiten Entwicklung: Spanne", np.round(np.quantile(p_dev, [0.05, 0.25, 0.5, 0.75, 0.95]), 3))

# COMMAND ----------
# Zelle 4: δ aus der Entwicklung ableiten und alle Anteile in beiden Zeiträumen auswerten.
absdev = np.abs(p_dev - 0.5)
zeilen = []
for q in Q_GRID:
    delta = float(np.quantile(absdev, q))            # nur Entwicklungsdaten
    for fenster in FENSTER:
        r = pd.DataFrame([bewerte_dreistufig(d, H, PROBA[(H, fenster)], delta) for H in SERIEN_LAUF])
        zeilen.append({"Anteil_Beobachten_Entw": q, "delta": round(delta, 4), "Fenster": fenster,
                       "Abdeckung": r["Abdeckung"].mean(), "DA_entschieden": r["DA_entschieden"].mean(),
                       "Precision_Kaufen": r["Precision_Kaufen"].mean(), "Precision_Warten": r["Precision_Warten"].mean(),
                       "Ersparnis_A": r["Ersparnis_A"].mean(), "Ersparnis_B": r["Ersparnis_B"].mean(),
                       "Serien_Ersp_A_lo_ueber_0": int((r["Ersp_A_lo"] > 0).sum())})
T = pd.DataFrame(zeilen)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
for fenster in FENSTER:
    print(f"\n=== {fenster}: Mittel über {len(SERIEN_LAUF)} Serien ===")
    print(T[T["Fenster"] == fenster].drop(columns="Fenster").round(3).to_string(index=False))
print("\nLesehilfe: Abdeckung = Anteil der Tage mit Kaufen oder Warten. DA_entschieden = Trefferquote nur auf diesen Tagen.")
print("Ersparnis in Euro pro 100 l und Tag gegenüber 'immer kaufen' (A: Beobachten = Warten, B: Beobachten = Kaufen).")

# COMMAND ----------
# Zelle 5: Auswahl auf der Entwicklung, einmalige Prüfung im Test.
dev = T[(T["Fenster"] == "Entwicklung") & (T["Anteil_Beobachten_Entw"].isin(Q_ERLAUBT))]
q_star = float(dev.sort_values("DA_entschieden", ascending=False)["Anteil_Beobachten_Entw"].iloc[0])
delta_star = float(T[(T["Anteil_Beobachten_Entw"] == q_star)]["delta"].iloc[0])
print(f"Auf der Entwicklung gewählt: {q_star:.0%} Beobachten (δ = {delta_star:.3f})")
basis = {f: T[(T["Anteil_Beobachten_Entw"] == 0.0) & (T["Fenster"] == f)].iloc[0] for f in FENSTER}
wahl = {f: T[(T["Anteil_Beobachten_Entw"] == q_star) & (T["Fenster"] == f)].iloc[0] for f in FENSTER}
for f in FENSTER:
    print(f"{f}: Trefferquote {basis[f].DA_entschieden:.3f} (immer entscheiden) -> {wahl[f].DA_entschieden:.3f} (nur sichere Tage, "
          f"Abdeckung {wahl[f].Abdeckung:.0%})")

print("\n=== Test, je Serie, gewählte Regel ===")
R = pd.DataFrame([bewerte_dreistufig(d, H, PROBA[(H, "Test")], delta_star) for H in SERIEN_LAUF])
R0 = pd.DataFrame([bewerte_dreistufig(d, H, PROBA[(H, "Test")], 0.0) for H in SERIEN_LAUF])
R["DA_ohne_Beobachten"] = R0["DA_entschieden"].values
R["Lift"] = R["DA_entschieden"] - R["DA_ohne_Beobachten"]
print(R[["Serie", "n", "n_Kaufen", "n_Warten", "n_Beobachten", "Abdeckung", "DA_entschieden", "Boot_lo", "Boot_hi",
         "DA_ohne_Beobachten", "Lift", "Precision_Kaufen", "Precision_Warten", "Ersparnis_A", "Ersp_A_lo", "Ersparnis_B",
         "Ersp_B_lo"]].round(3).to_string(index=False))
print(f"\nSerien mit höherer Trefferquote als ohne Beobachten: {int((R['Lift'] > 0).sum())} von {len(R)} "
      f"| mittlerer Zugewinn {R['Lift'].mean()*100:.1f} Prozentpunkte")
R.to_csv(f"{OUT}/db14_ergebnisse_test.csv", index=False)
T.to_csv(f"{OUT}/db14_alle_anteile.csv", index=False)

# COMMAND ----------
# Zelle 6: So könnte die tägliche Ausgabe für den Einkäufer aussehen (letzte 10 Handelstage im Testzeitraum, Heizöl).
REGION = {"heizoel_sued": "Süd", "heizoel_suedwest": "Südwest", "heizoel_suedost": "Südost", "heizoel_rheinmain": "Rhein-Main"}
ZEILEN = []
for H in [s for s in SERIEN_LAUF if s.startswith("heizoel")]:
    res = PROBA[(H, "Test")].tail(10)
    sig = signal(res["proba"].values, delta_star)
    for dt, p, s in zip(res["date"], res["proba"], sig):
        ZEILEN.append({"Datum": dt.date(), "Region": "Heizöl " + REGION[H], "Wahrscheinlichkeit_steigt": round(float(p), 3),
                       "Signal": NAMEN[int(s)]})
print(pd.DataFrame(ZEILEN).pivot(index="Datum", columns="Region", values="Signal").to_string())
print("\nHinweis: Beispiel aus dem alten Testzeitraum, kein aktuelles Signal.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Interpretation (nach dem Lauf gemeinsam ausfüllen)
# MAGIC - Steigt die Trefferquote auf den sicheren Tagen **im Test**, nicht nur auf der Entwicklung? Wenn nur auf der Entwicklung, ist es Zufall.
# MAGIC - Wie viel Prozent der Tage bleiben als Beobachten übrig, und ist das für den Einkäufer tragbar?
# MAGIC - Sinken die Kosten gegenüber „immer kaufen“ (Variante A und B)?
# MAGIC - Was zeigt das **nicht**? Gilt für S0 und diesen Zeitraum. Auf den sicheren Tagen sind es weniger Tage (etwa 200), also größere Unsicherheit. Kein neuer Zeitraum.
