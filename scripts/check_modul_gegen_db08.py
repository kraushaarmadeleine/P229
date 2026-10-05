"""Databricks-Check: Modul p229 muss die db_08-Ergebnisse reproduzieren.

Ausführung (Databricks-Notebook, eine Zelle pro Block), Repo muss im Workspace liegen:

    import sys; sys.path.insert(0, "/Workspace/Users/<dein-user>/P229")   # Pfad anpassen
    %pip install xgboost scikit-learn
    exec(open("/Workspace/Users/<dein-user>/P229/scripts/check_modul_gegen_db08.py").read())

Dauer ca. 3 Minuten (1 Serie, Test-Fenster, S0). Keine Zugangsdaten nötig.
"""
import numpy as np
import pandas as pd

from p229.config import LAST_ARGUS, SCEN, SERIES, TEST
from p229.evaluate import evaluate_run
from p229.features import build_features
from p229.leakage import leak_test
from p229.walk_forward import walk_forward

OUT = "/Volumes/dev_workspace/p229/p229_files/output"
H = "heizoel_sued"

df_gold = (spark.table("dev_workspace.p229.gold_df_features").toPandas()  # noqa: F821 (Databricks)
           .sort_values("date").reset_index(drop=True))
data = df_gold[df_gold["date"] <= LAST_ARGUS].reset_index(drop=True)

leak_test(data, H)
F = build_features(data, H)
el, fl = SCEN["S0 (Kauf zu Preis t)"]
res = walk_forward(data, H, F, TEST, entry_lag=el, feat_lag=fl)
neu = evaluate_run(data, H, F, res, el, fl)

alt = pd.read_csv(f"{OUT}/db08_alle_ergebnisse.csv")
alt = alt[(alt["Serie"] == H) & (alt["Fenster"] == "Test") & alt["Szenario"].str.startswith("S0")].iloc[0]

diff = {k: abs(neu[k] - alt[k]) for k in neu if k != "Serie"}
print("Größte Abweichung:", max(diff, key=diff.get), f"{max(diff.values()):.2e}")
assert max(diff.values()) < 1e-9, "Modul weicht von db_08 ab!"
print("OK: Modul reproduziert db_08 (", H, ", Test, S0 ).")
