"""Konstanten aus db_08 (unverändert übernommen)."""
import pandas as pd

SERIES = ["heizoel_sued", "heizoel_suedwest", "heizoel_suedost", "heizoel_rheinmain",
          "diesel_sued", "diesel_suedwest", "diesel_suedost", "diesel_rheinmain"]
EXO = ["ice_gasoil", "brent", "wti", "nymex_heating_oil", "usd_eur"]
LAST_ARGUS = pd.Timestamp("2026-01-20")   # danach OMR + fortgeschriebener ICE-Preis
DEV = (pd.Timestamp("2023-04-21"), pd.Timestamp("2024-11-01"))
TEST = (pd.Timestamp("2024-11-15"), pd.Timestamp("2026-01-20"))
HORIZON = 3
REG = dict(max_depth=2, n_estimators=150, learning_rate=0.03, min_child_weight=10,
           reg_lambda=10, subsample=0.7, colsample_bytree=0.5,
           eval_metric="logloss", random_state=42)
# Szenario -> (entry_lag, feat_lag)
SCEN = {"S0 (Kauf zu Preis t)": (0, 0),
        "S1 (Kauf zu Preis t+1)": (1, 0),
        "S2 (Daten nur bis t-1)": (0, 1)}

SERIEN_NAMEN = {"heizoel_sued": "Heizöl Süd", "heizoel_suedwest": "Heizöl Südwest", "heizoel_suedost": "Heizöl Südost",
                "heizoel_rheinmain": "Heizöl Rhein-Main", "diesel_sued": "Diesel Süd", "diesel_suedwest": "Diesel Südwest",
                "diesel_suedost": "Diesel Südost", "diesel_rheinmain": "Diesel Rhein-Main"}
