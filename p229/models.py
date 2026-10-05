"""Modelle für den Vergleich (db_10). Alle bekommen dieselben Daten, denselben Walk-Forward.

Imputation (Median) und Skalierung sind Teil des Modells und werden bei jedem Training nur auf den
Trainingsdaten gelernt (kein Look-ahead).
"""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import REG


def make_xgb():
    return XGBClassifier(**REG)


def make_logreg():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=1000))


def make_rf():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=20,
                                                max_features="sqrt", random_state=42, n_jobs=-1))


def make_majority():
    return DummyClassifier(strategy="prior")          # immer die im Training häufigere Klasse


def klein_features(H):
    """Kleines Modell: nur die wichtigsten Tagesrenditen und Nachlauf-Lücken."""
    return [f"{H}_r1", "ice_gasoil_r1", "nymex_heating_oil_r1", "brent_r1", "wti_r1",
            "gap1_ice_gasoil", "gap1_nymex_heating_oil"]


# Name -> (Fabrik, Merkmale: None = alle 44, "klein" = kleine Auswahl)
MODELLE = {
    "XGBoost": (make_xgb, None),
    "Logistische Regression": (make_logreg, None),
    "Random Forest": (make_rf, None),
    "Logistisch klein (7 Merkmale)": (make_logreg, "klein"),
    "Mehrheitsklasse": (make_majority, None),
}
