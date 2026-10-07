import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from p229 import plots
from p229.config import SERIES
from p229.features import build_features, merkmal_beschreibung
from p229.neue_daten import kombiniere_argus, lade_neue_argus, signale_neue_tage, suche_neueste_api_csv
from p229.walk_forward import walk_forward

H = "heizoel_sued"


def _df():
    n = len(SERIES)
    return pd.DataFrame({"Serie": SERIES, "DA": np.linspace(0.58, 0.66, n), "Wilson_lo": np.linspace(0.52, 0.60, n),
                         "Wilson_hi": np.linspace(0.64, 0.72, n), "Mehrheit": 0.52, "DA_NYMEX_Regel": 0.58,
                         "TP": 80, "FP": 50, "FN": 60, "TN": 100, "Precision": 0.6, "Recall": 0.58, "Basisrate": 0.47})


def test_alle_grafiken_erzeugen_figuren():
    df = _df()
    for fig in (plots.konfusion_raster(df), plots.trefferquote_mit_baselines(df), plots.precision_recall(df),
                plots.kalibrierung(np.random.default_rng(0).uniform(0.3, 0.7, 2000), np.random.default_rng(1).integers(0, 2, 2000)),
                plots.abdeckung_kurve(pd.DataFrame({"Fenster": ["Entwicklung"] * 3 + ["Test"] * 3, "Abdeckung": [1, 0.8, 0.6] * 2,
                                                    "DA_entschieden": [0.6, 0.62, 0.64, 0.62, 0.64, 0.66]}))):
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_merkmalsbeschreibung_deckt_alle_merkmale_ab(d):
    F = build_features(d, H)
    for c in F.columns:
        if c == "date":
            continue
        text = merkmal_beschreibung(c, H)
        assert text != c, f"keine Beschreibung für {c}"


def _api_csv(pfad, daten):
    df = pd.DataFrame({"date": daten})
    rng = np.random.default_rng(0)
    for c in SERIES + ["ice_gasoil"]:
        df[c] = 100 + rng.normal(size=len(daten)).cumsum()
    df.to_csv(pfad, index=False)
    return df


def test_neue_daten_laden_und_kombinieren(tmp_path):
    neu_csv = tmp_path / "argus_api_2025-10-01_bis_2026-10-02.csv"
    daten = pd.bdate_range("2026-01-12", periods=15)
    df = _api_csv(neu_csv, daten)
    assert suche_neueste_api_csv(str(tmp_path)) == str(neu_csv)
    neu = lade_neue_argus(str(neu_csv))
    alt = neu.iloc[:6].copy()                      # Überlappung 6 Tage, Rest ist neu
    alt[SERIES[0]] += 0.0
    komb, b = kombiniere_argus(alt, neu)
    assert b["Tage_ueberlappung"] == 6 and b["Tage_neu_dazu"] == 9 and len(komb) == 15
    assert b["max_Abweichung_Ueberlappung"] == 0.0
    assert komb["date"].is_monotonic_increasing


def test_csv_ohne_spalten_gibt_klare_fehlermeldung(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"date": ["2026-01-01"], "heizoel_sued": [1.0]}).to_csv(p, index=False)
    try:
        lade_neue_argus(str(p))
        assert False
    except ValueError as e:
        assert "fehlen Spalten" in str(e)


def test_signale_fuer_tage_ohne_ergebnis(d):
    start, ende = d["date"].iloc[1650], d["date"].iloc[-1]
    res, F = signale_neue_tage(d, H, start, ende, 0.05)
    assert len(res) == len(d) - 1650
    assert res["y_true"].iloc[-3:].isna().all() and (~res["Ergebnis_bekannt"].iloc[-3:]).all()
    assert set(res["Signal"]) <= {"Kaufen", "Warten", "Beobachten"}
    # Gleiche Vorhersagen wie der normale Lauf auf den Tagen mit Ergebnis
    normal = walk_forward(d, H, F, (start, ende))
    assert np.allclose(res["proba"].values[: len(normal)], normal["proba"].values)
