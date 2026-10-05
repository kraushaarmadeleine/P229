import numpy as np
import pandas as pd

from p229.evaluate import evaluate_run
from p229.features import build_features
from p229.walk_forward import walk_forward

H = "heizoel_sued"
SMALL = dict(max_depth=2, n_estimators=20, learning_rate=0.1, eval_metric="logloss", random_state=42)


def test_evaluate_run_konsistenz(d):
    F = build_features(d, H)
    window = (d["date"].iloc[1400], d["date"].iloc[1600])
    res = walk_forward(d, H, F, window, params=SMALL)
    out = evaluate_run(d, H, F, res, 0, 0)
    n = out["n"]
    assert n == len(res)
    assert out["TP"] + out["FP"] + out["TN"] + out["FN"] == n
    assert np.isclose(out["DA"], (out["TP"] + out["TN"]) / n)
    assert out["Wilson_lo"] < out["DA"] < out["Wilson_hi"]
    # Regret: Modell + Ersparnis = immer kaufen
    assert np.isclose(out["Regret_Modell"] + out["Ersparnis_vs_kaufen"], out["Regret_immer_kaufen"])
    assert np.isclose(out["davon_zu_frueh_gekauft"] + out["davon_zu_lange_gewartet"], out["Regret_Modell"])


def test_synthetischer_lag_effekt_wird_gefunden(d):
    # In den Testdaten folgt Heizöl dem Markt mit 1 Tag Verzögerung -> S0 muss deutlich über 50 % liegen.
    F = build_features(d, H)
    window = (d["date"].iloc[1400], d["date"].iloc[1600])
    res = walk_forward(d, H, F, window, params=SMALL)
    assert evaluate_run(d, H, F, res, 0, 0)["DA"] > 0.6
