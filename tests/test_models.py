import numpy as np
import pytest

from p229.config import EXO
from p229.features import build_features
from p229.models import MODELLE, klein_features
from p229.walk_forward import walk_forward

H = "heizoel_sued"


def _F(d, mode):
    F = build_features(d, H)
    return F if mode is None else F[["date"] + klein_features(H)]


@pytest.mark.parametrize("name", list(MODELLE))
def test_modell_haengt_nicht_von_zukunft_ab(d, name):
    fabrik, mode = MODELLE[name]
    window = (d["date"].iloc[1300], d["date"].iloc[1330])
    kw = dict(retrain_every=1, model_factory=fabrik)
    base = walk_forward(d, H, _F(d, mode), window, **kw)
    cut = 1315
    d2 = d.copy()
    d2.loc[cut + 1:, [H] + EXO] = d2.loc[cut + 1:, [H] + EXO] * 2.3
    alt = walk_forward(d2, H, _F(d2, mode), window, **kw)
    b = base[base["date"] <= d["date"].iloc[cut]].reset_index(drop=True)
    a = alt[alt["date"] <= d["date"].iloc[cut]].reset_index(drop=True)
    assert len(a) == len(b) > 0
    assert np.allclose(a["proba"].values, b["proba"].values)
    assert a["proba"].between(0, 1).all()


def test_standard_ohne_fabrik_ist_unveraendert(d):
    from p229.models import make_xgb
    window = (d["date"].iloc[1400], d["date"].iloc[1420])
    F = build_features(d, H)
    p = dict(max_depth=2, n_estimators=10, learning_rate=0.1, eval_metric="logloss", random_state=42)
    a = walk_forward(d, H, F, window, params=p)
    b = walk_forward(d, H, F, window, model_factory=lambda: __import__("xgboost").XGBClassifier(**p))
    assert np.allclose(a["proba"], b["proba"])
