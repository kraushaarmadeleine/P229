"""Leakage-Tests: Zukunftsdaten dürfen weder Features noch Modellvorhersagen beeinflussen."""
import numpy as np
import pytest

from p229.config import EXO, SERIES
from p229.features import build_features, london_features
from p229.leakage import leak_test
from p229.walk_forward import walk_forward

H = "heizoel_sued"
SMALL = dict(max_depth=2, n_estimators=10, learning_rate=0.1, eval_metric="logloss", random_state=42)


def test_leak_test_laeuft_fuer_alle_serien(d):
    for s in SERIES:
        leak_test(d, s)


def test_features_aendern_sich_nicht_bei_zukunftsaenderung(d):
    cut = 1000
    d2 = d.copy()
    cols = [H] + EXO
    d2.loc[cut + 1:, cols] = d2.loc[cut + 1:, cols] * 3.7        # Zukunft verändern
    a = build_features(d, H).iloc[:cut + 1].drop(columns="date").values
    b = build_features(d2, H).iloc[:cut + 1].drop(columns="date").values
    assert np.allclose(a, b, equal_nan=True)


def test_leak_test_erkennt_absichtliches_leck(d, monkeypatch):
    # Kontrolle: ein Feature mit Blick in die Zukunft muss den Test auslösen.
    import p229.leakage as lk
    orig = lk.build_features

    def leaky(df, H_):
        F = orig(df, H_)
        F["zukunft"] = np.log(df[H_]).shift(-1) - np.log(df[H_])
        return F

    monkeypatch.setattr(lk, "build_features", leaky)
    with pytest.raises(AssertionError):
        lk.leak_test(d, H)


def test_london_features_ohne_nymex_wti_brent(d):
    cols = london_features(H, build_features(d, H)).columns
    assert not any(("nymex" in c) or ("wti" in c) or ("brent" in c) for c in cols)


@pytest.mark.parametrize("entry_lag,feat_lag", [(0, 0), (1, 0), (0, 1)])
def test_walk_forward_vorhersage_haengt_nicht_von_zukunft_ab(d, entry_lag, feat_lag):
    """Preise nach Tag i ändern -> proba bis Tag i bleibt identisch (Szenarien S0/S1/S2)."""
    F = build_features(d, H)
    window = (d["date"].iloc[1300], d["date"].iloc[1330])
    kw = dict(entry_lag=entry_lag, feat_lag=feat_lag, retrain_every=1, params=SMALL)
    base = walk_forward(d, H, F, window, **kw)

    m = 15                                                       # ab Fensterposition m Zukunft ändern
    cut = 1300 + m
    d2 = d.copy()
    d2.loc[cut + 1:, [H] + EXO] = d2.loc[cut + 1:, [H] + EXO] * 2.3
    alt = walk_forward(d2, H, build_features(d2, H), window, **kw)

    # proba an Tagen <= cut dürfen sich nicht ändern. Zeilen, deren Label erst nach cut bekannt
    # wäre, fehlen in beiden Läufen nicht, aber y_true darf sich dort ändern; deshalb nur proba.
    b = base[base["date"] <= d["date"].iloc[cut]].reset_index(drop=True)
    a = alt[alt["date"] <= d["date"].iloc[cut]].reset_index(drop=True)
    assert len(a) == len(b) > 0
    assert np.allclose(a["proba"].values, b["proba"].values)


def test_walk_forward_trainiert_nur_auf_bekannten_labels(d, monkeypatch):
    """Direkter Purge-Test: letzte Trainingszeile = i - horizon - feat_lag."""
    import p229.walk_forward as wf
    seen = []

    class Spy:
        def __init__(self, **kw):
            self.kw = kw

        def fit(self, X, y):
            seen.append(len(X))
            return self

        def predict_proba(self, X):
            return np.array([[0.5, 0.5]])

    monkeypatch.setattr(wf, "XGBClassifier", Spy)
    F = build_features(d, H)
    i0 = 1300
    window = (d["date"].iloc[i0], d["date"].iloc[i0 + 4])
    for feat_lag in (0, 1):
        seen.clear()
        wf.walk_forward(d, H, F, window, entry_lag=0, feat_lag=feat_lag, retrain_every=5)
        # Erste Trainingsmenge: Zeilen 0..last, NaN-Labels am Ende werden verworfen (hier keine).
        last = i0 - 3 - feat_lag
        # Zeilen mit NaN-Features/Labels am Anfang bleiben im Array, werden aber nur über y gefiltert:
        # y ist ab Zeile 0 bekannt, daher len(X) == last + 1 (Features dürfen NaN sein).
        assert seen[0] == last + 1
