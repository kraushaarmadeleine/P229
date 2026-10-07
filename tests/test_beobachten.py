import numpy as np
import pandas as pd

from p229.beobachten import BEOBACHTEN, KAUFEN, WARTEN, bewerte_dreistufig, signal


def test_signal_regel():
    p = [0.70, 0.55, 0.50, 0.49, 0.30]
    assert list(signal(p, 0.0)) == [KAUFEN, KAUFEN, KAUFEN, WARTEN, WARTEN]
    assert list(signal(p, 0.10)) == [KAUFEN, BEOBACHTEN, BEOBACHTEN, BEOBACHTEN, WARTEN]
    assert list(signal(p, 0.25)) == [BEOBACHTEN] * 5          # 0,70 < 0,75 und 0,30 >= 0,25
    assert list(signal(p, 0.19)) == [KAUFEN, BEOBACHTEN, BEOBACHTEN, BEOBACHTEN, WARTEN]


def _daten():
    dates = pd.bdate_range("2024-01-01", periods=12)
    preis = np.array([100, 101, 99, 99, 102, 98, 97, 103, 104, 100, 99, 105], float)
    d = pd.DataFrame({"date": dates, "x": preis})
    # Entscheidungstage 0..7, y = Preis(t+3) > Preis(t)
    idx = np.arange(8)
    y = (preis[idx + 3] > preis[idx]).astype(float)
    proba = np.array([0.9, 0.2, 0.55, 0.45, 0.8, 0.1, 0.6, 0.4])
    res = pd.DataFrame({"date": dates[idx], "y_true": y, "proba": proba})
    return d, res, preis, y


def test_zaehlung_und_treffer():
    d, res, preis, y = _daten()
    r = bewerte_dreistufig(d, "x", res, 0.10)
    # delta 0.10: Kaufen p>=0.6: Tage 0, 4, 6 | Warten p<0.4: Tage 1, 5 | Beobachten: 2, 3, 7
    assert (r["n_Kaufen"], r["n_Warten"], r["n_Beobachten"]) == (3, 2, 3)
    assert np.isclose(r["Abdeckung"], 5 / 8)
    ok = [y[0] == 1, y[4] == 1, y[6] == 1, y[1] == 0, y[5] == 0]
    assert np.isclose(r["DA_entschieden"], np.mean(ok))


def test_kosten_varianten():
    d, res, preis, y = _daten()
    r = bewerte_dreistufig(d, "x", res, 0.10)
    idx = np.arange(8)
    best = np.minimum(preis[idx], preis[idx + 3])
    sig = signal(res["proba"], 0.10)
    cA = np.where(sig == KAUFEN, preis[idx], preis[idx + 3]) - best
    cB = np.where(sig == WARTEN, preis[idx + 3], preis[idx]) - best
    alle = preis[idx] - best
    assert np.isclose(r["Ersparnis_A"], (alle - cA).mean())
    assert np.isclose(r["Ersparnis_B"], (alle - cB).mean())


def test_delta_null_ist_zwei_stufen():
    d, res, preis, y = _daten()
    r = bewerte_dreistufig(d, "x", res, 0.0)
    assert r["n_Beobachten"] == 0 and r["Abdeckung"] == 1.0
    assert np.isclose(r["DA_entschieden"], ((res["proba"] >= 0.5).astype(int) == res["y_true"]).mean())
    assert np.isclose(r["Ersparnis_A"], r["Ersparnis_B"])
