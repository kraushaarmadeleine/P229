import numpy as np
import pandas as pd
import pytest

from p229.config import EXO, SERIES


def make_data(n=1700, seed=0):
    """Synthetische Preise: Heizöl folgt dem Markt mit 1 Tag Verzögerung (wie im echten Befund)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n)
    mkt = rng.normal(0, 0.012, n)
    df = pd.DataFrame({"date": dates})
    for k, c in enumerate(EXO):
        df[c] = 70 * np.exp(np.cumsum(mkt + rng.normal(0, 0.004, n)))
    lag = np.r_[0, mkt[:-1]]
    for s in SERIES:
        df[s] = 100 * np.exp(np.cumsum(lag + rng.normal(0, 0.004, n)))
    return df


@pytest.fixture(scope="session")
def d():
    return make_data()
