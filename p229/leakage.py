import numpy as np

from .features import build_features


def leak_test(d, H, n_cut=1500, n_check=50):
    """Features dürfen sich nicht ändern, wenn man die Zukunft abschneidet.

    Braucht mindestens n_cut Zeilen. Wirft AssertionError bei Look-ahead.
    """
    if len(d) < n_cut:
        raise ValueError(f"leak_test braucht >= {n_cut} Zeilen, d hat {len(d)}")
    a = build_features(d.iloc[:n_cut], H).iloc[n_cut - n_check:n_cut].drop(columns="date").values
    b = build_features(d, H).iloc[n_cut - n_check:n_cut].drop(columns="date").values
    assert np.allclose(a, b, equal_nan=True), f"Look-ahead in den Features ({H})!"
