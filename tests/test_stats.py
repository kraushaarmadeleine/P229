import numpy as np
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from p229.stats import block_boot_mean, holm, mcnemar, wil


def test_wilson_gegen_statsmodels():
    for k, n in [(177, 297), (150, 297), (5, 10), (0, 20)]:
        lo, hi = wil(k, n)
        ref = proportion_confint(k, n, alpha=0.05, method="wilson")
        assert np.allclose([lo, hi], ref, atol=2e-4)   # z=1.96 statt 1.959964


def test_holm_gegen_statsmodels():
    p = np.array([0.009, 0.044, 0.107, 0.159, 0.315, 0.547, 0.04])
    assert np.allclose(holm(p), multipletests(p, method="holm")[1])


def test_holm_beispiel_aus_db08():
    # diesel_sued: 0.009 bei m=24 Tests -> 0.207
    p = np.r_[0.009, np.full(23, 1.0)]
    assert np.isclose(holm(p)[0], 0.216) or np.isclose(holm(p)[0], 24 * 0.009)


def test_mcnemar_gegen_statsmodels():
    rng = np.random.default_rng(1)
    a = rng.random(300) < 0.6
    b = rng.random(300) < 0.55
    table = [[(a & b).sum(), (a & ~b).sum()], [(~a & b).sum(), (~a & ~b).sum()]]
    assert np.isclose(mcnemar(a, b), sm_mcnemar(table, exact=True).pvalue)


def test_mcnemar_ohne_diskordante_paare():
    a = np.array([True, False, True])
    assert mcnemar(a, a) == 1.0


def test_block_boot_enthaelt_mittelwert_und_ist_reproduzierbar():
    x = (np.random.default_rng(2).random(300) < 0.6).astype(float)
    lo, hi = block_boot_mean(x)
    assert lo < x.mean() < hi
    assert (lo, hi) == tuple(block_boot_mean(x))
