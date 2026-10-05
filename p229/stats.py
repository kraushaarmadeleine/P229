from math import sqrt

import numpy as np
from scipy.stats import binomtest


def wil(k, n, z=1.96):
    """Wilson-95-%-Intervall für eine Trefferquote k/n."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def block_boot_mean(x, block=5, n_boot=2000, seed=42):
    """95-%-Intervall des Mittelwerts per Block-Bootstrap (berücksichtigt überlappende Labels)."""
    x = np.asarray(x, float)
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = np.arange(n - block + 1)
    means = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(starts, nb)
        means[b] = x[(s[:, None] + np.arange(block)).ravel()[:n]].mean()
    return np.percentile(means, [2.5, 97.5])


def mcnemar(model_ok, other_ok):
    """Exakter McNemar-Test (gepaarte Treffer, zweiseitig)."""
    oa, ob = int((model_ok & ~other_ok).sum()), int((~model_ok & other_ok).sum())
    return binomtest(oa, oa + ob, 0.5).pvalue if oa + ob else 1.0


def holm(p):
    """Holm-Bonferroni-adjustierte p-Werte."""
    p = np.asarray(p, float)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        run = max(run, (m - rank) * p[idx])
        adj[idx] = min(1.0, run)
    return adj
