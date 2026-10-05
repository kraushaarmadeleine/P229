"""Grobe Power-Abschätzung für das Block-Bootstrap-Kriterium (Boot_lo > 0.5).

Annahme (vereinfacht, nicht an echten Daten gerechnet): Treffer-Indikator ist über 3 Tage
überlappend korreliert (gleitende Summe aus 3 Tagesschocks), wahre Trefferquote p.
Aufruf (Repo-Ordner): python scripts/power_holdout.py
"""
import numpy as np

from p229.stats import block_boot_mean


def power(n, p, sims=400, seed=1):
    rng = np.random.default_rng(seed)
    thr = np.quantile(rng.normal(size=200_000), 1 - p)
    hits = 0
    for _ in range(sims):
        z = rng.normal(size=n + 2)
        s = (z[:-2] + z[1:-1] + z[2:]) / np.sqrt(3)
        ok = (s > thr).astype(float)
        hits += block_boot_mean(ok, n_boot=300, seed=int(rng.integers(1e9)))[0] > 0.5
    return hits / sims


if __name__ == "__main__":
    for n in (87, 170, 297):
        print(n, {p: round(power(n, p), 2) for p in (0.50, 0.55, 0.60, 0.65)})
