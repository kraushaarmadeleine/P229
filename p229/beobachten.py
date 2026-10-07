"""Dreistufiges Signal: Kaufen / Warten / Beobachten.

Regel (vorab festgelegt):
    Kaufen      wenn p >= 0.5 + delta
    Warten      wenn p <  0.5 - delta
    Beobachten  sonst (Modell ist sich nicht sicher, keine Entscheidung)
delta = 0 entspricht dem bisherigen Zwei-Stufen-Signal (nie Beobachten).

Kostenannahme: Beobachten wird einmal wie "Warten" (Variante A) und einmal wie "Kaufen" (Variante B) gerechnet.
"""
import numpy as np
import pandas as pd

from .config import HORIZON
from .stats import block_boot_mean, wil

KAUFEN, WARTEN, BEOBACHTEN = 1, -1, 0
NAMEN = {KAUFEN: "Kaufen", WARTEN: "Warten", BEOBACHTEN: "Beobachten"}


def signal(proba, delta):
    p = np.asarray(proba, float)
    return np.where(p >= 0.5 + delta, KAUFEN, np.where(p < 0.5 - delta, WARTEN, BEOBACHTEN))


def bewerte_dreistufig(d, H, res, delta, entry_lag=0, horizon=HORIZON):
    """Kennzahlen eines dreistufigen Signals. res: Spalten date, y_true, proba (wie walk_forward)."""
    y, p = res["y_true"].values, res["proba"].values
    sig = signal(p, delta)
    n = len(res)
    kauf, wart = sig == KAUFEN, sig == WARTEN
    entsch = kauf | wart
    ok = np.where(kauf, y == 1, np.where(wart, y == 0, False))
    n_e = int(entsch.sum())
    out = {"Serie": H, "delta": delta, "n": n, "n_Kaufen": int(kauf.sum()), "n_Warten": int(wart.sum()),
           "n_Beobachten": int((~entsch).sum()), "Abdeckung": n_e / n}
    if n_e:
        hit = int(ok[entsch].sum())
        lo, hi = wil(hit, n_e)
        out.update({"DA_entschieden": hit / n_e, "Wilson_lo": lo, "Wilson_hi": hi})
        if n_e >= 20:
            blo, bhi = block_boot_mean(ok[entsch].astype(float))
            out.update({"Boot_lo": blo, "Boot_hi": bhi})
    out.setdefault("Boot_lo", np.nan); out.setdefault("Boot_hi", np.nan)
    out["Precision_Kaufen"] = float((y[kauf] == 1).mean()) if kauf.any() else np.nan
    out["Precision_Warten"] = float((y[wart] == 0).mean()) if wart.any() else np.nan
    out["Basisrate"] = float(y.mean())
    # Kosten wie in evaluate_run: kaufen = Preis(t+entry_lag), warten = Preis(t+horizon)
    pos = pd.Series(np.arange(len(d)), index=d["date"])
    i = pos.reindex(res["date"]).values.astype(int)
    p_now, p_fut = d[H].values[i + entry_lag], d[H].values[i + horizon]
    best = np.minimum(p_now, p_fut)
    c_alle = p_now - best
    c_a = np.where(kauf, p_now, p_fut) - best                 # Beobachten = Warten
    c_b = np.where(wart, p_fut, p_now) - best                 # Beobachten = Kaufen
    for name, c in (("A", c_a), ("B", c_b)):
        diff = c_alle - c
        lo, hi = block_boot_mean(diff)
        out.update({f"Ersparnis_{name}": float(diff.mean()), f"Ersp_{name}_lo": lo, f"Ersp_{name}_hi": hi})
    return out
