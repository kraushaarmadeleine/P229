import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .config import HORIZON
from .stats import block_boot_mean, mcnemar, wil


def evaluate_run(d, H, F, res, entry_lag, feat_lag, horizon=HORIZON):
    """Pflichtmetriken für einen Walk-Forward-Lauf (Logik 1:1 aus db_08).

    F muss das volle Merkmalsset enthalten (für die Baselines NYMEX-Regel und Momentum).
    """
    y, p = res["y_true"].values, res["proba"].values
    pred = (p >= 0.5).astype(int)
    ok = (pred == y)
    n = len(res)
    hit = int(ok.sum())
    lo, hi = wil(hit, n)
    blo, bhi = block_boot_mean(ok.astype(float))
    p_non = max(binomtest(int(ok[o::3].sum()), len(ok[o::3]), 0.5).pvalue for o in range(3))
    base = y.mean()
    brier, ref = np.mean((p - y) ** 2), np.mean((base - y) ** 2)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    # Vorzeichenregeln als Baselines (gleiche Tage)
    pos = pd.Series(np.arange(len(d)), index=d["date"])
    i = pos.reindex(res["date"]).values.astype(int)
    nym = (F["nymex_heating_oil_r1"].shift(feat_lag).values[i] > 0).astype(int)
    mom = (F[f"{H}_r1"].shift(feat_lag).values[i] > 0).astype(int)
    # Kosten (Regret): kaufen = Preis(t+entry_lag), warten = Preis(t+horizon)
    p_now, p_fut = d[H].values[i + entry_lag], d[H].values[i + horizon]
    best = np.minimum(p_now, p_fut)
    c_mod = np.where(pred == 1, p_now, p_fut) - best
    c_buy = p_now - best
    diff = c_buy - c_mod                                   # > 0: Modell billiger als immer kaufen
    dlo, dhi = block_boot_mean(diff)
    return {"Serie": H, "n": n, "DA": hit / n, "Wilson_lo": lo, "Wilson_hi": hi,
            "Boot_lo": blo, "Boot_hi": bhi, "p_nicht_ueberl": p_non,
            "Mehrheit": max(base, 1 - base), "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "Precision": tp / max(tp + fp, 1), "Recall": tp / max(tp + fn, 1),
            "Brier_Skill": 1 - brier / ref,
            "DA_NYMEX_Regel": float((nym == y).mean()), "DA_Momentum": float((mom == y).mean()),
            "p_McNemar_NYMEX": mcnemar(ok, nym == y), "p_McNemar_Momentum": mcnemar(ok, mom == y),
            "Regret_Modell": c_mod.mean(), "Regret_immer_kaufen": c_buy.mean(),
            "Ersparnis_vs_kaufen": diff.mean(), "Ersp_lo": dlo, "Ersp_hi": dhi,
            "davon_zu_frueh_gekauft": c_mod[pred == 1].sum() / n,
            "davon_zu_lange_gewartet": c_mod[pred == 0].sum() / n}
