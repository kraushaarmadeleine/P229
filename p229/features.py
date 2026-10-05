import numpy as np
import pandas as pd

from .config import EXO


def build_features(df, H, exo=EXO):
    """Stationäre Merkmale (Log-Returns, Nachlauf-Lücke, Volatilität, Wochentag) für Serie H.

    Jedes Merkmal in Zeile t nutzt nur Preise bis einschließlich t.
    """
    df = df.reset_index(drop=True)
    F = pd.DataFrame({"date": df["date"]})
    lp = {c: np.log(df[c].where(df[c] > 0)) for c in [H] + list(exo)}
    for c, s in lp.items():
        r1 = s.diff()
        F[f"{c}_r1"] = r1
        F[f"{c}_r1_l1"] = r1.shift(1)
        F[f"{c}_r1_l2"] = r1.shift(2)
        F[f"{c}_r3"] = s - s.shift(3)
        F[f"{c}_r5"] = s - s.shift(5)
    for c in exo:                                   # Nachlauf-Lücke: Marktbewegung minus Preisbewegung
        F[f"gap1_{c}"] = F[f"{c}_r1"] - F[f"{H}_r1"]
        F[f"gap3_{c}"] = F[f"{c}_r3"] - F[f"{H}_r3"]
    r1h = F[f"{H}_r1"]
    F["vol5_H"] = r1h.rolling(5).std()
    F["vol20_H"] = r1h.rolling(20).std()
    F["dev20_H"] = lp[H] - lp[H].rolling(20).mean()
    F["weekday"] = df["date"].dt.weekday
    return F


def london_features(H, F):
    """Argus-only-Variante: eigene Serie + ICE Gasoil + EUR/USD."""
    keep = ["date", "weekday", "vol5_H", "vol20_H", "dev20_H"]
    for c in [H, "ice_gasoil", "usd_eur"]:
        keep += [f"{c}_r1", f"{c}_r1_l1", f"{c}_r1_l2", f"{c}_r3", f"{c}_r5"]
    for c in ["ice_gasoil", "usd_eur"]:
        keep += [f"gap1_{c}", f"gap3_{c}"]
    return F[keep]
