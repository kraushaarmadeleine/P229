import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .config import HORIZON, REG


def walk_forward(d, H, F, window, entry_lag=0, feat_lag=0, retrain_every=5, params=REG, horizon=HORIZON,
                 model_factory=None):
    """Gepurgter Walk-Forward (expanding window).

    d: Datentabelle (Spalte date + Preisspalten), F: Features zu d, window: (start, ende).
    Ziel: Preis(t+horizon) > Preis(t+entry_lag). Features: Information bis t-feat_lag.
    Training nur mit Zeilen, deren Label am Entscheidungstag i schon bekannt ist:
    last = i - horizon - feat_lag.
    model_factory: Funktion ohne Argumente, die ein frisches Modell (fit/predict_proba) liefert.
    Ohne Angabe: XGBClassifier(**params) wie in db_08.
    """
    feats = [c for c in F.columns if c != "date"]
    P = d[H].astype(float)
    p0, p1 = P.shift(-entry_lag), P.shift(-horizon)
    y = (p1 > p0).astype(float).where(p0.notna() & p1.notna())
    X = F[feats].shift(feat_lag).values.astype(float)
    yv = y.values
    idx = np.where((d["date"] >= window[0]) & (d["date"] <= window[1]))[0]
    proba, model = [], None
    for k, i in enumerate(idx):
        if model is None or k % retrain_every == 0:
            last = i - horizon - feat_lag
            Xt, yt = X[:last + 1], yv[:last + 1]
            m = ~np.isnan(yt)
            model = (model_factory() if model_factory else XGBClassifier(**params)).fit(Xt[m], yt[m])
        proba.append(model.predict_proba(X[i:i + 1])[0, 1])
    res = pd.DataFrame({"date": d["date"].values[idx], "y_true": yv[idx], "proba": proba})
    return res.dropna(subset=["y_true"]).reset_index(drop=True)
