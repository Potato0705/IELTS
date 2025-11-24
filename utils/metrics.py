# utils/metrics.py
from __future__ import annotations
from typing import List, Dict
import numpy as np
from scipy.stats import pearsonr

def quadratic_weighted_kappa(y_true: List[float], y_pred: List[float]) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    y_true_i = np.round(y_true * 2).astype(int)  # 0..18
    y_pred_i = np.round(y_pred * 2).astype(int)

    min_r, max_r = 0, 18
    n = max_r - min_r + 1

    O = np.zeros((n, n), dtype=float)
    for a, b in zip(y_true_i, y_pred_i):
        if 0 <= a <= max_r and 0 <= b <= max_r:
            O[a, b] += 1

    hist_true = np.bincount(np.clip(y_true_i, 0, max_r), minlength=n).astype(float)
    hist_pred = np.bincount(np.clip(y_pred_i, 0, max_r), minlength=n).astype(float)

    E = np.outer(hist_true, hist_pred)
    if E.sum() == 0:
        return 0.0
    E = E / E.sum() * O.sum()

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            W[i, j] = ((i - j) ** 2) / ((max_r - min_r) ** 2)

    den = (W * E).sum()
    if den == 0:
        return 0.0
    num = (W * O).sum()
    return float(1.0 - num / den)

def compute_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)

    qwk = quadratic_weighted_kappa(yt.tolist(), yp.tolist())

    try:
        if np.std(yt) == 0 or np.std(yp) == 0:
            pear = 0.0
        else:
            pear = float(pearsonr(yt, yp)[0])
    except Exception:
        pear = 0.0

    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    exact_acc = float(np.mean(yt == yp))
    adj_acc = float(np.mean(np.abs(yt - yp) <= 0.5))

    return {
        "qwk": qwk,
        "pearson": pear,
        "rmse": rmse,
        "exact_acc": exact_acc,
        "adj_acc": adj_acc,
    }

def fitness_from_metrics(m: Dict[str, float]) -> float:
    qwk = m.get("qwk", 0.0)
    pear = max(m.get("pearson", 0.0), 0.0)
    rmse = m.get("rmse", 9.0)
    # plan priority: QWK > Pearson > RMSE
    return float(qwk + 0.3 * pear - 0.2 * rmse)
