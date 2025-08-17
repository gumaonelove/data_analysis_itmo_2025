from __future__ import annotations
import numpy as np
import pandas as pd

def split_time_aware(df: pd.DataFrame, ts_col: str='timestamp', test_size: float=0.2):
    df = df.sort_values(ts_col)
    n_test = int(len(df) * test_size)
    return df.iloc[:-n_test], df.iloc[-n_test:]

def bootstrap_pr_auc(y_true, p1, p2, B=1000, alpha=0.05, seed=42):
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        diffs.append(average_precision_score(y_true[idx], p2[idx]) -
                     average_precision_score(y_true[idx], p1[idx]))
    diffs = np.asarray(diffs)
    lo, hi = np.quantile(diffs, [alpha/2, 1-alpha/2])
    return float(diffs.mean()), float(lo), float(hi)
