
# src/validation.py


from __future__ import annotations
import numpy as np
import pandas as pd


def compute_time_cutoff(df: pd.DataFrame, ts_col: str = 'timestamp', test_size: float = 0.2):
    """Return timestamp cutoff so that roughly `test_size` of the dataset falls into test by time."""
    df_sorted = df.sort_values(ts_col)
    n_test = int(len(df_sorted) * test_size)
    if n_test <= 0:
        return df_sorted[ts_col].max()
    cutoff_idx = len(df_sorted) - n_test
    return df_sorted.iloc[cutoff_idx][ts_col]

def split_by_cutoff(df: pd.DataFrame, cutoff, ts_col: str = 'timestamp'):
    """Split df using provided `cutoff` timestamp so that df[ts_col] <= cutoff -> train, > cutoff -> test."""
    train = df[df[ts_col] <= cutoff]
    test = df[df[ts_col] > cutoff]
    return train, test

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


# Group-aware, time-ordered split by first seen
def time_group_split_by_first_seen(df: pd.DataFrame, group_col: str, ts_col: str = 'timestamp', test_frac: float = 0.2, ensure_two_classes_in_test: bool = True):
    """Split by groups with no leakage: newest groups by first_seen go to test.
    Ensures (optionally) that test contains at least two classes.
    """
    d = df.copy()
    d['_ts'] = pd.to_datetime(d[ts_col])
    g = (
        d[[group_col, '_ts', 'is_fraud']]
          .groupby(group_col, as_index=False)
          .agg(first_seen=('_ts', 'min'), pos=('is_fraud', 'sum'), n=('is_fraud', 'size'))
    )
    g = g.sort_values('first_seen')
    n_test_groups = max(1, int(len(g) * test_frac))
    def _make_split(k):
        test_groups = set(g.iloc[-k:][group_col])
        tr = d[~d[group_col].isin(test_groups)].drop(columns=['_ts'])
        te = d[d[group_col].isin(test_groups)].drop(columns=['_ts'])
        return tr, te
    k = n_test_groups
    train, test = _make_split(k)
    if ensure_two_classes_in_test:
        while test['is_fraud'].nunique() < 2 and k < len(g):
            k += 1
            train, test = _make_split(k)
    return train, test
