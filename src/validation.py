
# src/validation.py


from __future__ import annotations
import numpy as np
import pandas as pd


def compute_time_cutoff(df: pd.DataFrame, ts_col: str = 'timestamp', test_size: float = 0.2):
    """Return timestamp cutoff so that roughly `test_size` of the dataset falls into test by time.
    Использует квантили по времени вместо полной сортировки (быстрее на больших данных).
    """
    s = pd.to_datetime(df[ts_col])
    if test_size <= 0:
        return s.max()
    q = 1.0 - float(test_size)
    return s.quantile(q, interpolation='nearest')

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
    g = g.reset_index(drop=True)
    n_groups = len(g)
    n_test_groups = max(1, int(n_groups * test_frac))

    # Вычисляем минимальный k, чтобы в тест попали и позитивы, и негативы (если это требуется)
    k = n_test_groups
    if ensure_two_classes_in_test and n_groups > 0:
        g['neg'] = g['n'] - g['pos']
        has_pos = (g['pos'].sum() > 0)
        has_neg = (g['neg'].sum() > 0)
        if has_pos and has_neg:
            # Индексы последних (по времени) групп, содержащих хотя бы один положительный/отрицательный пример
            last_pos_idx = g.index[g['pos'] > 0].max()
            last_neg_idx = g.index[g['neg'] > 0].max()
            # Минимальный хвост (k), который гарантированно включает обе эти группы
            k_required_pos = n_groups - last_pos_idx
            k_required_neg = n_groups - last_neg_idx
            k = max(k, k_required_pos, k_required_neg)
        # если в данных нет одного из классов, оставляем k как есть

    k = min(max(1, int(k)), n_groups)

    test_groups = set(g.iloc[-k:][group_col])
    tr = d[~d[group_col].isin(test_groups)].drop(columns=['_ts'])
    te = d[d[group_col].isin(test_groups)].drop(columns=['_ts'])
    return tr, te
