# from __future__ import annotations


import pandas as pd
import numpy as np

CATEGORICALS = [
    'vendor_category', 'vendor_type', 'vendor', 'country', 'city', 'city_size',
    'card_type', 'device', 'channel'
]
BINARIES = [
    'is_card_present', 'is_outside_home_country', 'is_high_risk_vendor', 'is_weekend'
]

def unpack_last_hour_activity(df: pd.DataFrame, col: str = 'last_hour_activity') -> pd.DataFrame:
    df = df.copy()
    if col in df.columns:
        sub = pd.json_normalize(df[col])
        sub.columns = [f"{col}__{c}" for c in sub.columns]
        df = pd.concat([df.drop(columns=[col]), sub], axis=1)
    return df

def add_basic_time_features(df: pd.DataFrame, ts_col: str = 'timestamp') -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[ts_col])
    df['tx_hour'] = ts.dt.hour
    df['tx_dow'] = ts.dt.dayofweek
    df['tx_month'] = ts.dt.month
    return df

def customer_velocity(
    df: pd.DataFrame,
    customer_col: str = 'customer_id',
    ts_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Считает:
      - time_since_prev_s: сек с предыдущей транзакции клиента
      - cust_tx_count_{1,6,24}h: число транзакций клиента за окна 1/6/24 часа (без текущей)
    Без утечек: окна считаются с closed='left' (только прошлые события).
    Работает как через groupby.rolling(on=...), так и через fallback с временным индексом.
    """
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], utc=False, errors='coerce')
    out = out.sort_values([customer_col, ts_col])

    ts = out[ts_col]
    out['time_since_prev_s'] = (
        ts - ts.groupby(out[customer_col]).shift(1)
    ).dt.total_seconds()

    # Пытаемся использовать быстрый путь (pandas ≥ 1.3 supports on=...)
    def _count_window(hours: int) -> np.ndarray:
        try:
            cnt = (
                out.groupby(customer_col)
                   .rolling(f'{hours}h', on=ts_col, closed='left')[ts_col]
                   .count()
                   .reset_index(level=0, drop=True)
            )
            return cnt.to_numpy()
        except Exception:
            # Fallback: по группам с временным индексом
            vals = []
            for _, g in out.groupby(customer_col, sort=False):
                s = pd.Series(1, index=g[ts_col])
                rc = s.rolling(f'{hours}h', closed='left').sum()
                vals.append(rc.to_numpy())
            return np.concatenate(vals, axis=0)

    for h in (1, 6, 24):
        out[f'cust_tx_count_{h}h'] = _count_window(h)

    # Restore original row order so downstream time-based split aligns with the baseline
    out = out.sort_index()
    return out

def clip_and_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Простейшая очистка
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        q1, q99 = df[c].quantile([0.01, 0.99])
        df[c] = df[c].clip(q1, q99)
    return df.fillna({
        **{c: -1 for c in CATEGORICALS},
        **{b: 0 for b in BINARIES}
    })


# Device novelty features for H2
def add_device_novelty(df: pd.DataFrame, customer_col: str = 'customer_id', device_col: str = 'device_fingerprint', ts_col: str = 'timestamp') -> pd.DataFrame:
    """Adds per-customer device novelty: first use flag and usage index."""
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors='coerce')
    out = out.sort_values([customer_col, ts_col])
    out['cust_dev_tx_index'] = out.groupby([customer_col, device_col]).cumcount()
    out['is_new_device_for_customer'] = (out['cust_dev_tx_index'] == 0).astype(int)
    return out.sort_index()