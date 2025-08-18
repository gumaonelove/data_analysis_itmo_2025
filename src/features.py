# from __future__ import annotations


import pandas as pd
import numpy as np
from tqdm import tqdm  # type: ignore

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
    ts_col: str = 'timestamp',
    progress: bool = True,
) -> pd.DataFrame:
    """
    Быстрая реализация без Python-циклов: O(N) с векторизацией по каждому клиенту.
    Считает:
      - time_since_prev_s
      - cust_tx_count_{1,6,24}h (число ПРЕДЫДУЩИХ транзакций в окне, текущая не учитывается)
    Границы окон соответствуют closed='left' (равно граничному значению — исключается).
    """
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], utc=False, errors='coerce')

    # Сортируем и сохраняем исходные индексы
    out_sorted = out.sort_values([customer_col, ts_col]).reset_index()
    n = len(out_sorted)

    # Время в наносекундах для быстрых разностей
    t_ns = out_sorted[ts_col].values.astype('datetime64[ns]').astype('int64')
    ONE_H = 3600 * 10**9
    SIX_H = 6 * ONE_H
    DAY   = 24 * ONE_H

    # Результаты
    time_since_prev = np.empty(n, dtype='float64')
    cnt1  = np.zeros(n, dtype='int32')
    cnt6  = np.zeros(n, dtype='int32')
    cnt24 = np.zeros(n, dtype='int32')

    # Векторизованная обработка по каждому клиенту с прогресс-баром
    indices_dict = out_sorted.groupby(customer_col, sort=False).indices
    it = indices_dict.items()
    if progress:
        it = tqdm(it, total=len(indices_dict), desc='customer_velocity: groups')
    for _, pos in it:
        pos = np.asarray(pos)
        t = t_ns[pos]
        m = len(pos)
        if m == 0:
            continue

        # Время с предыдущей транзакции, сек
        dt = np.empty(m, dtype='float64')
        dt[0] = np.nan
        if m > 1:
            dt[1:] = (t[1:] - t[:-1]) / 1e9
        time_since_prev[pos] = dt

        # Векторизованно считаем число предыдущих транзакций в окнах 1/6/24 часа
        # Для closed='left' берём левую границу строго больше (tr - W)
        # l = searchsorted(t, t - W + 1, side='left')
        r_idx = np.arange(m, dtype=np.int64)
        l1  = np.searchsorted(t, t - ONE_H + 1, side='left')
        l6  = np.searchsorted(t, t - SIX_H + 1, side='left')
        l24 = np.searchsorted(t, t - DAY   + 1, side='left')

        cnt1[pos]  = (r_idx - l1).astype('int32')
        cnt6[pos]  = (r_idx - l6).astype('int32')
        cnt24[pos] = (r_idx - l24).astype('int32')

    out_sorted['time_since_prev_s'] = time_since_prev
    out_sorted['cust_tx_count_1h']  = cnt1
    out_sorted['cust_tx_count_6h']  = cnt6
    out_sorted['cust_tx_count_24h'] = cnt24

    # Возвращаем исходный порядок строк
    out_final = (
        out_sorted
        .set_index('index')
        .loc[out.index]
        .sort_index()
    )
    return out_final

def clip_and_fill(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        q = df[num_cols].quantile([0.01, 0.99])
        lower = q.loc[0.01]
        upper = q.loc[0.99]
        # Векторное клиппирование по столбцам
        df[num_cols] = df[num_cols].clip(lower=lower, upper=upper, axis=1)
    # Аккуратно заполняем только существующие колонки
    fill_map = {}
    for c in CATEGORICALS:
        if c in df.columns:
            fill_map[c] = -1
    for b in BINARIES:
        if b in df.columns:
            fill_map[b] = 0
    return df.fillna(fill_map)


# Device novelty features for H2
def add_device_novelty(df: pd.DataFrame, customer_col: str = 'customer_id', device_col: str = 'device_fingerprint', ts_col: str = 'timestamp') -> pd.DataFrame:
    """Adds per-customer device novelty: first use flag and usage index."""
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors='coerce')
    out = out.sort_values([customer_col, ts_col])
    out['cust_dev_tx_index'] = out.groupby([customer_col, device_col]).cumcount()
    out['is_new_device_for_customer'] = (out['cust_dev_tx_index'] == 0).astype(int)
    return out.sort_index()