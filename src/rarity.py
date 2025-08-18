# src/rarity.py


from __future__ import annotations
import pandas as pd

def train_freqs(train: pd.DataFrame):
    device_key = 'device' if 'device' in train.columns else 'device_fingerprint'
    return train_freqs_coarse(train, device_key=device_key, use_ip_prefix=True)

def apply_freqs(df: pd.DataFrame, dev_freq, ip_freq):
    device_key = 'device' if 'device' in df.columns else 'device_fingerprint'
    return apply_freqs_coarse(df, dev_freq, ip_freq, device_key=device_key, use_ip_prefix=True)


# New, safer-coarsened frequency utilities
import numpy as np

def train_freqs_coarse(train: pd.DataFrame, device_key: str = 'device', use_ip_prefix: bool = True):
    dev = train[device_key].value_counts(sort=False)
    if use_ip_prefix and 'ip_address' in train.columns:
        ip_series = train['ip_address'].astype(str)
        ip_key = ip_series.str.rsplit('.', n=1).str[0].fillna(ip_series)
        ip = ip_key.value_counts(sort=False)
    else:
        ip = train['ip_address'].value_counts(sort=False)
    return dev, ip

def apply_freqs_coarse(df: pd.DataFrame, dev_freq, ip_freq, device_key: str = 'device', use_ip_prefix: bool = True):
    out = df.copy()
    out['device_freq'] = out[device_key].map(dev_freq).fillna(0).astype(float)
    if use_ip_prefix and 'ip_address' in out.columns:
        ip_series = out['ip_address'].astype(str)
        ip_key = ip_series.str.rsplit('.', n=1).str[0].fillna(ip_series)
        out['ip_freq'] = ip_key.map(ip_freq).fillna(0).astype(float)
    else:
        out['ip_freq'] = out['ip_address'].map(ip_freq).fillna(0).astype(float)
    # Stable transforms for modeling
    out['device_freq_log1p'] = np.log1p(out['device_freq'])
    out['ip_freq_log1p'] = np.log1p(out['ip_freq'])
    return out
