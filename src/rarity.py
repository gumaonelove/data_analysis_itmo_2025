# src/rarity.py


from __future__ import annotations
import pandas as pd

def train_freqs(train: pd.DataFrame):
    dev = train['device_fingerprint'].value_counts()
    ip  = train['ip_address'].value_counts()
    return dev, ip

def apply_freqs(df: pd.DataFrame, dev_freq, ip_freq):
    out = df.copy()
    out['device_freq'] = out['device_fingerprint'].map(dev_freq).fillna(0).astype(float)
    out['ip_freq'] = out['ip_address'].map(ip_freq).fillna(0).astype(float)
    return out
