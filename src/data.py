# src/data.py 


from __future__ import annotations
import pandas as pd
from pathlib import Path

def load_transactions(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df

def load_fx(path: str | Path) -> pd.DataFrame:
    fx = pd.read_parquet(path)
    fx['date'] = pd.to_datetime(fx['date']).dt.date
    return fx
