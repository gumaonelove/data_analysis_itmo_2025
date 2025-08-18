# src/currency.py

 
from __future__ import annotations
import pandas as pd
import numpy as np

def load_fx(path: str) -> pd.DataFrame:
    fx = pd.read_parquet(path)
    fx['date'] = pd.to_datetime(fx['date']).dt.date
    return fx

def convert_to_usd(df_tx: pd.DataFrame, fx: pd.DataFrame, amount_col: str = 'amount',
                   currency_col: str = 'currency', timestamp_col: str = 'timestamp') -> pd.DataFrame:
    df = df_tx.copy()
    df['date'] = pd.to_datetime(df[timestamp_col]).dt.date

    # Forward-fill FX by date to avoid weekend/holiday gaps before melting
    fx2 = fx.sort_values('date').copy()
    num_cols = [c for c in fx2.columns if c != 'date']
    fx2[num_cols] = fx2[num_cols].ffill()

    # Melt FX to long for easy join
    fx_long = fx2.melt(id_vars=['date'], var_name='currency', value_name='rate_to_currency')

    df = df.merge(fx_long, on=['date', 'currency'], how='left')

    # Guard against zeros and infinities in rates
    df['rate_to_currency'] = df['rate_to_currency'].replace([np.inf, -np.inf, 0.0], np.nan)
    # If currency is USD and rate is missing, set to 1
    if currency_col in df.columns:
        mask_usd = df[currency_col] == 'USD'
        df.loc[mask_usd & df['rate_to_currency'].isna(), 'rate_to_currency'] = 1.0

    df['amount_usd'] = df[amount_col] / df['rate_to_currency']
    return df