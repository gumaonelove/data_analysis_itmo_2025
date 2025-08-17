from __future__ import annotations
import pandas as pd

def load_fx(path: str) -> pd.DataFrame:
    fx = pd.read_parquet(path)
    fx['date'] = pd.to_datetime(fx['date']).dt.date
    return fx

def convert_to_usd(df_tx: pd.DataFrame, fx: pd.DataFrame, amount_col: str = 'amount',
                   currency_col: str = 'currency', timestamp_col: str = 'timestamp') -> pd.DataFrame:
    df = df_tx.copy()
    df['date'] = pd.to_datetime(df[timestamp_col]).dt.date

    # Melt FX to long for easy join
    fx_long = fx.melt(id_vars=['date'], var_name='currency', value_name='rate_to_currency')
    # По README: столбцы — курсы валют относительно USD, USD=1.
    # Предполагаем: 1 USD = rate_to_currency * <currency>.
    # Тогда amount_in_usd = amount / rate_to_currency (если сумма в валюте).
    df = df.merge(fx_long, on=['date', 'currency'], how='left')

    df['amount_usd'] = df[amount_col] / df['rate_to_currency']
    return df