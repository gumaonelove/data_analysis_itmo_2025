from __future__ import annotations
import pandas as pd

def mask_super_risky(df: pd.DataFrame):
    return (
        (df['is_card_present'] == False) &
        (df['is_outside_home_country'] == True) &
        (df['is_high_risk_vendor'] == True)
    )
