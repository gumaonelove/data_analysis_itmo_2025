from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split

from features import unpack_last_hour_activity, add_basic_time_features, clip_and_fill
from currency import load_fx, convert_to_usd

DATA_DIR = Path('data')
REPORTS = Path('reports')
REPORTS.mkdir(exist_ok=True, parents=True)

TX_PATH = DATA_DIR / 'transaction_fraud_data.parquet'
FX_PATH = DATA_DIR / 'historical_currency_exchange.parquet'

def load_data() -> pd.DataFrame:
    df = pd.read_parquet(TX_PATH)
    fx = load_fx(str(FX_PATH))
    df = convert_to_usd(df, fx)
    df = unpack_last_hour_activity(df)
    df = add_basic_time_features(df)
    return df

def split_time_aware(df: pd.DataFrame, ts_col: str = 'timestamp', test_size: float = 0.2):
    df = df.sort_values(ts_col)
    n_test = int(len(df) * test_size)
    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]
    return train, test

def build_pipeline(df: pd.DataFrame) -> Pipeline:
    y = df['is_fraud'].astype(int)
    X = df.drop(columns=['is_fraud'])

    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
    # Ограничим high-cardinality (vendor, device_fingerprint, ip_address) — упростим baseline
    high_card = [c for c in categorical if c in ['vendor', 'device_fingerprint', 'ip_address', 'card_number']]
    categorical = [c for c in categorical if c not in high_card]

    pre = ColumnTransformer([
        ('num', 'passthrough', numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=50), categorical),
    ], remainder='drop')

    model = LogisticRegression(max_iter=200, class_weight='balanced', n_jobs=None)

    pipe = Pipeline([
        ('prep', pre),
        ('clf', model),
    ])
    return pipe

def evaluate(y_true, y_prob) -> dict:
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)

    # Threshold при Precision >= 0.9
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    target_p = 0.9
    mask = prec >= target_p
    best = np.argmax(rec[mask]) if mask.any() else None
    thr90 = float(thr[mask][best]) if best is not None and len(thr[mask])>0 else None
    rec90 = float(rec[mask][best]) if best is not None else None
    p90 = float(prec[mask][best]) if best is not None else None

    return {
        'roc_auc': float(roc),
        'pr_auc': float(pr),
        'threshold_at_precision_0_90': thr90,
        'recall_at_precision_0_90': rec90,
        'precision_achieved': p90
    }

def main():
    df = load_data()
    df = clip_and_fill(df)

    train, test = split_time_aware(df)
    y_tr = train['is_fraud'].astype(int)
    y_te = test['is_fraud'].astype(int)
    X_tr = train.drop(columns=['is_fraud'])
    X_te = test.drop(columns=['is_fraud'])

    pipe = build_pipeline(train)
    pipe.fit(X_tr, y_tr)
    y_prob = pipe.predict_proba(X_te)[:, 1]
    metrics = evaluate(y_te, y_prob)

    with open(REPORTS / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print('Saved reports/metrics.json')
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()