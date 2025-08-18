# src/pipeline.py
from __future__ import annotations
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import os

# Leak-like name patterns and helper
LEAK_PATTERNS = (
    'fraud', 'chargeback', 'cbk', 'cb', 'dispute',
    'refund', 'storno', 'charge_back',
    'label', 'target', 'outcome', 'decision',
    'risk', 'flag', 'score', 'alert', 'rule',
    'blacklist', 'blocked', 'ban'
)

# Exclude suspicious feature families by prefix (post-factum aggregates, etc)
EXCLUDE_PREFIXES = (
    'last_hour_activity__',  # потенциальные постфактум-агрегаты/флаги
)

# Safe whitelists for strict mode
SAFE_NUMERIC = [
    'amount_usd', 'tx_hour', 'tx_dow', 'tx_month',
    'time_since_prev_s', 'cust_tx_count_1h', 'cust_tx_count_6h', 'cust_tx_count_24h',
    'device_freq', 'ip_freq', 'device_freq_log1p', 'ip_freq_log1p'
]
SAFE_CATEGORICAL = [
    'vendor_category', 'vendor_type', 'country', 'card_type', 'channel'
]

def _is_leak_col(name: str) -> bool:
    n = str(name).lower()
    return any(p in n for p in LEAK_PATTERNS) and n != 'is_fraud'

def _replace_infinite(X):
    X = np.asarray(X, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

def build_preprocessor(X, drop_high_card: bool = True) -> ColumnTransformer:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

    # Filter out potential leakage columns by name
    numeric = [c for c in numeric if not _is_leak_col(c)]
    categorical = [c for c in categorical if not _is_leak_col(c)]

    # Exclude features by suspicious prefixes
    def _keep_ok(name: str) -> bool:
        n = str(name)
        return not any(n.startswith(p) for p in EXCLUDE_PREFIXES)

    numeric = [c for c in numeric if _keep_ok(c)]
    categorical = [c for c in categorical if _keep_ok(c)]

    # Strict safe whitelist (enabled by default). Set STRICT_WHITELIST=0 to disable.
    if os.getenv('STRICT_WHITELIST', '1') == '1':
        numeric = [c for c in numeric if c in SAFE_NUMERIC]
        categorical = [c for c in categorical if c in SAFE_CATEGORICAL]

    if drop_high_card:
        # Исключаем суррогаты идентификаторов из OHE
        high_card = [c for c in ['vendor', 'device_fingerprint', 'ip_address', 'card_number', 'device'] if c in categorical]
        categorical = [c for c in categorical if c not in high_card]

    num_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('fix_inf', FunctionTransformer(_replace_infinite, feature_names_out='one-to-one')),
        ('scale', RobustScaler()),
    ])

    cat_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        # Ограничиваем кардинальность, игнорируем неизвестные
        ('ohe', OneHotEncoder(handle_unknown='ignore', min_frequency=500)),
    ])

    pre = ColumnTransformer([
        ('num', num_pipe, numeric),
        ('cat', cat_pipe, categorical),
    ], remainder='drop')  # критично: дропаем остальные столбцы (в т.ч. сырые ID)

    return pre

def build_logreg(class_weight='balanced', max_iter=150, C=0.03, solver='saga', tol=5e-3, random_state=42):
    return LogisticRegression(
        penalty='l2',
        C=C,
        max_iter=max_iter,
        tol=tol,
        class_weight=class_weight,
        solver=solver,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )