# src/pipeline.py
from __future__ import annotations
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def _replace_infinite(X):
    X = np.asarray(X, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

def build_preprocessor(X, drop_high_card: bool = True) -> ColumnTransformer:
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

    if drop_high_card:
        # Исключаем суррогаты идентификаторов из OHE
        high_card = [c for c in ['vendor', 'device_fingerprint', 'ip_address', 'card_number'] if c in categorical]
        categorical = [c for c in categorical if c not in high_card]

    num_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('fix_inf', FunctionTransformer(_replace_infinite, feature_names_out='one-to-one')),
        ('scale', RobustScaler()),
    ])

    cat_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        # Ограничиваем кардинальность, игнорируем неизвестные
        ('ohe', OneHotEncoder(handle_unknown='ignore', min_frequency=50)),
    ])

    pre = ColumnTransformer([
        ('num', num_pipe, numeric),
        ('cat', cat_pipe, categorical),
    ], remainder='drop')  # критично: дропаем остальные столбцы (в т.ч. сырые ID)

    return pre

def build_logreg(class_weight='balanced', max_iter=2000, C=0.1, solver='lbfgs', random_state=42):
    return LogisticRegression(
        penalty='l2',
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        solver=solver,
        random_state=random_state,
    )