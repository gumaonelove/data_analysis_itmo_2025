# src/pipeline.py


from __future__ import annotations
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, RobustScaler

# Helper to safely replace NaN and inf values for robust model input
def _replace_infinite(X):
    import numpy as _np
    X = _np.asarray(X, dtype=_np.float64)
    # Replace NaN and +/- inf; NaN->0, +inf/-inf to large finite caps so solvers remain stable
    return _np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

def build_preprocessor(X, drop_high_card: bool = True) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'string', 'category', 'bool']).columns.tolist()

    if drop_high_card:
        for hc in ['vendor', 'device_fingerprint', 'ip_address', 'card_number']:
            if hc in cat_cols:
                cat_cols.remove(hc)

    num_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('fix_inf', FunctionTransformer(_replace_infinite, feature_names_out='one-to-one')),
        ('scale', RobustScaler()),
    ])

    # Совместимость: sklearn>=1.2 -> sparse_output, sklearn<1.2 -> sparse
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', min_frequency=50, sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', min_frequency=50, sparse=True)

    cat_pipe = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', ohe),
    ])

    pre = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols),
    ], remainder='drop')

    return pre

def build_logreg(class_weight='balanced', max_iter=1000, C=0.2, solver='lbfgs', random_state=42):
    return LogisticRegression(
        max_iter=max_iter,
        class_weight=class_weight,
        C=C,
        solver=solver,
        random_state=random_state,
    )