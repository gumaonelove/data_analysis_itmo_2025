# src/eval.py

from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

def thr_at_precision(y_true, y_prob, target_p=0.90):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    prec_t, rec_t, thr_t = prec[:-1], rec[:-1], thr
    mask = prec_t >= target_p
    if not np.any(mask):
        return None, None, None
    i = np.argmax(rec_t[mask])
    return float(thr_t[mask][i]), float(rec_t[mask][i]), float(prec_t[mask][i])

def eval_pack(y_true, y_prob, target_p=0.90, compute_threshold: bool = False):
    """Сводка по качеству. По умолчанию НЕ подбирает порог по самим же данным (чтобы избежать оптимизма).
    Если нужен порог по этим же данным — установите compute_threshold=True.
    """
    roc = float(roc_auc_score(y_true, y_prob))
    pr  = float(average_precision_score(y_true, y_prob))
    thr90 = rec90 = p90 = None
    if compute_threshold:
        thr90, rec90, p90 = thr_at_precision(y_true, y_prob, target_p)
    return {"roc_auc": roc, "pr_auc": pr, "threshold_at_precision": target_p,
            "thr_value": thr90, "recall_at_precision": rec90, "precision_achieved": p90}


# --- Leakage and superstrong feature diagnostics ---
import pandas as pd

def feature_auc_table(df: pd.DataFrame, target_col: str = 'is_fraud', topn: int = 30):
    """Возвращает таблицу по одиночным признакам (только числовые/бинарные): ROC-AUC, PR-AUC.
    Полезно для диагностики утечки: подозрительные признаки дадут AUC≈1.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col].astype(int).values
    rows = []
    for c in X.columns:
        s = X[c]
        if not (pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s)):
            continue
        try:
            s_clean = pd.to_numeric(s, errors='coerce').fillna(0.0).values
            if np.nanstd(s_clean) == 0:
                continue
            roc = roc_auc_score(y, s_clean)
            pr = average_precision_score(y, s_clean)
            rows.append((c, float(roc), float(pr), int(pd.Series(s_clean).nunique())))
        except Exception:
            continue
    out = pd.DataFrame(rows, columns=['feature', 'roc_auc_single', 'pr_auc_single', 'n_unique'])
    out = out.sort_values(['roc_auc_single', 'pr_auc_single'], ascending=[False, False]).head(topn)
    return out


def binary_agreement_with_target(df: pd.DataFrame, target_col: str = 'is_fraud', topn: int = 30):
    """Для бинарных колонок оценивает долю совпадения с целевой (подозрительно высокие ~1.0 — кандидаты на утечку)."""
    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col].astype(int).values
    rows = []
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_bool_dtype(s) or (pd.api.types.is_integer_dtype(s) and s.dropna().isin([0,1]).all()):
            sc = s.fillna(0).astype(int).values
            agree = (sc == y).mean()
            rows.append((c, float(agree)))
    out = pd.DataFrame(rows, columns=['feature', 'agreement_with_target'])
    out = out.sort_values('agreement_with_target', ascending=False).head(topn)
    return out


# --- Cold-start helpers ---

def new_entities_mask(train_df: pd.DataFrame, test_df: pd.DataFrame, key_col: str) -> pd.Series:
    """Маска новых сущностей в test: True, если ключ не встречался в train."""
    train_keys = set(train_df[key_col].dropna().astype(str).unique())
    test_keys = test_df[key_col].astype(str)
    return ~test_keys.isin(train_keys)


def cold_start_eval(y_test, p_test, mask_new) -> dict:
    """Метрики на подмножестве новых сущностей."""
    y_new = y_test[mask_new]
    p_new = p_test[mask_new]
    roc = float(roc_auc_score(y_new, p_new)) if y_new.nunique() == 2 else None
    pr  = float(average_precision_score(y_new, p_new)) if len(y_new) > 0 else None
    return {"roc_auc_new": roc, "pr_auc_new": pr, "share_new": float(mask_new.mean())}
