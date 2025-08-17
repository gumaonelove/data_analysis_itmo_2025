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

def eval_pack(y_true, y_prob, target_p=0.90):
    roc = float(roc_auc_score(y_true, y_prob))
    pr  = float(average_precision_score(y_true, y_prob))
    thr90, rec90, p90 = thr_at_precision(y_true, y_prob, target_p)
    return {"roc_auc": roc, "pr_auc": pr, "threshold_at_precision": target_p,
            "thr_value": thr90, "recall_at_precision": rec90, "precision_achieved": p90}
