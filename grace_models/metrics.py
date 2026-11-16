"""Evaluation metrics: AUROC, AUPRC, Brier with patient-clustered bootstrap CIs.

Stays are clustered within patients (repeat visits), so the bootstrap resamples PATIENTS, not stays;
resampling stays would make the CIs too narrow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()) if len(y) else np.nan}
    if len(y) == 0 or y.min() == y.max():  # undefined with one class
        out.update(auroc=np.nan, auprc=np.nan, brier=np.nan)
        return out
    out.update(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
               brier=float(brier_score_loss(y, p)))
    return out


def bootstrap_ci(y, p, groups, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict[str, tuple]:
    """Percentile CIs from a cluster bootstrap over `groups` (subject_id)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by_group = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    rng = np.random.default_rng(seed)
    stats = {"auroc": [], "auprc": [], "brier": []}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_group[g] for g in pick])
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():
            continue
        stats["auroc"].append(roc_auc_score(yb, pb))
        stats["auprc"].append(average_precision_score(yb, pb))
        stats["brier"].append(brier_score_loss(yb, pb))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) if v else (np.nan, np.nan)
            for k, v in stats.items()}


def evaluate(df: pd.DataFrame, target: str, prob_col: str, n_boot: int = 1000) -> dict:
    """Metrics on all stays and on admitted-only stays (ED-coded negatives are near-trivial; see
    LABELS_SPLITS_PROGRESS.md section 7)."""
    res = {}
    for subset, d in (("all", df), ("admitted", df[df.admitted.astype(bool)])):
        m = point_metrics(d[target], d[prob_col])
        if n_boot and not np.isnan(m["auroc"]):
            ci = bootstrap_ci(d[target], d[prob_col], d.subject_id, n_boot=n_boot)
            m.update({f"{k}_ci": v for k, v in ci.items()})
        res[subset] = m
    return res


def paired_auroc_test(y, p1, p2, groups, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired, patient-clustered bootstrap test of AUROC(p1) - AUROC(p2) on the same stays."""
    y, p1, p2, groups = map(np.asarray, (y, p1, p2, groups))
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by_group = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_group[g] for g in rng.integers(0, len(uniq), len(uniq))])
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    diffs = np.array(diffs)
    delta = roc_auc_score(y, p1) - roc_auc_score(y, p2)
    p_two_sided = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"delta_auroc": float(delta), "ci": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
            "p_value": float(min(1.0, p_two_sided))}
