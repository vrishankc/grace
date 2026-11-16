"""Generic encoder + head trainer used by every single-modality baseline.

    inputs (indexed by stay_id) --encoder.fit(train)/transform--> matrix --head--> P(label)

* Encoder: fit ONCE on the train split (e.g. TF-IDF vocabulary); identity for tabular features.
* Head: one binary classifier PER TARGET. Hyperparameters are chosen on an inner, patient-level
  holdout carved out of TRAIN (salted hash, like grace_data.splits), then the head is refit on all of
  train. So `val` is never used for fitting/selection and stays an honest dev set; `test` is only
  scored when eval_test=True.
* Output (derived/models/<run>/<W>h/):
    preds.parquet   stay_id, subject_id, split, admitted, y_*, p_<target>   (input for late fusion)
    metrics.json    per split x target x {all, admitted}: n, prevalence, AUROC/AUPRC/Brier (+95% CI)
    metrics.md      readable table
    run.json        config, selected hyperparameters, coverage
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .common import TARGETS, check_no_leakage, run_dir
from .metrics import evaluate

INNER_SALT = "grace-v1-inner"
INNER_FRAC = 0.15  # of train patients used for hyperparameter selection


def inner_holdout(subject_ids: pd.Series) -> np.ndarray:
    """Boolean mask: True = inner-validation patient (deterministic, patient-level)."""
    u = subject_ids.map(lambda s: int(hashlib.sha256(f"{INNER_SALT}:{int(s)}".encode()).hexdigest()[:15], 16) / 16**15)
    return (u < INNER_FRAC).to_numpy()


# ---------------------------------------------------------------------------
# Encoders: fit(inputs_train) -> self ; transform(inputs) -> 2-D array / sparse matrix / DataFrame
# ---------------------------------------------------------------------------
class IdentityEncoder:
    """For feature tables that are already numeric (tabular features, precomputed embeddings)."""
    name = "identity"

    def fit(self, X):
        return self

    def transform(self, X):
        return X


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------
@dataclass
class HeadSpec:
    name: str
    make: Callable[[dict], Any]          # params -> unfitted sklearn-style classifier
    grid: list[dict] = field(default_factory=lambda: [{}])


def _predict(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _rows(X, mask):
    return X.iloc[mask] if hasattr(X, "iloc") else X[mask]


def fit_eval(inputs, cohort: pd.DataFrame, encoder, head: HeadSpec, run_name: str, window_h: int,
             targets=TARGETS, eval_test: bool = False, n_boot: int = 200, meta: dict | None = None,
             seed: int = 0) -> dict:
    """Train/evaluate one modality.

    inputs : DataFrame or Series indexed by stay_id. Stays missing from it are treated as "modality
             absent" and get no prediction (late fusion handles missingness).
    cohort : grace_models.common.load_cohort(window_h) (optionally admitted_only).
    """
    t0 = time.time()
    if isinstance(inputs, pd.DataFrame):
        check_no_leakage(inputs)
    stays = cohort.index.intersection(inputs.index)
    co = cohort.loc[stays]
    inputs = inputs.loc[stays]
    split = co.split.to_numpy()
    if not eval_test:
        keep = split != "test"
        co, inputs, split = co[keep], _rows(inputs, keep), split[keep]

    coverage = {s: {"with_modality": int((co.split == s).sum()), "cohort": int((cohort.split == s).sum())}
                for s in ("train", "val", "test")}
    print(f"[{run_name} {window_h}h] coverage: " +
          ", ".join(f"{s} {v['with_modality']:,}/{v['cohort']:,}" for s, v in coverage.items()))

    tr = split == "train"
    enc = encoder.fit(_rows(inputs, tr))
    X = enc.transform(inputs)
    X_tr = _rows(X, tr)
    inner = inner_holdout(co.subject_id[tr])

    preds = co[["subject_id", "split", "admitted"] + [t for t in targets if t in co]].copy()
    selected = {}
    for target in targets:
        y_tr = co[target].to_numpy()[tr].astype(int)
        # --- select hyperparameters on the inner holdout (patient-level, inside train) ---
        best, best_auc = head.grid[0], -np.inf
        if len(head.grid) > 1:
            for params in head.grid:
                m = head.make(params).fit(_rows(X_tr, ~inner), y_tr[~inner])
                auc = roc_auc_score(y_tr[inner], _predict(m, _rows(X_tr, inner)))
                print(f"  {target} {params} inner-AUROC={auc:.4f}")
                if auc > best_auc:
                    best, best_auc = params, auc
        selected[target] = {"params": best, "inner_auroc": None if best_auc == -np.inf else float(best_auc)}
        # --- refit on all of train, predict every stay we're allowed to score ---
        model = head.make(best).fit(X_tr, y_tr)
        preds[f"p_{target.removeprefix('y_')}"] = _predict(model, X)

    # --- metrics ---
    metrics = {}
    for s in ("train", "val", "test"):
        d = preds[preds.split == s]
        if len(d) == 0:
            continue
        metrics[s] = {t: evaluate(d, t, f"p_{t.removeprefix('y_')}", n_boot=0 if s == "train" else n_boot)
                      for t in targets}

    out = run_dir(run_name, window_h)
    preds.reset_index().to_parquet(out / "preds.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (out / "metrics.md").write_text(metrics_markdown(metrics, run_name, window_h))
    run_info = {"run": run_name, "window_h": window_h, "built": datetime.now().isoformat(timespec="seconds"),
                "encoder": getattr(enc, "name", type(enc).__name__), "head": head.name,
                "selected": selected, "coverage": coverage, "eval_test": eval_test,
                "n_features": int(X.shape[1]) if hasattr(X, "shape") and len(X.shape) > 1 else None,
                "seconds": round(time.time() - t0, 1), **(meta or {})}
    (out / "run.json").write_text(json.dumps(run_info, indent=2, default=str))
    print(metrics_markdown(metrics, run_name, window_h, splits=("val", "test")))
    print(f"wrote {out}")
    return {"metrics": metrics, "run": run_info, "preds": preds}


def metrics_markdown(metrics: dict, run_name: str, window_h: int, splits=("train", "val", "test")) -> str:
    lines = [f"## {run_name} ({window_h}h)", "",
             "| split | target | subset | n | prev | AUROC [95% CI] | AUPRC [95% CI] | Brier |",
             "|---|---|---|---|---|---|---|---|"]

    def fmt(m, k):
        v = m.get(k, np.nan)
        ci = m.get(f"{k}_ci")
        return f"{v:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")

    for s in splits:
        for t, by in metrics.get(s, {}).items():
            for subset, m in by.items():
                lines.append(f"| {s} | {t} | {subset} | {m['n']:,} | {m['prevalence']:.3f} | "
                             f"{fmt(m, 'auroc')} | {fmt(m, 'auprc')} | {m['brier']:.4f} |")
    return "\n".join(lines) + "\n"
