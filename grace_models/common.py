"""Shared helpers for GRACE single-modality models: paths, cohort loading, leakage guard.

Conventions every modality follows (so late fusion can just join on stay_id):
  * Unit = one ED stay (stay_id) from derived/labels_splits/cohort_{W}h.csv.
  * Splits come from that file (grace_data.splits.assign_split); never re-split.
  * A feature builder returns a DataFrame indexed by stay_id that uses only data with
    timestamp <= t_pred_{W}h. Stays without the modality are simply absent from the index.
  * Predictions are written to derived/models/<run_name>/<W>h/preds.parquet (see trainer.py).
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import pandas as pd

from grace_data.paths import DERIVED as LABELS_DIR
from grace_data.paths import GRACE_ROOT, sources  # noqa: F401  (re-exported)

FEATURES_DIR = GRACE_ROOT / "derived" / "features"
MODELS_DIR = GRACE_ROOT / "derived" / "models"

WINDOWS_H = (6, 24)
TARGETS = ["y_ami", "y_ahf", "y_afib"]
SECONDARY_TARGETS = ["y2_hf_broad", "y2_aflutter"]

# Columns that reveal the outcome or the ED course after t_pred. A feature table containing any of
# these (or anything matching LEAKY_PATTERNS) is refused by check_no_leakage().
LEAKY_COLUMNS = {
    "disposition", "hadm_id", "admitted", "outtime", "ed_los_h", "label_source", "label_available",
    "in_cohort_6h", "in_cohort_24h", "split",
}
LEAKY_PATTERNS = [r"^y_", r"^y2_", r"^edcode_", r"^h_to_first_", r"^icd", r"^discharge_summ"]


def load_cohort(window_h: int, admitted_only: bool = False) -> pd.DataFrame:
    """Cohort for one prediction window, indexed by stay_id."""
    if window_h not in WINDOWS_H:
        raise ValueError(f"window_h must be one of {WINDOWS_H}")
    meta_path = LABELS_DIR / "label_definitions.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("provisional"):
        warnings.warn("Labels are from a PROVISIONAL build (hosp module missing at build time). "
                      "Fine for debugging; rerun grace_data.build_labels_splits before reporting results.",
                      stacklevel=2)
    co = pd.read_csv(LABELS_DIR / f"cohort_{window_h}h.csv",
                     parse_dates=["intime", "outtime", f"t_pred_{window_h}h"])
    co = co.rename(columns={f"t_pred_{window_h}h": "t_pred"}).set_index("stay_id")
    if admitted_only:
        co = co[co.admitted]
    return co


def check_no_leakage(X: pd.DataFrame) -> None:
    bad = [c for c in X.columns
           if c in LEAKY_COLUMNS or any(re.search(p, str(c), flags=re.I) for p in LEAKY_PATTERNS)]
    if bad:
        raise ValueError(f"Feature table contains outcome-revealing columns: {bad}")


def filter_before_tpred(events: pd.DataFrame, cohort: pd.DataFrame, time_col: str,
                        key: str = "stay_id", lookback_h: float | None = 0.0) -> pd.DataFrame:
    """Keep event rows whose time is in [intime - lookback_h, t_pred] for the stay they belong to.

    `events` must contain `key` (stay_id) and `time_col`. lookback_h=None keeps anything before t_pred.
    Rows for stays not in the cohort are dropped.
    """
    e = events.merge(cohort[["intime", "t_pred"]], left_on=key, right_index=True, how="inner")
    keep = e[time_col] <= e["t_pred"]
    if lookback_h is not None:
        keep &= e[time_col] >= e["intime"] - pd.Timedelta(hours=lookback_h)
    return e.loc[keep].drop(columns=["intime", "t_pred"])


def attach_events_by_subject(events: pd.DataFrame, cohort: pd.DataFrame, time_col: str,
                             lookback_h: float | None = 0.0) -> pd.DataFrame:
    """For sources keyed by subject_id + time only (radiology notes, echo studies): assign each event to
    every cohort stay of that patient whose window [intime - lookback_h, t_pred] contains it.
    Returns events with a stay_id column."""
    c = cohort[["subject_id", "intime", "t_pred"]].reset_index()
    e = events.merge(c, on="subject_id", how="inner")
    keep = e[time_col] <= e["t_pred"]
    if lookback_h is not None:
        keep &= e[time_col] >= e["intime"] - pd.Timedelta(hours=lookback_h)
    return e.loc[keep].drop(columns=["intime", "t_pred"])


def is_fresh(cache_path: Path, window_h: int) -> bool:
    """A cached feature file is valid only if it is newer than the cohort it was built from."""
    cohort_csv = LABELS_DIR / f"cohort_{window_h}h.csv"
    return cache_path.exists() and cache_path.stat().st_mtime > cohort_csv.stat().st_mtime


def run_dir(run_name: str, window_h: int) -> Path:
    d = MODELS_DIR / run_name / f"{window_h}h"
    d.mkdir(parents=True, exist_ok=True)
    return d
