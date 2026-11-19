"""ECG machine-measurement features for GRACE: one row per ED stay, from the LAST ECG acquired by t_pred.

Put this file at grace_models/features/ecg.py. Used by `train.py ecg` and `train.py structured-labs-ecg`.

Source: MIMIC-IV-ECG 1.0 machine_measurements.csv (open access). One row per ECG (study_id) with the
acquisition time (ecg_time), the cart's measured fiducial points/axes, and up to 18 machine
interpretation statements (report_0 ... report_17).

Design:
  * Timing: an ECG is available as soon as it is acquired, so ecg_time <= t_pred (and >= intime -
    lookback_h). ECGs are matched to stays by subject_id + time (attach_events_by_subject).
  * Which ECG: the most recent one by t_pred. No counts of ECGs and no timing features (leakage rule).
  * Interval features (always): heart rate, RR, PR, QRS duration, QT, QTc (Bazett), P-wave duration,
    P/QRS/T axes, and p_wave_detected. Device sentinel values (e.g. 29999 = "not measured") and
    implausible values become NaN. A missing P wave is real physiology (typical of AFib), not leakage,
    which is why p_wave_detected is kept as its own feature.
  * Interpretation features (only with interp=True): one-hot flags for the machine's statements
    (e.g. "atrial fibrillation", "sinus rhythm"). The vocabulary is the statements that appear in at
    least MIN_STMT_STAYS TRAIN stays (capped at MAX_STMTS), so val/test never shape it. These flags
    nearly hand the model the AFib label, so report intervals-only and intervals+interp separately.
  * Excluded on purpose: cart_id (identifies the machine, so the location: a care-process shortcut),
    bandwidth, filtering.

Caching (derived/features/, git-ignored, contains MIMIC data): ecg_<W>h_lb<L>[_interp].parquet for the
FULL window cohort; --admitted-only runs subset it.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from ..common import (FEATURES_DIR, GRACE_ROOT, attach_events_by_subject, check_no_leakage, is_fresh,
                      load_cohort)

REPORT_COLS = [f"report_{i}" for i in range(18)]
RAW_NUM_COLS = ["rr_interval", "p_onset", "p_end", "qrs_onset", "qrs_end", "t_end",
                "p_axis", "qrs_axis", "t_axis"]
SENTINEL_ABS = 10_000          # 29999 etc. mean "not measured"
MIN_STMT_STAYS = 50            # statement must occur in >= this many train stays
MAX_STMTS = 300


# ---------------------------------------------------------------- files

def _find_ecg(name: str):
    roots = [GRACE_ROOT / "datasets" / "files", GRACE_ROOT / "datasets" / "physionet.org" / "files"]
    hits = [p for r in roots for p in r.glob(f"mimic-iv-ecg/*/{name}")]
    if not hits:
        raise FileNotFoundError(f"mimic-iv-ecg/*/{name} not found under {[str(r) for r in roots]}")
    return max(hits, key=lambda p: tuple(int(x) for x in re.findall(r"\d+", p.parent.name)))


def _load_measurements(subjects: np.ndarray, with_reports: bool) -> pd.DataFrame:
    cols = ["subject_id", "study_id", "ecg_time"] + RAW_NUM_COLS + (REPORT_COLS if with_reports else [])
    dtype = {c: "string" for c in REPORT_COLS}
    df = pd.read_csv(_find_ecg("machine_measurements.csv"), usecols=cols, dtype=dtype, low_memory=False)
    df = df[df["subject_id"].isin(subjects)].drop_duplicates("study_id")
    df["ecg_time"] = pd.to_datetime(df["ecg_time"], errors="coerce")
    return df.dropna(subset=["ecg_time"])


# ---------------------------------------------------------------- features

def _interval_features(m: pd.DataFrame) -> pd.DataFrame:
    """m: one row per selected ECG (indexed by stay_id) with RAW_NUM_COLS."""
    v = {c: pd.to_numeric(m[c], errors="coerce") for c in RAW_NUM_COLS}
    v = {c: s.where(s.abs() < SENTINEL_ABS) for c, s in v.items()}

    rr = v["rr_interval"].where(v["rr_interval"].between(200, 3000))
    pr = v["qrs_onset"] - v["p_onset"]
    qrs = v["qrs_end"] - v["qrs_onset"]
    qt = v["t_end"] - v["qrs_onset"]
    p_dur = v["p_end"] - v["p_onset"]

    def ok(s, hi):
        return s.where((s > 0) & (s < hi))

    pr, qrs, qt, p_dur = ok(pr, 1000), ok(qrs, 400), ok(qt, 1000), ok(p_dur, 400)
    X = pd.DataFrame({
        "ecg_hr": 60_000 / rr,
        "ecg_rr": rr,
        "ecg_pr": pr,
        "ecg_qrs": qrs,
        "ecg_qt": qt,
        "ecg_qtc": qt / np.sqrt(rr / 1000),
        "ecg_p_dur": p_dur,
        "ecg_p_axis": v["p_axis"],
        "ecg_qrs_axis": v["qrs_axis"],
        "ecg_t_axis": v["t_axis"],
        "ecg_p_wave_detected": (v["p_onset"].notna() & v["p_end"].notna()).astype("float32"),
    }, index=m.index)
    return X.astype("float32")


def _normalize(s: pd.Series) -> pd.Series:
    return (s.str.lower().str.strip()
             .str.replace(r"\s+", " ", regex=True)
             .str.replace(r"[\s\.,;:]+$", "", regex=True))


def _statement_features(m: pd.DataFrame, train_stays: pd.Index) -> tuple[pd.DataFrame, list[str]]:
    """One-hot machine statements for the selected ECGs (m indexed by stay_id, has REPORT_COLS)."""
    long = (m[REPORT_COLS].reset_index()
            .melt(id_vars="stay_id", value_name="stmt")[["stay_id", "stmt"]]
            .dropna(subset=["stmt"]))
    long["stmt"] = _normalize(long["stmt"])
    long = long[long["stmt"].str.len() > 0].drop_duplicates()

    train_counts = long[long["stay_id"].isin(train_stays)]["stmt"].value_counts()
    vocab = list(train_counts[train_counts >= MIN_STMT_STAYS].index[:MAX_STMTS])

    names, used = {}, set()
    for s in vocab:
        base = "ecgstmt_" + (re.sub(r"[^a-z0-9]+", "_", s).strip("_")[:60] or "x")
        name, k = base, 1
        while name in used:
            k += 1
            name = f"{base}_{k}"
        used.add(name)
        names[s] = name

    hit = long[long["stmt"].isin(names)]
    X = (pd.crosstab(hit["stay_id"], hit["stmt"]).clip(upper=1)
         .rename(columns=names).reindex(index=m.index, fill_value=0))
    for col in names.values():          # statements absent from the selected ECGs
        if col not in X.columns:
            X[col] = 0
    return X[list(names.values())].astype("float32"), vocab


def select_last_ecg(cohort: pd.DataFrame, window_h: int, lookback_h: float = 0.0,
                    with_reports: bool = False) -> pd.DataFrame:
    """Most recent ECG acquired in [intime - lookback_h, t_pred] per stay, indexed by stay_id.
    Also useful for the waveform encoder: the study_id column says which waveform to load."""
    subjects = load_cohort(24)["subject_id"].unique()   # 24 h cohort is a superset of 6 h
    mm = _load_measurements(subjects, with_reports)
    ev = attach_events_by_subject(mm[["subject_id", "study_id", "ecg_time"]], cohort, "ecg_time",
                                  lookback_h=lookback_h)
    last = ev.sort_values(["stay_id", "ecg_time"]).groupby("stay_id").tail(1)[["stay_id", "study_id"]]
    sel = last.merge(mm.drop(columns=["subject_id"]), on="study_id", how="left").set_index("stay_id")
    return sel


def build(cohort: pd.DataFrame, window_h: int, cache: bool = True, lookback_h: float = 0.0,
          interp: bool = False) -> pd.DataFrame:
    """Stay-level ECG table for `cohort` (indexed by stay_id) from the last ECG acquired by t_pred."""
    tag = f"ecg_{window_h}h_lb{lookback_h:g}{'_interp' if interp else ''}"
    out = FEATURES_DIR / f"{tag}.parquet"

    if cache and is_fresh(out, window_h):
        X = pd.read_parquet(out)
    else:
        full = load_cohort(window_h)   # build for the full cohort; admitted-only runs subset below
        m = select_last_ecg(full, window_h, lookback_h, with_reports=interp)
        X = _interval_features(m)
        if interp:
            train_stays = full.index[full["split"] == "train"]
            S, vocab = _statement_features(m, train_stays)
            X = X.join(S)
            FEATURES_DIR.mkdir(parents=True, exist_ok=True)
            (FEATURES_DIR / f"{tag}_vocab.json").write_text(json.dumps(vocab, indent=1))
            print(f"ecg: {len(vocab)} machine statements in vocabulary "
                  f"(>= {MIN_STMT_STAYS} train stays)", flush=True)
        X.index = X.index.astype("int64")
        X.index.name = "stay_id"
        if cache:
            FEATURES_DIR.mkdir(parents=True, exist_ok=True)
            X.to_parquet(out)

    X = X.loc[X.index.intersection(cohort.index)]
    check_no_leakage(X)
    print(f"ecg: {X.shape[1]} features for {len(X):,}/{len(cohort):,} stays "
          f"({len(X) / max(len(cohort), 1):.1%})", flush=True)
    return X
