"""Stay-level narrative text for the notes baseline.

Sources:
  * "cc"        triage chief complaint (every ED stay; available now). Short, so it's mostly a pipeline
                check and a "triage text" baseline, not the proposal's notes model.
  * "radiology" MIMIC-IV-Note radiology reports (CXR, CT, echo-adjacent imaging, ...). A report is used
                only if it was AVAILABLE by t_pred: storetime (signed) <= t_pred, falling back to
                charttime when storetime is missing, matching grace_data's rad_note timing. Reports
                are linked by subject_id + time (the table has no stay_id).

Discharge summaries (MIMIC-IV-Note discharge.csv.gz) are never read: they describe the outcome.
"""
from __future__ import annotations

import re

import pandas as pd

from ..common import FEATURES_DIR, attach_events_by_subject, is_fresh, sources
from .triage import chief_complaint_text

DEID = re.compile(r"_{2,}")  # MIMIC de-identification placeholders "___"


def radiology_text(cohort: pd.DataFrame, lookback_h: float = 0.0, max_chars: int = 20000,
                   chunksize: int = 200_000) -> pd.Series:
    p = sources()["radiology"]
    if p is None:
        raise FileNotFoundError("MIMIC-IV-Note radiology.csv.gz not found under datasets/. "
                                "Download mimic-iv-note/2.2/note/radiology.csv.gz first.")
    subjects = set(cohort.subject_id)
    parts = []
    for ch in pd.read_csv(p, usecols=["note_id", "subject_id", "charttime", "storetime", "text"],
                          chunksize=chunksize):
        ch = ch[ch.subject_id.isin(subjects)]
        if ch.empty:
            continue
        ch["t_avail"] = pd.to_datetime(ch.storetime, errors="coerce").fillna(
            pd.to_datetime(ch.charttime, errors="coerce"))
        ch = attach_events_by_subject(ch.drop(columns=["storetime", "charttime"]), cohort, "t_avail",
                                      lookback_h=lookback_h)
        if not ch.empty:
            parts.append(ch[["stay_id", "note_id", "t_avail", "text"]])
    notes = pd.concat(parts, ignore_index=True).sort_values(["stay_id", "t_avail"])
    notes["text"] = notes.text.str.replace(DEID, " ", regex=True)
    doc = notes.groupby("stay_id").text.agg("\n\n".join).str.slice(0, max_chars)
    return doc.rename("text")


def build(cohort: pd.DataFrame, window_h: int, source: str = "radiology", cache: bool = True,
          lookback_h: float = 0.0) -> pd.Series:
    """Returns text per stay (index stay_id). Stays with no qualifying note are ABSENT, which the trainer
    treats as "modality missing" (no prediction) rather than as an empty document."""
    path = FEATURES_DIR / f"text_{source}_lb{lookback_h:g}_{window_h}h.parquet"
    if cache and is_fresh(path, window_h):
        s = pd.read_parquet(path)["text"]
        return s[s.index.isin(cohort.index)]
    if source == "cc":
        s = chief_complaint_text(cohort)
        s = s[s.str.strip().str.len() > 0].rename("text")
    elif source == "radiology":
        s = radiology_text(cohort, lookback_h=lookback_h)
    else:
        raise ValueError(f"unknown text source {source!r}")
    s.index.name = "stay_id"
    if cache:
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        s.to_frame().to_parquet(path)
    return s
