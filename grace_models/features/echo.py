"""Echo timing / linkage and stay-level EchoCLIP embeddings.

MIMIC-IV-ECHO (v0.1) files used:
  echo-record-list.csv : subject_id, study_id, acquisition_datetime, dicom_filepath  (one row per DICOM)
  echo-study-list.csv  : subject_id, study_id, study_datetime, note_id, ...
Echo images are available at acquisition, so a DICOM qualifies if acquisition_datetime is in
[intime - lookback_h, t_pred]. Linked to stays by subject_id + time (no stay_id in the source).

Workflow:
  1. python -m grace_models.train echo-manifest --window 24
       -> derived/features/echo_manifest_24h.csv and echo_dicoms_to_download.txt (only the DICOMs our
          cohort needs; the full ECHO set is far too large to pull).
  2. download those DICOMs into datasets/.../mimic-iv-echo/<ver>/<dicom_filepath>
  3. python -m grace_models.encoders.echoclip  (writes derived/features/echoclip_dicom_embeddings.parquet)
  4. python -m grace_models.train echo --window 24
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common import FEATURES_DIR, attach_events_by_subject, sources

EMB_FILE = FEATURES_DIR / "echoclip_dicom_embeddings.parquet"


def echo_records() -> pd.DataFrame:
    p = sources()["echo_records"]
    if p is None:
        raise FileNotFoundError("mimic-iv-echo echo-record-list.csv not found under datasets/.")
    r = pd.read_csv(p)
    r["t"] = pd.to_datetime(r["acquisition_datetime"], errors="coerce")
    return r


def manifest(cohort: pd.DataFrame, window_h: int, lookback_h: float = 0.0) -> pd.DataFrame:
    """One row per (stay, DICOM) usable for that stay's prediction."""
    r = echo_records()
    m = attach_events_by_subject(r, cohort, "t", lookback_h=lookback_h)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    m.to_csv(FEATURES_DIR / f"echo_manifest_{window_h}h.csv", index=False)
    (FEATURES_DIR / "echo_dicoms_to_download.txt").write_text("\n".join(sorted(m.dicom_filepath.unique())) + "\n")
    print(f"{m.stay_id.nunique():,} stays / {m.study_id.nunique():,} studies / {m.dicom_filepath.nunique():,} "
          f"DICOMs within window ({window_h}h). By split (stays):")
    print(cohort.loc[m.stay_id.unique()].split.value_counts().to_string())
    return m


def build(cohort: pd.DataFrame, window_h: int, lookback_h: float = 0.0) -> pd.DataFrame:
    """Stay-level EchoCLIP features: mean of DICOM embeddings over all qualifying DICOMs.
    (Each DICOM embedding is itself a mean over sampled frames; see encoders/echoclip.py.)"""
    if not EMB_FILE.exists():
        raise FileNotFoundError(f"{EMB_FILE} missing: run `python -m grace_models.encoders.echoclip` first.")
    emb = pd.read_parquet(EMB_FILE)  # dicom_filepath, e0..e{d-1}
    m = attach_events_by_subject(echo_records(), cohort, "t", lookback_h=lookback_h)
    m = m.merge(emb, on="dicom_filepath", how="inner")
    ecols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
    X = m.groupby("stay_id")[ecols].mean()
    # re-normalize (EchoCLIP embeddings are compared by cosine)
    X = X.div(np.linalg.norm(X.to_numpy(), axis=1, keepdims=True) + 1e-8)
    X["echo_n_dicoms"] = m.groupby("stay_id").size()
    return X
