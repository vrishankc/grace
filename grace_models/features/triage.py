"""Structured ED features (triage vitals, ED vital signs, chief complaint tokens, optional home meds).

All features use only information charted at or before t_pred:
  * edstays/triage: recorded at arrival (no timestamp; triage happens at intime).
  * vitalsign, medrecon: filtered on charttime <= t_pred.

Deliberately EXCLUDED:
  * disposition / hadm_id / outtime / ED LOS: outcome leakage (see grace_data README).
  * number or timing of vital-sign measurements: in the 24h window these encode ED LOS, which encodes
    disposition. Only value summaries (min/max/mean/last) are used.
  * pyxis (meds given in the ED): heparin, nitrates, diuretics, rate control are treatment decisions that
    proxy the diagnosis. Revisit only as an explicitly labeled "treatment-aware" ablation.
  * race: not used as a predictor (fairness; report subgroup performance instead).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from ..common import FEATURES_DIR, filter_before_tpred, is_fresh

VITALS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
# Physiologic plausibility ranges (values outside -> NaN). Temperature in Fahrenheit.
RANGES = {"temperature": (85, 110), "heartrate": (20, 300), "resprate": (4, 80), "o2sat": (50, 100),
          "sbp": (40, 300), "dbp": (20, 200), "pain": (0, 10)}
RHYTHM_PATTERNS = {
    "afib": r"a\.?\s?-?fib|atrial\s*fib|\baf\b|\bafib\b",
    "aflutter": r"flutter|\bafl\b",
    "paced": r"pac(?:ed|ing|er)|\bv-?pace",
    "sinus": r"\bsinus\b|\bnsr\b|\bsr\b|\bst\b|\bsb\b",
}
ED_DIR_GLOB = "mimic-iv-ed/*/ed/{}.csv.gz"


def _ed_table(name: str, **kw) -> pd.DataFrame:
    from grace_data.paths import _find
    p = _find(ED_DIR_GLOB.format(name))
    if p is None:
        raise FileNotFoundError(f"MIMIC-IV-ED table {name} not found")
    return pd.read_csv(p, **kw)


def clean_vitals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "temperature" in df:
        t = df["temperature"]
        df.loc[t.between(25, 45), "temperature"] = t * 9 / 5 + 32  # charted in Celsius
    if "pain" in df:
        df["pain"] = pd.to_numeric(df["pain"], errors="coerce")
    for c, (lo, hi) in RANGES.items():
        if c in df:
            df.loc[~df[c].between(lo, hi), c] = np.nan
    return df


def triage_block(cohort: pd.DataFrame) -> pd.DataFrame:
    tri = _ed_table("triage").set_index("stay_id")
    tri = tri.reindex(cohort.index)
    out = clean_vitals(tri[VITALS + ["pain"]]).add_prefix("tri_")
    out["tri_pain_nonnumeric"] = tri["pain"].notna() & pd.to_numeric(tri["pain"], errors="coerce").isna()
    out["tri_acuity"] = tri["acuity"]
    out["tri_shock_index"] = out.tri_heartrate / out.tri_sbp
    out["tri_pulse_pressure"] = out.tri_sbp - out.tri_dbp

    st = _ed_table("edstays", usecols=["stay_id", "arrival_transport"]).set_index("stay_id").reindex(cohort.index)
    for v in ("AMBULANCE", "HELICOPTER", "WALK IN", "OTHER"):
        out[f"arrival_{v.lower().replace(' ', '_')}"] = st.arrival_transport.eq(v)
    out["age"] = cohort["age"]
    out["male"] = cohort["gender"].eq("M")
    return out


def vitals_block(cohort: pd.DataFrame) -> pd.DataFrame:
    v = _ed_table("vitalsign", parse_dates=["charttime"])
    v = filter_before_tpred(v, cohort, "charttime", lookback_h=None)  # vitals all belong to this stay
    v = clean_vitals(v).sort_values("charttime")
    g = v.groupby("stay_id")
    cols = VITALS + ["pain"]
    agg = g[cols].agg(["min", "max", "mean", "last"])
    agg.columns = [f"vs_{c}_{s}" for c, s in agg.columns]
    rh = v.rhythm.fillna("").str.lower()
    for k, pat in RHYTHM_PATTERNS.items():
        v[f"rh_{k}"] = rh.str.contains(pat, regex=True)
    v["rh_documented"] = rh.str.len() > 0
    rflags = v.groupby("stay_id")[[c for c in v.columns if c.startswith("rh_")]].any()
    return agg.join(rflags, how="outer").reindex(cohort.index)


def chief_complaint_text(cohort: pd.DataFrame) -> pd.Series:
    tri = _ed_table("triage", usecols=["stay_id", "chiefcomplaint"]).set_index("stay_id")
    return tri.chiefcomplaint.reindex(cohort.index).fillna("").str.lower()


def cc_tokens_block(cohort: pd.DataFrame, min_df: int = 200, max_features: int = 300) -> pd.DataFrame:
    """Binary chief-complaint unigrams/bigrams. Vocabulary fit on TRAIN stays only."""
    text = chief_complaint_text(cohort)
    vec = CountVectorizer(binary=True, ngram_range=(1, 2), min_df=min_df, max_features=max_features,
                          token_pattern=r"(?u)\b[a-z][a-z/]+\b")
    vec.fit(text[cohort.split == "train"])
    M = vec.transform(text)
    names = [f"cc_{re.sub(r'[^a-z0-9]+', '_', t)}" for t in vec.get_feature_names_out()]
    return pd.DataFrame(M.toarray().astype(np.uint8), index=cohort.index, columns=names)


def medrecon_block(cohort: pd.DataFrame, min_stays: int = 500) -> pd.DataFrame:
    """Home-medication classes (ETC description) reconciled at or before t_pred.
    Strong for y_afib because chronic AF on anticoagulants counts as positive (label caveat)."""
    m = _ed_table("medrecon", usecols=["stay_id", "charttime", "etcdescription"], parse_dates=["charttime"])
    m = filter_before_tpred(m, cohort, "charttime", lookback_h=None)
    m = m.dropna(subset=["etcdescription"]).drop_duplicates(["stay_id", "etcdescription"])
    train_ids = cohort.index[cohort.split == "train"]
    counts = m[m.stay_id.isin(train_ids)].etcdescription.value_counts()
    keep = counts[counts >= min_stays].index
    wide = pd.crosstab(m.stay_id, m.etcdescription.where(m.etcdescription.isin(keep), "_other")).clip(upper=1)
    wide.columns = [f"med_{re.sub(r'[^a-z0-9]+', '_', c.lower()).strip('_')}" for c in wide.columns]
    n = m.groupby("stay_id").size().rename("med_n_classes")
    out = wide.join(n, how="outer").reindex(cohort.index)
    out["med_recon_done"] = out.med_n_classes.notna()
    return out.fillna(0)


BLOCKS = {"triage": triage_block, "vitals": vitals_block, "cc": cc_tokens_block, "medrecon": medrecon_block}
DEFAULT_BLOCKS = ("triage", "vitals", "cc")


def build(cohort: pd.DataFrame, window_h: int, blocks=DEFAULT_BLOCKS, cache: bool = True) -> pd.DataFrame:
    """Stay-level structured feature table (index stay_id). Every cohort stay gets a row
    (triage exists for all ED stays); missing values stay NaN for the GBM to handle."""
    tag = "-".join(blocks)
    path = FEATURES_DIR / f"structured_{tag}_{window_h}h.parquet"
    if cache and is_fresh(path, window_h):
        X = pd.read_parquet(path)
        if set(cohort.index) <= set(X.index):
            return X.loc[cohort.index]
    parts = [BLOCKS[b](cohort) for b in blocks]
    X = pd.concat(parts, axis=1)
    X = X.astype({c: "float32" for c in X.columns if X[c].dtype != bool})
    X = X.astype({c: "uint8" for c in X.columns if X[c].dtype == bool})
    if cache:
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        X.to_parquet(path)
    return X
