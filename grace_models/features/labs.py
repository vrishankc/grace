"""Blood-lab features for GRACE: one row per ED stay, using only labs RESULTED (storetime) by t_pred.

Put this file at grace_models/features/labs.py. Used by `train.py labs` and `train.py structured-labs`.

Design (matches grace_data/README.md and the leakage rules in ONBOARDING.md):
  * Source: MIMIC-IV hosp labevents. ED labs often have no hadm_id, so events are matched to stays by
    subject_id + time with common.attach_events_by_subject (window [intime - lookback_h, t_pred]).
  * Time column: storetime (when the result became available), never charttime.
  * Panel: LAB_SPEC below. Item IDs are resolved from d_labitems by exact label (fluid = Blood), and the
    resolved table is printed so you can sanity-check it. The four troponin/BNP item IDs used by the
    cohort rule (50963, 51002, 51003, 52642) must all be picked up, or a warning is raised.
  * Values: valuenum; when that is missing but the text is "<0.01" or ">35000", the threshold is used
    (e.g. a troponin below the detection limit is "low", not "missing").
  * Per lab and stay: last (by charttime), min, max. Optional troponin delta = last - first, only when
    there are >= 2 results (serial troponins are the standard AMI workup).
  * NOT included: number of tests, time since arrival, or any timing feature (they encode ED length of
    stay -> disposition). Labs never measured stay missing (XGBoost handles NaN natively).
  * Stays with no qualifying lab are absent from the index, as for every other modality.

Caching (derived/features/, git-ignored, contains MIMIC data):
  * labs_events_<hash>.parquet: labevents filtered to the panel and to 24 h-cohort patients (the 24 h
    cohort is a superset of the 6 h one). Rebuilt if the panel changes or the cohort is newer.
  * labs_<W>h_lb<L>[_nodelta].parquet: the stay-level table for the FULL window cohort; an
    --admitted-only run just subsets it.
"""
from __future__ import annotations

import hashlib
import json
import re
import warnings

import numpy as np
import pandas as pd

from ..common import (FEATURES_DIR, GRACE_ROOT, attach_events_by_subject, check_no_leakage, is_fresh,
                      load_cohort)

# feature name -> exact d_labitems labels (fluid = Blood). Several labels per feature merge the
# chemistry and blood-gas versions of the same analyte.
LAB_SPEC: dict[str, list[str]] = {
    # cardiac
    "troponin_t": ["Troponin T"],
    "troponin_i": ["Troponin I"],
    "ntprobnp": ["NTproBNP"],
    "ck_mb": ["Creatine Kinase, MB Isoenzyme"],
    # chemistry / electrolytes / renal
    "creatinine": ["Creatinine"],
    "bun": ["Urea Nitrogen"],
    "potassium": ["Potassium", "Potassium, Whole Blood"],
    "sodium": ["Sodium", "Sodium, Whole Blood"],
    "chloride": ["Chloride", "Chloride, Whole Blood"],
    "bicarbonate": ["Bicarbonate"],
    "glucose": ["Glucose"],
    "magnesium": ["Magnesium"],
    "calcium": ["Calcium, Total"],
    "lactate": ["Lactate"],
    "tsh": ["Thyroid Stimulating Hormone"],
    # hematology
    "wbc": ["White Blood Cells", "WBC Count"],
    "hemoglobin": ["Hemoglobin"],
    "platelets": ["Platelet Count"],
}
DELTA_FEATURES = ("troponin_t", "troponin_i")
COHORT_CARDIAC_ITEMIDS = {50963, 51002, 51003, 52642}  # used by the cohort rule; must be in the panel
STATS = ("last", "min", "max")
_THRESHOLD_RE = r"^\s*[<>]=?\s*([0-9]*\.?[0-9]+)"
_TIME_FMT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------- files and item IDs

def _find_hosp(name: str):
    """Highest-version mimiciv/<v>/hosp/<name> under either download root."""
    roots = [GRACE_ROOT / "datasets" / "files", GRACE_ROOT / "datasets" / "physionet.org" / "files"]
    hits = [p for r in roots for p in r.glob(f"mimiciv/*/hosp/{name}")]
    if not hits:
        raise FileNotFoundError(f"mimiciv/*/hosp/{name} not found under {[str(r) for r in roots]}")

    def version(p):
        return tuple(int(x) for x in re.findall(r"\d+", p.parent.parent.name))

    return max(hits, key=version)


def resolve_itemids(verbose: bool = True) -> dict[str, list[int]]:
    """feature -> item IDs, from d_labitems by exact label with fluid Blood."""
    d = pd.read_csv(_find_hosp("d_labitems.csv.gz"))
    d = d[d["fluid"].astype(str).str.lower().eq("blood")]
    mapping, rows = {}, []
    for feat, labels in LAB_SPEC.items():
        hit = d[d["label"].isin(labels)]
        mapping[feat] = sorted(int(i) for i in hit["itemid"])
        if not mapping[feat]:
            warnings.warn(f"labs: no d_labitems match for {feat} {labels}; feature will be absent")
        for _, r in hit.iterrows():
            rows.append((feat, int(r["itemid"]), r["label"], r.get("category", "")))

    missing = COHORT_CARDIAC_ITEMIDS - set().union(*mapping.values())
    if missing:
        warnings.warn(f"labs: cohort troponin/BNP item IDs not in the panel: {sorted(missing)}. "
                      "Check d_labitems and add their labels to LAB_SPEC.")
    if verbose:  # d_labitems is a dictionary table, not patient data
        print("labs: resolved item IDs")
        for feat, itemid, label, cat in rows:
            print(f"  {feat:<12} {itemid:>6}  {label}  [{cat}]")
    return mapping


# ---------------------------------------------------------------- raw events

def _load_events(mapping: dict[str, list[int]], subjects: np.ndarray, cache: bool) -> pd.DataFrame:
    """labevents rows for the panel and the given patients, compact dtypes."""
    key = hashlib.md5(json.dumps(mapping, sort_keys=True).encode()).hexdigest()[:8]
    path = FEATURES_DIR / f"labs_events_{key}.parquet"
    if cache and is_fresh(path, 24):
        return pd.read_parquet(path)

    id2feat = {i: f for f, ids in mapping.items() for i in ids}
    feat_cat = pd.CategoricalDtype(list(LAB_SPEC))
    src = _find_hosp("labevents.csv.gz")
    print(f"labs: reading {src} (several minutes)", flush=True)

    usecols = ["subject_id", "itemid", "charttime", "storetime", "value", "valuenum"]
    dtype = {"subject_id": "int64", "itemid": "int64", "charttime": "string", "storetime": "string",
             "value": "string", "valuenum": "float64"}
    parts, kept = [], 0
    for i, ch in enumerate(pd.read_csv(src, usecols=usecols, dtype=dtype, chunksize=5_000_000)):
        ch = ch[ch["itemid"].isin(id2feat.keys()) & ch["subject_id"].isin(subjects)]
        if ch.empty:
            continue
        val = ch["valuenum"].copy()
        need = val.isna() & ch["value"].notna()
        if need.any():
            val.loc[need] = pd.to_numeric(ch.loc[need, "value"].str.extract(_THRESHOLD_RE)[0],
                                          errors="coerce")
        out = pd.DataFrame({
            "subject_id": ch["subject_id"].astype("int32"),
            "feat": ch["itemid"].map(id2feat).astype(feat_cat),
            "charttime": pd.to_datetime(ch["charttime"], format=_TIME_FMT, errors="coerce"),
            "storetime": pd.to_datetime(ch["storetime"], format=_TIME_FMT, errors="coerce"),
            "valuenum": val.astype("float32"),
        }).dropna(subset=["storetime", "charttime", "valuenum"])
        parts.append(out)
        kept += len(out)
        print(f"  chunk {i + 1}: kept {kept:,} rows so far", flush=True)

    ev = pd.concat(parts, ignore_index=True)
    if cache:
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        ev.to_parquet(path, index=False)
    return ev


# ---------------------------------------------------------------- stay-level features

def build(cohort: pd.DataFrame, window_h: int, cache: bool = True, lookback_h: float = 0.0,
          deltas: bool = True) -> pd.DataFrame:
    """Stay-level lab table for `cohort` (indexed by stay_id), using labs with storetime <= t_pred."""
    tag = f"labs_{window_h}h_lb{lookback_h:g}{'' if deltas else '_nodelta'}"
    out = FEATURES_DIR / f"{tag}.parquet"

    if cache and is_fresh(out, window_h):
        X = pd.read_parquet(out)
    else:
        full = load_cohort(window_h)  # build for the full cohort; admitted-only runs subset below
        mapping = resolve_itemids(verbose=True)
        subjects = load_cohort(24)["subject_id"].unique()  # 24 h cohort is a superset of 6 h
        ev = _load_events(mapping, subjects, cache)

        ev = attach_events_by_subject(ev, full, "storetime", lookback_h=lookback_h)
        ev = ev.sort_values(["stay_id", "feat", "charttime"])
        g = ev.groupby(["stay_id", "feat"], observed=True)["valuenum"]
        agg = g.agg(["first", "last", "min", "max", "count"]).unstack("feat")

        cols = {}
        for feat in LAB_SPEC:
            if feat not in agg["last"].columns:
                continue
            for stat in STATS:
                cols[f"{feat}_{stat}"] = agg[stat][feat]
            if deltas and feat in DELTA_FEATURES:
                delta = agg["last"][feat] - agg["first"][feat]
                cols[f"{feat}_delta"] = delta.where(agg["count"][feat] >= 2)
        X = pd.DataFrame(cols).astype("float32")
        X.index = X.index.astype("int64")
        X.index.name = "stay_id"
        if cache:
            FEATURES_DIR.mkdir(parents=True, exist_ok=True)
            X.to_parquet(out)

    X = X.loc[X.index.intersection(cohort.index)]
    check_no_leakage(X)
    print(f"labs: {X.shape[1]} features for {len(X):,}/{len(cohort):,} stays "
          f"({len(X) / max(len(cohort), 1):.1%})", flush=True)
    return X
