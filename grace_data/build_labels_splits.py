"""Build GRACE ED-stay labels, cohort flags and patient-level splits.

Run from project/grace:
    ~/miniconda3/envs/torch/bin/python -m grace_data.build_labels_splits

Outputs (derived/labels_splits/):
    ed_stays_labeled.parquet   one row per MIMIC-IV-ED stay, all columns (see README)
    cohort_6h.csv, cohort_24h.csv   included stays for each prediction window
    subject_splits.csv         subject_id -> split for every ED patient
    label_definitions.json     ICD code lists + run metadata
    REPORT.md                  counts, prevalence, coverage, checks

Rules (see README.md in this package):
  * One row per ED stay; all stays of a patient kept and placed in the same split.
  * Prediction time t_pred = ED arrival (intime) + W, W in {6h, 24h}.
  * Labels: admitted (hadm_id present) -> ICD codes of the linked hospitalization;
            not admitted -> ED diagnosis ICD codes for the stay.
  * Cohort(W): valid times, adult (>=18 at arrival), label available,
               >=1 cardiac modality (ECG, CXR, echo, troponin/BNP) with time in [intime, intime+W].
"""
from __future__ import annotations

import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from . import icd
from .paths import DERIVED, sources
from .splits import FRACTIONS, SALT, assign_split

WINDOWS_H = (6, 24)
CARDIAC_MODALITIES = ["ecg", "cxr", "echo", "cardiac_lab"]
OTHER_MODALITIES = ["rad_note"]
CARDIAC_LAB_PATTERN = r"troponin|natriuretic|ntprobnp|nt-probnp|\bbnp\b"

log_lines: list[str] = []


def log(msg: str = ""):
    print(msg, flush=True)
    log_lines.append(msg)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_edstays(p) -> pd.DataFrame:
    ed = pd.read_csv(p["edstays"], parse_dates=["intime", "outtime"])
    ed["hadm_id"] = ed["hadm_id"].astype("Int64")
    ed["admitted"] = ed["hadm_id"].notna()
    ed["ed_los_h"] = (ed.outtime - ed.intime).dt.total_seconds() / 3600
    ed["valid_times"] = ed.intime.notna() & ed.outtime.notna() & (ed.ed_los_h > 0)
    return ed


def load_ecg_ext(p) -> pd.DataFrame | None:
    if p["ecg_ext"] is None:
        return None
    x = pd.read_csv(
        p["ecg_ext"],
        usecols=["subject_id", "study_id", "ecg_time", "ed_stay_id", "ed_diag_hosp",
                 "anchor_age", "anchor_year"],
        parse_dates=["ecg_time"],
    )
    return x


# ---------------------------------------------------------------------------
# Age
# ---------------------------------------------------------------------------
def add_age(ed, p, ecg_ext) -> str:
    if p["patients"] is not None:
        pt = pd.read_csv(p["patients"], usecols=["subject_id", "anchor_age", "anchor_year", "dod"])
        src = "hosp/patients"
    elif ecg_ext is not None:
        pt = ecg_ext[["subject_id", "anchor_age", "anchor_year"]].dropna().drop_duplicates("subject_id")
        src = "ecg_ext (PROVISIONAL: only patients with any ECG)"
    else:
        ed["age"] = np.nan
        return "none"
    ed_ = ed.merge(pt[["subject_id", "anchor_age", "anchor_year"]], on="subject_id", how="left")
    ed["age"] = (ed_.anchor_age + (ed_.intime.dt.year - ed_.anchor_year)).values
    return src


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def add_labels(ed, p, ecg_ext) -> str:
    # ED-code flags for every stay
    dx = pd.read_csv(p["ed_diagnosis"], usecols=["stay_id", "icd_code", "icd_version"])
    ed_flags = icd.aggregate_flags(dx, "stay_id")
    for f in icd.ALL_FLAGS:
        ed[f"edcode_{f}"] = ed.stay_id.map(ed_flags[f]).astype("boolean")
    has_ed_codes = ed.stay_id.isin(ed_flags.index)

    # Hospitalization-code flags for admitted stays
    hosp = pd.DataFrame(index=ed.index, columns=icd.ALL_FLAGS, dtype="boolean")
    hosp_src = pd.Series(pd.NA, index=ed.index, dtype="object")
    if p["diagnoses_icd"] is not None:
        d = pd.read_csv(p["diagnoses_icd"], usecols=["hadm_id", "icd_code", "icd_version"])
        d = d[d.hadm_id.isin(ed.hadm_id.dropna())]
        hf = icd.aggregate_flags(d, "hadm_id")
        m = ed.admitted & ed.hadm_id.isin(hf.index)
        for f in icd.ALL_FLAGS:
            hosp.loc[m, f] = ed.loc[m, "hadm_id"].map(hf[f]).values
        hosp_src[m] = "hosp"
        mode = "hosp/diagnoses_icd"
    elif ecg_ext is not None:
        e = ecg_ext.dropna(subset=["ed_stay_id"]).drop_duplicates("ed_stay_id")
        e = e[e.ed_diag_hosp.fillna("[]") != "[]"].reset_index(drop=True)
        long = icd.explode_list_column(e.ed_diag_hosp)
        long["stay_id"] = e.loc[long.idx, "ed_stay_id"].astype("int64").values
        long["icd_version"] = 10
        hf = icd.aggregate_flags(long, "stay_id")
        m = ed.admitted & ed.stay_id.isin(hf.index)
        for f in icd.ALL_FLAGS:
            hosp.loc[m, f] = ed.loc[m, "stay_id"].map(hf[f]).values
        hosp_src[m] = "hosp_via_ecg_ext"
        mode = "ecg_ext ed_diag_hosp (PROVISIONAL: admitted stays without an ED ECG are unlabeled)"
    else:
        mode = "none"

    # Combine
    src = pd.Series("missing", index=ed.index, dtype="object")
    src[~ed.admitted & has_ed_codes] = "ed"
    src[hosp_src.notna()] = hosp_src[hosp_src.notna()]
    # admitted, hosp module present, but hadm has no codes -> fall back to ED codes
    if p["diagnoses_icd"] is not None:
        fb = ed.admitted & hosp_src.isna() & has_ed_codes
        src[fb] = "ed_fallback"
    elif ed.admitted.any():
        src[ed.admitted & hosp_src.isna()] = "pending_hosp"
    ed["label_source"] = src

    for f in icd.ALL_FLAGS:
        col = pd.Series(pd.NA, index=ed.index, dtype="boolean")
        use_ed = src.isin(["ed", "ed_fallback"])
        use_hosp = src.isin(["hosp", "hosp_via_ecg_ext"])
        col[use_ed] = ed.loc[use_ed, f"edcode_{f}"].fillna(False)
        col[use_hosp] = hosp.loc[use_hosp, f]
        name = {"ami": "y_ami", "ahf": "y_ahf", "afib": "y_afib"}.get(f, f"y2_{f}")
        ed[name] = col
    ed["label_available"] = src.isin(["ed", "ed_fallback", "hosp", "hosp_via_ecg_ext"])
    return mode


# ---------------------------------------------------------------------------
# Modality event times
# ---------------------------------------------------------------------------
def modality_events(p, ecg_ext) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    ev, notes = {}, {}

    if ecg_ext is not None:
        ev["ecg"] = ecg_ext[["subject_id", "ecg_time"]].rename(columns={"ecg_time": "t"})
        notes["ecg"] = "mimic-iv-ecg-ext-icd-labels ecg_time (all MIMIC-IV-ECG records)"
    elif p["ecg_records"] is not None:
        r = pd.read_csv(p["ecg_records"], usecols=["subject_id", "ecg_time"], parse_dates=["ecg_time"])
        ev["ecg"] = r.rename(columns={"ecg_time": "t"})
        notes["ecg"] = "mimic-iv-ecg record_list ecg_time"

    if p["cxr_meta"] is not None:
        c = pd.read_csv(p["cxr_meta"], usecols=["subject_id", "study_id", "StudyDate", "StudyTime"])
        hhmmss = c.StudyTime.fillna(0).astype(float).astype(int).astype(str).str.zfill(6)
        c["t"] = pd.to_datetime(c.StudyDate.astype(int).astype(str) + hhmmss, format="%Y%m%d%H%M%S",
                                errors="coerce")
        ev["cxr"] = c.groupby("study_id").agg(subject_id=("subject_id", "first"), t=("t", "min")).reset_index(drop=True)
        notes["cxr"] = "mimic-cxr-jpg metadata StudyDate+StudyTime (image acquisition)"

    for key in ("echo_studies", "echo_records"):
        if p[key] is not None:
            e = pd.read_csv(p[key])
            tcol = next((c for c in ("study_datetime", "acquisition_datetime") if c in e.columns), None)
            if tcol is None:
                tcol = next((c for c in e.columns if "datetime" in c or "time" in c), None)
            if tcol is not None:
                e["t"] = pd.to_datetime(e[tcol], errors="coerce")
                ev["echo"] = e[["subject_id", "t"]]
                notes["echo"] = f"{p[key].name} {tcol}"
                break

    if p["labevents"] is not None and p["d_labitems"] is not None:
        items = pd.read_csv(p["d_labitems"])
        ids = items.loc[items.label.fillna("").str.lower().str.contains(CARDIAC_LAB_PATTERN, regex=True), "itemid"]
        parts = []
        for ch in pd.read_csv(p["labevents"], usecols=["subject_id", "itemid", "charttime", "storetime"],
                              chunksize=5_000_000):
            ch = ch[ch.itemid.isin(ids)]
            if len(ch):
                t = pd.to_datetime(ch.storetime, errors="coerce").fillna(pd.to_datetime(ch.charttime, errors="coerce"))
                parts.append(pd.DataFrame({"subject_id": ch.subject_id.values, "t": t.values}))
        ev["cardiac_lab"] = pd.concat(parts, ignore_index=True)
        notes["cardiac_lab"] = (f"labevents storetime (result available) for itemids {sorted(ids.tolist())} "
                                f"matching /{CARDIAC_LAB_PATTERN}/")

    if p["radiology"] is not None:
        r = pd.read_csv(p["radiology"], usecols=["subject_id", "charttime", "storetime"])
        t = pd.to_datetime(r.storetime, errors="coerce").fillna(pd.to_datetime(r.charttime, errors="coerce"))
        ev["rad_note"] = pd.DataFrame({"subject_id": r.subject_id, "t": t})
        notes["rad_note"] = "mimic-iv-note radiology storetime (report signed/available)"

    return ev, notes


def add_modality_flags(ed, events):
    base = ed[["stay_id", "subject_id", "intime"]].dropna(subset=["intime"]).sort_values("intime")
    for mod in CARDIAC_MODALITIES + OTHER_MODALITIES:
        if mod not in events:
            for W in WINDOWS_H:
                ed[f"has_{mod}_{W}h"] = pd.Series(pd.NA, index=ed.index, dtype="boolean")
            ed[f"h_to_first_{mod}"] = np.nan
            continue
        e = events[mod].dropna(subset=["t"]).sort_values("t")
        e = e.assign(subject_id=e.subject_id.astype("int64"))
        m = pd.merge_asof(base, e.rename(columns={"t": "t_first"}), left_on="intime", right_on="t_first",
                          by="subject_id", direction="forward")
        h = (m.t_first - m.intime).dt.total_seconds() / 3600
        h = pd.Series(h.values, index=m.stay_id.values)
        hrs = ed.stay_id.map(h)
        ed[f"h_to_first_{mod}"] = hrs.where(hrs <= max(WINDOWS_H))
        for W in WINDOWS_H:
            ed[f"has_{mod}_{W}h"] = (hrs <= W).fillna(False).astype("boolean")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = sources()
    missing_req = [k for k in ("edstays", "ed_diagnosis") if p[k] is None]
    if missing_req:
        sys.exit(f"Missing required inputs: {missing_req}")
    DERIVED.mkdir(parents=True, exist_ok=True)

    log("# GRACE labels & splits build report")
    log(f"\nBuilt: {datetime.now():%Y-%m-%d %H:%M}\n")
    log("## Inputs")
    for k, v in p.items():
        log(f"- {k}: {'`' + str(v.relative_to(v.parents[3])) + '`' if v else '**missing**'}")

    ed = load_edstays(p)
    ecg_ext = load_ecg_ext(p)
    age_src = add_age(ed, p, ecg_ext)
    label_mode = add_labels(ed, p, ecg_ext)
    events, ev_notes = modality_events(p, ecg_ext)
    add_modality_flags(ed, events)

    provisional = p["diagnoses_icd"] is None or p["patients"] is None
    log("\n## Mode")
    log(f"- Age source: {age_src}")
    log(f"- Admitted-stay label source: {label_mode}")
    for m in CARDIAC_MODALITIES + OTHER_MODALITIES:
        log(f"- Modality `{m}`: {ev_notes.get(m, '**not available yet** (flags are NA)')}")
    if provisional:
        log("\n> **PROVISIONAL BUILD.** hosp `patients` and/or `diagnoses_icd` are missing. "
            "Do not train/evaluate on these cohorts; rerun after downloading MIMIC-IV hosp.")
    avail_cardiac = [m for m in CARDIAC_MODALITIES if m in events]
    if set(avail_cardiac) != set(CARDIAC_MODALITIES):
        log(f"\n> Cohort modality criterion currently uses only: {avail_cardiac}. "
            "Cohort will grow when the other timing files are added.")

    # Splits
    ed["split"] = assign_split(ed.subject_id).values

    # Cohort flags
    ed["adult"] = ed.age >= 18
    for W in WINDOWS_H:
        anymod = pd.Series(False, index=ed.index)
        for m in avail_cardiac:
            anymod |= ed[f"has_{m}_{W}h"].fillna(False).astype(bool)
        ed[f"any_cardiac_mod_{W}h"] = anymod
        ed[f"t_pred_{W}h"] = ed.intime + pd.Timedelta(hours=W)
        ed[f"in_cohort_{W}h"] = (ed.valid_times & ed.adult.fillna(False) & ed.label_available & anymod)

    # ---------------- checks ----------------
    assert ed.stay_id.is_unique
    assert ed.groupby("subject_id").split.nunique().max() == 1, "patient in >1 split"
    for W in WINDOWS_H:
        assert not (ed[f"in_cohort_{W}h"] & ed.y_ami.isna()).any()
    assert (ed.in_cohort_6h <= ed.in_cohort_24h).all(), "6h cohort must be subset of 24h cohort"

    # ---------------- report ----------------
    log("\n## Attrition")
    n0 = len(ed)
    steps = [("All MIMIC-IV-ED stays", pd.Series(True, index=ed.index))]
    c = ed.valid_times.copy(); steps.append(("valid in/out times", c.copy()))
    c &= ed.adult.fillna(False); steps.append(("age >= 18 (age known)", c.copy()))
    c &= ed.label_available; steps.append(("label available", c.copy()))
    log("| step | stays | patients |\n|---|---|---|")
    for name, mask in steps:
        log(f"| {name} | {mask.sum():,} | {ed.loc[mask, 'subject_id'].nunique():,} |")
    for W in WINDOWS_H:
        mask = ed[f"in_cohort_{W}h"]
        log(f"| + >=1 cardiac modality within {W}h | {mask.sum():,} | {ed.loc[mask, 'subject_id'].nunique():,} |")

    log("\n## Label source (all stays)")
    log(ed.label_source.value_counts().rename_axis("source").to_frame("stays").to_markdown())

    for W in WINDOWS_H:
        co = ed[ed[f"in_cohort_{W}h"]]
        log(f"\n## Cohort {W}h")
        g = co.groupby("split").agg(stays=("stay_id", "size"), patients=("subject_id", "nunique"),
                                    ami=("y_ami", "mean"), ahf=("y_ahf", "mean"), afib=("y_afib", "mean"),
                                    hf_broad=("y2_hf_broad", "mean"), admitted=("admitted", "mean"))
        g.loc["all"] = [len(co), co.subject_id.nunique(), co.y_ami.mean(), co.y_ahf.mean(), co.y_afib.mean(),
                        co.y2_hf_broad.mean(), co.admitted.mean()]
        g = g.reindex(["train", "val", "test", "all"])
        log("Prevalence (fraction of stays):\n")
        log(g.to_markdown(floatfmt=".4f"))
        log("\nPositives by label source:\n")
        log(co.groupby("label_source")[["y_ami", "y_ahf", "y_afib"]].agg(["sum", "mean"])
            .to_markdown(floatfmt=".4f"))
        cov = {m: co[f"has_{m}_{W}h"].mean() if m in events else np.nan
               for m in CARDIAC_MODALITIES + OTHER_MODALITIES}
        log("\nModality coverage within window (fraction of cohort stays):\n")
        log(pd.Series(cov, name="coverage").to_frame().to_markdown(floatfmt=".3f"))

    sp = ed.drop_duplicates("subject_id").split.value_counts(normalize=True)
    log(f"\n## Split fractions (patients): " + ", ".join(f"{k} {sp.get(k, 0):.3f}" for k in FRACTIONS))
    log("\n## Checks passed\n- stay_id unique\n- every patient in exactly one split\n"
        "- cohort stays all have labels\n- 6h cohort subset of 24h cohort")

    # ---------------- write ----------------
    ed.to_parquet(DERIVED / "ed_stays_labeled.parquet", index=False)
    keep = ["subject_id", "stay_id", "hadm_id", "split", "intime", "outtime", "admitted", "disposition",
            "age", "gender", "label_source", "y_ami", "y_ahf", "y_afib", "y2_hf_broad", "y2_aflutter"]
    for W in WINDOWS_H:
        cols = keep + [f"t_pred_{W}h"] + [f"has_{m}_{W}h" for m in CARDIAC_MODALITIES + OTHER_MODALITIES]
        ed.loc[ed[f"in_cohort_{W}h"], cols].to_csv(DERIVED / f"cohort_{W}h.csv", index=False)
    ed.drop_duplicates("subject_id")[["subject_id", "split"]].sort_values("subject_id") \
        .to_csv(DERIVED / "subject_splits.csv", index=False)
    meta = {
        "built": datetime.now().isoformat(timespec="seconds"),
        "provisional": provisional,
        "split": {"method": "sha256(salt:subject_id)", "salt": SALT, "fractions": FRACTIONS},
        "windows_h": WINDOWS_H,
        "cardiac_modalities_used": avail_cardiac,
        "label_mode_admitted": label_mode,
        "age_source": age_src,
        "icd_definitions": icd.definitions_for_report(),
        "notes": {"ami_icd9": "410.x2 (subsequent episode) excluded"},
    }
    (DERIVED / "label_definitions.json").write_text(json.dumps(meta, indent=2))
    (DERIVED / "REPORT.md").write_text("\n".join(log_lines) + "\n")
    log(f"\nWrote outputs to {DERIVED}")


if __name__ == "__main__":
    main()
