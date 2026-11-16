# GRACE: Labels and Splits Progress Log

*Last updated: 2026-11-16 · Status: **FINAL for current data** (hosp 3.1 labels, troponin/BNP + ECG cohort rule). Cohort will grow again when CXR-JPG / ECHO timing files are added.*

---

## 1. Summary

I built and ran a pipeline that:
1. labels every MIMIC-IV-ED visit for **acute MI (AMI)**, **acute heart failure (AHF)** and **atrial fibrillation (AFib)**,
2. marks which visits belong to the **6 h** and **24 h** prediction cohorts, and
3. assigns every patient to **train / val / test** (70/15/15), with all of a patient's visits in the same split.

The first run (00:18) was provisional. It was rerun at 01:49 with MIMIC-IV hosp 3.1, labevents and MIMIC-IV-Note radiology (section 6). All admitted visits now get hospital-code labels. The provisional outputs are kept in `derived/labels_splits_provisional_20261116/` for comparison.

---

## 2. Data on disk

Location: `project/grace/datasets/physionet.org/files/`

| Dataset | What's there | What it gives us |
|---|---|---|
| **MIMIC-IV-ED 2.2** | Complete: `edstays`, `triage`, `vitalsign`, `diagnosis`, `medrecon`, `pyxis` | All 425,087 ED visits (205,504 patients), arrival/departure times, admitted or not (`hadm_id`), ED diagnosis ICD codes |
| **MIMIC-IV-ECG-Ext-ICD 1.0.1** | `records_w_diag_icd10.csv` (800,035 ECGs; checksum verified) | Timestamp of **every** MIMIC-IV ECG; hospital ICD-10 codes for admitted visits that had an ED ECG (98k of 203k admitted visits); patient age info |
| **MIMIC-CXR 2.1.0** | Record/study/provider lists and `mimic-cxr-reports.zip` only | Report text. **No images and no timestamps.** The `files/p10–p19` folders hold only HTML index pages and ~10.8k report `.txt` files, which duplicate the zip. |

**Notes from the inspection:**
- When I first read the ECG-Ext file it failed to parse because it was still downloading. Once it finished, the checksum matched PhysioNet's.
- **59 ED visits** have departure before arrival; they're excluded as invalid.
- **575 hospitalizations** are linked to two ED visits. Both visits get the same label, and they always share a split because they belong to the same patient.
- `hadm_id` doesn't line up exactly with `disposition` (e.g. 36k "HOME" visits have a `hadm_id`). "Admitted" is defined as **`hadm_id` present**.
- In the ECG-Ext data, the median ECG comes **22 minutes** after ED arrival (95th percentile 7.7 h).

---

## 3. Decisions

| Question | Decision |
|---|---|
| Patients sent home (no `hadm_id`) | Label them from **ED diagnosis ICD codes** |
| Prediction time | **Two windows:** ED arrival + **6 h** and arrival + **24 h** |
| Repeat visits | **Keep all visits**; every visit of a patient goes in the **same split** |

Team plan this builds on: MIMIC-IV ecosystem; adult ED stays with at least one cardiac-relevant modality within the window; labels from hospitalization ICD codes; no data after the prediction time; no discharge summaries; patient-level splits.

---

## 4. Design

### 4.1 Unit and timing
- **One row per ED visit** (`stay_id`).
- **Prediction time:** `t_pred = intime + W`, for W ∈ {6 h, 24 h}.
- A feature may only use data **timestamped ≤ `t_pred`**. For reports and labs, that means the time the result became available (`storetime`), not when the sample was taken.

### 4.2 Labels
| Visit type | Label source | `label_source` value |
|---|---|---|
| Admitted, hospital codes present | Hospital ICD codes (`hosp/diagnoses_icd`, ICD-9 + ICD-10) | `hosp` |
| Admitted, hospital has no codes | Fall back to ED codes | `ed_fallback` |
| Not admitted | ED diagnosis ICD codes | `ed` |
| *Provisional only:* admitted, hosp module missing, but had an ED ECG | Hospital codes from ECG-Ext `ed_diag_hosp` | `hosp_via_ecg_ext` |
| *Provisional only:* admitted, hosp module missing, no ECG | Not labeled yet, so excluded | `pending_hosp` |

### 4.3 ICD definitions (codes stored without dots)
| Label | ICD-10 | ICD-9 |
|---|---|---|
| **`y_ami`** | I21\*, I22\* (includes type 2 MI, I21A1) | 410.x0, 410.x1; **410.x2 ("subsequent episode") excluded** |
| **`y_ahf`** (strict) | I50.21, .23, .31, .33, .41, .43, I50.811, I50.813 | 428.21, .23, .31, .33, .41, .43 |
| `y2_hf_broad` (sensitivity analysis) | I50\*, I11.0, I13.0, I13.2 | 428\*, 402.01/.11/.91, 404.01/.03/.11/.13/.91/.93 |
| **`y_afib`** | I48.0, I48.1\*, I48.2\*, I48.91 | 427.31 |
| `y2_aflutter` (kept separate) | I48.3, I48.4, I48.92 | 427.32 |

I checked the matcher against edge cases (I21A1, I25.10, I50.9, I11.0, I48.92, 410.72, 428.0, 427.32, dotted "I22.0"), and every one was classified correctly.

### 4.4 Cohort (computed separately for 6 h and 24 h)
1. Valid times (departure after arrival)
2. Age ≥ 18 at arrival (`anchor_age + year(intime) − anchor_year`)
3. Label available
4. **At least one cardiac modality** with a timestamp in `[intime, intime + W]`. The modalities are ECG, chest X-ray, echo, and troponin/BNP labs. Radiology notes are tracked but don't qualify a visit on their own.

### 4.5 Split
- `assign_split(subject_id)`: a salted SHA-256 hash (salt `"grace-v1"`) mapped to **70% train / 15% val / 15% test**.
- It depends **only on the patient ID**, so it never changes when labels, the cohort rules or data versions change.
- Splits are assigned to **all** ED patients, not just cohort patients, so the 6 h and 24 h cohorts use the same assignment.
- It also works for patients outside the ED data. **Kristine (and I) should use it to split any extra encoder-training data**, such as all MIMIC-CXR or MIMIC-IV-ECG, so test patients never leak into training.
- ⚠️ **Never change the salt or fractions once results exist.**

### 4.6 Automatic checks (the script stops if any fails)
- `stay_id` is unique
- every patient is in exactly one split
- every cohort visit has a label
- the 6 h cohort is a subset of the 24 h cohort

---

## 5. Code and outputs

**Code** (`project/grace/grace_data/`):
| File | Purpose |
|---|---|
| `build_labels_splits.py` | Main pipeline |
| `icd.py` | ICD definitions and matching |
| `splits.py` | `assign_split()` |
| `paths.py` | Finds the PhysioNet files (any version); missing optional files are allowed |
| `README.md` | Short reference for the design and caveats |

**Run** (the base `python3` has no pandas; use the `torch` conda env):
```bash
cd "project/grace"
~/miniconda3/envs/torch/bin/python -m grace_data.build_labels_splits   # ~15 s
```

**Outputs** (`project/grace/derived/labels_splits/`):
| File | Contents |
|---|---|
| `cohort_6h.csv`, `cohort_24h.csv` | Included visits: IDs, split, labels, `t_pred`, modality-availability flags |
| `ed_stays_labeled.parquet` | Every ED visit, including excluded ones: ED-code flags (`edcode_*`), hours to first test of each modality, `label_source` |
| `subject_splits.csv` | `subject_id` → split |
| `label_definitions.json` | Code lists, salt, windows, sources used |
| `REPORT.md` | Exclusion counts, prevalence per split, modality coverage, checks |

> 🔒 `derived/` contains MIMIC data. Keep it local: never upload or commit it, and never paste it into an AI tool.

---

## 6. Results

### 6.1 Final build (2026-11-16 01:49)

Inputs: ED 2.2, hosp 3.1 (`patients`, `diagnoses_icd`, `labevents`, `d_labitems`), ECG-Ext timing, MIMIC-IV-Note radiology timing. Cardiac modalities used for the cohort rule: **ECG + troponin/BNP** (itemids 50963, 51002, 51003, 52642, by `storetime`). New data in `datasets/files/` (wget `-nH`) is found because `paths.py` now searches both download roots.

| Step | Visits | Patients |
|---|---|---|
| All ED visits | 425,087 | 205,504 |
| Valid times | 425,028 | 205,466 |
| Age ≥ 18 | 424,952 | 205,427 |
| Label available | 424,654 | 205,308 |
| + cardiac modality within **6 h** | **164,629** | **91,175** |
| + cardiac modality within **24 h** | **177,803** | **96,628** |

**Label sources:** `ed` 221,773 · `hosp` 202,934 · `ed_fallback` 82 · missing 298 (no `pending_hosp` left).

**Prevalence (6 h cohort):** AMI 3.40% · AHF 7.05% · AFib 14.5% · HF broad 15.8% · admitted 66.5%, nearly identical across train/val/test. 24 h: AMI 3.24% · AHF 6.85% · AFib 14.4%.

**Coverage within window (6 h / 24 h):** ECG 0.895 / 0.892 · troponin/BNP 0.477 / 0.517 · radiology report signed 0.704 / 0.814.

**Sanity checks:**
- Hosp labels vs the provisional ECG-Ext labels on the 98,085 visits in both: AHF and AFib agree 100%. AMI agrees 99.7%. All 305 differences are admissions coded ICD-9 **410.x2** (subsequent episode), which our definition excludes on purpose and ECG-Ext's ICD-10 conversion counts as AMI.
- Every patient's split is identical to the provisional build (205,504/205,504), as designed.

### 6.2 Provisional build (00:18, superseded)

Inputs used: ED + ECG-Ext only. Age comes from ECG-Ext, hospital labels from ECG-Ext, and **ECG is the only cardiac modality** available.

**Exclusions:**
| Step | Visits | Patients |
|---|---|---|
| All ED visits | 425,087 | 205,504 |
| Valid times | 425,028 | 205,466 |
| Age ≥ 18 (age known) | 301,839 | 110,334 |
| Label available | 231,895 | 98,223 |
| + cardiac modality within **6 h** | **145,807** | **82,367** |
| + cardiac modality within **24 h** | **149,222** | **83,717** |

**Label sources (all visits):** `ed` 221,773 · `pending_hosp` 104,922 · `hosp_via_ecg_ext` 98,094 · missing 298

**Prevalence (fraction of cohort visits):**
| Cohort | Split | AMI | AHF | AFib | HF broad | Admitted |
|---|---|---|---|---|---|---|
| 6 h | train | 3.28% | 6.58% | 14.29% | 15.09% | 65.5% |
| 6 h | val | 3.31% | 6.79% | 13.93% | 15.16% | 65.4% |
| 6 h | test | 3.28% | 6.42% | 13.03% | 14.69% | 65.3% |
| 24 h | all | 3.23% | 6.49% | 13.94% | 14.89% | 65.7% |

Patient split: **train 0.700 / val 0.150 / test 0.150**. All checks passed.

---

## 7. Findings to carry into the write-up

1. **Patients sent home are almost all easy negatives.** Among about 50k such visits in the cohort, ED codes gave only **32 AMI** and **2 AHF** positives (AFib 3%), because ED coders use "HF, unspecified" (I50.9) rather than acuity-specific codes. These visits inflate AUROC, so **also report metrics on admitted patients only**.
2. **Never use as features** (they reveal the outcome): `disposition`, `hadm_id`/admitted, `outtime`/ED length of stay (especially for the 6 h window), `label_source`, ED or hospital diagnosis codes, discharge summaries, or anything timestamped after `t_pred`.
3. **`y_afib` includes chronic or pre-existing AF**, not only new-onset AF, because it comes from hospitalization codes.
4. **Pretraining overlap:** many CXR models (and possibly MedGemma) were trained on MIMIC-CXR. Flag test patients who appear in public MIMIC-CXR training splits once the CXR-JPG split file is downloaded.
5. With ECG as the only modality, the 6 h and 24 h cohorts are nearly identical because ECGs come early. They should diverge once CXR, echo and lab timing are added.

---

## 8. Next steps

**Download** into `project/grace/datasets/` (PhysioNet credentials; run `wget` yourself):

| Priority | Files | Why |
|---|---|---|
| Status | Files | Why |
|---|---|---|
| ✅ done | MIMIC-IV hosp 3.1: `patients`, `admissions`, `diagnoses_icd`, `d_icd_diagnoses`, `labevents`, `d_labitems` | Labels, ages, troponin/BNP timing |
| ✅ done | MIMIC-IV-Note 2.2: `radiology`, `radiology_detail` | Report availability times and text |
| **Strongly recommended** | MIMIC-CXR-JPG: `mimic-cxr-2.0.0-metadata`, `-split`, `-chexpert` | CXR timestamps for the cohort rule; pretraining-overlap flag |
| **Strongly recommended** | MIMIC-IV-ECHO **1.0.1**: `echo-study-list`, `echo-record-list`, `structured_measurement` | Echo timing + structured echo measurements (LVEF etc.). Free-text echo reports are not released yet, and radiology notes contain no echo reports (only cardiac MR). |

**Then:**
1. ✅ Rerun the build; ✅ no `pending_hosp`; ✅ hosp vs ECG-Ext labels compared (section 6.1).
2. Add the MIMIC-CXR pretraining-overlap flag (needs the CXR-JPG split file).
3. Share `cohort_*.csv` and `subject_splits.csv` with Kristine (locally, not through a public service) before she builds the CXR/labs/ECG encoders (due 11/18). **These files changed at 01:49; any copy from before then is provisional.**
