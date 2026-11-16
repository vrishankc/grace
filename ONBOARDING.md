# GRACE onboarding: data access, setup, and handoff

This file covers everything needed to go from a fresh clone to the same labels, splits and baseline
results Vrishank has, using **your own PhysioNet credentials**. The repo holds **code and docs only**.
No MIMIC data is (or may ever be) committed.

*Written 2026-11-16 for Kristine. State of the project at handoff is in section 8.*

---

## 0. Ground rules for the data (read first)

- All MIMIC modules except MIMIC-IV-ECG are **credentialed** under the PhysioNet Credentialed Health
  Data Use Agreement. You may not share the data, including derived files like our cohort CSVs, with
  anyone who hasn't signed the same DUA. Kristine and Vrishank both being credentialed is fine.
- **This GitHub repo is currently PUBLIC.** Never commit anything from `datasets/` or `derived/`.
  `.gitignore` blocks them and every data-like extension (csv, gz, parquet, jpg, dcm, ipynb, ...).
  Don't `git add -f` past it. Before every commit run `git status` and read the list.
- Don't paste MIMIC rows, notes or derived tables into ChatGPT, Claude or other online tools.
  PhysioNet's DUA forbids sending the data to third parties. Aggregate counts and metrics are fine.
- Share data files with each other only machine-to-machine (USB, MIT Dropbox/OneDrive shared only
  between the two of us), never through a public link.

---

## 1. Get PhysioNet access (skip what you already have)

1. **Account:** create one at <https://physionet.org/register/>.
2. **Training:** complete CITI **"Data or Specimens Only Research"** (<https://physionet.org/about/citi-course/>):
   1. Create a CITI Program account. Under *My Courses*, click **Add affiliation** and pick
      **"Massachusetts Institute of Technology Affiliates"**.
   2. Answer questions 1–3 as the page instructs. For question 5, answer **Yes** to *Conflicts of Interest*.
   3. Complete the *Data or Specimens Only Research* and *Conflicts of Interest* modules.
   4. Download the **completion report** (not the certificate) from *Records → View-Print-Share*.
3. **Credentialing:** on PhysioNet, go to *Settings → Credentialing*, apply, and upload the CITI
   completion report. Approval can take days, so start early.
4. **Sign the DUA for each project** (each project page has a "sign the data use agreement" link;
   the URL pattern is `https://physionet.org/sign-dua/<project>/<version>/`). You need:

| Project | Version | Access | Why | DUA |
|---|---|---|---|---|
| MIMIC-IV-ED | 2.2 | credentialed | ED stays, triage, vitals, ED diagnoses (cohort, labels, structured features) | [sign](https://physionet.org/sign-dua/mimic-iv-ed/2.2/) |
| MIMIC-IV | 3.1 | credentialed | hosp: age, hospitalization ICD labels, labs | [sign](https://physionet.org/sign-dua/mimiciv/3.1/) |
| MIMIC-IV-Note | 2.2 | credentialed | radiology report text and sign times | [sign](https://physionet.org/sign-dua/mimic-iv-note/2.2/) |
| MIMIC-IV-ECG-Ext-ICD | 1.0.1 | credentialed | timestamps of every MIMIC-IV ECG (used for the cohort rule) | [sign](https://physionet.org/sign-dua/mimic-iv-ecg-ext-icd-labels/1.0.1/) |
| MIMIC-IV-ECG | 1.0 | **open** (ODbL) | ECG waveforms + machine measurements (your ECG encoder) | none needed |
| MIMIC-CXR-JPG | 2.1.0 | credentialed | CXR images + metadata/timestamps + official split (your CXR encoder) | [sign](https://physionet.org/sign-dua/mimic-cxr-jpg/2.1.0/) |
| MIMIC-CXR | 2.1.0 | credentialed | CXR report text (optional; MIMIC-IV-Note radiology already covers CXR reports with sign times) | [sign](https://physionet.org/sign-dua/mimic-cxr/2.1.0/) |
| MIMIC-IV-ECHO | 1.0.1 | credentialed | echo timing + structured echo measurements (LVEF etc.) | [sign](https://physionet.org/sign-dua/mimic-iv-echo/1.0.1/) |

If a DUA link 404s, open the project page (`https://physionet.org/content/<project>/<version>/`) and
use the link in its *Files* section. Versions above were the latest on 2026-11-16.

---

## 2. Download the data

### 2.1 Where files go

Everything goes under `datasets/` in the repo (git-ignored). Run **all wget commands from inside
`datasets/`** with `-x -nH`, which recreates PhysioNet's folder structure:

```
grace/datasets/files/<project>/<version>/...      e.g. datasets/files/mimiciv/3.1/hosp/labevents.csv.gz
```

`grace_data/paths.py` finds files by glob in both `datasets/files/` and `datasets/physionet.org/files/`
(the latter is what plain `wget -r` produces), and picks the highest version, so either layout works.

```bash
mkdir -p grace/datasets && cd grace/datasets      # not in the repo; you create it
PN_USER=<your-physionet-username>
WGET="wget -N -c -x -nH --user $PN_USER --ask-password"   # -c resumes, -N skips up-to-date files
```

`--ask-password` prompts once per wget call. Put several URLs in one call (or use `-i urls.txt`).
Big downloads die on flaky Wi-Fi; just rerun the same command (`-c` resumes).

### 2.2 Required to rebuild labels, cohorts and splits (~3.9 GB)

```bash
B=https://physionet.org/files

# MIMIC-IV-ED 2.2 (~116 MB)
$WGET $B/mimic-iv-ed/2.2/ed/{edstays,triage,vitalsign,diagnosis,medrecon,pyxis}.csv.gz \
      $B/mimic-iv-ed/2.2/SHA256SUMS.txt

# MIMIC-IV 3.1 hosp (~2.6 GB, of which labevents is 2.59 GB)
$WGET $B/mimiciv/3.1/hosp/{patients,admissions,diagnoses_icd,d_icd_diagnoses,labevents,d_labitems}.csv.gz \
      $B/mimiciv/3.1/SHA256SUMS.txt

# MIMIC-IV-ECG-Ext-ICD 1.0.1 (~339 MB)
$WGET $B/mimic-iv-ecg-ext-icd-labels/1.0.1/records_w_diag_icd10.csv \
      $B/mimic-iv-ecg-ext-icd-labels/1.0.1/SHA256SUMS.txt

# MIMIC-IV-Note 2.2 radiology (~820 MB). Do NOT download discharge.csv.gz (outcome leakage; never used)
$WGET $B/mimic-iv-note/2.2/note/{radiology,radiology_detail}.csv.gz \
      $B/mimic-iv-note/2.2/SHA256SUMS.txt
```

(zsh and bash both expand `{a,b}.csv.gz`.)

### 2.3 For your modalities (CXR, labs, ECG) and echo

Labs need nothing extra (`labevents` + `d_labitems` above). For the others, **first download the small
index files, rebuild labels (section 4), then download only the waveforms/images the cohort needs**:

```bash
# MIMIC-IV-ECG 1.0 (open access, so --user isn't needed but doesn't hurt).
# The index files are ~300 MB. Full waveforms are 90.4 GB; don't pull them all.
$WGET $B/mimic-iv-ecg/1.0/{record_list,machine_measurements,machine_measurements_data_dictionary,waveform_note_links}.csv \
      $B/mimic-iv-ecg/1.0/SHA256SUMS.txt          # SHA256SUMS here is ~168 MB (it lists every waveform)

# MIMIC-CXR-JPG 2.1.0 metadata (small). The JPGs total hundreds of GB; don't pull them all.
$WGET $B/mimic-cxr-jpg/2.1.0/{mimic-cxr-2.0.0-metadata,mimic-cxr-2.0.0-split,mimic-cxr-2.0.0-chexpert,mimic-cxr-2.0.0-negbio}.csv.gz \
      $B/mimic-cxr-jpg/2.1.0/SHA256SUMS.txt

# MIMIC-IV-ECHO 1.0.1 indexes + structured measurements (check exact names in the project's Files list)
$WGET $B/mimic-iv-echo/1.0.1/{echo-record-list,echo-study-list,structured_measurement}.csv \
      $B/mimic-iv-echo/1.0.1/SHA256SUMS.txt
```

Then rebuild labels (section 4; the cohort grows once CXR and echo timing exist) and generate
cohort-only download lists:

```bash
cd grace
python -m grace_data.download_lists ecg --window 24   # -> derived/download_lists/ecg_24h_urls.txt
python -m grace_data.download_lists cxr --window 24   # -> derived/download_lists/cxr_24h_urls.txt
cd datasets
$WGET -i ../derived/download_lists/ecg_24h_urls.txt
$WGET -i ../derived/download_lists/cxr_24h_urls.txt
```

With the current cohort, a partial index already gave about 75k ECG records for 52k stays, so expect
roughly 200k+ records (~25 GB) for the full list. Check free disk space first. If you also want to
**pretrain** an encoder on non-cohort patients, any extra patient must be split with
`grace_data.splits.assign_split(subject_id)`, and **test patients may never be used for training**.

### 2.4 Verify downloads

A truncated file looks like a normal file. Vrishank's first `record_list.csv` was cut off at 266k of
~800k lines. Check every download:

```bash
# Run inside each project's version folder, e.g. datasets/files/mimiciv/3.1
# SHA256SUMS.txt lists paths relative to that folder; this checks only the files you downloaded:
awk '{print $2}' SHA256SUMS.txt | while read f; do [ -f "$f" ] && grep " $f\$" SHA256SUMS.txt; done | shasum -a 256 -c -
# Every line should end in "OK". A quick extra check for .gz files:
gzip -t hosp/labevents.csv.gz && echo OK
```

---

## 3. Python environment

Developed with Python 3.9.16 on macOS (Apple silicon). Versions are in `requirements.txt`.

```bash
conda create -n grace python=3.9 -y && conda activate grace
pip install -r requirements.txt          # install torch/torchvision per pytorch.org for your machine/GPU
# macOS + xgboost: if import fails with libomp errors -> brew install libomp
```

All commands below are run **from the repo root** (`grace/`) as `python -m ...`.

---

## 4. Build labels, cohorts, splits

```bash
python -m grace_data.build_labels_splits     # ~10 min; mostly reading labevents
```

It writes `derived/labels_splits/` (`cohort_6h.csv`, `cohort_24h.csv`, `ed_stays_labeled.parquet`,
`subject_splits.csv`, `label_definitions.json`, `REPORT.md`). Design and decisions are in
`grace_data/README.md` and `LABELS_SPLITS_PROGRESS.md`.

**Check you reproduced Vrishank's build** (inputs: ED 2.2, hosp 3.1, ECG-Ext 1.0.1, Note 2.2; no
CXR-JPG/ECHO yet). `REPORT.md` should show:

| check | expected |
|---|---|
| ED stays / patients | 425,087 / 205,504 |
| 6 h cohort stays / patients | 164,629 / 91,175 (train 114,913 · val 24,886 · test 24,830) |
| 24 h cohort stays / patients | 177,803 / 96,628 (train 124,133 · val 26,838 · test 26,832) |
| 6 h positives AMI / AHF / AFib | 5,598 / 11,611 / 23,841 |
| 24 h positives AMI / AHF / AFib | 5,764 / 12,171 / 25,635 |
| patient split counts | train 143,881 · val 30,817 · test 30,806 |
| `shasum -a 256 derived/labels_splits/subject_splits.csv` | `b98547186e4d68fe090dc3b9560383567d2776cab77b841c2c16a42eb9637109` |

Once you add CXR-JPG metadata or ECHO lists, the cohort **grows** (more qualifying modalities) and
these numbers change. That's expected; tell each other and rerun all models on the new cohort.
**Splits never change** (they depend only on `subject_id`).

---

## 5. Run the baselines

```bash
python -m grace_models.train structured --window 6               # XGBoost: triage + ED vitals + chief-complaint tokens
python -m grace_models.train text --source cc --window 6         # TF-IDF + LR on chief complaint
python -m grace_models.train text --source radiology --window 6  # TF-IDF + LR on radiology reports signed <= t_pred
# repeat with --window 24; add --admitted-only for the admitted-only training variant
```

Outputs go to `derived/models/<run>/<W>h/`: `preds.parquet` (input to late fusion), `metrics.json`,
`metrics.md`, `run.json`. **`--eval-test` scores the test split. Don't use it until we freeze models
for check-in 2.**

Reference **val** AUROC (admitted-only stays) from Vrishank's run, to sanity-check yours:

| run | 6 h AMI / AHF / AFib | 24 h AMI / AHF / AFib |
|---|---|---|
| structured, XGBoost | 0.815 / 0.851 / 0.814 | 0.823 / 0.857 / 0.818 |
| chief complaint, TF-IDF + LR | 0.755 / 0.782 / 0.653 | 0.762 / 0.784 / 0.654 |
| radiology, TF-IDF + LR (only stays with a report by t_pred) | 0.760 / 0.895 / 0.761 | 0.793 / 0.902 / 0.776 |

---

## 6. Repo map and conventions

```
grace/
├── README.md, ONBOARDING.md, LABELS_SPLITS_PROGRESS.md, requirements.txt, .gitignore
├── docs/                  proposal PDF, study diagram
├── grace_data/            labels, cohorts, splits  (README.md = design decisions)
│   ├── build_labels_splits.py   icd.py   splits.py   paths.py   download_lists.py
├── grace_models/          single-modality baselines (README.md = how runs work)
│   ├── common.py  (cohort loading, leakage guard, time filters)   trainer.py  heads.py  metrics.py  train.py (CLI)
│   ├── features/  triage.py  notes.py  echo.py        <- one stay-level feature builder per modality
│   └── encoders/  text.py (TF-IDF, BioWordVec)  echoclip.py
├── datasets/  (git-ignored) raw PhysioNet downloads
└── derived/   (git-ignored) labels, features, model outputs. Contains MIMIC data.
```

**Rules every model follows** (details in `grace_data/README.md`, `grace_models/README.md`):

- **Unit:** one ED stay (`stay_id`). Prediction time `t_pred = intime + 6 h` or `+ 24 h`.
- **Time:** an input may only use data **available by `t_pred`**: lab `storetime`, report
  `storetime` (radiology reports are signed a median 2.25 h after the exam), CXR/ECG acquisition time.
- **Splits:** use the `split` column from the cohort CSV, or `grace_data.splits.assign_split(subject_id)`
  for anything outside it. Never re-split.
- **Never use as features:** `disposition`, `hadm_id`/admitted, `outtime`/ED LOS, `label_source`,
  diagnosis codes, discharge summaries, anything after `t_pred`. `common.check_no_leakage` enforces
  the column names. Also avoid ED-administered meds (`pyxis`) and counts/timing of measurements
  (they encode ED LOS → disposition).
- **Labels:** `y_ami`, `y_ahf`, `y_afib` (+ `y2_hf_broad`, `y2_aflutter` for sensitivity). Report
  metrics on **all** stays and on **admitted-only** stays (non-admitted stays are near-trivial negatives).

**Adding your modality** (CXR, labs, ECG) takes one feature builder plus the shared trainer:

```python
# grace_models/features/labs.py  (sketch)
from ..common import filter_before_tpred, attach_events_by_subject
def build(cohort, window_h):
    # cohort: indexed by stay_id, has subject_id, intime, t_pred, split
    # 1. load events with a time column (labs: storetime)
    # 2. events = attach_events_by_subject(events, cohort, "storetime", lookback_h=0)   # adds stay_id, keeps [intime, t_pred]
    # 3. aggregate to one row per stay_id (e.g. last/max troponin, BNP, creatinine, ...)
    return X   # DataFrame indexed by stay_id; stays without the modality may be absent
```

Then add a branch in `grace_models/train.py` (copy the `structured` one) with `heads.gbm()` for
tabular data, or `heads.logistic()` on frozen-encoder embeddings (CXR: DenseNet-121/TorchXRayVision
features; ECG: 1D-CNN or a pretrained ECG encoder). `trainer.fit_eval` handles hyperparameter
selection (inner holdout of train, so val stays clean), metrics with patient-clustered bootstrap CIs,
and writes `preds.parquet` in the shared format for late fusion.

---

## 7. Git workflow

```bash
git pull
git status            # confirm nothing under datasets/ or derived/, no .csv/.parquet/.ipynb
git add <specific files>
git commit -m "..."
git push
```

`.gitignore` blocks notebooks (`*.ipynb`) because outputs often contain patient rows. To commit one,
clear all outputs first, then `git add -f notebook.ipynb`.

---

## 8. Status at handoff (2026-11-16) and open items

**Done**
- Labels/cohorts/splits, final build (hosp 3.1). Checks: hosp vs ECG-Ext labels agree 100% for AHF/AFib;
  for AMI, 305 differences, all ICD-9 410.x2 "subsequent episode" (excluded by our definition).
- Baselines running on val: structured (XGBoost), chief-complaint text, radiology text (TF-IDF + LR).
- Echo code (EchoCLIP + linear head) written but not run (no echo data yet).

**Kristine: next (due 11/18)**
1. CXR encoder: download CXR-JPG metadata + split → rebuild labels → `download_lists cxr` → images.
2. Labs features: from `labevents` by `storetime` ≤ t_pred (template in section 6).
3. ECG encoder: machine measurements (tabular → XGBoost) and/or waveforms (1D CNN).
4. MIMIC-CXR pretraining-overlap flag: mark test patients that appear in public MIMIC-CXR training
   splits (`mimic-cxr-2.0.0-split.csv.gz`), since many CXR models and possibly MedGemma trained on them.

**Known issues / decisions pending**
- **Echo reports (free text) are not released** on PhysioNet, and MIMIC-IV-Note radiology has no echo
  reports (only cardiac MR). Plan: use MIMIC-IV-ECHO 1.0.1 `structured_measurement.csv` (206k studies,
  91k patients, incl. LVEF) instead of or alongside EchoCLIP. That changes the proposal, so we need to agree.
- **Prior-history window:** should inputs from *before* ED arrival (e.g. last year's LVEF) count? The
  proposal says "from the prediction window". `--lookback-h` supports both. Team decision.
- Radiology AHF AUROC ~0.90 partly reflects the ordering clinician's indication text ("eval for CHF"),
  which is legitimately known by t_pred but should be noted in the write-up.
- Vrishank's local `record_list.csv` was truncated and needs re-downloading (doesn't affect labels;
  ECG timing comes from ECG-Ext).
