# grace_models: single-modality baselines

Encoders + heads + a shared trainer for the proposal's single-modality baselines (section 3.2). Every model
writes predictions in the same format, so late fusion only needs to join on `stay_id`.

```bash
cd project/grace
PY=~/miniconda3/envs/torch/bin/python
$PY -m grace_models.train structured --window 6            # XGBoost/HGB: triage + ED vitals + chief-complaint tokens
$PY -m grace_models.train text --window 6 --source cc      # TF-IDF + logistic regression on chief complaint
$PY -m grace_models.train text --window 24 --source radiology                      # needs MIMIC-IV-Note radiology
$PY -m grace_models.train text --window 24 --source radiology --encoder biowordvec --vectors <BioWordVec .bin>
$PY -m grace_models.train echo-manifest --window 24        # needs MIMIC-IV-ECHO record list
$PY -m grace_models.encoders.echoclip                      # needs open_clip_torch + downloaded DICOMs
$PY -m grace_models.train echo --window 24
```
Flags: `--admitted-only` (train and evaluate on admitted stays), `--eval-test` (also score **test**; leave it
off during development), `--n-boot`, `--targets`, `--blocks triage vitals cc medrecon`, `--lookback-h`.

## How a run works (`trainer.fit_eval`)
1. **Inputs** = a table/Series indexed by `stay_id`, built only from data timestamped `<= t_pred`.
   Stays without the modality are absent → no prediction (missing modality, handled at fusion).
2. **Encoder** is fit once on the train split (TF-IDF vocabulary; BioWordVec vocabulary; identity for
   tabular features and precomputed EchoCLIP embeddings).
3. **Head**: one classifier per target. Hyperparameters are picked on an *inner* 15% patient-level
   holdout of **train** (salt `grace-v1-inner`), then refit on all of train. Val is never used for
   fitting or selection, so it stays an honest dev set.
4. **Metrics**: AUROC / AUPRC / Brier with **patient-clustered** bootstrap 95% CIs, on all stays and on
   **admitted-only** stays (ED-coded negatives are near-trivial). `metrics.paired_auroc_test` compares
   two models on the same stays.

Outputs: `derived/models/<run>/<W>h/{preds.parquet, metrics.json, metrics.md, run.json}`;
cached features in `derived/features/`. **Both contain MIMIC data: keep them local.**

## Modalities
| Run | Module | Status |
|---|---|---|
| Structured (triage vitals, ED vitals ≤ t_pred, chief-complaint tokens; optional home meds) | `features/triage.py` + `heads.gbm` | runs now |
| Chief complaint text (TF-IDF / BioWordVec + LR) | `features/notes.py` (`cc`) + `encoders/text.py` | runs now |
| Radiology reports (storetime ≤ t_pred) | `features/notes.py` (`radiology`) | waits on `mimic-iv-note/2.2/note/radiology.csv.gz` |
| Echo (frozen EchoCLIP + linear head) | `features/echo.py` + `encoders/echoclip.py` | waits on MIMIC-IV-ECHO + `pip install open_clip_torch` |
| CXR, labs, ECG (Kristine) | add a `features/<mod>.py` returning a stay-indexed table, then call `fit_eval` | — |

## Deliberate exclusions (leakage)
`check_no_leakage` refuses `disposition`, `hadm_id`/`admitted`, `outtime`/ED LOS, `label_source`, `y_*`,
`edcode_*`, `h_to_first_*`. In addition, the structured features do **not** use:
- counts or timing of vital-sign measurements (in the 24 h window these encode ED LOS → disposition);
- `pyxis` (ED-administered meds such as heparin, nitrates, diuretics, rate control are treatment decisions
  that stand in for the diagnosis; use only as a clearly labeled "treatment-aware" ablation);
- `race` (report subgroup performance instead);
- discharge summaries (never read).

`medrecon` (home meds) is opt-in: it is legitimately available, but makes `y_afib` partly a
"known chronic AF" detector (anticoagulants), which matches the label caveat in `grace_data/README.md`.

## Install notes (torch env)
- `xgboost` is not installed; `heads.gbm()` falls back to sklearn `HistGradientBoostingClassifier`.
  Install with `~/miniconda3/envs/torch/bin/pip install xgboost` to match the proposal.
- `gensim` is broken in this env; BioWordVec is read by a streaming reader (`encoders/text.py`) that
  keeps only the training vocabulary (the full 13 GB file doesn't fit in RAM).
- EchoCLIP: `pip install open_clip_torch pylibjpeg pylibjpeg-libjpeg` (compressed ultrasound DICOMs).
