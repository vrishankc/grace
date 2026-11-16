# grace_data: labels, cohorts, splits

```bash
cd project/grace
~/miniconda3/envs/torch/bin/python -m grace_data.build_labels_splits   # ~15 s; more once labevents is added
```
Rerun whenever new PhysioNet files land in `datasets/physionet.org/files/`. Inputs are found automatically (`paths.py`); missing optional inputs are listed in `derived/labels_splits/REPORT.md`.

**`derived/` contains MIMIC data. Keep it local: never upload it, commit it, or paste it into an LLM.**

## Design decisions
| Topic | Rule |
|---|---|
| Unit | One row per MIMIC-IV-ED stay. All of a patient's stays are kept. |
| Prediction time | `t_pred = intime + W` for W ∈ {6 h, 24 h}. A feature may only use data **timestamped ≤ t_pred** (for reports and labs, use the time the result became available: `storetime`). |
| Admitted (`hadm_id` present) | Label = ICD codes of the linked hospitalization (`hosp/diagnoses_icd`, ICD-9 + ICD-10). If the hadm has no codes, fall back to ED codes (`label_source=ed_fallback`). |
| Not admitted | Label = ED diagnosis ICD codes for that stay (`label_source=ed`). |
| AMI (`y_ami`) | ICD-10 I21*, I22*; ICD-9 410.x0/410.x1 (410.x2 "subsequent episode" excluded) |
| Acute HF (`y_ahf`) | Strict: ICD-10 I50.21/.23/.31/.33/.41/.43/.811/.813; ICD-9 428.21/.23/.31/.33/.41/.43. `y2_hf_broad` = any HF code (I50*, I11.0, I13.0/.2; 428*, 402.x1, 404.x1/x3) for sensitivity analysis |
| AFib (`y_afib`) | ICD-10 I48.0, I48.1*, I48.2*, I48.91; ICD-9 427.31. Flutter is separate (`y2_aflutter`). |
| Cohort(W) | Valid times, age ≥ 18 at arrival, label available, and ≥1 cardiac modality (ECG, CXR, echo, troponin/BNP) timestamped in [intime, intime+W] |
| Split | `splits.assign_split(subject_id)`: salted SHA-256 → 70/15/15 train/val/test. Depends only on the patient, so it never changes as labels or cohorts change. **Use it for any extra data too (e.g. encoder pretraining on all MIMIC-CXR/ECG) so test patients never leak into training.** |

## Outputs (`derived/labels_splits/`)
- `cohort_6h.csv`, `cohort_24h.csv`: included stays with split, labels, `t_pred_*`, and modality-availability flags
- `ed_stays_labeled.parquet`: every ED stay, including excluded ones, with `edcode_*` flags (ED-code labels for every stay), `h_to_first_<modality>`, and `label_source`
- `subject_splits.csv`: subject_id → split
- `label_definitions.json`: code lists, salt, windows, and the sources used
- `REPORT.md`: attrition, prevalence per split, coverage, checks

## Never use as features (they leak the outcome)
`disposition`, `hadm_id`/`admitted`, `outtime`/ED length of stay (for the 6 h window), `label_source`, ED diagnosis codes, any hosp diagnosis, discharge summaries, and anything timestamped after `t_pred`.

## Caveats to report
- Patients who were not admitted almost never carry acute ED codes (in the provisional build: 32 AMI and 2 AHF positives among ~50k such stays). They act as easy negatives, so also report metrics on the **admitted-only** subset.
- `y_afib` from hospitalization codes includes prevalent or chronic AF, not just new-onset AF.
- MIMIC-CXR overlap with pretrained models: flag test patients who are in public MIMIC-CXR training splits (to do once the `mimic-cxr-jpg` split file is downloaded).
