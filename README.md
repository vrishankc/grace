# GRACE: Graph-Reinforced Assessment of Cardiac Events with Multimodal Data &amp; Drug Repurposing
Final Project for 6.3930 AI and Decision-Making for Medicine: From Disease to Therapy
Project Teammates: Vrishank Chandrasekhar, Kristine Yang
Professors: Dr. Collin Stultz, Dr. Regina Barzilay

Predicting acute MI, acute heart failure and atrial fibrillation for adult ED stays in MIMIC-IV,
comparing MedGemma vs Gemma 3, multimodal integration strategies, knowledge-graph augmentation and
grounded reasoning. Proposal: [`docs/GRACE_Proposal.pdf`](docs/GRACE_Proposal.pdf).

> **New here? Start with [`ONBOARDING.md`](ONBOARDING.md)**: PhysioNet access, exact download
> commands, environment, how to reproduce the labels and baselines, and the current status.
>
> **No MIMIC data in this repo, ever.** Raw data lives in `datasets/` and derived files in `derived/`,
> both git-ignored. Check `git status` before every commit.

| Component | Where | Docs |
|---|---|---|
| Labels, cohorts (6 h / 24 h), patient-level splits | `grace_data/` | [`grace_data/README.md`](grace_data/README.md), [`LABELS_SPLITS_PROGRESS.md`](LABELS_SPLITS_PROGRESS.md) |
| Single-modality baselines (encoders, heads, shared trainer, metrics) | `grace_models/` | [`grace_models/README.md`](grace_models/README.md) |

Split any data with `grace_data.splits.assign_split(subject_id)`. A new modality is a
`grace_models/features/<mod>.py` that returns a stay_id-indexed table using only data <= t_pred, then
`trainer.fit_eval(...)`.

To-Do:
- Vrishank: finish labeling and splits by 11/13 3 PM; validate with Kristine by EOD ✅ (final build 11/16)
- Kristine: CXR, labs, ECG –> make encoders / processing -- by 11/18
- Vrishank: Notes, triage vitals/chief complaints, Echo -> make encoders / processing (structured + notes running; echo pending data/decision, see ONBOARDING §8)
- Meet tomorrow at 10:30 AM to like review
