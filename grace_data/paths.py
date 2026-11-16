"""Locations of raw PhysioNet downloads and derived outputs.

Raw files are discovered by glob so any downloaded version works (highest version wins).
A missing optional source returns None and the pipeline degrades gracefully.
"""
from __future__ import annotations

from pathlib import Path

GRACE_ROOT = Path(__file__).resolve().parents[1]
# wget without -nH writes datasets/physionet.org/files/...; with -nH it writes datasets/files/...
RAW_ROOTS = [GRACE_ROOT / "datasets" / "physionet.org" / "files", GRACE_ROOT / "datasets" / "files"]
RAW = RAW_ROOTS[0]  # kept for backwards compatibility
DERIVED = GRACE_ROOT / "derived" / "labels_splits"


def _find(pattern: str) -> Path | None:
    # Search every root; pick the highest version by the path *within* the root so roots don't bias the order.
    hits = [(h.relative_to(root).as_posix(), h) for root in RAW_ROOTS for h in root.glob(pattern)]
    return max(hits)[1] if hits else None


def sources() -> dict[str, Path | None]:
    return {
        # required
        "edstays": _find("mimic-iv-ed/*/ed/edstays.csv.gz"),
        "ed_diagnosis": _find("mimic-iv-ed/*/ed/diagnosis.csv.gz"),
        # labels / demographics (MIMIC-IV hosp)
        "patients": _find("mimiciv/*/hosp/patients.csv.gz"),
        "diagnoses_icd": _find("mimiciv/*/hosp/diagnoses_icd.csv.gz"),
        "labevents": _find("mimiciv/*/hosp/labevents.csv.gz"),
        "d_labitems": _find("mimiciv/*/hosp/d_labitems.csv.gz"),
        # modality timing
        "ecg_ext": _find("mimic-iv-ecg-ext-icd-labels/*/records_w_diag_icd10.csv"),
        "ecg_records": _find("mimic-iv-ecg/*/record_list.csv"),
        "cxr_meta": _find("mimic-cxr-jpg/*/mimic-cxr-2.0.0-metadata.csv.gz"),
        "cxr_split": _find("mimic-cxr-jpg/*/mimic-cxr-2.0.0-split.csv.gz"),
        "echo_studies": _find("mimic-iv-echo/*/echo-study-list.csv*"),
        "echo_records": _find("mimic-iv-echo/*/echo-record-list.csv*"),
        "echo_measurements": _find("mimic-iv-echo/*/structured_measurement.csv*"),
        "radiology": _find("mimic-iv-note/*/note/radiology.csv.gz"),
    }
