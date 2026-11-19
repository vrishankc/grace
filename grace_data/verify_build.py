"""Check a local labels/splits build against Vrishank's reference build (2026-11-16).

Reference inputs: MIMIC-IV-ED 2.2, MIMIC-IV hosp 3.1, MIMIC-IV-ECG-Ext-ICD 1.0.1,
MIMIC-IV-Note 2.2. No CXR-JPG or ECHO files present.

Put this file in grace/grace_data/ and run from the repo root:
    python -m grace_data.verify_build

It prints only aggregate counts, so its output is safe to share. It contains no data
and is fine to commit.

Note: once CXR-JPG metadata or ECHO lists are added, the cohort counts are EXPECTED
to change. The subject_splits.csv hash and patient split counts should never change.
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

OUT = Path("derived/labels_splits")

EXPECTED = {
    "subject_splits_sha256": "b98547186e4d68fe090dc3b9560383567d2776cab77b841c2c16a42eb9637109",
    "patient_split_counts": {"train": 143_881, "val": 30_817, "test": 30_806},
    "cohorts": {
        6: {
            "stays": 164_629,
            "patients": 91_175,
            "split_stays": {"train": 114_913, "val": 24_886, "test": 24_830},
            "positives": {"y_ami": 5_598, "y_ahf": 11_611, "y_afib": 23_841},
        },
        24: {
            "stays": 177_803,
            "patients": 96_628,
            "split_stays": {"train": 124_133, "val": 26_838, "test": 26_832},
            "positives": {"y_ami": 5_764, "y_ahf": 12_171, "y_afib": 25_635},
        },
    },
}

results = []  # (name, ok, got, expected)


def check(name, got, expected):
    results.append((name, got == expected, got, expected))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if not OUT.exists():
        sys.exit(f"{OUT} not found. Run from the repo root after build_labels_splits.")

    # 1. Patient-level splits (independent of cohort; must always match)
    ss_path = OUT / "subject_splits.csv"
    check("subject_splits.csv sha256", sha256(ss_path), EXPECTED["subject_splits_sha256"])
    ss = pd.read_csv(ss_path)
    check("patient split counts",
          ss["split"].value_counts().to_dict(), EXPECTED["patient_split_counts"])

    # 2. Cohorts
    stays = {}
    for w, exp in EXPECTED["cohorts"].items():
        c = pd.read_csv(OUT / f"cohort_{w}h.csv")
        stays[w] = set(c["stay_id"])
        check(f"{w}h stays", len(c), exp["stays"])
        check(f"{w}h patients", c["subject_id"].nunique(), exp["patients"])
        check(f"{w}h split stays", c["split"].value_counts().to_dict(), exp["split_stays"])
        for y, n in exp["positives"].items():
            check(f"{w}h positives {y}", int(c[y].sum()), n)
        check(f"{w}h stay_id unique", c["stay_id"].is_unique, True)
        check(f"{w}h one split per patient",
              bool((c.groupby("subject_id")["split"].nunique() == 1).all()), True)

    check("6h cohort subset of 24h", stays[6] <= stays[24], True)

    # Report
    width = max(len(r[0]) for r in results)
    n_fail = 0
    for name, ok, got, exp in results:
        status = "OK  " if ok else "FAIL"
        line = f"[{status}] {name:<{width}}"
        if not ok:
            n_fail += 1
            line += f"  got={got}  expected={exp}"
        print(line)

    print(f"\n{len(results) - n_fail}/{len(results)} checks passed.")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
