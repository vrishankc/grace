"""ICD code definitions for GRACE labels.

Codes are stored without dots (MIMIC convention), e.g. I21.4 -> "I214", 410.71 -> "41071".
Each label has a PRIMARY definition (used as the target) and, where useful, a BROAD
definition kept as a secondary column for sensitivity analyses.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

LABELS = ["ami", "ahf", "afib"]

# ---------------------------------------------------------------------------
# Definitions. Each entry: {icd_version: (tuple of prefixes, tuple of exact codes)}
# ---------------------------------------------------------------------------
DEFS: dict[str, dict[int, dict[str, tuple[str, ...]]]] = {
    # Acute myocardial infarction (incl. subsequent MI I22, type 2 MI I21A1).
    # ICD-9 410.x2 = "subsequent episode of care" (MI treated earlier) -> excluded below.
    "ami": {
        10: {"prefix": ("I21", "I22"), "exact": ()},
        9: {"prefix": ("410",), "exact": ()},
    },
    # Acute heart failure, STRICT: codes that state acute or acute-on-chronic.
    "ahf": {
        10: {"prefix": (), "exact": ("I5021", "I5023", "I5031", "I5033", "I5041", "I5043",
                                     "I50811", "I50813")},
        9: {"prefix": (), "exact": ("42821", "42823", "42831", "42833", "42841", "42843")},
    },
    # Any heart-failure code (acuity unspecified), incl. hypertensive heart/CKD with HF.
    "hf_broad": {
        10: {"prefix": ("I50",), "exact": ("I110", "I130", "I132")},
        9: {"prefix": ("428",), "exact": ("40201", "40211", "40291", "40401", "40403",
                                          "40411", "40413", "40491", "40493")},
    },
    # Atrial fibrillation (any type). Flutter kept separate.
    "afib": {
        10: {"prefix": ("I480", "I481", "I482", "I4891"), "exact": ()},
        9: {"prefix": (), "exact": ("42731",)},
    },
    "aflutter": {
        10: {"prefix": ("I483", "I484", "I4892"), "exact": ()},
        9: {"prefix": (), "exact": ("42732",)},
    },
}
ALL_FLAGS = list(DEFS)  # ami, ahf, hf_broad, afib, aflutter


def normalize(codes: pd.Series) -> pd.Series:
    return codes.astype(str).str.upper().str.replace(".", "", regex=False).str.strip()


def flag_codes(codes: pd.Series, versions: pd.Series | int) -> pd.DataFrame:
    """Return a boolean DataFrame (one column per flag in DEFS) for each code row."""
    codes = normalize(codes)
    if np.isscalar(versions):
        versions = pd.Series(int(versions), index=codes.index)
    versions = versions.astype(int)
    out = {}
    for name, by_ver in DEFS.items():
        hit = pd.Series(False, index=codes.index)
        for ver, d in by_ver.items():
            m = versions == ver
            h = pd.Series(False, index=codes.index)
            if d["prefix"]:
                h |= codes.str.startswith(d["prefix"])
            if d["exact"]:
                h |= codes.isin(d["exact"])
            hit |= m & h
        out[name] = hit
    df = pd.DataFrame(out)
    # ICD-9 410.x2 (subsequent episode) is not a new AMI.
    sub = (versions == 9) & codes.str.match(r"^410\d2$")
    df.loc[sub, "ami"] = False
    return df


def aggregate_flags(df: pd.DataFrame, key: str, code_col="icd_code", ver_col="icd_version") -> pd.DataFrame:
    """Any-code-positive per key (e.g. hadm_id or stay_id)."""
    f = flag_codes(df[code_col], df[ver_col] if ver_col in df else 10)
    f[key] = df[key].values
    return f.groupby(key)[ALL_FLAGS].any()


_CODE_RE = re.compile(r"'([A-Za-z0-9]+)'")


def explode_list_column(s: pd.Series) -> pd.DataFrame:
    """Parse ECG-Ext-ICD stringified lists like "['I214', 'I10']" -> long (index, code)."""
    lst = s.fillna("[]").map(_CODE_RE.findall)
    long = lst.explode().dropna()
    return pd.DataFrame({"idx": long.index, "icd_code": long.values})


def definitions_for_report() -> dict:
    return {k: {f"icd{v}": {"prefixes": list(d["prefix"]), "exact": list(d["exact"])}
                for v, d in by.items()} for k, by in DEFS.items()}
