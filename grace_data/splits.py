"""Patient-level split assignment.

The split is a pure function of subject_id (salted SHA-256), so:
  * every stay of a patient lands in the same split (no patient leakage),
  * it never changes when labels / cohort rules / data versions change,
  * ANY dataset (e.g. all MIMIC-CXR or MIMIC-IV-ECG records used to pretrain an encoder)
    can be split consistently by calling assign_split(subject_id).

Do NOT change SALT or FRACTIONS after results have been reported.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

SALT = "grace-v1"
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _u(subject_id) -> float:
    h = hashlib.sha256(f"{SALT}:{int(subject_id)}".encode()).hexdigest()
    return int(h[:15], 16) / 16**15  # uniform in [0, 1)


def assign_split(subject_ids) -> pd.Series | str:
    """Map subject_id(s) -> 'train' | 'val' | 'test'."""
    edges = np.cumsum(list(FRACTIONS.values()))
    names = np.array(list(FRACTIONS))
    if np.isscalar(subject_ids):
        return str(names[np.searchsorted(edges, _u(subject_ids), side="right")])
    s = pd.Series(subject_ids)
    uniq = pd.Series(s.unique())
    u = uniq.map(_u).to_numpy()
    lut = dict(zip(uniq, names[np.searchsorted(edges, u, side="right")]))
    return s.map(lut).rename("split")
