"""Write URL lists for the ECG waveforms / CXR JPGs that the GRACE cohort actually needs.

The full MIMIC-IV-ECG waveform set is ~90 GB and MIMIC-CXR-JPG is far larger, but only records
taken during a cohort ED stay's prediction window are usable as model inputs. This script reads the
small index files (which you download first) and writes one URL per line, ready for `wget -i`.

    cd project/grace
    python -m grace_data.download_lists ecg  [--window 24] [--lookback-h 0]
    python -m grace_data.download_lists cxr  [--window 24] [--lookback-h 0]

Needs: derived/labels_splits/cohort_<W>h.csv (run build_labels_splits first) and
  ecg: mimic-iv-ecg/<ver>/record_list.csv
  cxr: mimic-cxr-jpg/<ver>/mimic-cxr-2.0.0-metadata.csv.gz
Output: derived/download_lists/{ecg,cxr}_<W>h_urls.txt   (then: cd datasets && wget -N -c -x -nH
        --user <physionet-username> --ask-password -i ../derived/download_lists/<file>)

Use --window 24 (the 24 h cohort contains the 6 h one). Add --lookback-h if the team decides
prior studies (before ED arrival) are allowed as inputs.
"""
from __future__ import annotations

import argparse

import pandas as pd

from .paths import DERIVED, GRACE_ROOT, sources

BASE = "https://physionet.org/files"
OUT = GRACE_ROOT / "derived" / "download_lists"


def _cohort(window_h: int) -> pd.DataFrame:
    co = pd.read_csv(DERIVED / f"cohort_{window_h}h.csv", usecols=["stay_id", "subject_id", "intime", f"t_pred_{window_h}h"],
                     parse_dates=["intime", f"t_pred_{window_h}h"])
    return co.rename(columns={f"t_pred_{window_h}h": "t_pred"})


def _in_window(ev: pd.DataFrame, co: pd.DataFrame, lookback_h: float) -> pd.DataFrame:
    m = ev.merge(co, on="subject_id", how="inner")
    keep = (m.t <= m.t_pred) & (m.t >= m.intime - pd.Timedelta(hours=lookback_h))
    return m[keep]


def _version(p) -> str:
    return p.parent.name  # .../<project>/<version>/<file>


def ecg(window_h: int, lookback_h: float) -> list[str]:
    p = sources()["ecg_records"]
    if p is None:
        raise SystemExit("mimic-iv-ecg record_list.csv not found under datasets/")
    r = pd.read_csv(p, usecols=["subject_id", "ecg_time", "path"], parse_dates=["ecg_time"]).rename(columns={"ecg_time": "t"})
    m = _in_window(r, _cohort(window_h), lookback_h)
    paths = sorted(m.path.unique())
    print(f"ECG: {m.stay_id.nunique():,} stays, {len(paths):,} records (x2 files: .hea + .dat)")
    return [f"{BASE}/mimic-iv-ecg/{_version(p)}/{x}{ext}" for x in paths for ext in (".hea", ".dat")]


def cxr(window_h: int, lookback_h: float) -> list[str]:
    p = sources()["cxr_meta"]
    if p is None:
        raise SystemExit("mimic-cxr-jpg mimic-cxr-2.0.0-metadata.csv.gz not found under datasets/")
    c = pd.read_csv(p, usecols=["dicom_id", "subject_id", "study_id", "StudyDate", "StudyTime"])
    hhmmss = c.StudyTime.fillna(0).astype(float).astype(int).astype(str).str.zfill(6)
    c["t"] = pd.to_datetime(c.StudyDate.astype(int).astype(str) + hhmmss, format="%Y%m%d%H%M%S", errors="coerce")
    m = _in_window(c, _cohort(window_h), lookback_h)
    rel = ("files/p" + m.subject_id.astype(str).str[:2] + "/p" + m.subject_id.astype(str) +
           "/s" + m.study_id.astype(str) + "/" + m.dicom_id + ".jpg")
    rel = sorted(rel.unique())
    print(f"CXR: {m.stay_id.nunique():,} stays, {m.study_id.nunique():,} studies, {len(rel):,} JPGs")
    return [f"{BASE}/mimic-cxr-jpg/{_version(p)}/{x}" for x in rel]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modality", choices=["ecg", "cxr"])
    ap.add_argument("--window", type=int, choices=[6, 24], default=24)
    ap.add_argument("--lookback-h", type=float, default=0.0)
    a = ap.parse_args()
    urls = {"ecg": ecg, "cxr": cxr}[a.modality](a.window, a.lookback_h)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.modality}_{a.window}h_urls.txt"
    out.write_text("\n".join(urls) + "\n")
    print(f"wrote {len(urls):,} URLs to {out}")


if __name__ == "__main__":
    main()
