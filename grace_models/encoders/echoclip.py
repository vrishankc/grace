"""Frozen EchoCLIP image embeddings for MIMIC-IV-ECHO DICOMs (Christensen et al., Nat Med 2024).

    ~/miniconda3/envs/torch/bin/pip install open_clip_torch pylibjpeg pylibjpeg-libjpeg
    ~/miniconda3/envs/torch/bin/python -m grace_models.encoders.echoclip [--frames 16] [--limit N]

Per DICOM: sample up to `frames` evenly spaced frames -> EchoCLIP image encoder -> L2-normalized frame
embeddings -> mean -> one vector. Embeddings are cached by dicom_filepath (resumable).

Preprocessing follows the EchoCLIP README (frames -> PIL -> the model's `preprocess_val`, 224x224).
TODO before reporting: check against github.com/echonet/echo_CLIP whether they also crop/mask the
ultrasound sector, and whether they use a specific frame stride; we use the plain transform.
EchoCLIP's weights are downloaded from the Hugging Face hub; NO MIMIC data is sent anywhere.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import FEATURES_DIR, sources
from ..features.echo import EMB_FILE

HF_ID = "hf-hub:mkaichristensen/echo-clip"


def read_dicom_frames(path: Path, n_frames: int) -> np.ndarray:
    """-> uint8 array (T, H, W, 3) RGB."""
    import pydicom
    from pydicom.pixel_data_handlers.util import convert_color_space

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    if arr.ndim == 2:            # single grayscale frame
        arr = arr[None]
    if arr.ndim == 3 and getattr(ds, "SamplesPerPixel", 1) == 3:  # single colour frame (H, W, 3)
        arr = arr[None]
    if arr.ndim == 3:            # (T, H, W) grayscale clip
        arr = np.repeat(arr[..., None], 3, axis=-1)
    pi = str(getattr(ds, "PhotometricInterpretation", "RGB"))
    if pi.startswith("YBR"):
        arr = convert_color_space(arr, pi, "RGB", per_frame=True)
    idx = np.linspace(0, len(arr) - 1, num=min(n_frames, len(arr))).round().astype(int)
    return arr[idx].astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(FEATURES_DIR / "echo_manifest_24h.csv"),
                    help="from `train echo-manifest`; the 24h manifest covers the 6h one")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N DICOMs")
    ap.add_argument("--save-every", type=int, default=200)
    args = ap.parse_args()

    import open_clip
    import torch
    from PIL import Image

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(HF_ID, device=device)
    model.eval()

    rec_path = sources()["echo_records"]
    if rec_path is None:
        raise FileNotFoundError("echo-record-list.csv not found")
    echo_root = rec_path.parent  # dicom_filepath is relative to the dataset version folder
    todo = pd.read_csv(args.manifest).dicom_filepath.drop_duplicates()
    done = pd.read_parquet(EMB_FILE) if EMB_FILE.exists() else pd.DataFrame(columns=["dicom_filepath"])
    todo = todo[~todo.isin(done.dicom_filepath)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(todo):,} DICOMs to embed on {device}")

    rows, missing = [], 0
    for i, rel in enumerate(todo, 1):
        path = echo_root / rel
        if not path.exists():
            missing += 1
            continue
        try:
            frames = read_dicom_frames(path, args.frames)
        except Exception as e:  # compressed transfer syntax without a decoder, corrupt file, ...
            print(f"  skip {rel}: {type(e).__name__}: {e}")
            continue
        x = torch.stack([preprocess(Image.fromarray(f)) for f in frames]).to(device)
        with torch.no_grad():
            z = model.encode_image(x).float()
            z = torch.nn.functional.normalize(z, dim=-1).mean(0)
        rows.append({"dicom_filepath": rel, **{f"e{j}": float(v) for j, v in enumerate(z.cpu().numpy())}})
        if i % args.save_every == 0 or i == len(todo):
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_parquet(EMB_FILE, index=False)
            rows = []
            print(f"  {i:,}/{len(todo):,} embedded (missing files so far: {missing:,})", flush=True)
    if rows:
        pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_parquet(EMB_FILE, index=False)
    print(f"done; {missing:,} DICOMs in the manifest were not on disk")


if __name__ == "__main__":
    main()
