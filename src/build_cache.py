"""
Decode the dataset once into a memmapped uint8 array so that training reads
from RAM instead of re-decoding PNGs every epoch.

Two cache variants are built so the preprocessing pipeline itself can be
ablated:

  faithful  - the paper's pipeline: B-channel split, padding-aware auto-crop,
             resize, the DSI's (11,11,168,202) crop, resize, corner masking.
             This mirrors src/preprocessing.py + src/_tf1_reference/dsi.py.
  simple    - what src/train.py currently does: grayscale convert + resize.

Usage:
  python src/build_cache.py --data-path data/ --variant faithful
  python src/build_cache.py --data-path data/ --variant simple
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from common import CACHE_DIR, RESIZE_SHAPE
from preprocessing import preprocess_image

# Corner mask from preprocessing.py: hides burnt-in metadata in the top
# corners with the mean training intensity.
AVG_TRAIN_INTENSITY = 0.531459229


def _corner_mask():
    keep = np.ones(RESIZE_SHAPE, dtype=np.float32)
    fill = np.zeros(RESIZE_SHAPE, dtype=np.float32)
    rr, cc = np.mgrid[0:RESIZE_SHAPE[0], 0:RESIZE_SHAPE[1]]
    corners = (rr < 45) & ((cc < 45) | (cc > 178))
    keep[corners] = 0.0
    fill[corners] = AVG_TRAIN_INTENSITY
    return keep, fill


KEEP, FILL = _corner_mask()


def dsi_crop(img_uint8, tf_semantics=True):
    """
    The crop applied inside the original data interface.

    dsi.py calls tf.image.crop_to_bounding_box(img, 11, 11, 168, 202), whose
    last two arguments are target HEIGHT and WIDTH -> rows 11:179, cols 11:213.
    preprocessing.py instead slices [11:168, 11:202], treating them as end
    indices. The two differ; tf_semantics=True reproduces the original.
    """
    if tf_semantics:
        cropped = img_uint8[11:11 + 168, 11:11 + 202]
    else:
        cropped = img_uint8[11:168, 11:202]
    return cv2.resize(cropped, RESIZE_SHAPE, interpolation=cv2.INTER_LANCZOS4)


def process_faithful(path):
    cleaned = preprocess_image(path)          # uint8 HxWx3, B-channel repeated
    if cleaned.ndim == 3:
        cleaned = cleaned[:, :, 0]
    img = dsi_crop(cleaned).astype(np.float32) / 255.0
    img = img * KEEP + FILL
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return (img * 255.0).clip(0, 255).astype(np.uint8)


def process_simple(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, RESIZE_SHAPE, interpolation=cv2.INTER_LINEAR)


PROCESSORS = {"faithful": process_faithful, "simple": process_simple}


def _worker(args):
    path, variant = args
    try:
        return os.path.basename(path), PROCESSORS[variant](path)
    except Exception as exc:                      # noqa: BLE001 - report and skip
        print(f"  ! failed on {os.path.basename(path)}: {exc}")
        return os.path.basename(path), None


def build(data_path, variant, workers):
    paths = sorted(
        glob.glob(os.path.join(data_path, "**", "*.png"), recursive=True)
        + glob.glob(os.path.join(data_path, "**", "*.jpg"), recursive=True)
    )
    if not paths:
        raise SystemExit(f"No images found under {data_path}")
    print(f"Found {len(paths)} images; building '{variant}' cache "
          f"with {workers} workers")

    os.makedirs(CACHE_DIR, exist_ok=True)
    arr = np.zeros((len(paths), *RESIZE_SHAPE), dtype=np.uint8)
    names, row = [], 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (name, img) in enumerate(
                pool.map(_worker, [(p, variant) for p in paths], chunksize=32)):
            if img is None:
                continue
            arr[row] = img
            names.append(name)
            row += 1
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(paths)}")

    arr = arr[:row]
    np.save(os.path.join(CACHE_DIR, f"{variant}.npy"), arr)
    pd.DataFrame({"filename": names, "row": range(row)}).to_csv(
        os.path.join(CACHE_DIR, f"{variant}_index.csv"), index=False)
    print(f"Cached {row} images -> cache/{variant}.npy "
          f"({arr.nbytes / 1024 ** 2:.0f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data/")
    ap.add_argument("--variant", default="faithful",
                    choices=list(PROCESSORS) + ["both"])
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()

    for v in (list(PROCESSORS) if a.variant == "both" else [a.variant]):
        build(a.data_path, v, a.workers)
