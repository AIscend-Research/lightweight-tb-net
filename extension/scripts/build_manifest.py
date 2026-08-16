#!/usr/bin/env python3
"""Scan data/raw/ into data/processed/manifest.csv and report class balance.

Assumes images live somewhere under data/raw/ and that either:
  * the aggregated Kaggle layout is used (folders named Normal/ and Tuberculosis/), or
  * the raw NLM layout is used (filenames MCUCXR_*_0/1, CHNCXR_*_0/1).

Adjust `label_from_dir` / the provenance rules in tbtrust.data.manifest to match
however you actually laid things out. This script is deliberately thin.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tbtrust.data import manifest as M


def label_from_nlm_filename(path: str) -> int:
    stem = Path(path).stem
    if stem.endswith("_1"):
        return 1
    if stem.endswith("_0"):
        return 0
    return -1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed/manifest.csv")
    ap.add_argument("--layout", choices=["kaggle", "nlm", "auto"], default="auto")
    args = ap.parse_args()

    if args.layout in ("kaggle", "auto"):
        df = M.scan_directory(args.raw, label_from_dir={"Tuberculosis": 1, "Normal": 0})
    else:
        df = M.scan_directory(args.raw)

    # backfill labels from NLM filenames where the folder map missed them
    mask = df["label"] < 0
    if mask.any():
        df.loc[mask, "label"] = df.loc[mask, "path"].map(label_from_nlm_filename)

    unlabeled = int((df["label"] < 0).sum())
    if unlabeled:
        print(f"WARNING: {unlabeled} images have no label (-1). Fix the rules or add a metadata join.")

    M.save(df, args.out)
    print(f"Wrote {len(df)} rows -> {args.out}\n")
    print("Class balance per clinic:")
    with pd.option_context("display.max_rows", None):
        print(M.class_balance_report(df))
    print("\nReminder: NIAID is TB-only, RSNA normal-only, Belarus TB-only. "
          "Use Montgomery/Shenzhen as your two-class LOCO holdouts.")


if __name__ == "__main__":
    main()
