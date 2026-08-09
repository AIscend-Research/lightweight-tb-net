"""
Generate stratified train/val/test splits, and verify they are leak-free.

Replaces the previous generator, which produced splits containing 1400 TB
entries drawn from only 700 distinct images: the TB class was present twice,
once as 'Tuberculosis-N.png' and once as 'TB-N.png'. Because the split was
taken over rows rather than over distinct images, 222 of the 700 TB images
ended up in both the training set and a held-out set under their other name.
115 of the 140 TB images in the old test.csv were duplicates of training
images. Any metric computed on those splits is contaminated.

This version keys on a canonical image identity, so a duplicate can never
straddle a split boundary, and then asserts that property before writing.

  python src/make_splits.py --data-path data/ --out data_splits/
  python src/make_splits.py --verify-only --out data_splits/
"""

import argparse
import glob
import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

SPLITS = ("train", "val", "test")


def canonical_id(filename):
    """
    Map filenames that denote the same underlying image onto one key.
    'TB-14.png' and 'Tuberculosis-14.png' are the same radiograph.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"^(TB|Tuberculosis)-(\d+)$", stem, flags=re.IGNORECASE)
    if m:
        return f"tb-{int(m.group(2))}"
    m = re.match(r"^Normal-(\d+)$", stem, flags=re.IGNORECASE)
    if m:
        return f"normal-{int(m.group(1))}"
    return stem.lower()


def build(data_path, out_dir, seed, val_frac, test_frac):
    files = sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(data_path, "**", "*.png"), recursive=True)
    )
    if not files:
        raise SystemExit(f"no PNGs under {data_path}")

    df = pd.DataFrame({"filename": files})
    df["label"] = (~df["filename"].str.startswith("Normal")).astype(int)
    df["cid"] = df["filename"].map(canonical_id)

    n_before = len(df)
    # Keep one representative per distinct image.
    df = df.sort_values("filename").drop_duplicates("cid", keep="first")
    if len(df) < n_before:
        print(f"dropped {n_before - len(df)} duplicate filenames "
              f"({n_before} files -> {len(df)} distinct images)")

    counts = df["label"].value_counts().sort_index()
    print(f"distinct images: {len(df)}  normal={counts.get(0, 0)}  "
          f"TB={counts.get(1, 0)}  ratio={counts.get(0, 0) / max(counts.get(1, 1), 1):.2f}:1")

    train, temp = train_test_split(df, test_size=val_frac + test_frac,
                                   stratify=df["label"], random_state=seed)
    val, test = train_test_split(
        temp, test_size=test_frac / (val_frac + test_frac),
        stratify=temp["label"], random_state=seed)

    os.makedirs(out_dir, exist_ok=True)
    for name, part in zip(SPLITS, (train, val, test)):
        part[["filename", "label"]].to_csv(
            os.path.join(out_dir, f"{name}.csv"), index=False, header=False)
        pos = int(part["label"].sum())
        print(f"  {name:5s} {len(part):5d}  TB={pos}  normal={len(part) - pos}")

    verify(out_dir)


def verify(out_dir):
    """Fail loudly if any canonical image appears in more than one split."""
    members = {}
    for name in SPLITS:
        path = os.path.join(out_dir, f"{name}.csv")
        df = pd.read_csv(path, header=None, names=["filename", "label"])
        members[name] = set(df["filename"].map(canonical_id))
        if len(members[name]) != len(df):
            raise SystemExit(
                f"{name}.csv contains {len(df) - len(members[name])} "
                f"duplicate images under different filenames")

    ok = True
    for a in SPLITS:
        for b in SPLITS:
            if a >= b:
                continue
            overlap = members[a] & members[b]
            if overlap:
                ok = False
                print(f"LEAK: {len(overlap)} images shared between {a} and {b} "
                      f"(e.g. {sorted(overlap)[:5]})")
    if not ok:
        raise SystemExit("splits are contaminated - do not train on these")
    total = sum(len(v) for v in members.values())
    print(f"splits verified leak-free: {total} distinct images across "
          f"{'/'.join(f'{len(members[s])}' for s in SPLITS)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="data/")
    ap.add_argument("--out", default="data_splits/")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()

    if a.verify_only:
        verify(a.out)
    else:
        build(a.data_path, a.out, a.seed, a.val_frac, a.test_frac)
