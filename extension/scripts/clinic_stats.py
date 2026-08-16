#!/usr/bin/env python3
"""Measure the domain shift between clinics: resolution, brightness, contrast.

Run this once the manifest exists and before you interpret any LOCO gap -- it is
the evidence that the clinics really are different imaging conditions and not
just different filenames, and the table goes straight into the paper next to the
per-clinic generalization gaps.

    python scripts/clinic_stats.py --manifest data/processed/manifest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbtrust.data import manifest as M
from tbtrust.data.clinic_stats import clinic_shift_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--sample", type=int, default=100, help="images sampled per clinic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional CSV to write")
    args = ap.parse_args()

    df = M.load(args.manifest)
    table = clinic_shift_table(df, sample_per_clinic=args.sample, seed=args.seed)
    if table.empty:
        raise SystemExit(
            f"No readable images found via {args.manifest}. Check that the 'path' "
            "column points at files that exist on this machine."
        )
    print(table.to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
