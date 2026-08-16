#!/usr/bin/env python3
"""Thin wrapper so you can run `python scripts/train.py --config ...` from the repo
root without installing. Prefer the `tbtrust-train` console script after install."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tbtrust.train.loop import main

if __name__ == "__main__":
    main()
