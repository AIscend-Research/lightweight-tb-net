#!/usr/bin/env python3
"""Thin wrapper: `python scripts/evaluate.py --config ... --checkpoint ...`.
Prefer the `tbtrust-eval` console script after install."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tbtrust.eval.run import main

if __name__ == "__main__":
    main()
