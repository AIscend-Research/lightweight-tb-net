#!/usr/bin/env python3
"""Train a deep ensemble: N independent runs of the same config, different seeds.

    python scripts/train_ensemble.py --config configs/loco_montgomery.yaml --n-members 5

Each member is a full `tbtrust-train` run under the hood (see
`models/ensemble.train_deep_ensemble`); this just fans that out N times and
reports where the checkpoints landed. Compare its calibration/deferral numbers
(via `models.ensemble.evaluate_ensemble`) against the single-model and
evidential runs on the same held-out clinic to pick the featured uncertainty
method (docs/phase1_framing.md section 2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbtrust.config import load_experiment
from tbtrust.models.ensemble import train_deep_ensemble


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--config-dir", default="configs")
    ap.add_argument("--n-members", type=int, default=5)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("overrides", nargs="*", help="key.subkey=value overrides")
    args = ap.parse_args()

    cfg = load_experiment(args.config, config_dir=args.config_dir, overrides=args.overrides)
    checkpoints = train_deep_ensemble(cfg, n_members=args.n_members, base_seed=args.base_seed)
    print(f"Trained {len(checkpoints)} members:")
    for c in checkpoints:
        print(f"  {c}")


if __name__ == "__main__":
    main()
