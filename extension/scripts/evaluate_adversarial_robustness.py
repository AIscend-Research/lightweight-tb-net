#!/usr/bin/env python3
"""Phase 3 parallel track: worst-of-N-query black-box degradation search.

    python scripts/evaluate_adversarial_robustness.py \
        --config configs/loco_montgomery.yaml --checkpoint outputs/montgomery/best.ckpt

See eval/adversarial_degradation.py for what this actually measures and why it
is a worst-of-N search rather than a gradient-based attack (the degradation ops
are non-differentiable PIL transforms).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    import torch

    from tbtrust.config import load_experiment
    from tbtrust.data import manifest as M
    from tbtrust.data.splits import leave_one_clinic_out
    from tbtrust.eval.adversarial_degradation import evaluate_adversarial_robustness
    from tbtrust.models.baseline import build_model
    from tbtrust.models.evidential import build_evidential_model
    from tbtrust.models.tbnet import TBNet
    from tbtrust.utils.io import load_checkpoint, save_json

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--config-dir", default="configs")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--severity", type=float, default=0.7)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--sample-n", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_experiment(args.config, config_dir=args.config_dir, overrides=args.overrides)
    device = cfg["train"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    df = M.load(cfg["data"]["manifest"])
    df = leave_one_clinic_out(df, cfg["data"]["holdout_clinic"],
                              require_two_class_test=cfg["data"].get("require_two_class_test", True))
    test_df = df[df["split"] == "test"]

    arch = cfg["model"].get("arch")
    model = (TBNet() if arch == "tbnet" else build_evidential_model(cfg) if arch == "evidential" else build_model(cfg))
    model = model.to(device)
    load_checkpoint(model, args.checkpoint, map_location=device)

    report = evaluate_adversarial_robustness(
        model, test_df, severity=args.severity, n_trials=args.n_trials,
        sample_n=args.sample_n, image_size=cfg["data"]["image_size"], device=device,
    )
    print(json.dumps(report, indent=2))

    out_path = Path(args.out) if args.out else Path(args.checkpoint).parent / "adversarial_robustness.json"
    save_json(report, out_path)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
