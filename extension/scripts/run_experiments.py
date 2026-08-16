#!/usr/bin/env python3
"""Phase 4: run and aggregate the full leave-one-clinic-out sweep.

ONBOARDING.md's for-loop trains and evaluates each fold but leaves "turn the
collected metrics.json files into the actual cross-site comparison" as a manual
step. This script is that step: collect per-image [clinic, label, prob]
predictions across folds, then run `eval/crosssite.py` (generalization gap,
per-clinic calibration heatmap) and `eval/forecast_verification.py` (Murphy
decomposition, Brier skill score per clinic) over the combined table.

Deliberately does NOT re-tune the deferral threshold on this combined
test-set data -- that would reintroduce the val/test leak `eval/run.py` was
just fixed to avoid. Each fold's own `outputs/<clinic>/metrics.json` (from
`tbtrust-eval`) has the honest, val-tuned deferral operating point and
human-rescue rate; this script's job is strictly the cross-site comparison.

Usage:
    # train + evaluate every fold from scratch
    python scripts/run_experiments.py \
        --configs configs/loco_montgomery.yaml,configs/loco_shenzhen.yaml --train

    # aggregate already-trained checkpoints
    python scripts/run_experiments.py \
        --configs configs/loco_montgomery.yaml,configs/loco_shenzhen.yaml \
        --checkpoints outputs/montgomery/best.ckpt,outputs/shenzhen/best.ckpt \
        --reference-accuracy 0.97

`--reference-accuracy` is the in-distribution (random-split) reference number
for the generalization-gap table -- get it from a random-split run (e.g.
`configs/baseline_densenet.yaml` evaluated on a random train/test split, not a
LOCO fold). This script only aggregates LOCO folds; it doesn't compute that
reference itself, since doing so needs a different (non-LOCO) split entirely.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _fold_predictions(cfg: dict, checkpoint: str, severity: float, device: str | None = None):
    """Test-split predictions for one LOCO fold, temperature-scaled on that fold's val.

    Two things this deliberately shares with `eval/run.py` rather than
    reimplementing:

    * `_build_model` for the arch dispatch. This path used to call bare `TBNet()`,
      ignoring the config's `dropout` and `with_uncertainty_head` -- so a
      checkpoint trained with `with_uncertainty_head: false` could not be loaded
      back here at all.
    * The temperature, re-fitted per fold on that fold's own validation split
      (never on test). Without it the per-clinic ECE in the cross-site table
      would describe uncalibrated probabilities while each fold's metrics.json
      describes calibrated ones -- two different numbers under one name.
    """
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from tbtrust.data import manifest as M
    from tbtrust.data.dataset import TBDataset
    from tbtrust.data.splits import split_from_config
    from tbtrust.eval.calibration import apply_temperature, fit_temperature
    from tbtrust.eval.run import _build_model
    from tbtrust.utils.io import load_checkpoint

    device = device or cfg["train"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    # Same split the model was trained under -- see split_from_config.
    df = split_from_config(M.load(cfg["data"]["manifest"]), cfg)

    model = _build_model(cfg).to(device)
    load_checkpoint(model, checkpoint, map_location=device)
    model.eval()

    # Same fixed-seed rule as eval/run.py: the cross-site comparison has to be
    # reproducible, so degradation must not be re-randomised per fetch.
    eval_seed = int(cfg.get("eval", {}).get("seed", cfg.get("seed", 0)))

    def labels_and_logits(split: str, sev: float):
        ds = TBDataset(df, split=split, image_size=cfg["data"]["image_size"],
                       degradation_severity=sev, seed=eval_seed)
        loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"])
        ys, zs = [], []
        with torch.no_grad():
            for batch in loader:
                zs.append(model(batch["image"].to(device))["logit"].cpu().numpy())
                ys.append(batch["label"].numpy())
        return np.concatenate(ys), np.concatenate(zs)

    y_val, z_val = labels_and_logits("val", severity)
    temperature = fit_temperature(z_val, y_val)

    y, z = labels_and_logits("test", severity)
    p = apply_temperature(z, temperature)
    return pd.DataFrame({
        "clinic": cfg["data"]["holdout_clinic"],
        "label": y,
        "prob": p,
        "temperature": temperature,
    })


def run_fold(config_path: str, config_dir: str, checkpoint: str | None, train_first: bool,
            severity_override: float | None, overrides: list[str] | None = None):
    from tbtrust.config import load_experiment

    cfg = load_experiment(config_path, config_dir=config_dir, overrides=overrides)
    if train_first or checkpoint is None:
        from tbtrust.train.loop import train

        result = train(cfg)
        checkpoint = result["checkpoint"]
    severity = severity_override if severity_override is not None else cfg.get("eval", {}).get("primary_severity", 0.5)
    df_fold = _fold_predictions(cfg, checkpoint, severity)
    return df_fold, cfg, checkpoint, severity


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", required=True, help="comma-separated experiment yaml paths, one per LOCO fold")
    ap.add_argument("--config-dir", default="configs")
    ap.add_argument("--checkpoints", default=None,
                    help="comma-separated checkpoint paths aligned with --configs; omit with --train")
    ap.add_argument("--train", action="store_true", help="train each fold before evaluating")
    ap.add_argument("--severity", type=float, default=None,
                    help="fixed severity for the cross-site comparison; default = each config's eval.primary_severity")
    ap.add_argument("--reference-accuracy", type=float, default=None,
                    help="in-distribution reference accuracy for the generalization-gap table (see module docstring)")
    ap.add_argument("--out-dir", default="outputs/loco_sweep")
    ap.add_argument("overrides", nargs="*", help="key.subkey=value overrides applied to every fold's config")
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    if args.checkpoints:
        checkpoints = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
        if len(checkpoints) != len(configs):
            raise SystemExit("--checkpoints must list exactly one path per --configs entry")
    else:
        if not args.train:
            raise SystemExit("pass --checkpoints (aligned with --configs) or --train to train fresh")
        checkpoints = [None] * len(configs)

    frames, severities_used = [], {}
    # strict=True: both branches above guarantee equal lengths, so a mismatch is
    # a bug worth crashing on rather than silently truncating a fold out of the sweep.
    for config_path, checkpoint in zip(configs, checkpoints, strict=True):
        df_fold, cfg, used_checkpoint, severity = run_fold(
            config_path, args.config_dir, checkpoint, args.train, args.severity, overrides=args.overrides
        )
        clinic = cfg["data"]["holdout_clinic"]
        severities_used[clinic] = severity
        print(f"{clinic}: {len(df_fold)} test images @ severity={severity} (checkpoint: {used_checkpoint})")
        frames.append(df_fold)

    combined = __import__("pandas").concat(frames, ignore_index=True)

    from tbtrust.eval import crosssite as X
    from tbtrust.eval import forecast_verification as FV

    per_clinic = X.per_clinic_table(combined)
    print("\nPer-clinic (cross-site) results:")
    print(per_clinic)

    report: dict = {
        "severities_used": severities_used,
        "per_clinic": per_clinic.reset_index().to_dict(orient="records"),
    }

    if args.reference_accuracy is not None:
        gap = X.generalization_gap(per_clinic, args.reference_accuracy)
        report["reference_accuracy"] = args.reference_accuracy
        report["generalization_gap"] = gap.reset_index().to_dict(orient="records")
        print(f"\nGeneralization gap vs. in-distribution reference ({args.reference_accuracy}):")
        print(gap[["accuracy", "gap_vs_reference"]])
    else:
        print(
            "\nNo --reference-accuracy given -- skipping the generalization-gap table. "
            "Get that number from a random-split (non-LOCO) run, e.g. "
            "configs/baseline_densenet.yaml evaluated on a random train/test split."
        )

    loco_results = {clinic: sub for clinic, sub in combined.groupby("clinic")}
    heatmap = X.calibration_heatmap_data(loco_results)
    report["calibration_heatmap"] = heatmap.reset_index().to_dict(orient="records")

    report["forecast_verification_per_clinic"] = {
        clinic: {
            **FV.murphy_decomposition(sub["label"], sub["prob"]),
            "brier_skill_score": FV.brier_skill_score(sub["label"], sub["prob"]),
        }
        for clinic, sub in loco_results.items()
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "combined_predictions.csv", index=False)
    with open(out_dir / "loco_sweep_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    if args.reference_accuracy is not None:
        import matplotlib.pyplot as plt

        fig = X.plot_gap_bars(gap).figure
        fig.tight_layout()
        fig.savefig(out_dir / "generalization_gap.png", dpi=150)
        plt.close(fig)

    print(f"\nWrote {out_dir}/combined_predictions.csv and loco_sweep_report.json")


if __name__ == "__main__":
    main()
