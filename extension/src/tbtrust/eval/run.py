"""Evaluation entry point (`tbtrust-eval`).

Given a trained checkpoint and its config, produce the paper's numbers on the
held-out clinic: point metrics, calibration (ECE/MCE + reliability diagram), the
safe-deferral / risk-coverage curve with a tuned threshold, a conformal coverage
guarantee, and a robustness sweep across degradation severities. Writes a
metrics.json and figures to the run dir.

Two protocol rules this module enforces, because both are easy to get wrong and
both silently inflate the headline numbers:

1. **Everything fitted is fitted on val, applied to test.** The temperature, the
   uncertainty->confidence normalisation, the deferral threshold, and the
   conformal quantile are all estimated on the validation split of the *seen*
   clinics and then applied, never re-fit, on the held-out clinic. In deployment
   you do not have labels from the new clinic, so tuning on it measures nothing.

2. **The point prediction is held fixed across uncertainty methods.** Every
   method scores the same calibrated probabilities and differs only in the score
   the deferral policy *ranks* on. That isolates what is actually being compared
   -- the quality of the uncertainty ranking, which is what AURC measures --
   rather than confounding it with a change in the classifier's own predictions.

Usage:
    tbtrust-eval --config configs/loco_montgomery.yaml --checkpoint outputs/montgomery/best.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Uncertainty signals the deferral policy can rank on. "confidence" is the
# baseline (max(p, 1-p) off the point prediction); the rest are the real signals.
UNCERTAINTY_METHODS = ("confidence", "mc_dropout", "head", "ensemble")


def _collect(model, loader, device, mc_passes: int = 0):
    """Run a loader and return labels, logits, the uncertainty head, and MC spread.

    `logit` is always the deterministic (dropout-off) forward pass, so it is a
    stable input for temperature scaling regardless of which uncertainty method
    is being evaluated. When `mc_passes > 0` each batch is additionally run with
    dropout re-enabled to get the predictive spread.

    Everything is gathered in ONE pass over the loader, with the deterministic
    and stochastic forwards sharing the same batch tensor. A second pass would
    re-fetch the images, and `TBDataset` re-samples its smartphone degradation on
    every fetch, so the second pass sees *different* images: `mc_std[i]` would
    then describe a different photo than `logit[i]`, and the deferral policy
    would rank each prediction by another image's uncertainty. Sharing the batch
    makes the pairing correct whatever the dataset's seeding does.
    """
    import torch

    from ..models.uncertainty import enable_mc_dropout

    ys, logits, heads, mc_std = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)

            model.eval()                       # dropout off: the point prediction
            out = model(x)
            logits.append(out["logit"].cpu().numpy())
            ys.append(batch["label"].numpy())
            heads.append(
                out["uncertainty"].cpu().numpy()
                if "uncertainty" in out
                else np.full(len(batch["label"]), np.nan)
            )

            if mc_passes > 0:
                enable_mc_dropout(model)       # dropout on: the same x, T times
                samples = torch.stack(
                    [torch.sigmoid(model(x)["logit"]) for _ in range(mc_passes)], dim=0
                )
                mc_std.append(samples.std(dim=0).cpu().numpy())
    model.eval()                               # leave the model deterministic

    return {
        "y": np.concatenate(ys),
        "logit": np.concatenate(logits),
        "head": np.concatenate(heads),
        "mc_std": np.concatenate(mc_std) if mc_std else None,
    }


def _uncertainty_signal(collected: dict, method: str) -> np.ndarray | None:
    """Raw uncertainty (higher = defer sooner) for one method, or None if unavailable."""
    if method == "confidence":
        return None                       # deferral falls back to max(p, 1-p)
    if method == "mc_dropout":
        return collected.get("mc_std")
    if method == "head":
        head = collected.get("head")
        if head is None or np.all(np.isnan(head)):
            return None                   # model was built without an uncertainty head
        return head
    if method == "ensemble":
        return collected.get("ensemble_std")
    raise ValueError(f"unknown uncertainty method {method!r}; expected one of {UNCERTAINTY_METHODS}")


def _fit_confidence_map(val_uncertainty: np.ndarray):
    """Fit the uncertainty -> confidence normalisation on val; return a reusable map.

    `deferral.confidence_from_uncertainty` min-max normalises against whatever
    array it is handed. Calling it separately on val and on test would rescale
    each split by its own extremes, so a threshold tuned on val would mean
    something different on test. Freezing the bounds from val is what makes the
    tuned threshold transfer, and it is the same discipline as fitting the
    temperature on val.
    """
    u = np.asarray(val_uncertainty, dtype=float)
    lo, hi = float(np.min(u)), float(np.max(u))

    def to_confidence(x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if hi - lo < 1e-12:
            return np.full_like(x, 0.5)
        # Clip so a held-out clinic more extreme than anything seen in val still
        # maps into [0, 1] rather than outside the tuned threshold's domain.
        return np.clip(1.0 - (x - lo) / (hi - lo), 0.0, 1.0)

    return to_confidence


def _build_model(cfg: dict):
    """Arch dispatch, matching train/loop.py exactly."""
    from ..models.baseline import build_model
    from ..models.evidential import build_evidential_model
    from ..models.tbnet import TBNet

    arch = cfg["model"].get("arch", "baseline")
    if arch == "tbnet":
        m = cfg["model"]
        return TBNet(dropout=m.get("dropout", 0.3),
                     with_uncertainty_head=m.get("with_uncertainty_head", True))
    if arch == "evidential":
        return build_evidential_model(cfg)
    return build_model(cfg)


def evaluate(cfg: dict, checkpoint: str, ensemble_checkpoints: list[str] | None = None) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from ..data import manifest as M
    from ..data.dataset import TBDataset
    from ..data.splits import split_from_config
    from ..utils.io import load_checkpoint, save_json
    from ..utils.seed import seed_everything
    from . import calibration as C
    from . import conformal as CP
    from . import deferral as D
    from . import degradation_uncertainty as DU
    from . import forecast_verification as FV
    from .metrics import summary

    device = cfg["train"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    eval_cfg = cfg.get("eval", {})

    # MC-dropout draws from torch's global RNG. Without seeding here, two
    # evaluations of the same checkpoint return different mc_dropout AURCs and a
    # different tuned threshold -- the numbers would not be quotable.
    seed_everything(int(eval_cfg.get("seed", cfg.get("seed", 0))))

    # Same split the model was trained under -- see split_from_config.
    df = split_from_config(M.load(cfg["data"]["manifest"]), cfg)

    model = _build_model(cfg).to(device)
    load_checkpoint(model, checkpoint, map_location=device)

    out_dir = Path(checkpoint).parent
    split_mode = str(cfg["data"].get("split_mode", "loco")).lower()
    results: dict = {
        "split_mode": split_mode,
        # Under split_mode: random the test set is in-distribution, so this names
        # the config's clinic only for bookkeeping -- nothing was held out.
        "holdout_clinic": cfg["data"]["holdout_clinic"] if split_mode == "loco" else None,
        "arch": cfg["model"].get("arch", "baseline"),
    }

    primary_severity = eval_cfg.get("primary_severity", 0.5)
    mc_passes = int(eval_cfg.get("mc_dropout_passes", 20))
    batch_size = cfg["train"]["batch_size"]
    image_size = cfg["data"]["image_size"]

    # Fixed seed for every eval dataset: degradation must be reproducible here.
    # With seed=None the pipeline re-randomises artifacts on each fetch, so the
    # severity sweep would return different numbers on every run and could not be
    # compared across configs or checkpoints. TBDataset offsets this by row index,
    # so images still differ from one another. Training deliberately leaves it
    # unseeded -- there the randomness is the augmentation.
    eval_seed = int(eval_cfg.get("seed", cfg.get("seed", 0)))

    def loader_for(split: str, severity: float):
        ds = TBDataset(
            df, split=split, image_size=image_size,
            degradation_severity=severity, seed=eval_seed,
        )
        return DataLoader(ds, batch_size=batch_size)

    # ------------------------------------------------------------- val fits
    val = _collect(model, loader_for("val", primary_severity), device, mc_passes=mc_passes)
    if ensemble_checkpoints:
        val["ensemble_std"] = _ensemble_std(
            cfg, ensemble_checkpoints, loader_for("val", primary_severity), device
        )

    # Temperature scaling: fit on val logits, apply everywhere downstream.
    temp_bounds = tuple(eval_cfg.get("temperature_bounds", (0.05, 20.0)))
    temperature = C.fit_temperature(val["logit"], val["y"], bounds=temp_bounds)
    p_val = C.apply_temperature(val["logit"], temperature)
    results["temperature"] = float(temperature)
    # A temperature sitting on a bound means the NLL was still decreasing there:
    # the search wanted to go further and could not. That is a symptom (an
    # undertrained model whose logits carry almost no signal, or a val split too
    # small to calibrate on), not a calibrated temperature -- flag it rather than
    # quietly reporting whatever the bound happened to be.
    if min(abs(temperature - temp_bounds[0]), abs(temperature - temp_bounds[1])) < 1e-3:
        results["temperature_at_bound"] = True
        results["temperature_warning"] = (
            f"Fitted temperature {temperature:.4g} sits on the search bound {temp_bounds}. "
            "Treat the calibrated probabilities as unreliable and check that the model "
            "actually trained and that the val split is large enough."
        )
    results["val_ece_before_temperature"] = float(
        C.expected_calibration_error(val["y"], C.apply_temperature(val["logit"], 1.0))
    )
    results["val_ece_after_temperature"] = float(C.expected_calibration_error(val["y"], p_val))

    # Which uncertainty signals this checkpoint can actually produce.
    requested = eval_cfg.get("uncertainty", ["confidence", "mc_dropout", "head"])
    if isinstance(requested, str):
        requested = [requested]
    if ensemble_checkpoints and "ensemble" not in requested:
        requested = [*requested, "ensemble"]

    available, unavailable = [], []
    for method in requested:
        if method not in UNCERTAINTY_METHODS:
            raise ValueError(
                f"unknown eval.uncertainty entry {method!r}; expected one of {UNCERTAINTY_METHODS}"
            )
        if method == "confidence" or _uncertainty_signal(val, method) is not None:
            available.append(method)
        else:
            unavailable.append(method)
    results["uncertainty_methods_evaluated"] = available
    if unavailable:
        # e.g. `head` on a model built with with_uncertainty_head=False.
        results["uncertainty_methods_unavailable"] = unavailable

    # Per method: freeze the confidence map and the deferral threshold on val.
    tuned: dict[str, dict] = {}
    for method in available:
        u_val = _uncertainty_signal(val, method)
        conf_map = _fit_confidence_map(u_val) if u_val is not None else None
        conf_val = conf_map(u_val) if conf_map is not None else None
        point = D.tune_threshold(
            val["y"], p_val,
            target=eval_cfg.get("target", "accuracy"),
            target_value=eval_cfg.get("target_accuracy", 0.99),
            min_coverage=eval_cfg.get("min_coverage", 0.5),
            confidence=conf_val,
        )
        tuned[method] = {
            "threshold": point.threshold,
            "conf_map": conf_map,
            "val_coverage": point.coverage,
            "val_accuracy": point.accuracy,
        }

    results["deferral_threshold_tuned_on"] = "val"
    results["deferral_threshold_per_method"] = {
        m: {k: v for k, v in d.items() if k != "conf_map"} for m, d in tuned.items()
    }

    conformal_alpha = float(eval_cfg.get("conformal_alpha", 0.1))

    # ------------------------------------------------------- test-side sweep
    severities = eval_cfg.get("severity_sweep", [0.0, 0.25, 0.5, 0.75, 1.0])
    sweep: dict[str, dict] = {}
    mean_uncertainty_by_method: dict[str, list[float]] = {}
    for s in severities:
        test = _collect(model, loader_for("test", s), device, mc_passes=mc_passes)
        if ensemble_checkpoints:
            test["ensemble_std"] = _ensemble_std(
                cfg, ensemble_checkpoints, loader_for("test", s), device
            )

        y = test["y"]
        p = C.apply_temperature(test["logit"], temperature)   # calibrated, never re-fit

        m = summary(y, p)
        m["ece"] = C.expected_calibration_error(y, p)
        m["mce"] = C.max_calibration_error(y, p)
        m["brier_skill_score"] = FV.brier_skill_score(y, p)

        # The per-method comparison: same probabilities, different ranking signal.
        m["uncertainty_methods"] = {}
        for method in available:
            u = _uncertainty_signal(test, method)
            conf = tuned[method]["conf_map"](u) if tuned[method]["conf_map"] is not None else None
            # Track how uncertain this method is, on average, at this severity.
            # Correlated with severity after the sweep: see uncertainty_vs_severity.
            raw = 1.0 - np.maximum(p, 1.0 - p) if u is None else np.asarray(u, dtype=float)
            mean_uncertainty_by_method.setdefault(method, []).append(float(np.mean(raw)))
            m["uncertainty_methods"][method] = {
                "mean_uncertainty": float(np.mean(raw)),
                "aurc": D.area_under_risk_coverage(y, p, confidence=conf),
                "operating_point": D.risk_coverage_curve(
                    y, p, thresholds=np.array([tuned[method]["threshold"]]), confidence=conf
                )[0].__dict__,
                "human_rescue_rate": D.human_rescue_rate(
                    y, p, threshold=tuned[method]["threshold"], confidence=conf
                ),
            }
        sweep[str(s)] = m

        if s == primary_severity:
            results["murphy_decomposition"] = FV.murphy_decomposition(y, p)
            results["sharpness"] = FV.sharpness(p)
            results["conformal"] = CP.coverage_gap_report(
                val["y"], p_val, y, p, alpha=conformal_alpha
            )
            _save_figures(y, p, tuned, test, out_dir, C, D)

    results["robustness_sweep"] = sweep

    # Does each uncertainty signal actually respond to capture quality? A method
    # can rank ambiguous *labels* well (good AURC) while being blind to how bad
    # the photo is -- and the "retake the photo" half of the deferral message is
    # only justified for a method whose uncertainty rises with severity. Rank
    # correlation across the sweep points, so it costs nothing extra to compute.
    if len(severities) >= 2:
        results["uncertainty_vs_severity"] = {
            method: DU.uncertainty_vs_severity(severities, means)
            for method, means in mean_uncertainty_by_method.items()
        }
    if str(primary_severity) not in sweep:
        results["warning"] = (
            f"eval.primary_severity ({primary_severity}) is not in eval.severity_sweep "
            f"({severities}), so the deferral operating point, conformal report and "
            "figures were skipped. Add it to the sweep."
        )

    save_json(results, out_dir / "metrics.json")
    return results


def _ensemble_std(cfg: dict, checkpoint_paths: list[str], loader, device: str) -> np.ndarray:
    """Member-disagreement uncertainty from a deep ensemble."""
    import torch

    from ..models.ensemble import DeepEnsemble

    ensemble = DeepEnsemble.load(cfg, checkpoint_paths, device=device)
    stds = []
    with torch.no_grad():
        for batch in loader:
            _, std = ensemble.predict(batch["image"].to(device))
            stds.append(std.cpu().numpy())
    return np.concatenate(stds)


def _save_figures(y, p, tuned, test, out_dir, C, D):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    C.plot_reliability(y, p, ax=axes[0], title="Calibration (temperature-scaled)")
    for method, d in tuned.items():
        u = _uncertainty_signal(test, method)
        conf = d["conf_map"](u) if d["conf_map"] is not None else None
        D.plot_risk_coverage(y, p, ax=axes[1], label=method, confidence=conf)
    axes[1].set_title("Safe deferral by uncertainty method")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "reliability_and_deferral.png", dpi=150)
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate TB-Trust")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config-dir", default="configs")
    p.add_argument(
        "--ensemble-checkpoints",
        default=None,
        help="comma-separated member checkpoints; adds the 'ensemble' uncertainty method",
    )
    p.add_argument("overrides", nargs="*")
    return p


def main():
    from ..config import load_experiment

    args = build_argparser().parse_args()
    cfg = load_experiment(args.config, config_dir=args.config_dir, overrides=args.overrides)
    members = (
        [c.strip() for c in args.ensemble_checkpoints.split(",") if c.strip()]
        if args.ensemble_checkpoints
        else None
    )
    print(evaluate(cfg, args.checkpoint, ensemble_checkpoints=members))


if __name__ == "__main__":
    main()
