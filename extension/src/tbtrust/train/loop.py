"""Training loop and CLI entry point (`tbtrust-train`).

Trains the classifier (BCE) and, when the model has one, the uncertainty head
(MSE against the weak degradation-derived target). Randomizes degradation severity
during training so the model learns across image qualities; validates on a fixed
severity. Saves the best checkpoint by validation accuracy.

Optionally adds a domain-generalization penalty (`dg.method`: coral | dann | irm)
or clinic-conditional FiLM (`model.clinic_film`) -- the other half of the
cross-site story, attacking the generalization gap at training time rather than
absorbing it with deferral. See `_dg_loss` and `losses/dg.py`.

Model selection uses the validation split of the *seen* clinics, never the
held-out one: at deployment you would have no labels from the new clinic to
tune on.

Usage:
    tbtrust-train --config configs/loco_montgomery.yaml
    python scripts/train.py --config configs/baseline_densenet.yaml model.backbone=resnet50
    tbtrust-train --config configs/loco_montgomery_coral.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import load_experiment
from ..utils.io import save_checkpoint, save_json
from ..utils.seed import seed_everything


def _build_loaders(cfg: dict):
    from torch.utils.data import DataLoader

    from ..data import manifest as M
    from ..data.dataset import TBDataset, uniform_severity
    from ..data.splits import split_from_config

    # Shared with eval/run.py and scripts/run_experiments.py so all three derive
    # an identical split -- see split_from_config / loco_split_from_config for why that matters.
    df = split_from_config(M.load(cfg["data"]["manifest"]), cfg)

    sev = cfg["degradation"]
    seed = cfg.get("seed", 0)
    # Seed both datasets. seed_everything() covers torch/python/np global RNGs but
    # not the degradation pipeline, which builds its own Generator per call: unseeded,
    # that draws from OS entropy, so a run was irreproducible no matter what seed
    # was configured. The training set still varies across epochs via set_epoch().
    train_ds = TBDataset(
        df, split="train", image_size=cfg["data"]["image_size"],
        severity_sampler=uniform_severity(sev["train_low"], sev["train_high"], seed=seed),
        seed=seed,
    )
    val_ds = TBDataset(
        df, split="val", image_size=cfg["data"]["image_size"],
        degradation_severity=sev.get("val_fixed", 0.0),
        seed=seed,
    )
    # Fail fast on an empty split rather than at the far end of the pipeline.
    # A small clinic plus a small val_frac can round every stratum's val count to
    # zero. Without this the run trains happily, every epoch's val accuracy is
    # NaN, no checkpoint is ever written (NaN > best_acc is False), and the first
    # symptom is a FileNotFoundError in eval pointing at a checkpoint path the
    # training run reported as its result. Everything downstream -- temperature,
    # the deferral threshold, the conformal quantile -- is fitted on val, so an
    # empty val split is fatal regardless.
    counts = df["split"].value_counts()
    for split in ("train", "val"):
        if counts.get(split, 0) == 0:
            raise ValueError(
                f"The '{split}' split is empty for holdout clinic "
                f"'{cfg['data']['holdout_clinic']}' (split sizes: {counts.to_dict()}). "
                "Raise data.val_frac, or check the manifest has enough images per "
                "(clinic, label) stratum."
            )

    bs = cfg["train"]["batch_size"]
    nw = cfg["train"].get("num_workers", 2)
    if len(train_ds) < bs:
        # drop_last=True would silently yield zero batches and train nothing.
        raise ValueError(
            f"train split has {len(train_ds)} images but batch_size is {bs}; with "
            "drop_last=True that produces no batches at all. Lower train.batch_size."
        )
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=True),
        DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw),
        df,
    )


def _dg_loss(model, out, y, clinic_idx, cfg, step, total_steps):
    """Domain-generalization penalty added to the task loss.

    Chosen by cfg["dg"]["method"]:
        none    task loss only. This is the baseline that produces the
                cross-site generalization gap the paper reports.
        coral   align feature mean/covariance across the clinics in a batch.
        dann    a clinic classifier behind a gradient-reversal layer.
        irm     the IRM gradient penalty, computed per clinic and averaged.

    Orthogonal to the uncertainty work: it changes what the backbone learns, not
    how confidence is read off it, so any dg.method combines with any
    model.arch and with every uncertainty method in eval/run.py.
    """
    import torch
    import torch.nn.functional as F

    from ..losses.dg import coral_loss, irm_penalty
    from ..models.grl import dann_lambda

    dg = cfg.get("dg", {})
    method = str(dg.get("method", "none")).lower()
    weight = float(dg.get("weight", 1.0))
    zero = torch.zeros((), device=out["logit"].device)
    if method == "none" or weight == 0.0:
        return zero, {}

    clinics = clinic_idx.unique()
    if method == "coral":
        # A batch drawn from one clinic has nothing to align; skip rather than
        # return a spurious zero-gradient term.
        if len(clinics) < 2:
            return zero, {}
        groups = [out["features"][clinic_idx == c] for c in clinics]
        loss = weight * coral_loss(groups)
    elif method == "dann":
        lambd = dann_lambda(step, total_steps, gamma=float(dg.get("gamma", 10.0)))
        loss = weight * F.cross_entropy(model.domain_logits(out["features"], lambd), clinic_idx)
    elif method == "irm":
        penalties = [irm_penalty(out["logit"][clinic_idx == c], y[clinic_idx == c]) for c in clinics]
        loss = weight * torch.stack(penalties).mean()
    else:
        raise ValueError(
            f"unknown dg.method {method!r}; expected one of none | coral | dann | irm"
        )
    return loss, {"dg_loss": float(loss.item())}


def _evaluate(model, loader, device) -> dict:
    import numpy as np
    import torch

    from ..eval.metrics import summary

    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            logit = model(batch["image"].to(device))["logit"]
            ps.append(torch.sigmoid(logit).cpu().numpy())
            ys.append(batch["label"].numpy())
    if not ys:
        return {"accuracy": float("nan"), "n": 0}
    return summary(np.concatenate(ys), np.concatenate(ps))


def train(cfg: dict) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn

    from ..models.baseline import build_model
    from ..models.evidential import build_evidential_model, evidential_loss
    from ..models.tbnet import TBNet

    seed_everything(cfg.get("seed", 0))
    device = cfg["train"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = _build_loaders(cfg)

    arch = cfg["model"].get("arch", "baseline")
    dg_method = str(cfg.get("dg", {}).get("method", "none")).lower()
    # tbnet.py and evidential.py return neither pooled features nor a domain head,
    # so the DG penalties have nothing to attach to. Fail here with the reason
    # rather than at the first batch with a KeyError on "features".
    if dg_method != "none" and arch != "baseline":
        raise ValueError(
            f"dg.method={dg_method!r} is only implemented for model.arch=baseline "
            f"(got {arch!r}): TBNet and the evidential head expose no pooled "
            "features or domain head. Add one to that arch, or set dg.method=none."
        )
    if arch == "tbnet":
        m = cfg["model"]
        model = TBNet(dropout=m.get("dropout", 0.3),
                      with_uncertainty_head=m.get("with_uncertainty_head", True)).to(device)
    elif arch == "evidential":
        model = build_evidential_model(cfg).to(device)
    else:
        model = build_model(cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"].get("weight_decay", 1e-4))
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    u_weight = cfg["train"].get("uncertainty_loss_weight", 0.5)
    annealing_epochs = cfg["train"].get("evidential_annealing_epochs", 10)

    out_dir = Path(cfg["train"].get("output_dir", "outputs")) / cfg["data"]["holdout_clinic"]
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc, best_path = -1.0, out_dir / "best.ckpt"

    # DANN eases its adversary in over the whole run, so it needs the total step
    # count up front.
    step, total_steps = 0, max(cfg["train"]["epochs"] * max(len(train_loader), 1), 1)
    use_clinic = bool(cfg["model"].get("clinic_film", False))
    last_dg: dict = {}

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        # Advance the augmentation stream so each epoch sees different artifacts
        # while the whole run stays reproducible from cfg.seed.
        train_loader.dataset.set_epoch(epoch)
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].float().to(device)
            clinic_idx = batch["clinic_idx"].to(device)
            # FiLM is the only component that conditions the forward pass on the
            # clinic; CORAL/DANN/IRM use the label only in the loss. Passing it
            # unconditionally would be harmless but makes it look like the plain
            # baseline sees provenance at inference, which it must not.
            out = model(x, clinic_idx) if use_clinic else model(x)
            if "evidence" in out:
                # Dirichlet/Bayes-risk loss (models/evidential.py), not BCE: the
                # "uncertainty" here is the vacuity read off the evidence, not a
                # separately-trained head, so there's no MSE term to add.
                loss = evidential_loss(out["evidence"], y, epoch=epoch, annealing_epochs=annealing_epochs)
            else:
                loss = bce(out["logit"], y)
                if "uncertainty" in out:
                    loss = loss + u_weight * mse(out["uncertainty"], batch["uncertainty_target"].to(device))
            dg_term, last_dg = _dg_loss(model, out, y, clinic_idx, cfg, step, total_steps)
            loss = loss + dg_term
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1

        val = _evaluate(model, val_loader, device)
        dg_note = f" dg_loss={last_dg['dg_loss']:.4f}" if last_dg else ""
        print(
            f"epoch {epoch}: val_acc={val['accuracy']:.4f} "
            f"val_sens={val.get('sensitivity', float('nan')):.4f}{dg_note}"
        )
        # `not (nan > best)` is True, so a NaN val accuracy must not be allowed to
        # leave best.ckpt unwritten while train() still reports its path.
        if best_path.exists() and not np.isfinite(val["accuracy"]):
            continue
        if not best_path.exists() or val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            save_checkpoint(model, best_path, config=cfg, epoch=epoch, val=val)

    if not best_path.exists():
        raise RuntimeError(
            f"No checkpoint was written to {best_path}: every epoch's validation "
            "accuracy was non-finite. Check that the val split is non-empty and "
            "that the loss is not diverging."
        )

    result = {"best_val_accuracy": best_acc, "checkpoint": str(best_path)}
    save_json(result, out_dir / "train_summary.json")
    return result


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train TB-Trust")
    p.add_argument("--config", required=True, help="experiment yaml under configs/")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("overrides", nargs="*", help="key.subkey=value overrides")
    return p


def main():
    args = build_argparser().parse_args()
    cfg = load_experiment(args.config, config_dir=args.config_dir, overrides=args.overrides)
    print(train(cfg))


if __name__ == "__main__":
    main()
