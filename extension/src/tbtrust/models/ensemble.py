"""Deep ensembles (Lakshminarayanan, Pritzel & Blundell, 2017) -- the Phase 3
uncertainty-method comparison's calibration upper bound, not the deployed model.

N independently-initialized, independently-trained networks; average their
predicted probabilities for the point estimate, use the spread across members as
epistemic uncertainty. Usually the best-calibrated of the cheap options, at the
cost of N x inference (and N x storage), which is exactly why it isn't the
featured method for a device meant to run in a clinic with no reliable power --
see `docs/phase1_framing.md` section 2. Its job here is to be the reference the
cheap methods (MC-dropout, evidential) are judged against: if MC-dropout or the
evidential head calibrates nearly as well as a 5-member ensemble, that's the
result that justifies shipping the cheap one.

Deliberately does not reimplement training: `train_deep_ensemble` calls
`train.loop.train` once per member with a different seed, so a member is defined
as "the exact same recipe as the single-model baseline, minus the random seed" --
otherwise a calibration comparison against MC-dropout/evidential would be
comparing methods *and* training recipes at once, confounding the result.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn


def _build_member(cfg: dict) -> nn.Module:
    """One member, matching train/loop.py's arch dispatch exactly."""
    from .baseline import build_model
    from .evidential import build_evidential_model
    from .tbnet import TBNet

    arch = cfg["model"].get("arch", "baseline")
    if arch == "tbnet":
        m = cfg["model"]
        return TBNet(dropout=m.get("dropout", 0.3), with_uncertainty_head=m.get("with_uncertainty_head", True))
    if arch == "evidential":
        return build_evidential_model(cfg)
    return build_model(cfg)


@dataclass
class DeepEnsemble:
    members: list[nn.Module]
    device: str = "cpu"

    def __post_init__(self):
        self.members = [m.to(self.device).eval() for m in self.members]

    @property
    def n_members(self) -> int:
        return len(self.members)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Same (mean, std) signature as `models.uncertainty.mc_dropout_predict`,
        so the two are drop-in comparable in eval code."""
        probs = torch.stack([torch.sigmoid(m(x)["logit"]) for m in self.members], dim=0)  # (N, B)
        return probs.mean(0), probs.std(0)

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.members):
            torch.save(m.state_dict(), out_dir / f"member_{i}.ckpt")

    @classmethod
    def load(cls, cfg: dict, checkpoint_paths: list[str], device: str = "cpu") -> DeepEnsemble:
        members = []
        for path in checkpoint_paths:
            m = _build_member(cfg)
            ckpt = torch.load(path, map_location=device)
            m.load_state_dict(ckpt.get("state_dict", ckpt))
            members.append(m)
        return cls(members=members, device=device)


def train_deep_ensemble(cfg: dict, n_members: int = 5, base_seed: int = 0) -> list[str]:
    """Train `n_members` independent models; return their checkpoint paths.

    Each member's config is the input `cfg` with only `seed` and `train.output_dir`
    changed, so architecture, data, and hyperparameters are identical across
    members -- the only source of diversity is initialization + data shuffling
    order, which is the point (that diversity is what makes the ensemble spread a
    meaningful epistemic-uncertainty signal rather than noise).
    """
    from ..train.loop import (
        train,  # local import: train/ imports models/, avoid a cycle at module load
    )

    base_out = Path(cfg["train"].get("output_dir", "outputs")) / "ensemble" / cfg["data"]["holdout_clinic"]
    checkpoints = []
    for i in range(n_members):
        member_cfg = copy.deepcopy(cfg)
        member_cfg["seed"] = base_seed + i
        member_cfg["train"]["output_dir"] = str(base_out / f"member_{i}")
        result = train(member_cfg)
        checkpoints.append(result["checkpoint"])
    return checkpoints


def evaluate_ensemble(ensemble: DeepEnsemble, loader, device: str = "cpu") -> dict:
    """Point metrics (via mean prediction) + mean ensemble std, for comparing
    against the single-model + MC-dropout numbers on the same held-out set."""
    import numpy as np

    from ..eval.metrics import summary

    ys, means, stds = [], [], []
    for batch in loader:
        x = batch["image"].to(device)
        mean, std = ensemble.predict(x)
        means.append(mean.cpu().numpy())
        stds.append(std.cpu().numpy())
        ys.append(batch["label"].numpy())
    y, p, s = np.concatenate(ys), np.concatenate(means), np.concatenate(stds)
    out = summary(y, p)
    out["mean_ensemble_std"] = float(s.mean())
    return out
