"""Uncertainty estimation and the deferral decision.

Two independent sources of "how unsure am I", which the paper can compare:

1. MC-dropout  -- keep dropout on at inference, run T stochastic forward passes,
   use the spread of the TB probability as epistemic uncertainty. Cheap, works
   with the existing baseline, no retraining. Implemented here.

2. Learned uncertainty head -- the second head on TBClassifier trained on the
   weak degradation labels ("heavy degradation => high uncertainty is correct").
   Its output is read directly.

The other options are implemented in this repo too: deep ensembles in
`models/ensemble.py`, evidential deep learning in `models/evidential.py`, and
split-conformal coverage in `eval/conformal.py`. `eval/run.py` compares whichever
of them a given checkpoint can produce, head to head on one fixed set of
calibrated probabilities, and reports AURC per method. `torch-uncertainty` covers
the same ground and is installable as the optional `[uq]` extra for
cross-checking, but nothing here imports it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def enable_mc_dropout(model: nn.Module) -> None:
    """Put the model in eval mode but re-enable dropout layers (for MC sampling)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor, passes: int = 20):
    """Run `passes` stochastic forward passes.

    Returns
    -------
    mean_prob : (B,) averaged TB probability -- the point prediction
    std_prob  : (B,) std across passes       -- epistemic uncertainty
    """
    enable_mc_dropout(model)
    probs = []
    for _ in range(passes):
        logit = model(x)["logit"]
        probs.append(torch.sigmoid(logit))
    stacked = torch.stack(probs, dim=0)          # (passes, B)
    return stacked.mean(0), stacked.std(0)


@dataclass
class DeferralPolicy:
    """Decide report-vs-defer from an uncertainty signal.

    `mode` picks which signal drives the decision:
        "confidence" : defer when max(p, 1-p) < threshold
        "mc_std"     : defer when MC-dropout std > threshold
        "head"       : defer when the learned uncertainty head > threshold

    Tune `threshold` with eval.deferral.tune_threshold on the validation split.
    """

    mode: str = "confidence"
    threshold: float = 0.9

    def decide(self, *, prob=None, mc_std=None, head=None):
        """Return a boolean tensor/array: True = report, False = defer."""
        if self.mode == "confidence":
            if prob is None:
                raise ValueError("mode 'confidence' needs prob")
            conf = torch.maximum(prob, 1 - prob) if torch.is_tensor(prob) else None
            return conf >= self.threshold if conf is not None else (max(prob, 1 - prob) >= self.threshold)
        if self.mode == "mc_std":
            if mc_std is None:
                raise ValueError("mode 'mc_std' needs mc_std")
            return mc_std <= self.threshold
        if self.mode == "head":
            if head is None:
                raise ValueError("mode 'head' needs head")
            return head <= self.threshold
        raise ValueError(f"unknown deferral mode: {self.mode}")
