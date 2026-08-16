"""Gradient Reversal Layer (Ganin et al., 2016).

Forward pass is the identity. Backward pass multiplies the gradient by -lambda.
That single sign flip is what makes the feature extractor fight the domain
classifier: the domain head tries to guess which clinic a feature came from, and
the reversed gradient pushes the features to make that guess impossible. What is
left is a representation that does not carry the machine's fingerprint -- which
is exactly what a leave-one-clinic-out holdout punishes a model for having.
"""

from __future__ import annotations

import math

import torch
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


def dann_lambda(step: int, total_steps: int, gamma: float = 10.0) -> float:
    """Schedule that ramps lambda 0 -> 1 over training (from the DANN paper).

    Starting the adversary at full strength destabilises early training, so it
    is eased in as p = step / total_steps grows.
    """
    p = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return float(2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)
