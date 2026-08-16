"""Point metrics for binary TB classification.

Kept torch-free (numpy in, numbers out) so they work on any array of scores,
whether produced by the baseline, the MC-dropout ensemble, or a saved run.
Convention: label 1 = TB (the 'positive'), label 0 = normal.
"""

from __future__ import annotations

import numpy as np


def _binarize(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(probs) >= threshold).astype(int)


def confusion(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, int]:
    y = np.asarray(labels).astype(int)
    p = _binarize(probs, threshold)
    tp = int(np.sum((p == 1) & (y == 1)))
    tn = int(np.sum((p == 0) & (y == 0)))
    fp = int(np.sum((p == 1) & (y == 0)))
    fn = int(np.sum((p == 0) & (y == 1)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def sensitivity(labels, probs, threshold: float = 0.5) -> float:
    c = confusion(labels, probs, threshold)
    denom = c["tp"] + c["fn"]
    return c["tp"] / denom if denom else float("nan")


def specificity(labels, probs, threshold: float = 0.5) -> float:
    c = confusion(labels, probs, threshold)
    denom = c["tn"] + c["fp"]
    return c["tn"] / denom if denom else float("nan")


def accuracy(labels, probs, threshold: float = 0.5) -> float:
    y = np.asarray(labels).astype(int)
    p = _binarize(probs, threshold)
    return float(np.mean(p == y)) if len(y) else float("nan")


def brier_score(labels, probs) -> float:
    """Mean squared error between predicted TB probability and the 0/1 label.

    A proper score borrowed from forecast verification (Phase 4 meteorology
    extension). Lower is better; rewards calibrated, confident-when-right models.
    """
    y = np.asarray(labels).astype(float)
    p = np.asarray(probs, dtype=float)
    return float(np.mean((p - y) ** 2))


def summary(labels, probs, threshold: float = 0.5) -> dict[str, float]:
    return {
        "accuracy": accuracy(labels, probs, threshold),
        "sensitivity": sensitivity(labels, probs, threshold),
        "specificity": specificity(labels, probs, threshold),
        "brier": brier_score(labels, probs),
        "n": len(np.asarray(labels)),
    }
