"""Calibration metrics and plots (Phase 4 core metric 2: honest confidence).

Everything here is numpy-only so you can point it at any (labels, probs) arrays.
These are the implementations the paper's calibration numbers and the reliability
diagram are built from -- there is no third-party metrics dependency behind them.
`torch-uncertainty` offers richer variants (SmoothECE, adaptive binning) and is
available as the optional `[uq]` extra for cross-checking, but is not imported.

Definitions
-----------
confidence(x) = max(p, 1-p)          # how sure the model is, either way
correct(x)    = prediction matches label
ECE = sum_bins (n_b / N) * | acc_b - conf_b |     (expected calibration error)
MCE = max_bins | acc_b - conf_b |                  (worst-bin error)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReliabilityBins:
    edges: np.ndarray          # bin boundaries, length n_bins+1
    conf: np.ndarray           # mean confidence per bin
    acc: np.ndarray            # accuracy per bin
    count: np.ndarray          # samples per bin


def _confidence_and_correct(labels: np.ndarray, probs: np.ndarray):
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs, dtype=float)
    pred = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1 - p)
    correct = (pred == y).astype(float)
    return conf, correct


def reliability_bins(labels, probs, n_bins: int = 15) -> ReliabilityBins:
    conf, correct = _confidence_and_correct(labels, probs)
    edges = np.linspace(0.5, 1.0, n_bins + 1)  # confidence for binary preds lives in [0.5, 1]
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    conf_b = np.full(n_bins, np.nan)
    acc_b = np.full(n_bins, np.nan)
    count_b = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = idx == b
        count_b[b] = int(m.sum())
        if count_b[b]:
            conf_b[b] = conf[m].mean()
            acc_b[b] = correct[m].mean()
    return ReliabilityBins(edges, conf_b, acc_b, count_b)


def expected_calibration_error(labels, probs, n_bins: int = 15) -> float:
    b = reliability_bins(labels, probs, n_bins)
    n = b.count.sum()
    if n == 0:
        return float("nan")
    gaps = np.abs(b.acc - b.conf)
    weights = b.count / n
    valid = ~np.isnan(gaps)
    return float(np.sum(weights[valid] * gaps[valid]))


def max_calibration_error(labels, probs, n_bins: int = 15) -> float:
    b = reliability_bins(labels, probs, n_bins)
    gaps = np.abs(b.acc - b.conf)
    gaps = gaps[~np.isnan(gaps)]
    return float(np.max(gaps)) if len(gaps) else float("nan")


def temperature_nll(logits: np.ndarray, labels: np.ndarray, T: float) -> float:
    """Mean binary NLL of sigmoid(logit / T), computed without overflow.

    Using logaddexp instead of forming sigmoid() and taking its log keeps this
    finite for large |z/T|, which is exactly the regime a temperature search
    walks into when it probes small T.
    """
    z = np.asarray(logits, dtype=float)
    y = np.asarray(labels, dtype=float)
    signed = (2.0 * y - 1.0) * (z / T)          # +z/T when y=1, -z/T when y=0
    return float(np.mean(np.logaddexp(0.0, -signed)))


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    bounds: tuple[float, float] = (0.05, 20.0),
    iters: int = 100,
) -> float:
    """Fit a single temperature T that rescales binary logits to fix over/under-confidence.

    Minimizes NLL of sigmoid(logit / T) over `bounds` by golden-section search.
    `logits` are the raw pre-sigmoid scores for the TB class. Returns T; apply as
    `sigmoid(logit / T)` (see `apply_temperature`). Fit on the validation split only.

    Why a bounded search rather than gradient descent: the NLL is unimodal in
    log T, so golden-section converges without a learning rate, a convergence
    check, or any risk of stepping into the overflow region. The previous
    fixed-step gradient implementation had its gradient sign inverted, so it
    ascended the NLL -- it drove T toward 0 (returning NaN once exp overflowed)
    on overconfident logits, which is precisely the case temperature scaling
    exists to fix. `test_fit_temperature_*` pins the corrected behaviour against
    a brute-force grid optimum so it cannot regress silently.
    """
    lo, hi = float(bounds[0]), float(bounds[1])
    if not 0 < lo < hi:
        raise ValueError(f"bounds must satisfy 0 < lo < hi, got {bounds}")
    labels = np.asarray(labels, dtype=float)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        # Single-class (or empty) validation data carries no calibration signal;
        # the NLL is monotone in T and the search would just return a bound.
        return 1.0

    # Search in log space: the NLL is symmetric-ish in log T, not in T.
    log_lo, log_hi = np.log(lo), np.log(hi)
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    c = log_hi - inv_phi * (log_hi - log_lo)
    d = log_lo + inv_phi * (log_hi - log_lo)
    fc = temperature_nll(logits, labels, float(np.exp(c)))
    fd = temperature_nll(logits, labels, float(np.exp(d)))
    for _ in range(iters):
        if fc < fd:
            log_hi, d, fd = d, c, fc
            c = log_hi - inv_phi * (log_hi - log_lo)
            fc = temperature_nll(logits, labels, float(np.exp(c)))
        else:
            log_lo, c, fc = c, d, fd
            d = log_lo + inv_phi * (log_hi - log_lo)
            fd = temperature_nll(logits, labels, float(np.exp(d)))
        if log_hi - log_lo < 1e-8:
            break
    return float(np.exp((log_lo + log_hi) / 2.0))


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Calibrated P(TB) = sigmoid(logit / T). Overflow-safe."""
    z = np.asarray(logits, dtype=float) / float(T)
    # sigmoid via tanh is stable at both tails, unlike 1/(1+exp(-z)).
    return 0.5 * (1.0 + np.tanh(z / 2.0))


def plot_reliability(labels, probs, n_bins: int = 15, ax=None, title: str = "Reliability"):
    """Draw a reliability diagram. Import matplotlib lazily so the module stays light."""
    import matplotlib.pyplot as plt

    b = reliability_bins(labels, probs, n_bins)
    centers = (b.edges[:-1] + b.edges[1:]) / 2
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0.5, 1], [0.5, 1], "--", color="gray", label="perfect")
    ax.bar(centers, np.nan_to_num(b.acc), width=(b.edges[1] - b.edges[0]) * 0.9,
           alpha=0.7, edgecolor="black", label="accuracy")
    ax.plot(centers, b.conf, "o-", color="crimson", label="confidence")
    ece = expected_calibration_error(labels, probs, n_bins)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{title}  (ECE={ece:.3f})")
    ax.set_xlim(0.5, 1)
    ax.set_ylim(0.5, 1)
    ax.legend(loc="lower right", fontsize=8)
    return ax
