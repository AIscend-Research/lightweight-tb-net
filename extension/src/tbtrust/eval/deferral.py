"""Safe Deferral Rate and the risk-coverage curve (Phase 4 core metric 1).

The deferral policy: if the model's confidence is below a threshold T, don't
report a prediction to the clinician -- flag "retake photo / refer to specialist".
The question this module answers: how much accuracy (and sensitivity/specificity)
do you recover by deferring the least-confident cases, and where should T sit?

This is exactly selective classification. The risk-coverage curve is the formal
object; "Safe Deferral Rate" is its clinical framing.

The functions here are what the paper's numbers and figures are computed from --
there is no third-party metrics dependency behind them. `torch-uncertainty` ships
equivalent AURC / AUGRC / CovAt5%Risk metrics and is available as the optional
`[uq]` extra if you want to cross-check these implementations against a
maintained reference, but nothing in this repo imports it.

Every function takes an optional `confidence=` array. That is the hook the
uncertainty methods plug into: pass MC-dropout spread, the learned uncertainty
head, evidential vacuity or ensemble disagreement (via
`confidence_from_uncertainty`) and the deferral policy ranks on that instead of
on max(p, 1-p). `eval/run.py` uses it to compare all of them head to head on one
fixed set of calibrated probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import accuracy, sensitivity, specificity


@dataclass
class RiskCoveragePoint:
    threshold: float       # confidence cutoff; predict only if conf >= threshold
    coverage: float        # fraction of cases the model still answers
    accuracy: float        # accuracy on answered cases
    sensitivity: float
    specificity: float
    n_kept: int


def _confidence(probs: np.ndarray, confidence: np.ndarray | None = None) -> np.ndarray:
    """The score the deferral policy ranks on. Higher = more trustworthy.

    Defaults to the softmax-style confidence max(p, 1-p) read off the point
    prediction. Pass `confidence` explicitly to rank on a real uncertainty
    signal instead -- MC-dropout spread, the learned uncertainty head, evidential
    vacuity, or ensemble disagreement (see `confidence_from_uncertainty`). That
    substitution is the point of having uncertainty methods at all: max(p, 1-p)
    is only a proxy, and comparing methods head to head requires the deferral
    decision to actually consume the signal each method produces.
    """
    if confidence is not None:
        return np.asarray(confidence, dtype=float)
    p = np.asarray(probs, dtype=float)
    return np.maximum(p, 1 - p)


def confidence_from_uncertainty(uncertainty) -> np.ndarray:
    """Turn an uncertainty score (higher = worse) into a confidence (higher = better).

    Monotone decreasing, so it preserves the ranking the risk-coverage curve
    depends on, and min-max maps into [0, 1] so a tuned threshold stays
    interpretable across signals whose raw scales differ (MC-dropout std lives in
    [0, 0.5], evidential vacuity in [0, 1], the learned head in [0, 1]).

    Note the normalisation is fit on the array it is handed. Fit it on validation
    and reuse those bounds if you need a threshold that transfers across splits;
    `eval/run.py` does exactly that.
    """
    u = np.asarray(uncertainty, dtype=float)
    if u.size == 0:
        return u
    lo, hi = float(np.min(u)), float(np.max(u))
    if hi - lo < 1e-12:
        return np.full_like(u, 0.5)   # no discrimination available
    return 1.0 - (u - lo) / (hi - lo)


def _default_thresholds(conf: np.ndarray, n: int = 51) -> np.ndarray:
    """Threshold grid drawn from the score's own quantiles.

    A fixed linspace only samples coverage evenly if the score is uniform. Real
    confidence scores pile up near 1.0, so a fixed grid puts most of its points
    at coverage ~0 or ~1 and barely samples the middle of the risk-coverage
    curve -- which is the region AURC integrates over. Quantiles give an even
    sweep in *coverage* for any score scale.
    """
    if conf.size == 0:
        return np.linspace(0.0, 1.0, n)
    return np.unique(np.quantile(conf, np.linspace(0.0, 1.0, n)))


def risk_coverage_curve(
    labels,
    probs,
    thresholds: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
) -> list[RiskCoveragePoint]:
    """Sweep confidence thresholds; report coverage and quality on kept cases."""
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs, dtype=float)
    conf = _confidence(p, confidence)
    if thresholds is None:
        thresholds = _default_thresholds(conf)
    curve = []
    for t in thresholds:
        keep = conf >= t
        n = int(keep.sum())
        if n == 0:
            curve.append(RiskCoveragePoint(float(t), 0.0, float("nan"), float("nan"), float("nan"), 0))
            continue
        yk, pk = y[keep], p[keep]
        curve.append(
            RiskCoveragePoint(
                threshold=float(t),
                coverage=n / len(y),
                accuracy=accuracy(yk, pk),
                sensitivity=sensitivity(yk, pk),
                specificity=specificity(yk, pk),
                n_kept=n,
            )
        )
    return curve


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """np.trapz was removed in numpy 2.0 and renamed np.trapezoid; support both."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def area_under_risk_coverage(
    labels,
    probs,
    thresholds: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
) -> float:
    """AURC-style scalar: mean error over coverage (lower is better).

    Integrates (1 - accuracy) against coverage. A model whose confidence ranks
    its errors well will drop error quickly as it starts deferring -> low AURC.
    This is the scalar that actually separates the uncertainty methods, so it is
    the one to report per method in the paper's comparison table.
    """
    curve = [pt for pt in risk_coverage_curve(labels, probs, thresholds, confidence) if pt.n_kept > 0]
    if not curve:
        return float("nan")
    cov = np.array([pt.coverage for pt in curve])
    err = np.array([1 - pt.accuracy for pt in curve])
    order = np.argsort(cov)
    cov, err = cov[order], err[order]
    if len(cov) < 2:
        return float(err[0])
    return _trapezoid(err, cov)


def tune_threshold(
    labels,
    probs,
    target: str = "accuracy",
    target_value: float = 0.99,
    min_coverage: float = 0.5,
    confidence: np.ndarray | None = None,
) -> RiskCoveragePoint:
    """Pick the operating threshold T for the deferral policy.

    Strategy: among thresholds that still cover at least `min_coverage` of cases,
    choose the smallest T that reaches `target_value` on `target` (accuracy /
    sensitivity / specificity). If none reach the target, return the point with the
    best value at >= min_coverage. Tune this on validation, report on test.
    """
    curve = risk_coverage_curve(labels, probs, confidence=confidence)
    feasible = [pt for pt in curve if pt.coverage >= min_coverage and pt.n_kept > 0]
    if not feasible:
        feasible = [pt for pt in curve if pt.n_kept > 0]
    if not feasible:
        raise ValueError("no threshold keeps any case; check the confidence scores")
    hits = [pt for pt in feasible if getattr(pt, target) >= target_value]
    if hits:
        return min(hits, key=lambda pt: pt.threshold)  # least aggressive deferral that works
    return max(feasible, key=lambda pt: getattr(pt, target))


def human_rescue_rate(
    labels,
    probs,
    threshold: float,
    confidence: np.ndarray | None = None,
) -> dict[str, float]:
    """Phase 4 human-in-the-loop metric.

    Of the cases we defer (conf < threshold), what fraction would a clinician
    actually flip vs. were already correct? High 'would_correct' means deferral is
    catching genuine mistakes rather than wasting clinician time.
    """
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs, dtype=float)
    conf = _confidence(p, confidence)
    deferred = conf < threshold
    n_def = int(deferred.sum())
    if n_def == 0:
        return {"deferred": 0, "would_correct_frac": float("nan"), "already_correct_frac": float("nan")}
    pred = (p >= 0.5).astype(int)
    wrong_among_deferred = int(np.sum(pred[deferred] != y[deferred]))
    return {
        "deferred": n_def,
        "would_correct_frac": wrong_among_deferred / n_def,
        "already_correct_frac": 1 - wrong_among_deferred / n_def,
    }


def plot_risk_coverage(labels, probs, ax=None, label: str = "model", confidence: np.ndarray | None = None):
    import matplotlib.pyplot as plt

    curve = [pt for pt in risk_coverage_curve(labels, probs, confidence=confidence) if pt.n_kept > 0]
    cov = [pt.coverage for pt in curve]
    acc = [pt.accuracy for pt in curve]
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.plot(cov, acc, "o-", label=label)
    ax.set_xlabel("coverage (fraction answered)")
    ax.set_ylabel("accuracy on answered cases")
    ax.set_title("Safe deferral (risk-coverage)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax
