"""Split-conformal prediction over the deferral decision (Phase 1 promise, Phase 4 result).

`docs/phase1_framing.md` picks conformal prediction as the *post-hoc coverage
guarantee* layered on top of whichever uncertainty score drives day-to-day
deferral -- not as the deployed uncertainty head. This module is that layer.

The construction is LAC / "least ambiguous set-valued classifier" (Sadinle,
Lei & Wasserman, 2019), the standard split-conformal recipe for classification:

  1. On a held-out calibration split, score each case by how much probability
     mass the model put on the *true* class:  s_i = 1 - p_{y_i}(x_i).
     A confidently-correct case scores near 0; a confidently-wrong one near 1.
  2. Take q_hat = the ceil((n+1)(1-alpha))/n empirical quantile of those scores.
     The (n+1) and the "round up" are what buy the finite-sample guarantee --
     the plain empirical quantile is anti-conservative and undercovers.
  3. At test time the prediction set is  C(x) = {k : p_k(x) >= 1 - q_hat}.

Guarantee: P(Y in C(X)) >= 1 - alpha, marginally over the draw of calibration
and test data. Distribution-free and finite-sample -- no assumption about the
model being well-specified, well-calibrated, or even any good.

**Mapping sets onto the deferral decision.** Binary TB screening has two labels,
so C(x) can only be one of four things:

    {TB}          -> report TB
    {normal}      -> report normal
    {TB, normal}  -> ambiguous: defer ("refer to specialist")
    {}            -> no label is plausible: defer ("retake the photo")

So "defer when |C(x)| != 1" *is* the deferral policy, and the conformal
guarantee becomes a statement about it: among cases where we commit to a single
label, we are wrong at most alpha of the time in the limit of the guarantee.

**The exchangeability caveat, which is the whole point under LOCO.** The
guarantee holds when calibration and test data are exchangeable. Under
leave-one-clinic-out they are emphatically *not*: the test clinic is a different
imaging site, which is the domain shift this project exists to measure. So the
honest claim is not "we have a 1-alpha guarantee on the held-out clinic." It is:

    the guarantee is exact on data exchangeable with calibration (same clinics),
    and the *shortfall* between guaranteed coverage (1-alpha) and coverage
    actually achieved on the held-out clinic is a direct, calibrated measurement
    of how far that clinic has drifted.

`coverage_gap_report` computes exactly that pair, which makes conformal a
cross-site diagnostic here rather than a rubber stamp. A large shortfall is a
finding, not a bug in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LABELS = (0, 1)  # normal, TB


@dataclass
class ConformalCalibration:
    """A fitted conformal threshold, plus the numbers needed to report it honestly."""

    q_hat: float                # nonconformity quantile
    alpha: float                # target miscoverage
    n_calibration: int
    guarantee_attainable: bool  # False when n is too small for the requested alpha

    @property
    def target_coverage(self) -> float:
        return 1.0 - self.alpha

    def prediction_sets(self, probs) -> np.ndarray:
        """(N, 2) boolean membership matrix; column k = "label k is in the set"."""
        return prediction_sets(probs, self.q_hat)

    def decide(self, probs) -> dict[str, np.ndarray]:
        return decide(probs, self.q_hat)


def _class_probs(probs) -> np.ndarray:
    """(N,) P(TB) -> (N, 2) columns [P(normal), P(TB)], matching LABELS order."""
    p = np.asarray(probs, dtype=float).ravel()
    return np.stack([1.0 - p, p], axis=1)


def nonconformity_scores(labels, probs) -> np.ndarray:
    """s_i = 1 - p_{true class}. Higher = the model was more wrong about this case."""
    y = np.asarray(labels).astype(int).ravel()
    pk = _class_probs(probs)
    if len(y) != len(pk):
        raise ValueError(f"labels ({len(y)}) and probs ({len(pk)}) length mismatch")
    if not np.isin(y, LABELS).all():
        raise ValueError(f"labels must be in {LABELS}, got {sorted(set(y.tolist()))}")
    return 1.0 - pk[np.arange(len(y)), y]


def calibrate(labels, probs, alpha: float = 0.1) -> ConformalCalibration:
    """Fit q_hat on a calibration split. Run this on validation, never on test."""
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    scores = nonconformity_scores(labels, probs)
    n = len(scores)
    if n == 0:
        raise ValueError("conformal calibration needs at least one calibration point")

    # Rank of the conservative finite-sample quantile. If it exceeds n, the
    # calibration set is too small to certify this alpha at all (you need
    # n >= 1/alpha - 1); fall back to the vacuous set rather than silently
    # reporting a guarantee that isn't there.
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return ConformalCalibration(q_hat=1.0, alpha=alpha, n_calibration=n, guarantee_attainable=False)

    q_hat = float(np.sort(scores)[rank - 1])
    return ConformalCalibration(q_hat=q_hat, alpha=alpha, n_calibration=n, guarantee_attainable=True)


def prediction_sets(probs, q_hat: float) -> np.ndarray:
    """C(x) = {k : p_k(x) >= 1 - q_hat}, as an (N, 2) boolean matrix."""
    return _class_probs(probs) >= (1.0 - q_hat)


def decide(probs, q_hat: float) -> dict[str, np.ndarray]:
    """Turn prediction sets into the report/defer decision.

    Returns
    -------
    report      : (N,) bool, True when the set is a singleton (commit to a label)
    prediction  : (N,) int, the committed label; meaningless where report is False
    set_size    : (N,) int in {0, 1, 2}
    defer_reason: (N,) object, "" | "ambiguous" (size 2) | "no_plausible_label" (size 0)
    """
    sets = prediction_sets(probs, q_hat)
    set_size = sets.sum(axis=1).astype(int)
    report = set_size == 1
    # argmax gives the single member where the set is a singleton; elsewhere the
    # value is unused (report is False there).
    prediction = np.argmax(sets, axis=1).astype(int)
    reason = np.full(len(set_size), "", dtype=object)
    reason[set_size == 2] = "ambiguous"
    reason[set_size == 0] = "no_plausible_label"
    return {
        "report": report,
        "prediction": prediction,
        "set_size": set_size,
        "defer_reason": reason,
    }


def evaluate(labels, probs, cal: ConformalCalibration) -> dict:
    """Measure what the conformal layer actually did on a set of predictions.

    `coverage` is the quantity the guarantee is about: the fraction of cases whose
    prediction set contained the true label. Everything else describes the price
    paid for it.
    """
    y = np.asarray(labels).astype(int).ravel()
    sets = prediction_sets(probs, cal.q_hat)
    if len(y) == 0:
        return {
            "n": 0, "coverage": float("nan"), "target_coverage": cal.target_coverage,
            "coverage_shortfall": float("nan"), "abstention_rate": float("nan"),
            "singleton_rate": float("nan"), "selective_accuracy": float("nan"),
            "mean_set_size": float("nan"), "ambiguous_rate": float("nan"),
            "empty_rate": float("nan"), "q_hat": cal.q_hat,
            "guarantee_attainable": cal.guarantee_attainable,
        }

    covered = sets[np.arange(len(y)), y]
    d = decide(probs, cal.q_hat)
    report, set_size = d["report"], d["set_size"]
    n_report = int(report.sum())
    coverage = float(covered.mean())

    return {
        "n": len(y),
        "coverage": coverage,
        "target_coverage": cal.target_coverage,
        # positive => we covered less than promised (the domain-shift signal)
        "coverage_shortfall": float(cal.target_coverage - coverage),
        "abstention_rate": float(1.0 - report.mean()),
        "singleton_rate": float(report.mean()),
        "selective_accuracy": (
            float(np.mean(d["prediction"][report] == y[report])) if n_report else float("nan")
        ),
        "mean_set_size": float(set_size.mean()),
        "ambiguous_rate": float(np.mean(set_size == 2)),
        "empty_rate": float(np.mean(set_size == 0)),
        "q_hat": cal.q_hat,
        "guarantee_attainable": cal.guarantee_attainable,
    }


def coverage_gap_report(
    val_labels, val_probs, test_labels, test_probs, alpha: float = 0.1, seed: int = 0
) -> dict:
    """Calibrate on validation, then contrast coverage in-distribution vs. held-out clinic.

    This is the cross-site diagnostic from the module docstring: coverage on data
    exchangeable with calibration should land at ~1-alpha, and however far the
    held-out clinic falls below that is a distribution-free readout of the domain
    shift, on the same scale as the guarantee.

    The validation set is split in half -- one half fits q_hat, the other measures
    coverage. Measuring on the same points that fit the quantile would be
    in-sample: coverage there is ~1-alpha *by construction* (it is what the
    quantile rank was chosen to make true), so it would demonstrate arithmetic
    rather than the guarantee. The held-out half is the honest exchangeable
    reference the shifted clinic gets compared against.
    """
    y_val = np.asarray(val_labels).astype(int).ravel()
    p_val = np.asarray(val_probs, dtype=float).ravel()
    n = len(y_val)
    if n < 4:
        raise ValueError(f"need at least 4 validation points to split calibration/measurement, got {n}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    half = n // 2
    cal_idx, ref_idx = perm[:half], perm[half:]

    cal = calibrate(y_val[cal_idx], p_val[cal_idx], alpha=alpha)
    in_dist_eval = evaluate(y_val[ref_idx], p_val[ref_idx], cal)
    test_eval = evaluate(test_labels, test_probs, cal)

    return {
        "alpha": alpha,
        "q_hat": cal.q_hat,
        "n_calibration": cal.n_calibration,
        "n_in_distribution_reference": len(ref_idx),
        "guarantee_attainable": cal.guarantee_attainable,
        "in_distribution_reference": in_dist_eval,
        "heldout_clinic": test_eval,
        # The headline: how much coverage the shift cost, beyond what alpha allows.
        "coverage_shortfall_vs_target": test_eval["coverage_shortfall"],
        "coverage_drop_in_dist_to_heldout": float(
            in_dist_eval["coverage"] - test_eval["coverage"]
        ),
    }
