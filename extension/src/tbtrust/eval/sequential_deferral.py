"""Sequential/surveillance-style deferral (Phase 3 extension).

`models/uncertainty.py`'s `DeferralPolicy` is a point rule: fixed T MC-dropout
passes, then threshold once. This module borrows two ideas from sequential
analysis and epidemiological surveillance instead of a single point threshold:

1. `sequential_mc_dropout_decide` -- run MC-dropout passes one at a time and stop
   as soon as there's *enough accumulated evidence* to commit to report-or-defer,
   rather than always spending the same fixed compute budget (Wald, 1945,
   sequential probability ratio testing; the same "look repeatedly, stop early
   once the evidence is decisive" idea behind group-sequential monitoring in
   clinical trials). This is not a literal SPRT likelihood-ratio test -- it's a
   practical analogue, a running z-test of the MC-dropout mean probability
   against the 0.5 decision boundary -- but it inherits the useful property: easy
   cases stop in a handful of passes, hard/ambiguous cases spend the full budget
   and then correctly defer. On a device with no reliable power, that adaptive
   compute is a real advantage over always running a fixed T=20 passes.

2. `CUSUMMonitor` -- a classic one-sided CUSUM chart (Page, 1954), the same
   family of method public-health surveillance systems (e.g. CDC's EARS) use to
   detect a sustained shift in a monitored signal rather than react to single
   noisy readings. Applied here to a *stream* of per-image confidence/
   uncertainty scores from one clinic over time: it answers a different question
   than per-image deferral -- not "is this one photo trustworthy" but "has this
   clinic's capture quality drifted" (a camera degrading, a lighting change, a
   protocol slip), which single-image deferral can't see because it only ever
   looks at one image at a time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SequentialResult:
    decision: str            # "report" or "defer"
    prob: float               # running mean TB probability at the stopping point
    passes_used: int
    z_statistic: float
    standard_error: float


def sequential_mc_dropout_decide(
    model,
    x,
    passes_min: int = 3,
    passes_max: int = 20,
    z_threshold: float = 1.96,
    decision_boundary: float = 0.5,
) -> SequentialResult:
    """One image (`x` is a 1xCxHxW tensor). Take MC-dropout passes until the
    running mean's z-score against `decision_boundary` clears `z_threshold`
    (default 1.96 ~ two-sided alpha=0.05) or `passes_max` is reached.

    Stopping early on a confident case is the point -- most of the savings this
    is supposed to offer only materialize if `passes_min` is small (3-5) so easy
    cases really do exit early, not just occasionally.
    """
    import torch

    from ..models.uncertainty import enable_mc_dropout

    enable_mc_dropout(model)
    probs: list[float] = []
    z = 0.0
    se = float("inf")
    t = 0
    with torch.no_grad():
        for t in range(1, passes_max + 1):
            p = torch.sigmoid(model(x)["logit"]).item()
            probs.append(p)
            if t >= passes_min:
                mean = float(np.mean(probs))
                se = float(np.std(probs, ddof=1) / math.sqrt(t)) if t > 1 else float("inf")
                z = (mean - decision_boundary) / se if se > 0 else 0.0
                if abs(z) >= z_threshold:
                    break

    mean = float(np.mean(probs))
    decision = "report" if abs(z) >= z_threshold else "defer"
    return SequentialResult(decision=decision, prob=mean, passes_used=t, z_statistic=z, standard_error=se)


@dataclass
class CUSUMMonitor:
    """One-sided-both-ways CUSUM chart over a stream of scalar readings.

    Parameters mirror the standard Page (1954) formulation:
        target    - the in-control mean (e.g. mean predicted uncertainty measured
                    on a clean validation set for this clinic)
        slack     - "k", how big a shift to tolerate before it counts against the
                    sum (usually ~half the shift size you want to detect)
                    expressed in the same units as the readings
        threshold - "h", the decision interval; C exceeding this raises an alarm

    `update` feeds one new reading (e.g. this image's predicted uncertainty, or
    the clinic's running mean over a day's images) and returns whether an alarm
    fired. `state_high`/`state_low` track sustained upward/downward drift
    separately, since for this project only "capture quality got worse"
    (uncertainty trending up / confidence trending down) matters operationally,
    but both are computed so a downward-drift use case isn't precluded.
    """

    target: float
    slack: float
    threshold: float
    state_high: float = field(default=0.0, init=False)
    state_low: float = field(default=0.0, init=False)
    history: list[float] = field(default_factory=list, init=False)

    def update(self, value: float) -> dict[str, bool]:
        self.state_high = max(0.0, self.state_high + (value - self.target - self.slack))
        self.state_low = max(0.0, self.state_low + (self.target - self.slack - value))
        self.history.append(value)
        return {"alarm_high": self.state_high > self.threshold, "alarm_low": self.state_low > self.threshold}

    def reset(self) -> None:
        self.state_high = 0.0
        self.state_low = 0.0
        self.history.clear()
