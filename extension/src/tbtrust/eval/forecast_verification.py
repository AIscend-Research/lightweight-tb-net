"""Forecast-verification metrics (Phase 4 meteorology extension).

`eval/metrics.brier_score` already covers the headline proper score. This
module adds the machinery meteorologists use to *diagnose* a Brier score rather
than just report it, borrowed wholesale from weather forecast verification
(Murphy, 1973; Wilks, "Statistical Methods in the Atmospheric Sciences") since
binary TB screening is mathematically the same object as binary weather-event
forecasting: a probability forecast, verified against a binary outcome.

Deliberately not implemented: the Ranked Probability Score. RPS generalizes
Brier score to ordinal multi-category forecasts; TB screening only has two
categories (normal/TB), so RPS collapses to exactly the Brier score already in
`metrics.py`. Adding a same-number-different-name function would misrepresent
this as more rigorous than it is -- the honest version of "add RPS" for a
binary problem is "note that it's redundant here," not reimplement it.
"""

from __future__ import annotations

import numpy as np

from .metrics import brier_score


def brier_skill_score(labels, probs) -> float:
    """1 - BS / BS_ref, against the 'climatological' reference forecast that
    always predicts the base rate (TB prevalence in this sample).

    >0 : better than always guessing the prevalence.
    <=0: worse -- a real risk on an imbalanced held-out clinic (e.g. a
         low-TB-prevalence fold), where a lazily-miscalibrated model can still
         post a deceptively low raw Brier score just because the base rate
         itself is low. This is the reason to report BSS alongside Brier, not
         instead of it.
    """
    y = np.asarray(labels, dtype=float)
    bs = brier_score(y, probs)
    base_rate = y.mean()
    bs_ref = brier_score(y, np.full_like(y, base_rate))
    if bs_ref == 0:
        return float("nan")
    return float(1.0 - bs / bs_ref)


def murphy_decomposition(labels, probs, n_bins: int = 10) -> dict:
    """Murphy (1973) three-term decomposition: BS = reliability - resolution + uncertainty.

        reliability : mean_bins (n_b/N) * (forecast_b - observed_freq_b)^2
                      lower is better -- penalizes bins where the forecast
                      probability doesn't match the observed TB frequency in
                      that bin (the same thing ECE measures, but signed per bin
                      and in probability space rather than confidence space).
        resolution  : mean_bins (n_b/N) * (observed_freq_b - base_rate)^2
                      higher is better -- rewards the forecast for usefully
                      separating cases away from the base rate. A model that
                      always predicts the base rate has resolution = 0 (and
                      reliability = 0, i.e. perfectly "calibrated" and useless
                      -- this is exactly why sharpness has to be reported too,
                      see `sharpness` below).
        uncertainty : base_rate * (1 - base_rate), irreducible, a property of
                      the *event* (how balanced this clinic's TB prevalence
                      is), not of the model.

    Binned over the full forecast range [0, 1] (not eval.calibration's
    confidence range [0.5, 1]), since the decomposition needs the raw forecast
    probability, not "how sure either way."
    """
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probs, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    base_rate = y.mean()
    n = len(y)

    reliability, resolution = 0.0, 0.0
    for b in range(n_bins):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            continue
        forecast_b = p[m].mean()
        observed_freq_b = y[m].mean()
        reliability += (nb / n) * (forecast_b - observed_freq_b) ** 2
        resolution += (nb / n) * (observed_freq_b - base_rate) ** 2

    uncertainty = base_rate * (1 - base_rate)
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier_reconstructed": float(reliability - resolution + uncertainty),
        "brier_actual": float(brier_score(y, p)),
    }


def sharpness(probs, n_bins: int = 10) -> dict:
    """Histogram spread of forecast probabilities (a 'sharpness diagram').

    A model that only ever predicts near the base rate is well-calibrated by
    construction (reliability=0 above) but useless. Forecast verification
    reports sharpness alongside calibration for exactly this reason, and it
    matters here specifically: the deferral policy only has something to work
    with if "confident" is sometimes true, not always the base rate hedged.
    """
    p = np.asarray(probs, dtype=float)
    counts, edges = np.histogram(p, bins=n_bins, range=(0, 1))
    return {"edges": edges.tolist(), "counts": counts.tolist(), "std": float(p.std()), "mean": float(p.mean())}
