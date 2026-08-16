"""Does the model's own uncertainty actually rise with image degradation?

(Ported from the `xctb` prototype's `eval/degradation_uncertainty.py`.)

The weak uncertainty target in `data/manifest.py` trains the confidence head to
say "being unsure here is correct" on heavily degraded images. This is the check
that it worked -- and, importantly, it applies to *any* uncertainty signal, not
just the trained head: MC-dropout spread, ensemble disagreement, or evidential
vacuity all get scored the same way, so it doubles as a comparison across the
methods in `eval/run.py`.

The "retake the photo" half of the deferral message depends on this specific
correlation. A model can have excellent AURC by being uncertain about *label*
ambiguity while being blind to *capture quality*; that model defers the right
images for the wrong reason, and telling its user to retake the photo would be
useless advice. Near-zero or negative rho means that story is not supported yet.

Torch-free (numpy only), like the rest of `eval/`.
"""

from __future__ import annotations

import numpy as np


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties get the mean rank), the input to Spearman's rho."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # break ties by averaging ranks within equal-value runs
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks


def spearman_correlation(a, b) -> float:
    """Spearman rank correlation in [-1, 1]. NaN if either input is constant
    or the two arrays have fewer than 2 points.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if len(a) < 2:
        return float("nan")
    ranks_a, ranks_b = _rank(a), _rank(b)
    ra, rb = ranks_a - ranks_a.mean(), ranks_b - ranks_b.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((ra * rb).sum() / denom)


def uncertainty_vs_severity(severity, uncertainty) -> dict:
    """Summarise the relationship between degradation severity (the `severity`
    the Dataset actually applied, or `uncertainty_target_from_severity` of it --
    the rank correlation is identical, that map is monotonic) and the model's
    predicted uncertainty for the same images.

    Also reports the mean uncertainty in the cleanest and most degraded thirds,
    because a positive rho driven entirely by a handful of extreme images is a
    weaker claim than a shifted distribution.
    """
    sev = np.asarray(severity, dtype=float).ravel()
    unc = np.asarray(uncertainty, dtype=float).ravel()
    rho = spearman_correlation(sev, unc)
    out = {
        "n": len(sev),
        "spearman_rho": round(rho, 4) if np.isfinite(rho) else float("nan"),
    }
    if len(sev) >= 3:
        order = np.argsort(sev, kind="mergesort")
        k = max(len(sev) // 3, 1)
        out["mean_uncertainty_cleanest_third"] = round(float(unc[order[:k]].mean()), 4)
        out["mean_uncertainty_most_degraded_third"] = round(float(unc[order[-k:]].mean()), 4)
    return out
