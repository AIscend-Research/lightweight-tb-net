"""Cross-site generalization analysis (Phase 4).

Two questions point-accuracy alone can't answer:
  1. Which held-out clinic is hardest? (generalization gap per LOCO fold)
  2. Does the model calibrate well on clinic A but stay overconfident on clinic B?
     (hidden domain shift that ECE-on-the-pool would hide)

Feed this a long-form results table with one row per test image and columns:
    clinic, label, prob   (prob = predicted TB probability)
plus the in-distribution reference accuracy to compute the gap against.
"""

from __future__ import annotations

import pandas as pd

from .calibration import expected_calibration_error, max_calibration_error
from .metrics import accuracy, sensitivity, specificity


def per_clinic_table(results: pd.DataFrame) -> pd.DataFrame:
    """results: columns [clinic, label, prob]. One row per clinic with metrics."""
    rows = []
    for clinic, grp in results.groupby("clinic"):
        y, p = grp["label"].to_numpy(), grp["prob"].to_numpy()
        rows.append(
            {
                "clinic": clinic,
                "n": len(grp),
                "accuracy": accuracy(y, p),
                "sensitivity": sensitivity(y, p),
                "specificity": specificity(y, p),
                "ece": expected_calibration_error(y, p),
                "mce": max_calibration_error(y, p),
            }
        )
    return pd.DataFrame(rows).set_index("clinic")


def generalization_gap(per_clinic: pd.DataFrame, reference_accuracy: float) -> pd.DataFrame:
    """How far each held-out clinic falls below the in-distribution reference."""
    out = per_clinic.copy()
    out["gap_vs_reference"] = (reference_accuracy - out["accuracy"]).round(4)
    return out.sort_values("gap_vs_reference", ascending=False)


def calibration_heatmap_data(loco_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build an ECE matrix: rows = held-out clinic (the LOCO fold), value = ECE.

    loco_results maps holdout_clinic -> its per-image results DataFrame.
    Reveals whether calibration is uneven across sites.
    """
    rows = {}
    for holdout, res in loco_results.items():
        y, p = res["label"].to_numpy(), res["prob"].to_numpy()
        rows[holdout] = {
            "ece": expected_calibration_error(y, p),
            "accuracy": accuracy(y, p),
        }
    return pd.DataFrame(rows).T


def plot_gap_bars(gap_table: pd.DataFrame, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    gap_table = gap_table.sort_values("gap_vs_reference")
    ax.barh(gap_table.index, gap_table["gap_vs_reference"], color="steelblue", edgecolor="black")
    ax.set_xlabel("accuracy gap vs. in-distribution reference")
    ax.set_title("Cross-site generalization gap by held-out clinic")
    ax.grid(alpha=0.3, axis="x")
    return ax
