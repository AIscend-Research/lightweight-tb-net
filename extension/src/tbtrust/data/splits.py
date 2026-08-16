"""Leave-one-clinic-out (LOCO) splits.

Hold one clinic out entirely as the test set; train+validate on the others.
Rotating the held-out clinic measures cross-site domain shift directly, which is
the whole point of the multi-site evaluation.

The single-class guard is the important bit: because some sources are one-class
(see manifest.py), a naive holdout can produce a test set with only normals or
only TB, making sensitivity/specificity undefined. `leave_one_clinic_out` flags
that and, by default, refuses to build such a fold.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def leave_one_clinic_out(
    df: pd.DataFrame,
    holdout_clinic: str,
    val_frac: float = 0.15,
    seed: int = 0,
    require_two_class_test: bool = True,
) -> pd.DataFrame:
    """Return a copy of `df` with the `split` column set to train/val/test.

    test = every row whose clinic == holdout_clinic
    the remaining clinics are split into train/val, stratified by (clinic, label)
    so val mirrors train's composition.
    """
    if holdout_clinic not in set(df["clinic"]):
        raise ValueError(f"clinic '{holdout_clinic}' not in manifest")

    out = df.copy()
    out["split"] = "train"
    is_test = out["clinic"] == holdout_clinic
    out.loc[is_test, "split"] = "test"

    test = out[is_test]
    n_classes = test["label"].nunique()
    if n_classes < 2:
        present = sorted(test["label"].unique().tolist())
        msg = (
            f"LOCO test fold '{holdout_clinic}' has only class(es) {present}. "
            "Sensitivity or specificity will be undefined on it. "
            "Montgomery and Shenzhen are the two-class holdouts; NIAID/Belarus are "
            "TB-only and RSNA is normal-only."
        )
        if require_two_class_test:
            raise ValueError(msg + " Pass require_two_class_test=False to allow anyway.")
        warnings.warn(msg, stacklevel=2)

    # stratified train/val over the non-held-out rows
    rng = np.random.default_rng(seed)
    train_pool = out[~is_test]
    val_idx = []
    for _, grp in train_pool.groupby(["clinic", "label"]):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n_val = round(len(idx) * val_frac)
        val_idx.extend(idx[:n_val].tolist())
    out.loc[val_idx, "split"] = "val"
    return out


def random_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> pd.DataFrame:
    """Cohort-blind stratified train/val/test split -- the in-distribution reference.

    This is the "best case" number the cross-site gap is measured *against*: the
    test set is drawn from the same clinics as training, so it answers "how well
    does this model do when deployment looks like development."

    It has to be a real held-out test split. Reporting the generalization gap
    against a *validation* accuracy instead would use the score the checkpoint
    was selected on -- a maximum over epochs, biased upward -- which inflates the
    gap by exactly that selection bias and makes cross-site degradation look
    worse than it is. Stratified by (clinic, label) so both stay balanced.
    """
    if not 0 <= val_frac + test_frac < 1:
        raise ValueError(f"val_frac + test_frac must be in [0, 1), got {val_frac + test_frac}")
    out = df.copy()
    out["split"] = "train"
    rng = np.random.default_rng(seed)
    for _, grp in out.groupby(["clinic", "label"]):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_test = round(n * test_frac)
        n_val = round(n * val_frac)
        out.loc[idx[:n_test], "split"] = "test"
        out.loc[idx[n_test:n_test + n_val], "split"] = "val"
    return out


def split_from_config(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Build the split from an experiment config, honouring `data.split_mode`.

    `split_mode: loco` (the default) holds one clinic out entirely -- the
    deployment number. `split_mode: random` ignores clinic and splits at random --
    the in-distribution reference number to compare it against.
    """
    mode = str(cfg["data"].get("split_mode", "loco")).lower()
    if mode == "random":
        return random_split(
            df,
            val_frac=cfg["data"].get("val_frac", 0.15),
            test_frac=cfg["data"].get("test_frac", 0.15),
            seed=cfg.get("seed", 0),
        )
    if mode == "loco":
        return loco_split_from_config(df, cfg)
    raise ValueError(f"unknown data.split_mode {mode!r}; expected 'loco' or 'random'")


def loco_split_from_config(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Build the LOCO split from an experiment config. Every caller should use this.

    Training and evaluation must derive *identical* splits, or the "val" split at
    eval time contains images the model was trained on -- and val is what fits the
    temperature, the uncertainty->confidence map, the deferral threshold and the
    conformal quantile. Fitting any of those on training data makes the model look
    better calibrated, and more confidently-right, than it is.

    That was not hypothetical: `train/loop.py` passed `cfg.data.val_frac` and
    `cfg.seed`, while `eval/run.py` and `scripts/run_experiments.py` called
    `leave_one_clinic_out` with neither, silently taking the defaults (0.15 / 0).
    Any config that set either key -- or any non-zero seed -- produced two
    different splits. Routing every caller through one helper removes the
    opportunity to forget an argument.
    """
    return leave_one_clinic_out(
        df,
        holdout_clinic=cfg["data"]["holdout_clinic"],
        val_frac=cfg["data"].get("val_frac", 0.15),
        seed=cfg.get("seed", 0),
        require_two_class_test=cfg["data"].get("require_two_class_test", True),
    )


def all_loco_folds(
    df: pd.DataFrame,
    clinics: list[str] | None = None,
    two_class_only: bool = True,
) -> list[str]:
    """List the clinics worth rotating as holdouts.

    With `two_class_only`, returns only clinics whose data has both classes
    (the ones you can compute a full confusion matrix on).
    """
    clinics = clinics or sorted(df["clinic"].unique().tolist())
    keep = []
    for c in clinics:
        sub = df[df["clinic"] == c]
        if two_class_only and sub["label"].nunique() < 2:
            continue
        keep.append(c)
    return keep


def summarize_split(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of normal/TB per split, for a quick sanity check."""
    g = (
        df.assign(normal=(df.label == 0).astype(int), tb=(df.label == 1).astype(int))
        .groupby("split")[["normal", "tb"]]
        .sum()
    )
    g["total"] = g.sum(axis=1)
    return g
