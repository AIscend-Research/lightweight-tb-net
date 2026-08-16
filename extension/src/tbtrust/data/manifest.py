"""Build and inspect the dataset manifest.

The manifest is a single CSV that every other module reads. One row per image:

    path, clinic, label, split, degradation_severity, uncertainty_target

* clinic  - provenance source: montgomery | shenzhen | niaid | rsna | belarus
* label   - 0 = normal, 1 = tuberculosis
* split   - filled in later by splits.py (train/val/test under a LOCO config)
* degradation_severity - 0.0 for clean originals; set per-image when you
  pre-generate a degraded copy, or left 0 and applied on-the-fly by the Dataset.
* uncertainty_target   - weak supervision for the confidence head:
  high when the image is heavily degraded ("being unsure here is correct").

IMPORTANT PROVENANCE FACT (read before designing your LOCO splits)
------------------------------------------------------------------
The popular aggregated Kaggle TB set (tawsifurrahman/tuberculosis-tb-chest-xray-
dataset) is a *mixture* of sources with very skewed per-source class balance:

    montgomery : normal + TB   (both classes, ~138 imgs)   <- good holdout
    shenzhen   : normal + TB   (both classes, ~662 imgs)   <- good holdout
    niaid      : almost all TB positive
    rsna       : almost all normal (pneumonia-challenge normals)
    belarus    : all TB positive

So holding out NIAID or RSNA alone gives a single-class test set, on which
sensitivity OR specificity is undefined. Montgomery and Shenzhen are the clean
two-class cross-site holdouts. splits.py guards against single-class test folds
and warns you. Plan your leave-one-clinic-out rotation with this in mind.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLINICS = ["montgomery", "shenzhen", "niaid", "rsna", "belarus"]
COLUMNS = ["path", "clinic", "label", "split", "degradation_severity", "uncertainty_target"]

# Stable integer id per clinic, for the domain-adversarial head and the
# clinic-conditional FiLM embedding. Index len(CLINICS) is the catch-all for
# 'unknown' (infer_clinic_from_path did not match) and for any clinic added to a
# manifest but not to CLINICS, so a stray provenance string cannot index out of
# bounds mid-epoch. Hence NUM_CLINIC_SLOTS, not len(CLINICS), sizes the embedding.
CLINIC_INDEX = {name: i for i, name in enumerate(CLINICS)}
NUM_CLINIC_SLOTS = len(CLINICS) + 1


def clinic_index(clinic: str) -> int:
    """Map a clinic name to its embedding/domain-head index (unknown -> last slot)."""
    return CLINIC_INDEX.get(str(clinic).strip().lower(), len(CLINICS))


def uncertainty_target_from_severity(severity: float, floor: float = 0.05) -> float:
    """Map degradation severity -> a soft 'appropriate uncertainty' target in [0,1].

    Clean images get a small floor (never perfectly certain); heavily degraded
    images approach 1. This is the weak label the confidence head regresses to
    in Phase 3. Keep it monotonic and simple; tune the shape later.
    """
    return float(np.clip(floor + (1 - floor) * severity, 0.0, 1.0))


def infer_clinic_from_path(path: str) -> str:
    """Best-effort provenance from a filename/folder.

    Works for the raw NLM sets (MCUCXR_* = Montgomery, CHNCXR_* = Shenzhen) and
    for foldered dumps. Falls back to 'unknown' so you can audit what didn't match.
    Adjust the rules to match however you laid out data/raw.
    """
    p = path.lower()
    name = Path(p).name
    if name.startswith("mcucxr") or "montgomery" in p:
        return "montgomery"
    if name.startswith("chncxr") or "shenzhen" in p or "china" in p:
        return "shenzhen"
    if "niaid" in p or "tbportal" in p:
        return "niaid"
    if "rsna" in p:
        return "rsna"
    if "belarus" in p:
        return "belarus"
    return "unknown"


def build_manifest(rows: list[dict]) -> pd.DataFrame:
    """Assemble a manifest DataFrame from row dicts and fill derived columns."""
    df = pd.DataFrame(rows)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["degradation_severity"] = df["degradation_severity"].fillna(0.0)
    df["uncertainty_target"] = df["degradation_severity"].map(uncertainty_target_from_severity)
    df["split"] = df["split"].fillna("unassigned")
    return df[COLUMNS]


def scan_directory(root: str | Path, label_from_dir: dict[str, int] | None = None) -> pd.DataFrame:
    """Walk a directory of images into a manifest.

    Parameters
    ----------
    root : path with images somewhere under it (*.png/*.jpg/*.jpeg).
    label_from_dir : optional map from a parent-folder name to a label, e.g.
        {"Tuberculosis": 1, "Normal": 0}. If None, label is left as -1 for you
        to fill from the dataset's own metadata CSV.
    """
    root = Path(root)
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    rows = []
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() not in exts:
            continue
        label = -1
        if label_from_dir:
            for part in f.parts:
                if part in label_from_dir:
                    label = label_from_dir[part]
                    break
        rows.append(
            {
                "path": str(f),
                "clinic": infer_clinic_from_path(str(f)),
                "label": label,
                "degradation_severity": 0.0,
            }
        )
    return build_manifest(rows)


def class_balance_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-clinic normal/TB counts. Run this before choosing LOCO folds."""
    rep = (
        df.assign(is_tb=(df["label"] == 1).astype(int), is_normal=(df["label"] == 0).astype(int))
        .groupby("clinic")[["is_normal", "is_tb"]]
        .sum()
        .rename(columns={"is_normal": "normal", "is_tb": "tb"})
    )
    rep["total"] = rep["normal"] + rep["tb"]
    rep["tb_frac"] = (rep["tb"] / rep["total"].clip(lower=1)).round(3)
    return rep.sort_values("total", ascending=False)


def save(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
