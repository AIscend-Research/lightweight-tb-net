"""Small IO helpers for checkpoints and metric logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def save_checkpoint(model, path: str | Path, **extra) -> None:
    """Save torch model weights plus any extra metadata (config, epoch, metrics)."""
    import torch

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(model, path: str | Path, map_location: str = "cpu"):
    """Load weights into `model` in place; return the checkpoint dict.

    A common way to trip this up: reusing the same `train.output_dir` +
    `data.holdout_clinic` (so the same `<output_dir>/<clinic>/best.ckpt` path)
    across two configs with different `model.arch`. The second training run
    silently overwrites the first's checkpoint, and the failure only surfaces
    later, at eval time, as a confusing "missing/unexpected key" error from
    `load_state_dict` with no indication of *why* the shapes don't match. Since
    `train/loop.py` already saves the training config alongside the weights,
    surface that here instead of leaving it to the raw torch error.
    """
    import torch

    ckpt = torch.load(path, map_location=map_location)
    try:
        model.load_state_dict(ckpt["state_dict"])
    except RuntimeError as e:
        saved_arch = ckpt.get("config", {}).get("model", {}).get("arch", "unknown")
        raise RuntimeError(
            f"Couldn't load checkpoint '{path}' (trained with model.arch='{saved_arch}') into a "
            f"differently-shaped model. Check that the --config you're passing now has the same "
            f"model.arch as the one used to train this checkpoint, or that train.output_dir isn't "
            f"shared across two configs with different archs for the same holdout clinic.\n{e}"
        ) from e
    return ckpt
