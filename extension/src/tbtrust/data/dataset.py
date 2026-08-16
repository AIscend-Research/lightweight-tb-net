"""Torch Dataset over the manifest, with on-the-fly smartphone degradation.

Returns, per item:
    image             : float tensor CxHxW, normalized
    label             : long (0 normal / 1 TB)
    uncertainty_target: float in [0,1]  (weak supervision for the confidence head)
    severity          : float, the degradation strength actually applied
    clinic_idx        : long, provenance id (domain label for CORAL/DANN/IRM/FiLM)

Set `degradation_severity` to a fixed value for a controlled robustness sweep
(Phase 4: accuracy vs. severity), or pass `severity_sampler` to randomize during
training so the model sees a range of image qualities.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

from .degradation import SmartphoneDegradation
from .manifest import clinic_index, uncertainty_target_from_severity

try:  # torch is optional at import time so the rest of the package stays light
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object  # type: ignore


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TBDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str | None = None,
        image_size: int = 224,
        degradation_severity: float = 0.0,
        severity_sampler: Callable[[], float] | None = None,
        grayscale_to_rgb: bool = True,
        normalize: bool = True,
        seed: int | None = None,
    ):
        if torch is None:
            raise ImportError("TBDataset needs torch. `pip install -e .`")
        self.df = manifest if split is None else manifest[manifest["split"] == split].reset_index(drop=True)
        self.image_size = image_size
        self.degradation_severity = degradation_severity
        self.severity_sampler = severity_sampler
        self.grayscale_to_rgb = grayscale_to_rgb
        self.normalize = normalize
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.df)

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation stream for a new training epoch.

        With `seed` set, the per-item RNG is derived from (seed, epoch, index).
        Without the epoch term every image would get the identical degradation in
        every epoch, which defeats the point of randomised augmentation; without
        the seed the stream comes from OS entropy and the run is unreproducible.
        Call this once per epoch (the training loop does) and leave it alone at
        eval time, where a fixed stream is exactly what you want.
        """
        self.epoch = int(epoch)

    def _item_seed(self, i: int) -> int | None:
        """Reproducible per-(epoch, item) seed, or None to stay fully random.

        `SmartphoneDegradation` constructs a fresh Generator(seed) on every call,
        so handing it one dataset-wide seed would give every image an identical
        draw. Mixing in the index (and the epoch) keeps images different from each
        other and from themselves across epochs, while staying reproducible.
        """
        if self.seed is None:
            return None
        return (self.seed * 1_000_003 + self.epoch * 10_007 + i) % (2**32)

    def _load(self, path: str) -> np.ndarray:
        img = Image.open(path).convert("L").resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.asarray(img)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        arr = self._load(row["path"])

        item_seed = self._item_seed(i)
        if self.severity_sampler is not None:
            # Drive the sampler from the same per-item stream when seeded, so the
            # severity does not depend on the order the DataLoader happens to
            # fetch items in -- which varies with shuffling and worker count, and
            # would otherwise make a seeded run still irreproducible.
            severity = (
                self.severity_sampler(np.random.default_rng(item_seed))
                if item_seed is not None
                else self.severity_sampler()
            )
        else:
            severity = self.degradation_severity

        if severity > 0:
            arr, _ = SmartphoneDegradation(severity=severity, seed=item_seed)(arr)

        arr = arr.astype(np.float32) / 255.0
        if self.grayscale_to_rgb:
            arr = np.stack([arr, arr, arr], axis=0)          # 3xHxW
        else:
            arr = arr[None, ...]                             # 1xHxW
        tensor = torch.from_numpy(arr)

        if self.normalize and self.grayscale_to_rgb:
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            tensor = (tensor - mean) / std

        u_target = uncertainty_target_from_severity(severity)
        return {
            "image": tensor,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "uncertainty_target": torch.tensor(u_target, dtype=torch.float32),
            "severity": torch.tensor(float(severity), dtype=torch.float32),
            # Domain label for the DG objectives (CORAL/DANN/IRM) and the
            # clinic-conditional FiLM embedding. Always emitted -- it costs one
            # int per image and means switching dg.method needs no data-side
            # change. Ignored entirely when dg.method is 'none'.
            "clinic_idx": torch.tensor(clinic_index(row["clinic"]), dtype=torch.long),
        }


@dataclass
class constant_severity:
    """Callable, not a closure/lambda: DataLoader with num_workers > 0 pickles the
    severity_sampler to hand it to worker processes. A `lambda: value` closure
    fails that pickling under the spawn/forkserver start method (macOS/Windows
    always, and POSIX too as of Python 3.14) -- it looks fine in local testing
    on Linux only because fork doesn't need to pickle anything, then breaks the
    first time someone trains on a Mac or a newer Python. A dataclass with
    __call__ pickles fine everywhere."""

    value: float

    def __call__(self) -> float:
        return self.value


@dataclass
class uniform_severity:
    """Same picklability fix as `constant_severity`, for the randomized-severity
    training sampler. Owns its own `np.random.Generator` instead of closing over
    one, for the same reason.

    Accepts an optional `rng`: when `TBDataset` is seeded it passes a per-item
    generator so the severity is a function of (seed, epoch, index) rather than of
    how many times this sampler happens to have been called. The instance's own
    generator advances in DataLoader fetch order, which changes with shuffling and
    with `num_workers` (each worker gets its own copy of the sampler and replays
    the same sequence), so relying on it alone leaves a "seeded" run irreproducible.
    """

    low: float = 0.0
    high: float = 1.0
    seed: int | None = None

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, rng: np.random.Generator | None = None) -> float:
        return float((rng or self._rng).uniform(self.low, self.high))
