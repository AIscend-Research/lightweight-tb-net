"""TB-Net reproduction slot.

The published TB-Net (Wong et al., 2022, Front. Artif. Intell.) is a machine-
designed self-attention CNN built from *visual attention condensers*, ~4.24M
params / 0.42 GMACs at 224x224, reaching 99.86% on the clean Kaggle TB benchmark.
The official code is TensorFlow 1.15, checkpoint format:
    https://github.com/darwinai/TuberculosisNet   (paper arXiv:2104.03165)

Reproducing it in PyTorch is a deliverable, not a given. Two honest paths:

  A. Port the exact macro/micro architecture from the released checkpoint. The
     generative-synthesis design is irregular, so this means reading the graph out
     of the TF checkpoint and rebuilding it layer for layer. Highest fidelity,
     most work.
  B. Reimplement the *idea* -- a compact CNN whose blocks are attention condensers
     -- and treat "matches TB-Net's accuracy/efficiency" as the target. Faster,
     and enough to serve as the efficient, low-compute model the paper needs.

Below is a starting-point attention-condenser block and a skeleton network for
path B, matching TBClassifier's output dict so it is a drop-in replacement in the
training loop. It is NOT yet the verified TB-Net; treat the accuracy numbers as
unvalidated until real training runs happen -- but `scripts/benchmark_efficiency.py`
has already measured one axis: this network's param count is tuned to TB-Net's
~4.24M (see below), yet it costs ~0.83 GMACs at 224x224 against TB-Net's reported
~0.42 GMACs -- about 2x the compute per parameter. That's a real, informative gap,
not noise: the standard 3x3 convolutions in `TBNetBlock` are less MAC-efficient
per parameter than the real attention condenser's design (which leans on
depthwise/grouped convolutions in its condense/expand path, not just inside the
attention branch). Worth closing before quoting a "matches TB-Net's efficiency"
claim -- e.g. make the block's main conv depthwise-separable, not just the
attention condenser's internal embed conv -- rather than re-tuning widths to hit
0.42 GMACs directly, which would just trade the params-match for a MACs-match
without fixing the actual design gap.

Attention condenser (Wong et al., 2020, "AttendNets"/"TinySpeech"): condense the
activation map into a compact embedding capturing joint spatial + cross-channel
structure, produce a self-attention map from it, and selectively amplify features.
The block here is a lightweight stand-in for that mechanism.
"""

from __future__ import annotations

from itertools import pairwise

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionCondenser(nn.Module):
    """Lightweight stand-in for a visual attention condenser.

    condense (pool + 1x1) -> embed -> expand to a spatial attention map -> gate.
    Replace with a faithful condenser once you port the real design.
    """

    def __init__(self, channels: int, reduction: int = 4, pool: int = 2):
        super().__init__()
        self.pool = pool
        hidden = max(channels // reduction, 8)
        self.condense = nn.Conv2d(channels, hidden, kernel_size=1)
        self.embed = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.expand = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _b, _c, h, w = x.shape
        # At 224x224 with 5 downsampling stages the last block's map is only
        # 7x7, and any deliberately-small smoke-test input can go below that --
        # clamp so a pool=2 request on a 1x1 map doesn't crash the whole forward
        # pass with a "output size is too small" error.
        pool = max(1, min(self.pool, h, w))
        s = F.avg_pool2d(x, pool)                  # condense spatially
        s = F.relu(self.condense(s))
        s = F.relu(self.embed(s))
        s = torch.sigmoid(self.expand(s))         # attention in condensed space
        att = F.interpolate(s, size=(h, w), mode="bilinear", align_corners=False)
        return x * att                            # selective attention


class TBNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.att = AttentionCondenser(out_ch)

    def forward(self, x):
        return self.att(F.relu(self.bn(self.conv(x))))


class TBNet(nn.Module):
    """Compact attention-condenser CNN (reproduction path B). Swappable with TBClassifier.

    Default widths were picked by a small grid search over (stage widths) to land
    on TB-Net's reported ~4.24M parameters: `(64, 128, 256, 448, 608)` gives
    4,228,282 params (see `test_tbnet_param_count_matches_tbnet_paper` in
    tests/test_smoke.py, which pins this so it doesn't silently drift). Matching
    parameter count is a necessary, not sufficient, condition for "reproduces
    TB-Net" -- it's evidence the macro-architecture (5 downsampling stages,
    attention condenser per stage) is in the right regime, not proof the accuracy
    matches; that's still an open validation once real training runs happen.
    5 stages (stem + 4 blocks, each stride 2) taking 224x224 down to 7x7 mirrors
    the depth of typical chest-X-ray backbones (e.g. DenseNet-121's 224->7 pool),
    rather than the shallower 4-stage default this started from.
    """

    def __init__(self, in_channels: int = 3, widths=(64, 128, 256, 448, 608),
                 dropout: float = 0.3, with_uncertainty_head: bool = True):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_channels, widths[0], 3, stride=2, padding=1),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU())
        blocks = [TBNetBlock(in_ch, out_ch, stride=2) for in_ch, out_ch in pairwise(widths)]
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(widths[-1], 1)
        self.with_uncertainty_head = with_uncertainty_head
        if with_uncertainty_head:
            self.uncertainty_head = nn.Sequential(nn.Linear(widths[-1], 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x) -> dict[str, torch.Tensor]:
        f = self.blocks(self.stem(x))
        f = torch.flatten(self.pool(f), 1)
        f = self.dropout(f)
        out = {"logit": self.classifier(f).squeeze(-1)}
        if self.with_uncertainty_head:
            out["uncertainty"] = torch.sigmoid(self.uncertainty_head(f).squeeze(-1))
        return out


def load_tf_checkpoint_notes() -> str:
    """Pointer for path A (porting the official TF checkpoint)."""
    return (
        "Official weights (TF1.15 ckpt): linked from docs/models.md in "
        "github.com/darwinai/TuberculosisNet. To port: load the ckpt with TF, "
        "enumerate ops/shapes, and rebuild the graph in torch. Verify layer "
        "outputs match on a fixed input before trusting the port."
    )
