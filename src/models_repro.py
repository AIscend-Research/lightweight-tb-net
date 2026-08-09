"""
Model registry for the reproducibility study.

  compact — the original 0.27M-parameter re-implementation (src/model.py).
  full    — a capacity-matched reconstruction sized to the ~4.24M parameters
            reported by Wong et al. (2022).

IMPORTANT, and this belongs in the paper: 'full' is NOT a faithful port. The
original TB-Net graph is not present in the released source. The TF1 training
script loads it via tf.train.import_meta_graph(model_train.meta) - the
architecture lives inside the checkpoint, and the checkpoint download is dead.
'full' therefore matches the reported parameter budget and the described
motifs (depthwise-separable convolutions + attention condensers) but the exact
layer topology cannot be verified against the original.
"""

import torch
import torch.nn as nn

from model import AttentionCondenser, DepthwiseSeparable, TBNet  # noqa: F401


class TBNetFull(nn.Module):
    """Capacity-matched reconstruction (~4.2M parameters at base=112)."""

    def __init__(self, num_classes=2, base=112, blocks_per_stage=2):
        super().__init__()
        widths = [base * (2 ** i) for i in range(5)]   # 112 224 448 896 1792

        self.stem = nn.Sequential(
            nn.Conv2d(1, widths[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )

        stages = []
        in_ch = widths[0]
        for si, out_ch in enumerate(widths[1:]):
            for bi in range(blocks_per_stage):
                # Downsample once per stage, on the first block.
                stride = 2 if bi == 0 else 1
                tgt = out_ch if bi == blocks_per_stage - 1 else in_ch
                stages.append(DepthwiseSeparable(in_ch, tgt, stride=stride))
                in_ch = tgt
            stages.append(AttentionCondenser(in_ch))
        self.features = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.features(self.stem(x))
        x = self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))


def build_model(arch, num_classes=2):
    if arch == "compact":
        return TBNet(num_classes=num_classes)
    if arch == "full":
        return TBNetFull(num_classes=num_classes)
    raise ValueError(f"unknown arch: {arch}")


def build_student(pretrained=True, num_classes=2):
    from torchvision.models import (MobileNet_V3_Small_Weights,
                                    mobilenet_v3_small)
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    m = mobilenet_v3_small(weights=weights)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m


if __name__ == "__main__":
    for arch in ("compact", "full"):
        m = build_model(arch)
        n = sum(p.numel() for p in m.parameters())
        out = m(torch.randn(2, 1, 224, 224))
        print(f"{arch:8s} {n / 1e6:.2f}M params  out={tuple(out.shape)}")
    s = build_student(pretrained=False)
    print(f"student  {sum(p.numel() for p in s.parameters()) / 1e6:.2f}M params")
