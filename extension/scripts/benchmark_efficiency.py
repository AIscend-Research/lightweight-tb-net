#!/usr/bin/env python3
"""Phase 3: back (or refute) the low-compute claim with numbers, not architecture prose.

Measures, for each model, on CPU with a fixed thread count to approximate a
low-end clinic device (no GPU, no reliable power, so "runs on a phone/cheap
laptop" has to mean something measured, not assumed):

  params    - trainable parameter count
  gmacs     - multiply-accumulate ops for one 224x224 forward pass, counted via
              forward hooks on Conv2d/Linear (the layers that dominate CNN cost;
              BN/activation/pooling are not counted -- this is the same
              convention papers use when they report "GMACs", including the
              TB-Net number this project is compared against, so the numbers
              are comparable even though they're not a literal total-FLOPs count)
  latency_ms - wall-clock time for one batch=1 forward pass, mean/median/p95
              over repeated runs after a warmup, single CPU thread

Usage:
    python scripts/benchmark_efficiency.py --out outputs/efficiency_benchmark.json
    python scripts/benchmark_efficiency.py --models baseline_densenet121,baseline_resnet50,tbnet,evidential \
        --threads 1 --repeats 30
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn as nn


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(model: nn.Module, input_shape: tuple[int, int, int, int] = (1, 3, 224, 224)) -> int:
    """Sum of Conv2d/Linear MACs for one forward pass at `input_shape`, via hooks."""
    macs = 0

    def conv_hook(module: nn.Conv2d, inp, out):
        nonlocal macs
        b, _, oh, ow = out.shape
        kh, kw = module.kernel_size
        in_ch_per_group = module.in_channels // module.groups
        macs += b * oh * ow * kh * kw * in_ch_per_group * module.out_channels

    def linear_hook(module: nn.Linear, inp, out):
        nonlocal macs
        macs += inp[0].shape[0] * module.in_features * module.out_features

    handles = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
    try:
        model.eval()
        with torch.no_grad():
            model(torch.randn(*input_shape))
    finally:
        for h in handles:
            h.remove()
    return macs


def benchmark_latency(model: nn.Module, input_shape=(1, 3, 224, 224), repeats: int = 30, warmup: int = 5) -> dict:
    model.eval()
    x = torch.randn(*input_shape)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "p95_ms": times[int(0.95 * len(times)) - 1],
        "min_ms": times[0],
        "max_ms": times[-1],
    }


def _build(name: str) -> nn.Module:
    from tbtrust.models.baseline import TBClassifier
    from tbtrust.models.evidential import EvidentialClassifier
    from tbtrust.models.tbnet import TBNet

    if name == "tbnet":
        return TBNet()
    if name == "evidential":
        return EvidentialClassifier(pretrained=False)
    if name.startswith("baseline_"):
        backbone = name.removeprefix("baseline_")
        return TBClassifier(backbone=backbone, pretrained=False)
    raise ValueError(f"unknown model '{name}' (try tbnet, evidential, baseline_<timm/torchvision name>)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="baseline_densenet121,tbnet,evidential",
                    help="comma-separated: tbnet, evidential, baseline_<backbone name>")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads; 1 approximates a low-end device")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--out", default="outputs/efficiency_benchmark.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    shape = (1, 3, args.image_size, args.image_size)

    report = {"threads": args.threads, "input_shape": list(shape), "models": {}}
    print(f"{'model':<22} {'params':>12} {'gmacs':>8} {'latency mean/median/p95 (ms)':>32}")
    for name in args.models.split(","):
        name = name.strip()
        model = _build(name)
        n_params = count_params(model)
        macs = count_macs(model, shape)
        lat = benchmark_latency(model, shape, repeats=args.repeats)
        report["models"][name] = {"params": n_params, "gmacs": macs / 1e9, "latency_ms": lat}
        print(f"{name:<22} {n_params:>12,} {macs / 1e9:>8.3f} "
              f"{lat['mean_ms']:>10.2f}/{lat['median_ms']:>7.2f}/{lat['p95_ms']:>7.2f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")
    print(
        "\nReference: TB-Net reports ~4.24M params / 0.42 GMACs at 224x224 "
        "(Wong et al. 2022). Compare tbnet's gmacs above against that -- params "
        "are tuned to match (see models/tbnet.py), MACs are a consequence of the "
        "architecture, not separately tuned, so a mismatch here is informative."
    )


if __name__ == "__main__":
    main()
