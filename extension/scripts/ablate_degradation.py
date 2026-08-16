#!/usr/bin/env python3
"""Phase 2 extension: which degradation strategy actually looks like a phone photo?

Compares up to three sources of "degraded" images:
  physics  - the hand-parametrized pipeline in data/degradation.py
  learned  - the adversarially-trained generator in data/degradation_learned.py
  real     - actual phone recaptures, if you have them (see data/real_recapture/)

The question this answers is not "which looks nicer" but "which is harder to
tell apart from a real phone photo": for each synthetic source, a small
classifier is trained on cheap no-reference image-quality features (blur,
brightness, contrast, edge density) to distinguish it from `real`. Classifier
accuracy near chance (0.5) means the synthetic images are statistically close to
real captures on these axes; accuracy near 1.0 means they're easy to spot as fake.
This is the ablation itself -- a lower "tell apart from real" accuracy is the
evidence that a degradation strategy is a better stand-in for the real thing.

Without --real-dir (no real recaptures on hand yet, the common case right now)
the script still runs: it reports the feature distributions for each synthetic
source and how separable *they* are from each other, which at least confirms
`physics` and `learned` are producing meaningfully different artifacts worth
comparing later, and tells you exactly what to point at once real data exists.

Usage:
    python scripts/ablate_degradation.py --out outputs/degradation_ablation.json
    python scripts/ablate_degradation.py --source data/raw --real-dir data/real_recapture \
        --learned-checkpoint outputs/learned_degrader.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbtrust.data.degradation import SmartphoneDegradation


def _fake_xray(rng: np.random.Generator, size: int = 128) -> np.ndarray:
    """Same crude synthetic lung image the smoke test uses, so this script is
    self-contained when there's no real data directory to point it at yet."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = 60 + 40 * np.sin(xx / 12) * np.cos(yy / 15)
    for _ in range(2):
        cy, cx = rng.uniform(0.35, 0.65, 2) * size
        base += 80 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (size / 6) ** 2)))
    return np.clip(base + rng.normal(0, 8, (size, size)), 0, 255).astype(np.uint8)


def _load_dir(path: str, n: int, size: int = 128) -> list[np.ndarray]:
    from PIL import Image

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted(f for f in Path(path).rglob("*") if f.suffix.lower() in exts)[:n]
    return [np.asarray(Image.open(f).convert("L").resize((size, size))) for f in files]


# --------------------------------------------------------------------------- #
# no-reference image-quality features (cheap, numpy-only, no GT needed)
# --------------------------------------------------------------------------- #
_LAPLACIAN_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def _laplacian(img: np.ndarray) -> np.ndarray:
    p = np.pad(img.astype(np.float32), 1, mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    for (dy, dx), w in np.ndenumerate(_LAPLACIAN_KERNEL):
        if w == 0:
            continue
        out += w * p[dy : dy + img.shape[0], dx : dx + img.shape[1]]
    return out


def image_features(img: np.ndarray) -> dict[str, float]:
    """Blur, brightness, contrast, edge density -- the axes phone capture disturbs most."""
    lap = _laplacian(img)
    gy, gx = np.gradient(img.astype(np.float32))
    grad_mag = np.hypot(gy, gx)
    return {
        "blur_laplacian_var": float(lap.var()),          # low = blurrier
        "brightness_mean": float(img.mean()),
        "brightness_std": float(img.std()),               # low = flat/glare-washed
        "edge_density": float((grad_mag > grad_mag.mean() + grad_mag.std()).mean()),
        "clipped_frac": float(((img <= 2) | (img >= 253)).mean()),  # blown highlights/shadows
    }


def feature_matrix(images: list[np.ndarray]) -> np.ndarray:
    feats = [image_features(im) for im in images]
    keys = sorted(feats[0].keys())
    return np.array([[f[k] for k in keys] for f in feats]), keys


def separability(a: np.ndarray, b: np.ndarray, seed: int = 0) -> dict[str, float]:
    """5-fold CV accuracy of a logistic classifier telling group a from group b.

    ~0.5 = statistically indistinguishable on these features (good, for
    synthetic-vs-real); ~1.0 = trivially separable.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    X = np.concatenate([a, b], axis=0)
    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    n_splits = max(2, min(5, len(y) // 2))
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    scores = cross_val_score(clf, Xs, y, cv=n_splits)
    return {"cv_accuracy_mean": float(scores.mean()), "cv_accuracy_std": float(scores.std()), "n_splits": n_splits}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=None, help="dir of clean source images; default = synthetic fakes")
    ap.add_argument("--real-dir", default=None, help="dir of real phone recaptures, if you have them")
    ap.add_argument("--learned-checkpoint", default=None, help="LearnedDegrader checkpoint; default = untrained")
    ap.add_argument("--skip-learned", action="store_true", help="skip the learned strategy (e.g. no torch)")
    ap.add_argument("--n", type=int, default=40, help="number of images per group")
    ap.add_argument("--severity", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/degradation_ablation.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.source:
        clean = _load_dir(args.source, args.n)
        if not clean:
            raise SystemExit(f"no images found under {args.source}")
    else:
        print(f"--source not given; generating {args.n} synthetic placeholder X-rays instead.")
        clean = [_fake_xray(rng) for _ in range(args.n)]

    groups: dict[str, list[np.ndarray]] = {}

    physics = SmartphoneDegradation(severity=args.severity, seed=args.seed)
    groups["physics"] = [physics(im.copy())[0] for im in clean]

    if not args.skip_learned:
        try:
            from tbtrust.data.degradation_learned import LearnedDegrader

            if args.learned_checkpoint:
                degrader = LearnedDegrader.load(args.learned_checkpoint)
            else:
                warnings.warn(
                    "No --learned-checkpoint given: using an UNTRAINED generator. This only "
                    "proves the pipeline runs; the output is not a meaningful degradation yet. "
                    "Train one with degradation_learned.train_learned_degradation on a real "
                    "recapture set (data/real_recapture/) before trusting these numbers.",
                    stacklevel=2,
                )
                degrader = LearnedDegrader()
            groups["learned"] = [degrader(im, args.severity) for im in clean]
        except ImportError:
            print("torch not available -- skipping the learned-degradation arm.")

    if args.real_dir:
        real = _load_dir(args.real_dir, args.n)
        if real:
            groups["real"] = real
        else:
            print(f"WARNING: --real-dir {args.real_dir} had no images; dropping the real-vs-synthetic comparison.")

    feats = {name: feature_matrix(imgs) for name, imgs in groups.items()}
    keys = next(iter(feats.values()))[1]

    report: dict = {"n_per_group": args.n, "severity": args.severity, "feature_summary": {}, "separability": {}}
    for name, (X, _) in feats.items():
        report["feature_summary"][name] = {
            k: {"mean": float(X[:, i].mean()), "std": float(X[:, i].std())} for i, k in enumerate(keys)
        }

    if "real" in feats:
        real_X, _ = feats["real"]
        for name in [g for g in groups if g != "real"]:
            report["separability"][f"{name}_vs_real"] = separability(feats[name][0], real_X, seed=args.seed)
        print("\nSeparability from real recaptures (lower cv_accuracy = more realistic):")
        for k, v in report["separability"].items():
            print(f"  {k}: {v['cv_accuracy_mean']:.3f} +/- {v['cv_accuracy_std']:.3f}")
    else:
        others = [g for g in groups if g != "real"]
        if len(others) >= 2:
            a, b = others[0], others[1]
            report["separability"][f"{a}_vs_{b}"] = separability(feats[a][0], feats[b][0], seed=args.seed)
        print(
            "\nNo --real-dir given, so this only compares synthetic strategies to each "
            "other -- see data/real_recapture/README.md to collect a pilot set, then "
            "re-run with --real-dir to get the actual ablation (synthetic vs. real)."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
