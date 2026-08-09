"""
End-to-end smoke test on synthetic images.

Runs every stage of experiments.py at toy scale (60 fake X-rays, 1 epoch) in a
temporary directory. It proves the plumbing works - caching, seeding, training,
pruning, distillation, quantization, ONNX export, phone perturbation, summary
aggregation - without touching the real dataset or the real results/.

  python src/smoke_test.py

Takes about a minute on CPU. Run this before every long GPU session; a typo
that surfaces here costs a minute instead of five hours.
"""

import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_cache  # noqa: E402
import common  # noqa: E402
import experiments  # noqa: E402


def make_fake_dataset(root, n_normal=40, n_tb=20):
    """Synthetic CXR-ish images: a bright blob on a dark field, plus the
    black padding borders that the auto-crop is supposed to strip."""
    data = os.path.join(root, "data")
    os.makedirs(data, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_normal + n_tb):
        is_tb = i >= n_normal
        name = f"{'Tuberculosis' if is_tb else 'Normal'}-{i}.png"
        img = np.zeros((300, 300), np.uint8)
        cv2.ellipse(img, (150, 150), (90, 120), 0, 0, 360,
                    int(rng.integers(90, 140)), -1)
        if is_tb:                       # a bright "lesion" the model can learn
            cx, cy = int(rng.integers(110, 190)), int(rng.integers(90, 160))
            cv2.circle(img, (cx, cy), int(rng.integers(10, 20)), 235, -1)
        img = np.clip(img.astype(np.int16) + rng.integers(-8, 8, img.shape), 0, 255)
        img = img.astype(np.uint8)
        img[:20] = 0                    # padding borders
        img[-20:] = 0
        cv2.imwrite(os.path.join(data, name), cv2.merge([img, img, img]))
        rows.append((name, int(is_tb)))

    rng.shuffle(rows)
    splits_dir = os.path.join(root, "data_splits")
    os.makedirs(splits_dir, exist_ok=True)
    n = len(rows)
    bounds = {"train": rows[:int(0.6 * n)],
              "val": rows[int(0.6 * n):int(0.8 * n)],
              "test": rows[int(0.8 * n):]}
    for split, items in bounds.items():
        with open(os.path.join(splits_dir, f"{split}.csv"), "w") as fh:
            for name, y in items:
                fh.write(f"{name},{y}\n")
    return data, splits_dir


class Args:
    """Stand-in for the argparse namespace, at toy scale."""
    arch = ["compact"]
    caches = ["faithful"]
    seeds = [0]
    weighted = [False]
    epochs = 1
    ft_epochs = 1
    distill_epochs = 1
    iter_steps = 2
    iter_amount = 0.2
    iter_ft_epochs = 1
    lr = 1e-3
    temperature = 4.0
    alpha = 0.7
    cosine = False
    augment = True
    pretrained_student = 0          # no network access needed
    phone_finetune = True
    external_path = ""
    force = True


def main():
    with tempfile.TemporaryDirectory() as tmp:
        print(f"scratch dir: {tmp}")
        data, splits_dir = make_fake_dataset(tmp)

        # Redirect every path the modules use into the temp dir.
        common.CACHE_DIR = os.path.join(tmp, "cache")
        build_cache.CACHE_DIR = common.CACHE_DIR
        common.RESULTS_DIR = os.path.join(tmp, "results")
        experiments.CKPT_DIR = os.path.join(tmp, "checkpoints")
        experiments.SPLITS = {s: os.path.join(splits_dir, f"{s}.csv")
                              for s in ("train", "val", "test")}

        print("\n[1/8] build_cache")
        build_cache.build(data, "faithful", workers=2)
        build_cache.build(data, "simple", workers=2)

        for i, stage in enumerate(
                ["baseline", "prune", "distill", "quantize", "phone",
                 "summary"], start=2):
            print(f"\n[{i}/8] {stage}")
            experiments.STAGES[stage](Args())

        print("\n[8/8] checking outputs")
        expected = ["baseline.csv", "prune.csv", "distill.csv",
                    "quantize.csv", "phone.csv"]
        missing = [f for f in expected
                   if not os.path.exists(os.path.join(common.RESULTS_DIR, f))]
        if missing:
            raise SystemExit(f"FAILED - missing result files: {missing}")
        for f in expected:
            path = os.path.join(common.RESULTS_DIR, f)
            n = sum(1 for _ in open(path)) - 1
            print(f"  {f}: {n} rows")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
