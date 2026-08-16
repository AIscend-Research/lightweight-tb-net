#!/usr/bin/env python3
"""Execute every notebook end to end on synthetic data. Run by CI.

The notebooks are the documented entry point for the whole pipeline, and they
rotted once already: they were written against an older API, never executed, and
silently referenced metrics.json keys and a `data/download_data.py` that did not
exist. Reading them is not enough; the only way they stay correct is if something
runs them.

This makes that cheap. It builds a small synthetic dataset in the aggregated
Kaggle layout, copies the repo with the configs shrunk (1 epoch, 64px, tiny
batches, resnet18 with no pretrained download), points the notebooks' environment
variables at it, and executes all five in order. The *code* under test is the real
package -- only the data and the hyperparameters are miniaturised, so an API
change that breaks a notebook still fails here.

    python scripts/test_notebooks.py [--keep]

Needs the dev extras (`pip install -e ".[dev]"`) for nbclient/nbformat/ipykernel.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Small enough to train on a CPU runner in minutes; big enough that every split is
# non-empty and both classes appear in each clinic.
IMAGES_PER_CLINIC = 48

SHRUNK_DEFAULTS = {
    "data": {"image_size": 64, "val_frac": 0.25, "test_frac": 0.25},
    "train": {"epochs": 1, "batch_size": 8, "num_workers": 0, "device": "cpu"},
    "eval": {"severity_sweep": [0.0, 0.5], "primary_severity": 0.5,
             "mc_dropout_passes": 2, "conformal_alpha": 0.2},
    # resnet18/pretrained=false keeps CI off the network and out of a densenet download.
    "model": {"backbone": "resnet18", "pretrained": False},
}


def _deep_update(base: dict, extra: dict) -> None:
    for k, v in extra.items():
        if isinstance(v, dict):
            _deep_update(base.setdefault(k, {}), v)
        else:
            base[k] = v


def build_synthetic_dataset(data_dir: Path) -> int:
    """Aggregated-Kaggle layout: folder gives the label, filename gives the clinic."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    (data_dir / "Normal").mkdir(parents=True, exist_ok=True)
    (data_dir / "Tuberculosis").mkdir(parents=True, exist_ok=True)

    for prefix, brightness_offset in (("MCUCXR", 0), ("CHNCXR", 30)):   # montgomery, shenzhen
        for i in range(IMAGES_PER_CLINIC):
            label = i % 2
            arr = np.clip(rng.normal(110 + brightness_offset, 18, (96, 96)), 0, 255)
            if label:                                   # a crude "TB" focus so the task is learnable
                yy, xx = np.mgrid[0:96, 0:96]
                arr = np.clip(arr + 60 * np.exp(-(((yy - 48) ** 2 + (xx - 48) ** 2) / 400)), 0, 255)
            folder = "Tuberculosis" if label else "Normal"
            Image.fromarray(arr.astype(np.uint8), "L").save(
                data_dir / folder / f"{prefix}_{i:04d}_{label}.png")
    return len(list(data_dir.rglob("*.png")))


def stage_repo(dest: Path) -> None:
    """Copy the repo and shrink every config so the run finishes on a CPU runner."""
    import yaml

    for item in ("src", "scripts", "configs", "tests", "notebooks", "pyproject.toml"):
        src = REPO_ROOT / item
        (shutil.copytree if src.is_dir() else shutil.copy)(src, dest / item)

    cfg_dir = dest / "configs"
    default = yaml.safe_load((cfg_dir / "default.yaml").read_text())
    _deep_update(default, SHRUNK_DEFAULTS)
    (cfg_dir / "default.yaml").write_text(yaml.safe_dump(default, sort_keys=False))

    for f in cfg_dir.glob("*.yaml"):
        if f.name == "default.yaml":
            continue
        cfg = yaml.safe_load(f.read_text()) or {}
        _deep_update(cfg, {"train": {"epochs": 1, "batch_size": 8,
                                     "num_workers": 0, "device": "cpu"}})
        if cfg.get("model", {}).get("backbone"):
            cfg["model"]["backbone"] = "resnet18"
            cfg["model"]["pretrained"] = False
        f.write_text(yaml.safe_dump(cfg, sort_keys=False))


def run_notebooks(repo: Path, data: Path, work: Path) -> list[tuple[str, str]]:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    env_overrides = {
        "TBTRUST_REPO": str(repo),
        "TBTRUST_DATA": str(data),
        "TBTRUST_WORK": str(work),
        "MPLBACKEND": "Agg",            # no display on a CI runner
        "PYTHONPATH": str(repo / "src"),
        # The physics notebooks default to a realistic 1024 px working resolution,
        # which is right for a real run and far too slow for CI at ~2 s per image.
        # Shrinking it changes what the certificate *concludes* -- a 2 mm nodule is
        # sub-pixel at 256 -- but this harness is testing that the code runs against
        # the current API, not that the verdicts are meaningful.
        "TBTRUST_PHYSICS_SIZE": "256",
        "TBTRUST_PHYSICS_N": "8",
    }
    saved = os.environ.copy()
    os.environ.update(env_overrides)
    failures: list[tuple[str, str]] = []
    try:
        for nb_path in sorted((repo / "notebooks").glob("*.ipynb")):
            print(f"\n=== executing {nb_path.name} ===", flush=True)
            nb = nbformat.read(nb_path, as_version=4)
            client = NotebookClient(nb, timeout=2400, kernel_name="python3",
                                    resources={"metadata": {"path": str(repo)}})
            try:
                client.execute()
                print(f"  OK  {nb_path.name}", flush=True)
            except CellExecutionError as e:
                failures.append((nb_path.name, str(e)[-4000:]))
                print(f"  FAILED  {nb_path.name}", flush=True)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true", help="keep the temp workspace for inspection")
    args = ap.parse_args()

    root = Path(tempfile.mkdtemp(prefix="tbtrust-nb-"))
    repo, data, work = root / "repo", root / "data", root / "work"
    for d in (repo, data, work):
        d.mkdir(parents=True)

    try:
        n = build_synthetic_dataset(data)
        print(f"synthetic dataset: {n} images")
        stage_repo(repo)
        failures = run_notebooks(repo, data, work)
    finally:
        if args.keep:
            print(f"\nworkspace kept at {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} notebook(s) failed:\n")
        for name, msg in failures:
            print(f"--- {name} ---\n{msg}\n")
        return 1
    print("All notebooks executed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
