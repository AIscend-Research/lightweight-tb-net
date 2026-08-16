#!/usr/bin/env python3
"""Generate every physics figure into one directory. No data or GPU required.

    python scripts/make_figures.py --out outputs/figures
    python scripts/make_figures.py --out outputs/figures --manifest data/processed/manifest.csv

Without a manifest it runs entirely on the synthetic film in `physics/film.py`,
which is enough for the schematic diagrams, the inversion panels, the certificate
and the detectability strip. With one it additionally renders real dataset
images: a normal-vs-TB gallery, the fiducial overlay on an actual archived
radiograph, and the degradation ladder applied to a real chest film.

The two figures worth putting earliest in a paper
-------------------------------------------------
`02_sign_convention` because the whole method reads backwards without it, and
`08_detectability_strip` because it is the only figure that lets a reader check
the central claim with their own eyes rather than taking a number on trust.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import numpy as np

from tbtrust.physics import figures as F
from tbtrust.physics.certificate import certify
from tbtrust.physics.film import capture, sample_params, synthetic_chest_density
from tbtrust.physics.findings import all_findings, get
from tbtrust.physics.invert import invert
from tbtrust.physics.triage import triage


def _real_images(manifest: str, n: int):
    """A stratified handful of real images, or None if the manifest is unusable."""
    import pandas as pd

    try:
        df = pd.read_csv(manifest)
    except (OSError, ValueError):
        return None
    if df.empty or "path" not in df:
        return None
    by = [c for c in ("clinic", "label") if c in df.columns]
    sel = (df.groupby(by, group_keys=False).sample(frac=min(1.0, n / len(df)), random_state=0)
           if by else df.sample(min(n, len(df)), random_state=0))
    return sel.head(n).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--manifest", default=None, help="adds the real-image figures")
    ap.add_argument("--size", type=int, default=768,
                    help="working resolution for the simulated film")
    ap.add_argument("--severity", type=float, default=0.5)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    rng = np.random.default_rng(args.seed)
    made: list[str] = []

    def emit(fig, name):
        p = F.save(fig, out / f"{name}.png", dpi=args.dpi)
        made.append(str(p))
        print(f"  {p}")

    print("schematics (no data needed)")
    emit(F.capture_chain_diagram(), "01_capture_chain")
    emit(F.sign_convention_panel(), "02_sign_convention")

    print("simulated capture")
    base, ftruth = synthetic_chest_density(size=args.size, rng=np.random.default_rng(args.seed))
    params = sample_params(args.severity, np.random.default_rng(args.seed + 1))
    photo, truth = capture(base, params, fiducial_truth=ftruth,
                           rng=np.random.default_rng(args.seed + 2))
    cal = invert(photo)
    cert = certify(cal, findings=all_findings())
    decision = triage(cert, cal, model_confidence=0.85)

    emit(F.fiducial_anatomy(photo, cal.fiducials,
                            title="Calibration targets on a simulated re-photographed film"),
         "03_fiducials_simulated")
    emit(F.finding_atlas(cert, cal.px_per_mm), "04_finding_atlas")
    emit(F.inversion_panels(photo, cal, truth), "05_inversion")
    emit(F.certificate_card(cert, cal, decision), "06_certificate")
    emit(F.retake_instruction(cal, decision), "07_retake_instruction")

    print("detectability strips (the falsifiable claim, visually)")
    for key in ("infiltrate", "cavity_wall"):
        emit(F.detectability_strip(base, params, get(key), cal=cal, fiducial_truth=ftruth,
                                   seed=args.seed + 7),
             f"08_detectability_{key}")

    if args.manifest:
        print("real dataset images")
        sel = _real_images(args.manifest, 12)
        if sel is None or sel.empty:
            print("  (manifest unreadable or empty; skipped)")
        else:
            emit(F.radiograph_gallery(sel["path"], sel.get("label"), sel.get("clinic"),
                                      title="Real chest radiographs from the LOCO clinics"),
                 "09_real_gallery")
            from PIL import Image

            first = np.asarray(Image.open(sel["path"].iloc[0]).convert("L"))
            emit(F.degradation_ladder(first, seed=int(rng.integers(1 << 30))),
                 "10_degradation_ladder")
            # The fiducial overlay on a *real* archived image is the honest version
            # of figure 03: whatever it shows is what the audit will report.
            emit(F.fiducial_anatomy(
                np.asarray(Image.open(sel["path"].iloc[0]).convert("L").resize((768, 768))),
                title="Calibration targets on a real archived radiograph"),
                "11_fiducials_real")

    print(f"\n{len(made)} figures -> {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
