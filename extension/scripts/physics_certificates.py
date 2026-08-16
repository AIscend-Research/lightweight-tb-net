#!/usr/bin/env python3
"""Compute a physics certificate per image and write a table the eval path can join.

This is the bridge between the physics track and the rest of the repo. It runs the
blind inversion over a manifest and emits one row per image -- certificate verdict,
margin in dB, per-finding floors, the limiting factor, and the triage action --
keyed by `path`, so `eval/physics_deferral.py` can join it to the model's
predictions without either side importing the other.

Two modes, and the distinction matters for what the numbers mean:

**--simulate** (default when the corpus lacks fiducials). Each manifest image is
treated as a clean film, has fiducials painted onto it by `physics/film.py`, and is
re-photographed through the forward capture model at a chosen severity. Everything
is then measured blind. This is the controlled experiment: capture quality is a
knob, and the certificate can be scored against a known ground truth.

**--real**. The manifest images are treated as already being photographs and are
inverted directly. This is the deployment-realistic path, and it is only available
where the archive kept the fiducials -- run `scripts/audit_fiducials.py` first and
expect a lot of ABSTAIN rows otherwise.

    python scripts/physics_certificates.py --manifest data/processed/manifest.csv \\
        --out outputs/certificates.csv --severities 0,0.25,0.5,0.75,1.0

    python scripts/physics_certificates.py --manifest ... --real --out outputs/certs_real.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from PIL import Image

from tbtrust.physics import findings as FIND
from tbtrust.physics.certificate import certificate_confidence, certify
from tbtrust.physics.film import simulate
from tbtrust.physics.floor import FloorSpec
from tbtrust.physics.invert import invert
from tbtrust.physics.triage import triage


def _load(path: str, size: int | None) -> np.ndarray | None:
    try:
        img = Image.open(path).convert("L")
    except (OSError, ValueError):
        return None
    if size and max(img.size) != size:
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img)


def process(
    path: str,
    severity: float,
    simulate_capture: bool,
    size: int,
    findings,
    spec: FloorSpec,
    seed: int,
) -> dict | None:
    img = _load(path, size if simulate_capture else None)
    if img is None:
        return None
    rng = np.random.default_rng(abs(hash((path, severity, seed))) % (2**32))

    if simulate_capture:
        photo, _truth = simulate(img, severity=severity, rng=rng, size=size)
    else:
        photo = img

    cal = invert(photo)
    cert = certify(cal, findings=findings, spec=spec)
    dec = triage(cert, cal)

    return {
        "path": path,
        "severity": float(severity),
        "mode": "simulated" if simulate_capture else "real",
        **cert.as_dict(),
        "physics_confidence": certificate_confidence(cert),
        "abstained": cert.abstained,
        "triage_action": dec.action.value,
        "triage_reason": dec.reason,
        "triage_instruction": dec.instruction,
        "expected_retake_gain_db": dec.expected_gain_db,
        **{f"cal_{k}": v for k, v in cal.summary().items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="outputs/certificates.csv")
    ap.add_argument("--real", action="store_true",
                    help="treat manifest images as photographs; do not simulate capture")
    ap.add_argument("--severities", default="0.0,0.25,0.5,0.75,1.0",
                    help="comma-separated severity sweep (simulation mode only)")
    ap.add_argument("--size", type=int, default=1024,
                    help="working resolution, and it matters more than any other flag. A phone "
                         "photographing a 35 cm film at 3000 px gets ~8 px/mm; at 320 px it gets "
                         "0.8, so a 2 mm miliary nodule is under two pixels and the certificate "
                         "correctly but uselessly calls every image insufficient. At 1024 the "
                         "sweep separates properly -- miliary marginal on a clean capture, "
                         "insufficient once degraded, larger findings still carried. Costs about "
                         "2 s per image.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default=None, help="restrict to one manifest split")
    ap.add_argument("--findings", default=None,
                    help="path to a YAML/CSV finding table replacing the nominal defaults")
    ap.add_argument("--rose-k", type=float, default=5.0)
    ap.add_argument("--include-anatomical", action="store_true",
                    help="fold anatomical clutter into the floor (no longer a pure channel bound)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if args.split:
        df = df[df["split"] == args.split]
    if args.limit and args.limit < len(df):
        # Stratified sample, not head(). A manifest is built by walking the
        # dataset directory, so it arrives grouped by class and by source -- the
        # first N rows are all Normal images from one clinic, which makes a
        # limited run silently unrepresentative and, on a small limit, single-class.
        by = [c for c in ("clinic", "label") if c in df.columns]
        # `GroupBy.sample`, not `GroupBy.apply(lambda g: g.sample(...))`. Since
        # pandas 2.x, `apply` hands the callback a frame with the grouping columns
        # *removed*, so the sampled result silently loses `clinic` and `label` --
        # and the downstream consumer sees every label as the -1 default and fails
        # somewhere far away with a single-class error.
        df = (df.groupby(by, group_keys=False).sample(frac=args.limit / len(df), random_state=0)
              if by else df.sample(args.limit, random_state=0))
        if df.empty:
            df = pd.read_csv(args.manifest).sample(args.limit, random_state=0)
    if df.empty:
        print("manifest selection is empty", file=sys.stderr)
        return 1

    if args.findings:
        FIND.install(FIND.load_findings(args.findings))
    findings = FIND.core()
    spec = FloorSpec(rose_k=args.rose_k, include_anatomical=args.include_anatomical)

    severities = ([0.0] if args.real
                  else [float(s) for s in args.severities.split(",") if s.strip()])

    rows = []
    total = len(df) * len(severities)
    for i, r in enumerate(df.itertuples()):
        for s in severities:
            row = process(r.path, s, not args.real, args.size, findings, spec, args.seed)
            if row is None:
                continue
            row["clinic"] = getattr(r, "clinic", "unknown")
            row["label"] = getattr(r, "label", -1)
            row["split"] = getattr(r, "split", "unknown")
            rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  {len(rows)}/{total}", flush=True)

    if not rows:
        print("no images processed", file=sys.stderr)
        return 1

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"\nwrote {args.out}  ({len(out)} rows)\n")
    print(f"  {'severity':>9} {'abstain':>8} {'insuff':>8} {'marginal':>9} {'detect':>8} "
          f"{'margin_db':>10} {'retake':>7} {'refer':>7}")
    print("  " + "-" * 74)
    for s, g in out.groupby("severity"):
        print(f"  {s:>9.2f} {g['abstained'].mean():>8.2f} "
              f"{(g['certificate'] == 'insufficient').mean():>8.2f} "
              f"{(g['certificate'] == 'marginal').mean():>9.2f} "
              f"{(g['certificate'] == 'detectable').mean():>8.2f} "
              f"{g['margin_db'].median():>10.1f} "
              f"{(g['triage_action'] == 'retake').mean():>7.2f} "
              f"{(g['triage_action'] == 'refer').mean():>7.2f}")

    if out["abstained"].mean() > 0.5 and args.real:
        print("\n  Over half of these images have no optical beam stop, so the certificate\n"
              "  abstained. Run scripts/audit_fiducials.py to see the per-clinic coverage,\n"
              "  and consider the --simulate path for a controlled experiment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
