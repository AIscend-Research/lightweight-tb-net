#!/usr/bin/env python3
"""Resolution dose-response: turn the 512-vs-1024 threshold into a curve.

`scripts/physics_certificates.py --size N` flips its verdicts somewhere between
512 and 1024 on synthetic film, which is a threshold reported as an anecdote.
This script sweeps capture resolution properly and reports the quantity a
deployment actually needs: **the px/mm -- and so the phone sensor resolution --
at which each TB finding's contrast rises above the measured density floor.**

No data, no GPU, no labels: every image is the synthetic film from
`physics/film.py`, so the answer is a statement about the *channel*, not about
any dataset.

Two modes, because "more pixels" is two different physical claims
-----------------------------------------------------------------
Every length in `CaptureParams` is in pixels of the output photo, so changing
`--size` alone silently asserts that the lens, the hand shake and the photon
budget per pixel all improve in lockstep with the sensor. They do not. The two
modes bracket the truth instead of picking one and hoping:

**sampling-limited** (`--mode sampling`) -- blur stays fixed *in pixels* and each
pixel keeps its own full photon well. This is what `--size` does today. It is the
optimistic end: every added pixel is a real gain, and the curve keeps rising.

**optics-limited** (`--mode optics`) -- blur is fixed *in millimetres of film*
(the lens and the operator's hand do not get better because the sensor got
denser, so the PSF and the motion smear scale with px/mm), and the photon budget
is conserved across the frame rather than per pixel (a denser sensor divides the
same light into more, smaller wells, so per-pixel shot noise rises). This is the
pessimistic and more realistic end, and it is the one that can show saturation --
the resolution past which more megapixels buy nothing.

Quote the bracket, not one end of it. If a finding clears the floor at the same
px/mm in both modes, that number is robust to the modelling choice; if the two
disagree, the honest deployment spec is the optics-limited one.

    python scripts/resolution_sweep.py --quick             # ~3 min, one mode, coarse
    python scripts/resolution_sweep.py                     # ~30 min, both modes

Reading the output
------------------
`resolution_sweep.csv` is one row per (mode, image, severity, size).
`summary.json` carries the headline: `crossings`, the px/mm at which each
finding's median margin crosses 0 dB (below it the certificate says the
information is not in the photograph) and +3 dB (above it the certificate calls
the finding carried), with each crossing also expressed as the equivalent phone
sensor resolution in megapixels for a 35.5 x 43.2 cm film filling the frame.

The px/mm axis is `CalibratedFilm.px_per_mm`, which is *inferred* from the
detected collimation field against a standard cassette diagonal at about +-20%
(`physics/invert._px_per_mm`). Every megapixel figure below inherits that 20%,
which is roughly +-40% in area. A ruler in the frame removes it; see
`docs/DEPLOYMENT_CHECKLIST.md` section B2.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from tbtrust.physics.certificate import certify
from tbtrust.physics.film import CaptureParams, capture, sample_params, synthetic_chest_density
from tbtrust.physics.findings import CORE_FINDINGS, get
from tbtrust.physics.floor import FloorSpec
from tbtrust.physics.invert import CASSETTE_MM, invert

# The resolution at which `film.sample_params`' pixel-referenced defaults are
# taken to be calibrated. Scaling in `--mode optics` is relative to this.
REF_SIZE = 512


def scaled_params(p: CaptureParams, size: int, mode: str, ref: int = REF_SIZE) -> CaptureParams:
    """Re-express one draw of capture parameters at a different sensor resolution.

    The draw itself (the same rng stream) is held fixed across the sweep so that
    resolution is the only thing that moves -- otherwise a rougher capture at one
    size would masquerade as a resolution effect.
    """
    if mode == "sampling":
        return p
    k = float(size) / float(ref)
    d = dict(p.__dict__)
    # Angular blur: fixed on the film, so it covers k times as many pixels.
    d["psf_sigma"] = float(p.psf_sigma * k)
    d["motion_length"] = float(p.motion_length * k)
    # Same light, more wells: per-pixel capacity falls as the pixel area does.
    # Read noise is a per-pixel property and stays.
    d["full_well"] = float(p.full_well / (k * k))
    return CaptureParams(**d)


def megapixels(px_per_mm: float) -> float:
    """Sensor pixels needed for a film of `CASSETTE_MM` filling the frame."""
    return float(px_per_mm**2 * CASSETTE_MM[0] * CASSETTE_MM[1] / 1e6)


def run_cell(image: int, severity: float, size: int, mode: str, findings, spec: FloorSpec,
             seed: int) -> dict:
    t0 = time.perf_counter()
    base, ftruth = synthetic_chest_density(size=size, rng=np.random.default_rng(5000 + image))
    # One parameter draw per (image, severity), reused at every size.
    p0 = sample_params(float(severity), np.random.default_rng(9000 + 31 * image + int(1000 * severity)))
    p = scaled_params(p0, size, mode)
    photo, _ = capture(base, p, fiducial_truth=ftruth,
                       rng=np.random.default_rng(seed + 17 * image))
    t1 = time.perf_counter()
    cal = invert(photo)
    t2 = time.perf_counter()
    cert = certify(cal, findings=findings, spec=spec)
    t3 = time.perf_counter()

    row = {
        "mode": mode,
        "image": image,
        "severity": float(severity),
        "size": int(size),
        "px_per_mm": float(cal.px_per_mm),
        "megapixel_equiv": megapixels(cal.px_per_mm),
        "psf_sigma_applied_px": float(p.psf_sigma),
        "motion_length_applied_px": float(p.motion_length),
        "full_well_applied": float(p.full_well),
        **cert.as_dict(),
        "abstained": bool(cert.abstained),
        "t_capture_s": t1 - t0,
        "t_invert_s": t2 - t1,
        "t_certify_s": t3 - t2,
    }
    row.update({f"cal_{k}": v for k, v in cal.summary().items()})
    return row


def _crossing(x: np.ndarray, y: np.ndarray, level: float) -> tuple[float, str]:
    """Smallest x at which y rises through `level`, log-interpolated in x.

    Returns the crossing and a status word, because three different things all
    look like "no number" and only one of them is a failure:

    `crossed`      -- the margin genuinely rises through the level inside the sweep.
    `clear_at_min` -- it was already above the level at the coarsest resolution
                      tried, so the spec is "at most this, and the sweep did not
                      look lower". Not the same as needing that resolution.
    `not_reached`  -- still below the level at the finest resolution tried. Under
                      `--mode optics` this is often the real answer rather than a
                      sweep that stopped too early: the curve saturates, so no
                      resolution recovers the finding.

    The last upward crossing is taken rather than the first point above the
    level: the margin is monotone in resolution only up to sampling noise, and a
    single noisy cell early in the sweep should not be reported as the spec.
    """
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return float("nan"), "no_data"
    if np.all(y < level):
        return float("nan"), "not_reached"
    if np.all(y >= level):
        return float(x[0]), "clear_at_min"
    below = np.nonzero(y < level)[0]
    i = int(below[-1])
    if i + 1 >= x.size:
        return float("nan"), "not_reached"
    lx0, lx1 = np.log10(x[i]), np.log10(x[i + 1])
    y0, y1 = y[i], y[i + 1]
    if y1 == y0:
        return float(x[i + 1]), "crossed"
    return float(10 ** (lx0 + (level - y0) * (lx1 - lx0) / (y1 - y0))), "crossed"


def summarize(df: pd.DataFrame) -> dict:
    out: dict = {"crossings": [], "timing": [], "saturation": []}

    for (mode, sev, key), g in _iter_finding_cells(df):
        med = g.groupby("size").agg(px_per_mm=("px_per_mm", "median"),
                                    margin=(f"margin_db_{key}", "median")).reset_index()
        for level, label in ((0.0, "insufficient_boundary"), (3.0, "detectable_boundary")):
            xc, status = _crossing(med["px_per_mm"].to_numpy(), med["margin"].to_numpy(), level)
            out["crossings"].append({
                "mode": mode, "severity": float(sev), "finding": key, "level_db": level,
                "boundary": label,
                "px_per_mm": xc,
                "megapixels": megapixels(xc) if np.isfinite(xc) else float("nan"),
                "status": status,
                "abstain_frac": float(g[f"margin_db_{key}"].isna().mean()),
                "margin_db_by_size": {int(r["size"]): float(r["margin"]) for _, r in med.iterrows()},
                "px_per_mm_by_size": {int(r["size"]): float(r["px_per_mm"]) for _, r in med.iterrows()},
            })
        # Dose-response slope between the two coarsest and two finest points:
        # a flattening slope is the saturation the optics-limited mode predicts.
        m = med.sort_values("size")
        if len(m) >= 4:
            def slope(a, b):
                dx = np.log2(m["px_per_mm"].iloc[b] / m["px_per_mm"].iloc[a])
                return float((m["margin"].iloc[b] - m["margin"].iloc[a]) / dx) if dx else float("nan")
            out["saturation"].append({
                "mode": mode, "severity": float(sev), "finding": key,
                "db_per_octave_low": slope(0, 1),
                "db_per_octave_high": slope(len(m) - 2, len(m) - 1),
            })

    for (mode, size), g in df.groupby(["mode", "size"]):
        out["timing"].append({
            "mode": mode, "size": int(size),
            "capture_s_median": float(g["t_capture_s"].median()),
            "invert_s_median": float(g["t_invert_s"].median()),
            "certify_s_median": float(g["t_certify_s"].median()),
            "total_s_median": float((g["t_capture_s"] + g["t_invert_s"] + g["t_certify_s"]).median()),
            "n": int(len(g)),
        })
    out["abstain_rate"] = float(df["abstained"].mean())
    out["abstain_by_cell"] = [
        {"mode": m, "severity": float(s), "size": int(z), "abstain_frac": float(g["abstained"].mean()),
         "n": int(len(g))}
        for (m, s, z), g in df.groupby(["mode", "severity", "size"])
    ]
    return out


def _iter_finding_cells(df: pd.DataFrame):
    keys = [c[len("margin_db_"):] for c in df.columns if c.startswith("margin_db_")]
    for (mode, sev), g in df.groupby(["mode", "severity"]):
        for key in keys:
            yield (mode, float(sev), key), g


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/resolution_sweep")
    ap.add_argument("--sizes", default="256,384,512,768,1024,1536,2048")
    ap.add_argument("--severities", default="0.0,0.5,1.0")
    ap.add_argument("--images", type=int, default=3)
    ap.add_argument("--modes", default="sampling,optics",
                    help="sampling = blur fixed in pixels (optimistic); "
                         "optics = blur fixed in mm of film and photons conserved (realistic)")
    ap.add_argument("--findings", default=",".join(CORE_FINDINGS))
    ap.add_argument("--rose-k", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="one mode, three sizes, two severities, one film")
    args = ap.parse_args()

    if args.quick:
        sizes, sevs, modes, n_img = [256, 512, 1024], [0.0, 0.5], ["optics"], 1
    else:
        sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
        sevs = [float(s) for s in args.severities.split(",") if s.strip()]
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        n_img = int(args.images)

    findings = [get(k.strip()) for k in args.findings.split(",") if k.strip()]
    spec = FloorSpec(rose_k=args.rose_k)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    total = len(modes) * n_img * len(sevs) * len(sizes)
    print(f"resolution sweep: {total} cells "
          f"({len(modes)} modes x {n_img} films x {len(sevs)} severities x {len(sizes)} sizes)")
    print(f"contrast table source: {findings[0].source}\n")

    rows, done, t_start = [], 0, time.perf_counter()
    for mode in modes:
        for image in range(n_img):
            for sev in sevs:
                for size in sizes:
                    rows.append(run_cell(image, sev, size, mode, findings, spec, args.seed))
                    done += 1
                    r = rows[-1]
                    print(f"  [{done:>3}/{total}] {mode:<8} film={image} sev={sev:<4} "
                          f"size={size:<5} px/mm={r['px_per_mm']:5.2f} "
                          f"{r['certificate']:<12} margin={r['margin_db']:+6.1f} dB "
                          f"({r['t_capture_s'] + r['t_invert_s'] + r['t_certify_s']:.1f}s)",
                          flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "resolution_sweep.csv", index=False)
    summary = summarize(df)
    summary["config"] = {"sizes": sizes, "severities": sevs, "modes": modes, "images": n_img,
                         "findings": [f.key for f in findings],
                         "contrast_source": findings[0].source, "rose_k": args.rose_k,
                         "ref_size": REF_SIZE, "cassette_mm": list(CASSETTE_MM),
                         "wall_clock_s": time.perf_counter() - t_start}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 84)
    print("resolution at which each finding's contrast clears the measured floor")
    print("  status: crossed = genuine crossing inside the sweep | clear_at_min = already above at "
          "the\n  coarsest size tried | not_reached = still below at the finest size tried\n")
    print(f"  {'mode':<9} {'sev':>4} {'finding':<15} "
          f"{'0 dB: px/mm':>12} {'MP':>6} {'status':<13} {'+3 dB: px/mm':>13} {'MP':>6} {'status':<13}")
    print("  " + "-" * 96)

    def cell(d):
        v, mp = d.get("px_per_mm", float("nan")), d.get("megapixels", float("nan"))
        vs = f"{v:>12.2f}" if np.isfinite(v) else f"{'--':>12}"
        ms = f"{mp:>6.2f}" if np.isfinite(mp) else f"{'--':>6}"
        return f"{vs} {ms} {d.get('status', '--'):<13}"

    for mode in modes:
        for sev in sevs:
            for f in findings:
                c = {x["boundary"]: x for x in summary["crossings"]
                     if x["mode"] == mode and x["severity"] == sev and x["finding"] == f.key}
                print(f"  {mode:<9} {sev:>4.2f} {f.key:<15} "
                      f"{cell(c.get('insufficient_boundary', {}))} "
                      f"{cell(c.get('detectable_boundary', {}))}")
    print(f"\n  abstained (no measurable beam stop): {summary['abstain_rate']:.0%} of cells")
    print("\n  measured cost per image (median seconds, single CPU core)")
    for t in sorted(summary["timing"], key=lambda t: (t["mode"], t["size"])):
        print(f"    {t['mode']:<9} size={t['size']:<5} capture={t['capture_s_median']:5.2f} "
              f"invert={t['invert_s_median']:6.2f} certify={t['certify_s_median']:5.2f} "
              f"total={t['total_s_median']:6.2f}")
    print(f"\n  NOTE: absolute verdicts inherit the finding-contrast table "
          f"({findings[0].source}); the px/mm axis carries the +-20% px_per_mm inference error.")
    print(f"\nwrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
