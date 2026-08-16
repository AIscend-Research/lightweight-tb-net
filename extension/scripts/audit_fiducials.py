#!/usr/bin/env python3
"""Do the public archives still carry the calibration targets? Run this first.

The entire physics track rests on one empirical assumption: that enough chest
radiographs in Montgomery, Shenzhen, NIAID and RSNA retain a lead side marker, a
collimation border and a direct-exposure region. Whoever assembled each archive
was free to crop, window and rescale, and a tight crop to the lung fields removes
all three in one stroke.

So this is the load-bearing check, and it is deliberately the cheapest thing in
the repo to run -- no model, no GPU, no training, a few milliseconds per image.
Run it before you build anything on top.

    python scripts/audit_fiducials.py --manifest data/processed/manifest.csv --out outputs/fiducial_audit
    python scripts/audit_fiducials.py --raw data/raw --limit 500        # no manifest yet

It writes:
  fiducial_audit.csv      one row per image, every detector diagnostic
  fiducial_audit.json     per-clinic coverage summary
  fiducial_audit.png      coverage bar chart + a contact sheet of detections

How to read the result
----------------------
`coverage=full` means marker + beam stop + a usable slanted edge: the whole
inversion is available. `partial` means a beam stop but no marker or no edge --
glare is still measured directly, which is the dominant term, but the tone scale
leans on the gamma prior. `none` means no beam stop, so the certificate must
abstain, and those images are simply outside this method's reach.

A low `full` rate is a finding, not a failure. It tells you which archives can
support the physics track and which cannot, and the honest paper reports the
coverage table alongside the results rather than quietly evaluating on the subset
that happened to work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from PIL import Image

from tbtrust.physics import fiducials as FID

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _load(path: str, max_side: int) -> np.ndarray | None:
    """Load greyscale, downscaled so the audit stays fast on 3000px archives.

    Downscaling is safe for *detection* -- a collimation border and a lead marker
    are large structures -- but it is not safe for the PSF, so the audit reports
    whether a usable edge exists rather than what its MTF is. `physics_certificates.py`
    works at native resolution for that reason.
    """
    try:
        img = Image.open(path).convert("L")
    except (OSError, ValueError):
        return None
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BILINEAR)
    return np.asarray(img)


def rows_from_manifest(manifest: Path, limit: int | None, max_side: int) -> list[dict]:
    df = pd.read_csv(manifest)
    if limit and limit < len(df):
        # Sample per clinic rather than head(): a manifest is built by walking the
        # dataset directory, so it arrives grouped by source and by class, and
        # head() would audit one clinic's normals and report it as the corpus.
        #
        # `GroupBy.sample`, not `GroupBy.apply(lambda g: g.sample(...))`: since
        # pandas 2.x, `apply` hands the callback a frame with the grouping columns
        # removed, so the result would come back without a `clinic` column and the
        # per-clinic table below would be empty.
        df = df.groupby("clinic", group_keys=False).sample(frac=limit / len(df), random_state=0)
    out = []
    for r in df.itertuples():
        img = _load(r.path, max_side)
        if img is None:
            continue
        f = FID.detect(img)
        out.append({"path": r.path, "clinic": getattr(r, "clinic", "unknown"),
                    "label": getattr(r, "label", -1), **f.summary()})
    return out


def rows_from_raw(raw: Path, limit: int | None, max_side: int) -> list[dict]:
    from tbtrust.data.manifest import infer_clinic_from_path

    paths = sorted(p for p in raw.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if limit:
        paths = paths[:: max(1, len(paths) // limit)][:limit]
    out = []
    for p in paths:
        img = _load(str(p), max_side)
        if img is None:
            continue
        f = FID.detect(img)
        out.append({"path": str(p), "clinic": infer_clinic_from_path(str(p)), "label": -1, **f.summary()})
    return out


def summarize(df: pd.DataFrame) -> dict:
    def _block(g: pd.DataFrame) -> dict:
        n = len(g)
        return {
            "n": n,
            "coverage_full": float((g["coverage"] == "full").mean()),
            "coverage_partial": float((g["coverage"] == "partial").mean()),
            "coverage_none": float((g["coverage"] == "none").mean()),
            "has_marker": float(g["has_marker"].mean()),
            "has_beamstop": float(g["has_beamstop"].mean()),
            "usable_mtf_edge": float((g["n_mtf_edges"] > 0).mean()),
            "beamstop_from_collimation": float((g["beamstop_source"] == "collimated_rim").mean()),
            "beamstop_from_dark_surround": float((g["beamstop_source"] == "dark_surround").mean()),
            "median_marker_confidence": float(g["marker_confidence"].median()),
            # The operational number: what fraction of this clinic's images the
            # certificate can say anything at all about.
            "certifiable": float((g["coverage"] != "none").mean()),
        }

    return {"overall": _block(df), "per_clinic": {c: _block(g) for c, g in df.groupby("clinic")}}


def plot(df: pd.DataFrame, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clinics = sorted(df["clinic"].unique())
    order = ["full", "partial", "none"]
    frac = np.array([[float((df[df.clinic == c]["coverage"] == k).mean()) for k in order] for c in clinics])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bottom = np.zeros(len(clinics))
    for j, k in enumerate(order):
        axes[0].bar(clinics, frac[:, j], bottom=bottom, label=k)
        bottom += frac[:, j]
    axes[0].set_ylabel("fraction of images")
    axes[0].set_title("Fiducial coverage per clinic")
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis="x", rotation=20)

    for c in clinics:
        v = df[df.clinic == c]["marker_confidence"].to_numpy()
        if v.size:
            axes[1].hist(v, bins=25, histtype="step", label=f"{c} (n={v.size})")
    axes[1].axvline(0.5, color="k", ls="--", lw=1)
    axes[1].set_xlabel("lead-marker detection confidence")
    axes[1].set_ylabel("images")
    axes[1].set_title("Marker confidence (0.5 = accepted)")
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="data/processed/manifest.csv")
    src.add_argument("--raw", help="directory of images, clinic inferred from filename")
    ap.add_argument("--out", default="outputs/fiducial_audit", help="output prefix")
    ap.add_argument("--limit", type=int, default=None, help="audit at most this many images")
    ap.add_argument("--max-side", type=int, default=768,
                    help="downscale longest side before detection (detection only, not MTF)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    rows = (rows_from_manifest(Path(args.manifest), args.limit, args.max_side)
            if args.manifest else rows_from_raw(Path(args.raw), args.limit, args.max_side))
    if not rows:
        print("no images found", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    summary = summarize(df)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    if not args.no_plot:
        plot(df, out.with_suffix(".png"))

    o = summary["overall"]
    print(f"\naudited {o['n']} images\n")
    print(f"  {'clinic':<14} {'n':>6} {'full':>7} {'partial':>8} {'none':>7} {'marker':>8} {'edge':>7}")
    print(f"  {'-' * 14} {'-' * 6} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 8} {'-' * 7}")
    for c, b in sorted(summary["per_clinic"].items()):
        print(f"  {c:<14} {b['n']:>6} {b['coverage_full']:>7.2f} {b['coverage_partial']:>8.2f} "
              f"{b['coverage_none']:>7.2f} {b['has_marker']:>8.2f} {b['usable_mtf_edge']:>7.2f}")
    print(f"  {'-' * 14} {'-' * 6} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 8} {'-' * 7}")
    print(f"  {'ALL':<14} {o['n']:>6} {o['coverage_full']:>7.2f} {o['coverage_partial']:>8.2f} "
          f"{o['coverage_none']:>7.2f} {o['has_marker']:>8.2f} {o['usable_mtf_edge']:>7.2f}")
    print(f"\n  certifiable (coverage != none): {o['certifiable']:.1%}")
    if o["certifiable"] < 0.5:
        print("\n  Most images in this corpus lack an optical beam stop, so the physics\n"
              "  certificate must abstain on them. That is a result to report, not a bug:\n"
              "  say so in the paper and evaluate the physics track on the certifiable\n"
              "  subset, stating its size. The simulated re-photography path in\n"
              "  physics/film.py paints fiducials back on and remains fully available.")
    print(f"\nwrote {out.with_suffix('.csv')}, {out.with_suffix('.json')}"
          + ("" if args.no_plot else f", {out.with_suffix('.png')}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
