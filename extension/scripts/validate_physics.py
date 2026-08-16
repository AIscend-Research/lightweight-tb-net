#!/usr/bin/env python3
"""Run the experiments that could falsify the physics track. No data, no GPU.

Three checks, in the order they should be believed:

1. **Fiducial recovery** -- does the detector find the marker and the collimation
   corners where they actually are?
2. **Channel recovery** -- handed only an 8-bit JPEG, does the blind estimator
   recover the veil fraction, the PSF width and the density map that generated it?
3. **Detectability** -- and the one that matters: does the density resolution floor
   predict the contrast at which an *optimal* detector starts to fail? This is the
   claim that makes the certificate a physical bound rather than another confidence
   score, and it is free to come out wrong.

    python scripts/validate_physics.py --quick        # ~1 minute
    python scripts/validate_physics.py                # ~10 minutes, publication run

Writes CSVs and a summary JSON to --out. Exit code is 1 if the detectability
calibration falls outside the tolerance band, so this can gate CI.

Interpreting the calibration ratio
----------------------------------
`ratio = predicted_floor / empirical_threshold`. Above 1 the bound is conservative
-- it declares information lost slightly before an optimal detector actually
loses it. Below 1 it is optimistic, which is the dangerous direction, because the
certificate would pass a photograph that cannot carry the finding. Report the
median ratio with the results; a bound quoted without its measured calibration is
just an assertion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from tbtrust.physics import validate as V
from tbtrust.physics.findings import CORE_FINDINGS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/physics_validation")
    ap.add_argument("--quick", action="store_true", help="small run for CI / a smoke check")
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--trials", type=int, default=None, help="paired captures per contrast level")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance", type=float, nargs=2, default=(0.4, 3.0),
                    help="acceptable band for the median predicted/empirical ratio")
    args = ap.parse_args()

    quick = args.quick
    size = args.size or (192 if quick else 320)
    trials = args.trials or (8 if quick else 24)
    n_images = 2 if quick else 8
    severities = (0.0, 0.5) if quick else (0.0, 0.25, 0.5, 0.75, 1.0)
    det_sevs = (0.0, 0.5) if quick else (0.0, 0.25, 0.5, 0.75)
    findings = ("infiltrate",) if quick else CORE_FINDINGS

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"config": {"size": size, "trials": trials, "n_images": n_images,
                                "severities": list(severities), "quick": quick}}

    # ---------------------------------------------------------------- 1. fiducials
    print("1/3  fiducial recovery ...", flush=True)
    fid = pd.DataFrame(V.fiducial_recovery(n_images=n_images, size=size, seed=args.seed,
                                           severities=severities))
    fid.to_csv(out / "fiducial_recovery.csv", index=False)
    summary["fiducials"] = {
        "coverage_full_frac": float((fid["coverage"] == "full").mean()),
        "marker_iou_median": float(fid["marker_iou"].median(skipna=True)),
        "corner_err_px_median": float(fid["corner_err_px"].median(skipna=True)),
        "usable_mtf_edge_frac": float((fid["n_mtf_edges"] > 0).mean()),
    }
    print("     " + json.dumps(summary["fiducials"]))

    # ----------------------------------------------------------------- 2. channel
    print("2/3  channel recovery ...", flush=True)
    rec = pd.DataFrame(V.recovery_experiment(n_images=n_images, severities=severities,
                                             size=size, seed=args.seed))
    rec.to_csv(out / "channel_recovery.csv", index=False)
    summary["channel"] = V.summarize_recovery(rec.to_dict("records"))
    print("     " + json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                for k, v in summary["channel"].items()}))

    # ----------------------------------------------------------- 3. detectability
    print("3/3  detectability (the falsification test) ...", flush=True)
    det_rows = []
    for sev in det_sevs:
        for f in findings:
            r = V.detectability_experiment(severity=sev, finding=f, n_trials=trials,
                                           size=size, seed=args.seed)
            det_rows.append(r.as_dict())
            print(f"     sev={sev:<5} {f:<16} predicted={r.predicted_floor:8.4f} "
                  f"empirical={r.empirical_threshold:8.4f} ratio={r.ratio:6.2f} "
                  f"r2={r.linearity_r2:5.2f} {'PASS' if r.passes else 'fail'}", flush=True)
    det = pd.DataFrame(det_rows)
    det.to_csv(out / "detectability.csv", index=False)

    finite = det[np.isfinite(det["ratio"]) & (det["ratio"] > 0)]
    med = float(finite["ratio"].median()) if len(finite) else float("nan")
    summary["detectability"] = {
        "n_conditions": len(det),
        "n_measurable": len(finite),
        "median_ratio": med,
        "iqr_ratio": [float(finite["ratio"].quantile(0.25)), float(finite["ratio"].quantile(0.75))]
        if len(finite) > 1 else None,
        "pass_frac": float(det["passes"].mean()),
        "median_linearity_r2": float(finite["linearity_r2"].median()) if len(finite) else float("nan"),
    }

    # --------------------------------------------------------------- 4. ordering
    print("     certificate ordering ...", flush=True)
    cons = pd.DataFrame(V.certificate_consistency(severities=severities,
                                                  n_images=max(2, n_images // 2), size=size,
                                                  seed=args.seed))
    cons.to_csv(out / "certificate_consistency.csv", index=False)
    by_sev = cons.groupby("severity")["margin_db"].median()
    monotone = bool(np.all(np.diff(by_sev.to_numpy()) <= 1e-9))
    summary["ordering"] = {
        "margin_db_by_severity": {float(k): float(v) for k, v in by_sev.items()},
        "margin_falls_monotonically": monotone,
        "insufficient_frac_by_severity": {
            float(k): float((g["certificate"] == "insufficient").mean())
            for k, g in cons.groupby("severity")
        },
    }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lo, hi = args.tolerance
    ok = bool(np.isfinite(med) and lo <= med <= hi) and monotone
    print("\n" + "=" * 66)
    print(f"median predicted/empirical ratio : {med:.2f}   (tolerance {lo}-{hi})")
    print(f"certificate margin monotone in severity : {monotone}")
    print(f"conditions passing individually  : {summary['detectability']['pass_frac']:.0%}")
    if np.isfinite(med) and med > 1:
        print("the bound is conservative -- it calls information lost slightly before an\n"
              "optimal detector actually loses it, which is the safe direction")
    elif np.isfinite(med):
        print("WARNING: the bound is OPTIMISTIC -- it certifies photographs an optimal\n"
              "detector cannot actually read. Do not deploy the certificate as a safety\n"
              "valve until this is above 1.")
    print(f"\nwrote {out}/")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
