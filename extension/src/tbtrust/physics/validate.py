"""Does any of this actually work? Two experiments that can fail.

The physics track makes two claims, and both are checkable in simulation where
ground truth exists by construction.

**Claim 1 -- the blind estimator recovers the channel.** Photograph a known film
with known glare, blur and tone curve, hand the estimator only the 8-bit JPEG, and
compare. `recovery_experiment` scores veil fraction, PSF width and density-map
error against truth across the severity sweep.

**Claim 2 -- the floor predicts detectability.** This is the one that matters, and
it is the reason the certificate is not just another confidence score.
`detectability_experiment` inserts lesions of known contrast, runs them through
the simulated capture, and measures how well an *optimal* detector -- a matched
filter that already knows the lesion's position, size and shape, so no real system
can beat it -- can tell present from absent. It then asks whether the empirical
detectability threshold lands where the blindly-computed floor said it would.

Why claim 2 is a real test and not a tautology
----------------------------------------------
A fair objection: the floor is *defined* as the matched-filter threshold, so of
course a matched filter hits it. But the floor is computed from quantities the
estimator had to *measure blind* -- sigma_D from a noise model fitted to the
image, E from an MTF read off the collimation edge, and the veil amplification
from the beam stop. The empirical threshold is measured from the detector's actual
behaviour on actual noisy captures. Nothing forces them to agree. If the veil is
under-measured, the floor comes out optimistic and the empirical threshold lands
above it. If the MTF is over-smoothed, the floor is pessimistic and the threshold
lands below. The ratio of the two is a direct score on the estimator, and it is
free to be wrong -- which is the whole point.

The experiments run on the synthetic film by default, so they need no data and no
GPU, and `scripts/validate_physics.py` runs them on a laptop in a couple of
minutes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _ops
from .certificate import certify
from .density import FilmModel
from .film import CaptureParams, capture, insert_lesion, lesion_template, sample_params, synthetic_chest_density
from .findings import FindingSpec, get
from .floor import FloorSpec, density_floor
from .invert import invert

# --------------------------------------------------------------------------- #
# claim 1: channel recovery
# --------------------------------------------------------------------------- #


def _veil_fraction_est(cal) -> float:
    m = cal.lung_field_mask()
    if not m.any():
        return float("nan")
    return float(np.median(cal.veil[m] / np.maximum(cal.signal[m], 1e-9)))


def recovery_experiment(
    n_images: int = 12,
    severities: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    size: int = 384,
    seed: int = 0,
    film: FilmModel | None = None,
) -> list[dict]:
    """Score the blind estimator against ground truth across the severity sweep.

    Returns one row per (image, severity). The columns to look at first are
    `veil_fraction_err` and `psf_sigma_rel_err`: those two carry the certificate.
    `density_diff_rmse` is the differential density error -- density measured
    relative to a local mean -- which is the quantity the floor actually bounds,
    as opposed to `density_abs_rmse`, which is dominated by the gamma prior and is
    reported mostly to show that it is.
    """
    rng = np.random.default_rng(seed)
    film = film or FilmModel()
    rows: list[dict] = []

    for i in range(int(n_images)):
        base, ftruth = synthetic_chest_density(size=size, rng=np.random.default_rng(1000 + i), film=film)
        for s in severities:
            params = sample_params(float(s), rng)
            # Geometry is exercised here but not in the detectability test, which
            # isolates the radiometric claim; see that function's docstring.
            photo, truth = capture(base, params, fiducial_truth=ftruth, film=film,
                                   rng=np.random.default_rng(rng.integers(1 << 31)))
            cal = invert(photo, film=film)

            m = cal.lung_field_mask()
            d_true_photo = _ops.warp_perspective(base, truth.homography, base.shape, fill=film.d_min)
            if m.any():
                err = cal.density[m] - d_true_photo[m]
                abs_rmse = float(np.sqrt(np.mean(err**2)))
                # Differential error: remove the common offset the gamma prior
                # causes, which cancels in any density difference.
                diff_rmse = float(np.std(err))
            else:
                abs_rmse = diff_rmse = float("nan")

            vt, ve = truth.veil_fraction_true, _veil_fraction_est(cal)
            rows.append({
                "image": i,
                "severity": float(s),
                "coverage": cal.fiducials.coverage.value,
                "marker_found": bool(cal.fiducials.has_bright_anchor),
                "n_mtf_edges": len(cal.fiducials.mtf_edges),
                "psf_sigma_true": truth.psf_sigma_effective,
                "psf_sigma_est": float(cal.psf.sigma),
                "psf_sigma_rel_err": float((cal.psf.sigma - truth.psf_sigma_effective)
                                           / max(truth.psf_sigma_effective, 1e-6)),
                "psf_method": cal.psf.method,
                "veil_fraction_true": vt,
                "veil_fraction_est": ve,
                "veil_fraction_err": float(ve - vt) if np.isfinite(ve) and np.isfinite(vt) else float("nan"),
                "glare_method": cal.glare.method,
                "gamma_true": params.tone_gamma,
                "gamma_est": float(cal.tone.gamma),
                "tone_method": cal.tone.method,
                "density_abs_rmse": abs_rmse,
                "density_diff_rmse": diff_rmse,
            })
    return rows


def fiducial_recovery(n_images: int = 12, size: int = 384, seed: int = 0,
                      severities: tuple[float, ...] = (0.0, 0.5, 1.0)) -> list[dict]:
    """Score the *detector* against where the fiducials truly are.

    Separate from `recovery_experiment` because a detection failure and an
    estimation failure need different fixes, and lumping them together hides which
    one is happening. Reports marker IoU and the collimation corner error in
    pixels.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(int(n_images)):
        base, ftruth = synthetic_chest_density(size=size, rng=np.random.default_rng(2000 + i))
        for s in severities:
            params = sample_params(float(s), rng)
            photo, truth = capture(base, params, fiducial_truth=ftruth,
                                   rng=np.random.default_rng(rng.integers(1 << 31)))
            cal_fid = invert(photo, iterations=1).fiducials

            iou = float("nan")
            if cal_fid.marker_mask is not None:
                tm = _ops.warp_perspective(ftruth["marker_mask"].astype(float), truth.homography,
                                           photo.shape, fill=0.0) > 0.5
                inter = float((cal_fid.marker_mask & tm).sum())
                union = float((cal_fid.marker_mask | tm).sum())
                iou = inter / union if union > 0 else float("nan")

            corner_err = float("nan")
            if cal_fid.field_quad is not None:
                tq = _ops.order_quad(truth.field_quad_photo)
                eq = _ops.order_quad(cal_fid.field_quad)
                corner_err = float(np.mean(np.hypot(*(eq - tq).T)))

            rows.append({
                "image": i, "severity": float(s),
                "coverage": cal_fid.coverage.value,
                "marker_iou": iou,
                "marker_confidence": float(cal_fid.marker_confidence),
                "corner_err_px": corner_err,
                "n_edges": len(cal_fid.edges),
                "n_mtf_edges": len(cal_fid.mtf_edges),
                "beamstop_source": cal_fid.beamstop_source,
            })
    return rows


# --------------------------------------------------------------------------- #
# claim 2: the floor predicts detectability
# --------------------------------------------------------------------------- #


@dataclass
class DetectabilityResult:
    """Predicted floor versus the threshold an optimal detector actually achieves."""

    severity: float
    finding: str
    predicted_floor: float          # floor at the lesion site
    floor_lung_median: float        # floor over the whole lung field, for context
    empirical_threshold: float
    ratio: float                    # predicted / empirical; 1.0 is perfect
    dprime_slope: float             # d' per unit delta D
    linearity_r2: float             # d' should be linear in delta D; check that it is
    n_trials: int
    deltas: list[float]
    dprimes: list[float]
    veil_fraction: float
    psf_sigma: float

    @property
    def passes(self) -> bool:
        """Within a factor of two, and d' genuinely linear in contrast.

        A factor of two on a bound derived blind from an 8-bit JPEG is the right
        tolerance: it is tight enough that a mis-measured veil or a wrong MTF
        fails the test, and loose enough not to trip on the template-shape
        mismatch between the Gaussian profile the floor assumes and the lesion
        that was actually inserted.

        The linearity gate is deliberately the looser of the two. It exists to
        catch a detector that is not behaving linearly at all, not to measure R^2
        precisely -- with a couple of dozen trials per contrast the estimate is
        itself noisy, and gating at 0.9 would fail runs that are perfectly sound.
        Raise `n_trials` before reading much into a value near the threshold.
        """
        return bool(0.5 <= self.ratio <= 2.0 and self.linearity_r2 >= 0.85)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("deltas", "dprimes")}
        d["passes"] = self.passes
        return d


def _matched_filter_weights(shape, center, size_px, psf_sigma):
    """The optimal linear detector for a known lesion: its blurred template.

    Zero-mean so the statistic is insensitive to a local density offset, and
    unit-norm so its scale is fixed; both matter because the statistic is compared
    across trials whose absolute calibration differs slightly.
    """
    t = lesion_template(shape, center, size_px)
    tb = _ops.gaussian_blur(t, max(psf_sigma, 1e-3))
    r = max(3.0 * size_px, 8.0)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    local = np.hypot(yy - center[0], xx - center[1]) <= r
    w = np.where(local, tb, 0.0)
    w = w - (w[local].mean() if local.any() else 0.0) * local
    n = np.sqrt((w**2).sum())
    return w / n if n > 0 else w


def detectability_experiment(
    severity: float = 0.5,
    finding: FindingSpec | str = "infiltrate",
    deltas: tuple[float, ...] | None = None,
    n_trials: int = 24,
    size: int = 320,
    seed: int = 0,
    film: FilmModel | None = None,
    spec: FloorSpec | None = None,
    params: CaptureParams | None = None,
) -> DetectabilityResult:
    """Measure the empirical detectability threshold and compare it to the floor.

    Protocol, per contrast level: `n_trials` paired captures of the same film with
    and without a lesion, differing only in sensor noise. The matched filter is
    applied to the *recovered density map* of each, and d' is computed between the
    two populations. Because both arms share the identical anatomy, the anatomical
    background contributes the same offset to both and cancels exactly in d' --
    which is what makes this a clean measurement of the *channel's* detectability
    rather than of how confusing ribs are.

    Geometry is switched off (no rotation, no keystone). The claim under test is
    radiometric, and resampling a warped image would add an interpolation blur
    that neither the floor nor the truth accounts for, muddying a test that is
    meant to be sharp. `recovery_experiment` exercises geometry separately.
    """
    film = film or FilmModel()
    spec = spec or FloorSpec()
    f = get(finding) if isinstance(finding, str) else finding
    rng = np.random.default_rng(seed)

    base, ftruth = synthetic_chest_density(size=size, rng=np.random.default_rng(7), film=film)
    p = params or sample_params(float(severity), rng)
    p = CaptureParams(**{**p.__dict__, "rotation_deg": 0.0, "keystone": 0.0})

    # A lesion site in the middle of the lung density band, clear of fiducials.
    #
    # Not the darkest lung pixels, which is the obvious choice and the wrong one:
    # the floor scales as 1/signal, so the darkest pixels carry a floor an order of
    # magnitude above the lung-field median, and siting every lesion there tests
    # the pipeline's worst corner while reporting it as the typical case. The
    # middle of the band is representative, and `floor_at_site` versus
    # `floor_lung_median` in the result makes the difference visible either way.
    inner = np.zeros(base.shape, dtype=bool)
    q = int(0.3 * size)
    inner[q : size - q, q : size - q] = True
    lo, hi = np.quantile(base[inner], [0.40, 0.70])
    cand = inner & (base >= lo) & (base <= min(hi, film.d_max - 0.3))
    ys, xs = np.nonzero(cand)
    if ys.size == 0:
        ys, xs = np.nonzero(inner)
    k = int(rng.integers(len(ys)))
    center = (float(ys[k]), float(xs[k]))

    # Reference inversion of the lesion-free film: this is what a deployed system
    # would have, and it is where the *predicted* floor comes from.
    ref_photo, _ = capture(base, p, fiducial_truth=ftruth, film=film, rng=np.random.default_rng(11))
    cal_ref = invert(ref_photo, film=film)
    size_px = f.size_px(cal_ref.px_per_mm)
    fm = density_floor(cal_ref, f, spec)
    disc = np.zeros(base.shape, dtype=bool)
    yy, xx = np.mgrid[0:size, 0:size]
    disc[np.hypot(yy - center[0], xx - center[1]) <= max(size_px, 4.0)] = True
    predicted = float(np.median(fm.floor[disc]))
    lung_median = float(np.median(fm.floor[cal_ref.lung_field_mask()]))

    w = _matched_filter_weights(base.shape, center, size_px, cal_ref.psf.sigma)

    if deltas is None:
        # Bracket the prediction so the linearity check has leverage on both sides,
        # but keep every probe inside a density step a real lesion could produce.
        # When the predicted floor is itself larger than any physical contrast the
        # honest answer is that the finding is unmeasurable in this capture, not a
        # simulation of an impossible lesion.
        deltas = tuple(min(predicted * mult, 1.0) for mult in (0.5, 1.0, 2.0))
        deltas = tuple(sorted(set(d for d in deltas if d > 1e-4)))
        if len(deltas) < 2:
            return DetectabilityResult(
                float(severity), f.key, predicted, lung_median, float("inf"), 0.0, 0.0, 0.0,
                int(n_trials), list(deltas), [],
                _veil_fraction_est(cal_ref), float(cal_ref.psf.sigma),
            )

    dprimes, used = [], []
    for dd in deltas:
        lesioned = insert_lesion(base, center, size_px, float(dd))
        ta, tp = [], []
        for t in range(int(n_trials)):
            r_a = np.random.default_rng(int(rng.integers(1 << 31)))
            r_p = np.random.default_rng(int(rng.integers(1 << 31)))
            pa, _ = capture(base, p, fiducial_truth=ftruth, film=film, rng=r_a)
            pp, _ = capture(lesioned, p, fiducial_truth=ftruth, film=film, rng=r_p)
            # Reuse the reference fiducials: they are a property of the film, not
            # of the noise, and re-detecting them 2*n_trials times would dominate
            # the runtime without changing the answer.
            ca = invert(pa, film=film, fid=cal_ref.fiducials, iterations=1)
            cp = invert(pp, film=film, fid=cal_ref.fiducials, iterations=1)
            ta.append(float((w * ca.density).sum()))
            tp.append(float((w * cp.density).sum()))
            del t
        a, b = np.asarray(ta), np.asarray(tp)
        pooled = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
        if pooled <= 0:
            continue
        # The lesion lowers density, so the present-arm statistic moves negative
        # for a positive delta; take the magnitude.
        dprimes.append(float(abs(b.mean() - a.mean()) / pooled))
        used.append(float(dd))

    if len(used) < 2:
        return DetectabilityResult(
            float(severity), f.key, predicted, lung_median, float("nan"), float("nan"),
            float("nan"), float("nan"), int(n_trials), list(used), list(dprimes),
            _veil_fraction_est(cal_ref), float(cal_ref.psf.sigma),
        )

    # d' is linear in contrast through the origin for a linear detector, so fit a
    # slope with no intercept; the R^2 against that model is itself a check that
    # the detector is behaving as the theory says.
    x, y = np.asarray(used), np.asarray(dprimes)
    slope = float((x @ y) / (x @ x))
    ss_res = float(((y - slope * x) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    empirical = float(spec.rose_k / slope) if slope > 0 else float("inf")

    return DetectabilityResult(
        severity=float(severity),
        finding=f.key,
        predicted_floor=predicted,
        floor_lung_median=lung_median,
        empirical_threshold=empirical,
        ratio=float(predicted / empirical) if np.isfinite(empirical) and empirical > 0 else float("nan"),
        dprime_slope=slope,
        linearity_r2=float(r2),
        n_trials=int(n_trials),
        deltas=list(used),
        dprimes=list(dprimes),
        veil_fraction=_veil_fraction_est(cal_ref),
        psf_sigma=float(cal_ref.psf.sigma),
    )


def certificate_consistency(
    severities: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    n_images: int = 6,
    size: int = 320,
    seed: int = 0,
) -> list[dict]:
    """Sanity check the certificate's *ordering*, which needs no ground-truth contrasts.

    Whatever the finding table says, three things must hold or the pipeline is
    broken: the floor rises monotonically with capture severity, the margin falls,
    and the fraction of images certified INSUFFICIENT increases. These hold
    independently of whether the nominal contrasts in `findings.py` are right,
    which makes this the check to run when someone swaps that table.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(int(n_images)):
        base, ftruth = synthetic_chest_density(size=size, rng=np.random.default_rng(3000 + i))
        for s in severities:
            p = sample_params(float(s), rng)
            photo, _ = capture(base, p, fiducial_truth=ftruth,
                               rng=np.random.default_rng(int(rng.integers(1 << 31))))
            cal = invert(photo)
            cert = certify(cal)
            rows.append({
                "image": i, "severity": float(s),
                **cert.as_dict(),
                "veil_fraction": _veil_fraction_est(cal),
                "psf_sigma": float(cal.psf.sigma),
            })
    return rows


def summarize_recovery(rows: list[dict]) -> dict:
    """Collapse `recovery_experiment` output into the numbers worth quoting."""
    import statistics as st

    def _col(name):
        return [r[name] for r in rows if isinstance(r.get(name), float) and np.isfinite(r[name])]

    out = {"n_rows": len(rows)}
    for name in ("veil_fraction_err", "psf_sigma_rel_err", "density_abs_rmse", "density_diff_rmse"):
        vals = _col(name)
        if vals:
            out[f"{name}_median"] = float(st.median(vals))
            out[f"{name}_mad"] = float(st.median([abs(v - st.median(vals)) for v in vals]))
            out[f"{name}_abs_median"] = float(st.median([abs(v) for v in vals]))
    cov = [r["coverage"] for r in rows]
    out["coverage_full_frac"] = cov.count("full") / max(len(cov), 1)
    return out
