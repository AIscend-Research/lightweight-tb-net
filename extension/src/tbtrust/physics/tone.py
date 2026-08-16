"""Two-point densitometry: pin the phone's unknown tone curve with the film's anchors.

A phone's image signal processor applies a tone curve nobody publishes, that
varies with scene, exposure and manufacturer, and that is baked into the JPEG
before you ever see it. Undo it and pixel values become luminance; leave it and
every density you compute is wrong by an unknown, scene-dependent, *nonlinear*
amount. Densitometry has always solved this the same way -- measure a target of
known density -- and a chest film carries two:

    lead marker, and film outside the collimation border   ->  D_min (base+fog)
    direct-exposure region                                 ->  D_max

Two anchors, so two free parameters
-----------------------------------
The estimator's model is a three-parameter monotone power law

    v  =  c0  +  c1 * L**(1/gamma)

and two anchors determine exactly two of them. Something has to give, and the
honest choice is to pin `gamma` to a prior -- sRGB's 2.2, which every consumer
ISP is built around -- solve `c0` and `c1` exactly, and then *propagate the
prior's uncertainty into the reported density* rather than pretending it is zero.
`ToneModel.gamma_sigma` is that admission, and `density_map` splits its
consequence out as a systematic term.

The reason this is tolerable, and it is the crux of the whole approach: a gamma
error is a *scale* error on density, and the quantity that matters -- the
resolution floor in `floor.py` -- is a bound on a density *difference* measured
between a lesion and the lung field a few millimetres away. A common scale error
moves the floor and the finding contrast together and largely cancels. Absolute
density is uncertain at the few-percent level; density *differences* are not, and
differences are what radiology reads.

Third anchors, if you want gamma for real
-----------------------------------------
`fit_tone` fits gamma properly the moment a third distinct density is available.
The cheapest way to get one in the field is to tape a small step wedge to the
lightbox beside the film -- a few pence of exposed, processed film, and it turns
the weakest assumption in this pipeline into a measurement. That is the one
hardware ask in `docs/DEPLOYMENT_CHECKLIST.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _ops
from .density import FilmModel, density_to_transmittance
from .fiducials import MARKER_ACCEPT

# sRGB encoding gamma. Consumer ISPs deviate with contrast S-curves and scene
# adaptation, hence the generous prior width rather than a tight one.
GAMMA_PRIOR = 2.2
GAMMA_PRIOR_SIGMA = 0.35


@dataclass
class Anchor:
    """One region of known optical density, reduced to a pixel-value measurement."""

    name: str
    pixel_mean: float
    pixel_sigma: float
    density: float
    n_px: int
    veil_luminance: float = 0.0     # veil sitting on this anchor, in luminance units
    illumination: float = 1.0       # lightbox level at this anchor, relative

    def target_luminance(self, film: FilmModel) -> float:
        """Total luminance this anchor should present: film transmission plus veil."""
        _ = film
        return self.illumination * float(density_to_transmittance(self.density)) + self.veil_luminance


@dataclass
class ToneModel:
    """v = c0 + c1 * L**(1/gamma), monotone and invertible on [c0, c0+c1]."""

    c0: float
    c1: float
    gamma: float = GAMMA_PRIOR
    gamma_sigma: float = GAMMA_PRIOR_SIGMA
    c1_sigma: float = 0.0
    c0_sigma: float = 0.0
    n_anchors: int = 0
    method: str = "two_point"       # 'three_point' | 'two_point' | 'one_point' | 'prior'
    residual: float = 0.0
    diagnostics: dict = field(default_factory=dict)

    def to_pixel(self, luminance):
        L = np.maximum(np.asarray(luminance, dtype=np.float64), 0.0)
        return self.c0 + self.c1 * L ** (1.0 / self.gamma)

    def to_luminance(self, pixel):
        """Inverse tone curve. Values at or below the black point map to 0."""
        v = np.asarray(pixel, dtype=np.float64)
        u = (v - self.c0) / max(self.c1, 1e-12)
        return np.maximum(u, 0.0) ** self.gamma

    def dL_dv(self, pixel):
        """Sensitivity of recovered luminance to a pixel-value perturbation.

        Needed to turn quantisation and sensor noise, which live in pixel units,
        into density uncertainty. It blows up nowhere except at the black point,
        where the curve is flat and a pixel step buys no luminance information --
        which is exactly the physical statement that a veiled or crushed region
        carries no density resolution, and it falls out of the algebra rather
        than being asserted.
        """
        v = np.asarray(pixel, dtype=np.float64)
        u = np.maximum((v - self.c0) / max(self.c1, 1e-12), 1e-12)
        return (self.gamma / max(self.c1, 1e-12)) * u ** (self.gamma - 1.0)


def _solve_linear(anchors: list[Anchor], gamma: float, film: FilmModel, fix_black: bool = False):
    """Weighted least squares for (c0, c1) -- or for c1 alone when the black point
    is pinned -- at fixed gamma."""
    L = np.array([a.target_luminance(film) for a in anchors], dtype=np.float64)
    v = np.array([a.pixel_mean for a in anchors], dtype=np.float64)
    s = np.array([max(a.pixel_sigma, 1e-4) for a in anchors], dtype=np.float64)
    basis = np.maximum(L, 0.0) ** (1.0 / gamma)
    A = basis[:, None] if fix_black else np.stack([np.ones_like(L), basis], axis=1)
    W = 1.0 / s
    coef, *_ = np.linalg.lstsq(A * W[:, None], v * W, rcond=None)
    pred = A @ coef
    chi2 = float(np.sum(((v - pred) / s) ** 2))
    c0, c1 = (0.0, float(coef[0])) if fix_black else (float(coef[0]), float(coef[1]))
    return c0, c1, chi2, float(np.sqrt(np.mean((v - pred) ** 2)))


def fit_tone(
    anchors: list[Anchor],
    film: FilmModel | None = None,
    gamma_prior: float = GAMMA_PRIOR,
    gamma_sigma: float = GAMMA_PRIOR_SIGMA,
) -> ToneModel:
    """Fit the tone curve from whatever anchors this image gave us.

    With two densities the black point is pinned at zero, and that is a physical
    decision rather than a convenience. `c0` and a spatially constant veil are
    *exactly degenerate*: both add a pedestal to every recorded value, and two
    anchors cannot tell them apart. Left free, `c0` absorbs the veil, the
    linearised beam stop then reads zero, the next iteration measures no glare,
    and the inversion converges to a confident, veil-free, wrong answer.

    Pinning `c0 = 0` assigns the entire pedestal to the veil instead, which is
    both the safe direction and the useful one: the veil is measured as a
    *surface* over the frame by `glare.py`, whereas a black level could only ever
    have been a scalar. Nothing downstream is harmed, because contrast
    compression and the density floor depend on the sum `black + veil` and not on
    its attribution. So the two anchors are each used for what only they can do --
    the bright one sets the density scale, the beam stop measures the pedestal --
    and including both leaves a residual that is a genuine self-consistency check.

    With three or more distinct densities the degeneracy is broken by the data:
    `gamma` is fitted on a grid with a free black point, and its uncertainty read
    off the chi-squared curvature. That is the case a step wedge buys you.
    """
    film = film or FilmModel()
    anchors = [a for a in anchors if a.n_px > 0 and np.isfinite(a.pixel_mean)]
    if not anchors:
        return ToneModel(c0=0.0, c1=1.0, gamma=gamma_prior, gamma_sigma=gamma_sigma,
                         n_anchors=0, method="prior",
                         diagnostics={"reason": "no usable anchors"})

    distinct = len({round(a.density, 3) for a in anchors})

    if distinct >= 3:
        grid = np.linspace(max(1.0, gamma_prior - 3 * gamma_sigma), gamma_prior + 3 * gamma_sigma, 61)
        best = None
        chis = []
        for g in grid:
            c0, c1, chi2, rms = _solve_linear(anchors, float(g), film)
            # Keep the prior in play as a weak regulariser so a noisy third anchor
            # cannot drag gamma somewhere unphysical.
            pen = chi2 + ((g - gamma_prior) / (3 * gamma_sigma)) ** 2
            chis.append(pen)
            if best is None or pen < best[0]:
                best = (pen, float(g), c0, c1, rms)
        chis = np.asarray(chis)
        _, g, c0, c1, rms = best
        # Delta chi^2 = 1 interval around the minimum, floored so it never claims
        # more precision than the anchors can support.
        ok = grid[chis <= chis.min() + 1.0]
        g_sig = float(max((ok.max() - ok.min()) / 2.0, 0.02)) if ok.size > 1 else gamma_sigma
        return ToneModel(c0=c0, c1=c1, gamma=g, gamma_sigma=g_sig, n_anchors=len(anchors),
                         method="three_point", residual=rms,
                         diagnostics={"distinct_densities": distinct})

    c0, c1, _, rms = _solve_linear(anchors, gamma_prior, film, fix_black=True)
    method = "two_point" if distinct == 2 else "one_point"
    diag: dict = {"distinct_densities": distinct, "black_point": "pinned at zero"}
    if distinct < 2:
        diag["warning"] = ("only one distinct density anchor; the density scale rests on it alone "
                           "and the beam-stop consistency check is unavailable")
    return ToneModel(c0=c0, c1=max(c1, 1e-9), gamma=gamma_prior, gamma_sigma=gamma_sigma,
                     n_anchors=len(anchors), method=method, residual=rms, diagnostics=diag)


# --------------------------------------------------------------------------- #
# anchors from a photo
# --------------------------------------------------------------------------- #


def anchors_from_fiducials(
    photo_norm: np.ndarray,
    fid,
    film: FilmModel | None = None,
    veil: np.ndarray | None = None,
    illumination: np.ndarray | float = 1.0,
) -> list[Anchor]:
    """Reduce each detected fiducial to an (pixel value, known density) anchor.

    Note the two D_min regions are kept as separate anchors even though they share
    a density. They do not constrain gamma any further -- but they *do* sit at
    different places in the frame, so the difference between them is a direct
    read on the illumination gradient and the veil gradient, which is why
    `fit_illumination` wants both.
    """
    film = film or FilmModel()
    img = np.asarray(photo_norm, dtype=np.float64)
    out: list[Anchor] = []

    def _mean_sigma(mask):
        v = img[mask]
        # Median, not mean: a dust speck or a JPEG ring inside the mask should not
        # move an anchor that the whole density scale hangs from.
        return float(np.median(v)), float(max(_ops.robust_std(v) / np.sqrt(max(v.size, 1)), 1e-4)), int(v.size)

    def _local(field_arr, mask, default=0.0):
        if field_arr is None:
            return float(default)
        if np.ndim(field_arr) == 0:
            return float(field_arr)
        return float(np.median(np.asarray(field_arr)[mask]))

    if fid.marker_mask is not None and fid.marker_mask.any() and fid.marker_confidence >= MARKER_ACCEPT:
        core = _ops.binary_erode(fid.marker_mask, 1)
        m = core if core.sum() >= 8 else fid.marker_mask
        mean, sig, n = _mean_sigma(m)
        out.append(Anchor("lead_marker", mean, sig, film.d_min, n,
                          _local(veil, m), _local(illumination, m, 1.0)))

    if fid.outside_mask is not None and fid.outside_mask.any():
        m = _ops.binary_erode(fid.outside_mask, 3)
        if m.sum() < 50:
            m = fid.outside_mask
        mean, sig, n = _mean_sigma(m)
        out.append(Anchor("film_base", mean, sig, film.d_min, n,
                          _local(veil, m), _local(illumination, m, 1.0)))

    if fid.beamstop_mask is not None and fid.beamstop_mask.any():
        m = fid.beamstop_mask
        mean, sig, n = _mean_sigma(m)
        out.append(Anchor("direct_exposure", mean, sig, film.d_max, n,
                          _local(veil, m), _local(illumination, m, 1.0)))
    return out


def fit_illumination(
    linear_img: np.ndarray,
    fid,
    tone: ToneModel,
    film: FilmModel | None = None,
    veil: np.ndarray | None = None,
    degree: int = 1,
) -> tuple[np.ndarray, dict]:
    """Recover the lightbox's illumination field from the D_min regions.

    The unexposed film outside the collimation border is a flat field of known
    density wrapping the entire frame, and the lead marker is one sample of the
    same density in the *interior*. Together they constrain a low-order
    illumination surface -- the periphery sets the edges, the marker keeps the
    middle from floating. This is the one job only the marker can do, and it is
    why an image with a border but no marker gets a first-order fit while one with
    both can support a second-order one.

    A caveat worth stating plainly: with a single interior sample the interior of
    the surface is interpolated, not measured. For absolute density that is a real
    error. For the density *differences* the floor is built on it is nearly
    harmless, because illumination varies on the scale of the whole lightbox and a
    lesion and its neighbouring lung are millimetres apart.
    """
    film = film or FilmModel()
    L = np.asarray(linear_img, dtype=np.float64)
    h, w = L.shape
    V = np.zeros_like(L) if veil is None else np.asarray(veil, dtype=np.float64)
    tau_min = float(density_to_transmittance(film.d_min))

    masks = []
    if fid.outside_mask is not None and fid.outside_mask.any():
        m = _ops.binary_erode(fid.outside_mask, 3)
        masks.append(m if m.sum() > 50 else fid.outside_mask)
    if fid.marker_mask is not None and fid.marker_confidence >= MARKER_ACCEPT:
        m = _ops.binary_erode(fid.marker_mask, 1)
        masks.append(m if m.sum() >= 8 else fid.marker_mask)

    if not masks:
        return np.ones((h, w)), {"method": "assumed_uniform",
                                 "reason": "no D_min region detected"}

    probe = np.logical_or.reduce(masks)
    ys, xs = np.nonzero(probe)
    vals = np.clip((L[ys, xs] - V[ys, xs]) / tau_min, 1e-6, None)
    if len(vals) > 4000:
        idx = np.random.default_rng(0).choice(len(vals), 4000, replace=False)
        ys, xs, vals = ys[idx], xs[idx], vals[idx]

    span_y = (ys.max() - ys.min()) / max(h - 1, 1)
    span_x = (xs.max() - xs.min()) / max(w - 1, 1)
    if len(vals) < 40 or min(span_y, span_x) < 0.2:
        level = float(np.median(vals))
        return np.full((h, w), level), {"method": "constant", "level": level,
                                        "n_probes": len(vals)}

    deg = degree if len(masks) > 1 else min(degree, 1)
    surface, _ = _ops.fit_poly_surface(ys.astype(float), xs.astype(float), vals, (h, w), degree=deg)
    surface = np.clip(surface, 1e-6, None)
    med = float(np.median(surface))
    return surface, {
        "method": f"poly{deg}",
        "n_probes": len(vals),
        "n_regions": len(masks),
        "nonuniformity": float((np.quantile(surface, 0.95) - np.quantile(surface, 0.05)) / max(med, 1e-9)),
    }
