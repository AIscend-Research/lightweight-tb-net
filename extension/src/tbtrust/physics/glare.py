"""Veiling glare, measured with the beam stop the film already provides.

Radiology QA measures veiling glare by putting a lead disc in the beam and
reading whatever signal appears underneath it: the disc transmits nothing, so any
recorded signal is stray light, full stop. No model, no assumption, just
subtraction. Astronomers do the same with a coronagraph -- occult the star, and
what remains is the instrument's scattered light.

A chest film hands us that disc for free. The direct-exposure region sits at
D_max, transmitting about 6e-4 of the lightbox. It is, optically, black. So the
luminance a phone records there is the veil, directly:

    L_measured(x)  =  I(x) * tau_max  +  bleed(x)  +  V(x)
                      \\-- known, ~0 --/   \\-- computable --/   \\- what we want -/

Why the middle term must be separated
-------------------------------------
Two different things put photons where there should be none. Short-range **blur
bleed** carries light a few pixels across the collimation edge; long-range
**veiling glare** carries it across the whole frame. They are the core and the
halo of one PSF, and confusing them wrecks the estimate in both directions: on a
sharp image you would credit the veil with light that was only defocus, and on a
blurry one you would let real glare hide inside the blur budget.

They are separable because we have already measured the core -- that is what
`psf.py` is for. Convolve the current scene estimate with the measured core,
read it at the probes, and subtract. Whatever survives is long-range, and it is
the veil.

What the veil then costs you
----------------------------
An additive veil V on top of a signal I does not change a density *difference* in
luminance, but it does compress it in the recorded contrast:

    contrast_recorded / contrast_true  =  1 / (1 + V/I)

which is the term that drives the density resolution floor in `floor.py`. A 20%
veil over a lung field costs you a sixth of your density resolution everywhere it
lands. That is the number this module exists to produce -- and to produce
*per pixel*, because glare from a window or a ceiling light is never uniform, and
"the glare is in the upper left, move the phone" is an actionable instruction
while "the image is glary" is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _ops
from .fiducials import Fiducials


@dataclass
class GlareEstimate:
    """The measured veil field and everything triage needs to act on it."""

    veil: np.ndarray                      # V(x, y) in luminance units, same shape as the photo
    veil_sigma: float                     # 1-sigma uncertainty on V, from probe scatter
    n_probes: int
    method: str                           # 'beamstop_surface' | 'beamstop_constant' | 'none'
    probe_residual: float = 0.0           # RMS of the surface fit at the probes
    degree: int = 2
    diagnostics: dict = field(default_factory=dict)

    def fraction(self, signal: np.ndarray) -> np.ndarray:
        """V / I, the veil relative to the local signal it is sitting on."""
        s = np.maximum(np.asarray(signal, dtype=np.float64), 1e-9)
        return np.clip(self.veil, 0.0, None) / s

    def contrast_compression(self, signal: np.ndarray) -> np.ndarray:
        """1 / (1 + V/I): the factor every density difference is shrunk by."""
        return 1.0 / (1.0 + self.fraction(signal))


@dataclass
class GlareHotspot:
    """Where the veil is worst, phrased so an operator can do something about it."""

    localized: bool
    peak_over_median: float
    affected_fraction: float              # fraction of the field above 1.5x median veil
    centroid_yx: tuple[float, float] | None
    direction: str                        # 'upper-left' ... 'centre', or 'diffuse'
    advice: str


def _probe_values(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    return ys.astype(np.float64), xs.astype(np.float64), img[ys, xs]


def _subsample(ys, xs, vals, max_n: int = 4000, rng=None):
    """Cap probe count. A rim can hold 10^5 pixels and the surface fit needs
    hundreds, not all of them; the lstsq cost is otherwise the slowest step in
    the whole inversion."""
    if len(vals) <= max_n:
        return ys, xs, vals
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(len(vals), size=max_n, replace=False)
    return ys[idx], xs[idx], vals[idx]


def estimate_veil(
    linear_img: np.ndarray,
    fid: Fiducials,
    psf_sigma: float,
    tau_max: float,
    tau_min: float = 0.631,
    scene_estimate: np.ndarray | None = None,
    illumination: np.ndarray | float = 1.0,
    degree: int = 2,
    trim_quantile: float = 0.75,
) -> GlareEstimate:
    """Measure the veil at the beam stop and fit its spatial shape over the frame.

    Parameters
    ----------
    linear_img
        The photo in luminance units (tone curve already inverted, however
        provisionally). `invert.py` calls this repeatedly as its tone estimate
        improves.
    psf_sigma
        The measured core width, used to predict and remove blur bleed.
    tau_max
        Transmittance of the film at D_max: the residual the beam stop legitimately
        does pass, which must not be counted as glare.
    scene_estimate
        Current best guess of the veil-free scene, used to compute blur bleed.
        On the first iteration pass None and the raw image is used, which
        slightly over-estimates bleed and therefore under-estimates glare -- a
        conservative first step that the iteration then corrects.
    trim_quantile
        Outlier rejection strength, in units of robust sigma above the fitted
        surface (the default 0.75 gives a 2.25-sigma one-sided cut). Rejection is
        one-sided by design: a rim probe overlapping a scratch, a staple or a bit
        of the patient reads bright and must go, while nothing physical reads
        darker than the veil.
    """
    img = np.asarray(linear_img, dtype=np.float64)
    h, w = img.shape
    if fid.beamstop_mask is None or not fid.beamstop_mask.any():
        return GlareEstimate(np.zeros((h, w)), 0.0, 0, "none",
                             diagnostics={"reason": "no beam stop found"})

    illum = np.asarray(illumination, dtype=np.float64) if np.ndim(illumination) else float(illumination)
    legit = illum * float(tau_max)                     # what the film really does transmit
    scene = np.asarray(scene_estimate, dtype=np.float64) if scene_estimate is not None else img

    # Handling blur bleed by *predicting and subtracting* it turns out to be a bad
    # idea, and it took several attempts to see why. Any prediction has to convolve
    # something with the core, and the only thing available is the already-blurred
    # measurement, so the prediction is doubly blurred and over-states how much
    # light actually reached a point several sigma from an edge. Subtracting it
    # then eats the veil itself -- and a confident V = 0 is indistinguishable,
    # everywhere downstream, from a genuinely clean photograph, so the failure is
    # silent and every certificate comes out veil-free.
    #
    # Geometry is the reliable instrument instead, and it is cheap to reason about
    # exactly. Light leaking across a straight boundary from a neighbour of
    # brightness B, measured k sigma inside the dark side, is B * erfc(k/sqrt2) / 2:
    # 0.6% at 3 sigma, 2.3% at 2 sigma, 16% at 1 sigma. Against a veil that is
    # typically tens of percent of the dark-region signal, anything at 2 sigma or
    # beyond is negligible and needs no correction at all.
    #
    # So take the largest guard the direct-exposure band can actually support
    # rather than insisting on 3 sigma. Insisting was the bug: a 14-pixel band
    # cannot be eroded by 3 sigma on a blurred capture, every image fell through to
    # the corrected path, and the correction's safety cap then halved the veil on
    # all of them.
    guard_sigmas = (3.0, 2.0, 1.5, 1.0)
    probes, used_sigma = None, None
    for ks in guard_sigmas:
        cand = _ops.binary_erode(fid.beamstop_mask, int(np.clip(np.ceil(ks * psf_sigma), 1, 40)))
        if cand.sum() >= 25:
            probes, used_sigma = cand, ks
            break

    if probes is None:
        # Even a 1-sigma guard is unaffordable. Fall back to the darkest pixels of
        # the dark region -- least contaminated by definition, since bleed only
        # ever adds light -- with a bounded correction on top. The cap is what
        # stops the doubly-blurred prediction from erasing the veil, and it is set
        # loose rather than tight because under-reporting glare is the dangerous
        # direction: it makes the density floor look better than it is.
        thr = float(np.quantile(img[fid.beamstop_mask], 0.30))
        probes = fid.beamstop_mask & (img <= thr)
        if probes.sum() < 12:
            probes = fid.beamstop_mask
        dark_floor = float(np.median(scene[probes]))
        bleed = _ops.gaussian_blur(np.maximum(scene - dark_floor, 0.0), max(psf_sigma, 1e-3))
        resid = img - legit - np.minimum(bleed, 0.5 * np.maximum(img - legit, 0.0))
    else:
        resid = img - legit

    # Residual leak at the achieved guard, carried into the veil's uncertainty so
    # a thin band is reported as a less certain measurement rather than as an
    # equally confident but quietly biased one.
    from math import erfc

    leak_frac = 0.5 * erfc((used_sigma or 0.5) / np.sqrt(2.0))

    ys, xs, vals = _probe_values(resid, probes)
    if vals.size < 12:
        return GlareEstimate(np.zeros((h, w)), 0.0, int(vals.size), "none",
                             diagnostics={"reason": "too few probes after guarding"})
    ys, xs, vals = _subsample(ys, xs, vals)
    vals = np.maximum(vals, 0.0)

    spread = _ops.robust_std(vals)
    # A surface needs probes spread over the frame. A single-blob dark surround
    # gives no leverage on the shape, so fall back to a constant rather than
    # extrapolate a quadratic off a small patch across the whole image.
    span = (ys.max() - ys.min()) / max(h - 1, 1), (xs.max() - xs.min()) / max(w - 1, 1)
    if min(span) < 0.25 or len(vals) < 60:
        level = float(np.median(vals))
        flat = _add_impossible_brightness(np.full((h, w), max(level, 0.0)), img, illum, tau_min, psf_sigma)
        return GlareEstimate(
            flat, max(spread, 1e-9), len(vals),
            "beamstop_constant", probe_residual=spread, degree=0,
            diagnostics={"probe_span": span, "level": level,
                         "guard_sigma": used_sigma, "leak_frac": leak_frac},
        )

    # Fit, then reject probes sitting well *above* the surface and refit once.
    # Trimming on the residual rather than on a global quantile of the values is
    # what keeps the spatial shape unbiased: a plain low-quantile trim would throw
    # away every probe in the genuinely glary corner and flatten the very gradient
    # the retake instruction depends on. Only upward outliers are cut, because a
    # scratch, a staple or a stray bit of anatomy inside the rim reads bright, and
    # nothing physical reads darker than the veil.
    surface, _ = _ops.fit_poly_surface(ys, xs, vals, (h, w), degree=degree)
    pred = surface[ys.astype(int), xs.astype(int)]
    resid_p = vals - pred
    cut = trim_quantile * 3.0 * _ops.robust_std(resid_p)
    keep = resid_p <= max(cut, 1e-12)
    if 12 <= keep.sum() < len(vals):
        ys, xs, vals = ys[keep], xs[keep], vals[keep]
        surface, _ = _ops.fit_poly_surface(ys, xs, vals, (h, w), degree=degree)
        pred = surface[ys.astype(int), xs.astype(int)]

    # Bound the surface to the range the probes actually support. The beam stop is
    # an annulus, so the entire interior of the field is extrapolation, and a
    # degree-2 surface extrapolated that far can run away -- overshooting the
    # measured luminance itself, which drives the recovered film signal to the
    # clamp and sends sigma_D, and with it the density floor, to absurd values. A
    # veil larger than anything measured anywhere on the frame is not a measurement,
    # it is the polynomial's opinion, so it is capped.
    probe_ceiling = float(np.quantile(vals, 0.99)) * 1.5
    surface = np.clip(surface, 0.0, max(probe_ceiling, 1e-12))
    residual = float(np.sqrt(np.mean((vals - pred) ** 2)))
    level = float(np.median(vals))
    surface = _add_impossible_brightness(surface, img, illum, tau_min, psf_sigma)
    return GlareEstimate(
        veil=surface,
        # Fit scatter and the un-corrected boundary leak, added in quadrature: a
        # band too thin to guard properly reports a less certain veil rather than
        # an equally confident but quietly biased one.
        veil_sigma=max(np.hypot(residual, leak_frac * level), 1e-9),
        n_probes=len(vals),
        method="beamstop_surface",
        probe_residual=residual,
        degree=degree,
        diagnostics={"probe_span": span, "median_probe": level,
                     "guard_sigma": used_sigma, "leak_frac": leak_frac},
    )


def hotspot(glare: GlareEstimate, field_mask: np.ndarray | None = None) -> GlareHotspot:
    """Classify the veil as a localized reflection or a diffuse wash, and say where.

    The distinction is the whole basis of a useful retake instruction. A specular
    reflection of a window is a compact blob that moves when the phone moves --
    "step to the side". A diffuse wash is ambient room light hitting the film
    everywhere -- moving the phone does nothing, you have to shade the film or
    turn the lights off. Telling an operator to do the wrong one wastes the
    retake, and a wasted retake in a rural clinic is a patient who leaves.
    """
    V = np.asarray(glare.veil, dtype=np.float64)
    m = field_mask if field_mask is not None else np.ones(V.shape, dtype=bool)
    if not m.any() or glare.method == "none":
        return GlareHotspot(False, 1.0, 0.0, None, "unknown", "no veil measurement available")

    v = V[m]
    med = float(np.median(v))
    peak = float(np.quantile(v, 0.99))
    ratio = peak / max(med, 1e-9)
    hot = m & (1.5 * max(med, 1e-12) < V)
    frac = float(hot.sum()) / float(m.sum())

    if hot.any():
        ys, xs = np.nonzero(hot)
        wgt = V[hot]
        cy = float((ys * wgt).sum() / max(wgt.sum(), 1e-12))
        cx = float((xs * wgt).sum() / max(wgt.sum(), 1e-12))
    else:
        cy = cx = None

    localized = bool(ratio > 1.8 and frac < 0.35)
    if cy is None:
        direction = "diffuse"
    else:
        h, w = V.shape
        vert = "upper" if cy < 0.38 * h else ("lower" if cy > 0.62 * h else "")
        horiz = "left" if cx < 0.38 * w else ("right" if cx > 0.62 * w else "")
        direction = "-".join([p for p in (vert, horiz) if p]) or "centre"

    if not localized:
        advice = ("Diffuse veiling light across the whole film. Moving the phone will not help: "
                  "shade the lightbox from room lighting or windows, or dim the room, then retake.")
    else:
        advice = (f"Specular reflection concentrated {direction}. Move the phone "
                  f"{'left' if 'right' in direction else 'right' if 'left' in direction else 'sideways'} "
                  "or tilt the film a few degrees so the reflection falls outside the lung fields, then retake.")
    return GlareHotspot(localized, ratio, frac, (cy, cx) if cy is not None else None, direction, advice)


def veil_fraction_report(glare: GlareEstimate, signal: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Scalar summary of the veil over a region of interest, for tables and logs."""
    m = mask if mask is not None else np.ones(np.shape(signal), dtype=bool)
    if not m.any() or glare.method == "none":
        return {"veil_fraction_median": float("nan"), "veil_fraction_p90": float("nan"),
                "contrast_retained_median": float("nan"), "method": glare.method}
    frac = glare.fraction(signal)[m]
    comp = 1.0 / (1.0 + frac)
    return {
        "veil_fraction_median": float(np.median(frac)),
        "veil_fraction_p90": float(np.quantile(frac, 0.9)),
        "contrast_retained_median": float(np.median(comp)),
        "contrast_retained_p10": float(np.quantile(comp, 0.1)),
        "method": glare.method,
        "n_probes": glare.n_probes,
    }


def _add_impossible_brightness(veil, img, illum, tau_min, psf_sigma):
    """Second glare probe: light that no film could have transmitted.

    The beam stop is an annulus, so it samples the veil only around the edge of
    the field and the interior is extrapolated by the surface fit. That is fine
    for a broad wash, which really is smooth, and wrong for a specular reflection
    of a window sitting in the middle of the film -- there are no probes under it,
    and the fit sails straight past.

    But the film supplies one more constraint for free. Nothing on a developed
    sheet is clearer than base+fog, so no pixel can legitimately transmit more
    than `I * tau_min`. Any excess over that is light which did not come through
    the film, which is to say glare, measured directly and with no model. It is a
    *lower* bound on the interior veil, and lower bounds are the right direction
    here: it can only ever raise the estimated veil, raise the density floor, and
    make the certificate stricter.

    It is smoothed before being added because glare is physically smooth, and an
    unsmoothed correction would chase sensor noise and quietly clip real
    highlights on the film.

    The honest limit: a reflection that is bright enough to matter but not bright
    enough to push a *lung-field* pixel above base+fog stays invisible to both
    probes, and the veil there is under-reported. `docs/PHYSICS.md` lists this as
    the leading known optimism in the bound.
    """
    ceiling = np.asarray(illum, dtype=np.float64) * float(tau_min)
    excess = np.maximum(np.asarray(img, dtype=np.float64) - np.asarray(veil) - ceiling, 0.0)
    if not np.any(excess > 0):
        return veil
    return np.asarray(veil) + _ops.gaussian_blur(excess, max(2.0 * psf_sigma, 2.0))
