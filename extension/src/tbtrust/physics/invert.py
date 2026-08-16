"""Blind inversion: phone photo -> calibrated optical density, with an error budget.

This is the orchestrator. It takes a single 8-bit photograph of a film on a
lightbox, with no side information, no reference shot and no knowledge of the
phone, and returns the optical density map plus a per-pixel statement of how well
that density is known.

The estimators are mutually entangled -- the tone curve needs the veil, the veil
needs the PSF, the PSF needs a linearised image, which needs the tone curve -- so
they are run as a short fixed-point iteration rather than a pipeline:

    provisional tone (gamma prior, raw anchors, no veil)
      repeat:
        linearise  ->  PSF from the slanted collimation edge
                   ->  illumination surface from the D_min regions
                   ->  veil from the beam stop, with blur bleed removed
                   ->  re-fit tone with veil- and illumination-corrected anchors

Two iterations are enough in practice; the first one is doing almost all the work
and the second mostly moves the veil, because the provisional tone curve is
already close wherever the anchors bracket the data.

The error budget is split, deliberately
---------------------------------------
`sigma_random` collects the terms that differ between two nearby pixels --
photon and read noise, quantisation, the scatter of the veil surface fit.
`sigma_systematic` collects the terms shared across the frame -- the gamma prior,
the gain, the illumination interpolation, the film's D_min/D_max tolerances.

Only the random terms enter the resolution floor in `floor.py`, because the floor
bounds a *difference* of densities measured millimetres apart, and a common
scale or offset error cancels in a difference. Reporting one merged sigma would
make the floor several times too pessimistic and the whole certificate useless.
Keeping them separate is what makes the bound both correct and tight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _ops
from . import glare as G
from . import psf as P
from . import tone as T
from .density import D_LUNG_RANGE, FilmModel, density_to_transmittance
from .fiducials import Coverage, Fiducials, detect

# Standard chest cassette diagonal, used only to turn detected pixels into
# millimetres so that a finding quoted in mm can be turned into a spatial
# frequency. Approximate by construction -- see `CalibratedFilm.px_per_mm`.
CASSETTE_MM = (355.0, 432.0)


@dataclass
class CalibratedFilm:
    """One inverted photo: density, its error budget, and everything that produced it."""

    density: np.ndarray                 # optical density, photo frame
    sigma_random: np.ndarray            # per-pixel differential density uncertainty
    sigma_systematic: np.ndarray        # common-mode density uncertainty
    luminance: np.ndarray               # total measured luminance (film + veil)
    signal: np.ndarray                  # veil-removed film luminance
    pixel_values: np.ndarray            # the normalised photo, kept so floor.py can
                                        # rebuild individual error terms in isolation
    veil: np.ndarray
    illumination: np.ndarray
    tone: T.ToneModel
    psf: P.PSFEstimate
    glare: G.GlareEstimate
    fiducials: Fiducials
    film: FilmModel
    noise_model: dict = field(default_factory=dict)
    quantization_step: float = 1.0 / 255.0
    px_per_mm: float = float("nan")
    px_per_mm_rel_sigma: float = 0.20
    canonical_homography: np.ndarray | None = None
    canonical_size: int = 384
    iterations: int = 0
    diagnostics: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- masks
    def analysable_mask(self) -> np.ndarray:
        """Pixels that carry patient information: in-field, and not a fiducial."""
        f = self.fiducials
        m = f.field_mask if f.field_mask is not None else np.ones(self.density.shape, dtype=bool)
        for extra in (f.beamstop_mask, f.marker_mask, f.outside_mask):
            if extra is not None:
                m = m & ~_ops.binary_dilate(extra, 2)
        return m

    def lung_field_mask(self, band: tuple[float, float] = D_LUNG_RANGE) -> np.ndarray:
        """Heuristic lung-field mask: analysable pixels in the lung density band.

        Density-based rather than anatomical, and it will include some abdomen and
        miss some retrocardiac lung. It is used to *aggregate* the floor over the
        region a TB finding could plausibly occupy, so a soft boundary costs a
        little dilution of the statistics, not a wrong bound -- the per-pixel floor
        map underneath is unaffected. Swap in a real segmentation if you have one;
        every consumer takes the mask as an argument.
        """
        lo, hi = band
        m = self.analysable_mask() & (self.density >= lo) & (self.density <= hi)
        if m.sum() < 0.02 * m.size:                       # inversion went sideways
            return self.analysable_mask()
        return _ops.binary_erode(_ops.binary_dilate(m, 2), 2)

    # ------------------------------------------------------------ geometry
    def to_canonical(self, arr: np.ndarray, fill: float = np.nan) -> np.ndarray:
        """Warp a photo-frame map into the rectified field frame, for comparison."""
        if self.canonical_homography is None:
            return np.asarray(arr, dtype=np.float64)
        return _ops.warp_perspective(arr, self.canonical_homography,
                                     (self.canonical_size, self.canonical_size), fill=fill)

    def summary(self) -> dict:
        """Flat JSON-friendly record. One row per image in the certificate tables."""
        lung = self.lung_field_mask()
        sig = self.signal[lung] if lung.any() else self.signal.ravel()
        veil = self.veil[lung] if lung.any() else self.veil.ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            vf = np.median(veil / np.maximum(sig, 1e-9))
        return {
            "coverage": self.fiducials.coverage.value,
            "tone_method": self.tone.method,
            "tone_gamma": float(self.tone.gamma),
            "tone_gamma_sigma": float(self.tone.gamma_sigma),
            "psf_method": self.psf.method,
            "psf_sigma_px": float(self.psf.sigma),
            "psf_anisotropy": float(self.psf.anisotropy),
            "motion_dominant": bool(self.psf.motion_dominant),
            "mtf50_cy_px": float(self.psf.mtf50),
            "glare_method": self.glare.method,
            "veil_fraction_median": float(vf),
            "contrast_retained_median": float(1.0 / (1.0 + max(vf, 0.0))),
            "sigma_random_median": float(np.median(self.sigma_random[lung])) if lung.any() else float("nan"),
            "sigma_systematic_median": float(np.median(self.sigma_systematic[lung])) if lung.any() else float("nan"),
            "px_per_mm": float(self.px_per_mm),
            "iterations": self.iterations,
        }


# --------------------------------------------------------------------------- #
# noise
# --------------------------------------------------------------------------- #


def estimate_pixel_noise(img: np.ndarray, mask: np.ndarray | None = None, nbins: int = 12) -> dict:
    """Fit sigma_v^2 = a + b*v from the image itself, level by level.

    Method: take horizontal neighbour differences over small blocks, which cancel
    smooth structure but keep noise, and read the *low* quantile of the per-block
    MAD within each brightness bin. Structured blocks read high and flat blocks
    read true, so the low quantile within a bin is the noise floor -- an old trick
    and a reliable one, and it needs no flat field and no repeated exposure.

    The linear form is not a curve fit for its own sake: read noise is constant
    and photon shot noise is proportional to the collected signal, so variance is
    affine in level. Recovering `b > 0` is a check that the model is describing a
    real sensor rather than JPEG blocking.
    """
    x = np.asarray(img, dtype=np.float64)
    h, w = x.shape
    m = np.ones((h, w), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)

    d = (x[:, 1:] - x[:, :-1]) / np.sqrt(2.0)
    dm = m[:, 1:] & m[:, :-1]
    lvl = 0.5 * (x[:, 1:] + x[:, :-1])

    def _flat(sigma, n_blocks):
        s = float(max(sigma, 1e-6))
        return {"a": s**2, "b": 0.0, "var_floor": s**2, "method": "global", "n_blocks": int(n_blocks)}

    bs = 8
    bh, bw = h // bs, (w - 1) // bs
    if bh < 2 or bw < 2:
        return _flat(_ops.robust_std(d[dm]) if dm.any() else 1.0 / 255.0, 0)

    db = d[: bh * bs, : bw * bs].reshape(bh, bs, bw, bs)
    mb = dm[: bh * bs, : bw * bs].reshape(bh, bs, bw, bs)
    lb = lvl[: bh * bs, : bw * bs].reshape(bh, bs, bw, bs)
    valid = mb.all(axis=(1, 3))
    sig = 1.4826 * np.median(np.abs(db), axis=(1, 3))
    lev = lb.mean(axis=(1, 3))
    sig, lev = sig[valid], lev[valid]

    if sig.size < 16:
        return _flat(np.median(sig) if sig.size else 1.0 / 255.0, sig.size)

    edges = np.unique(np.quantile(lev, np.linspace(0, 1, nbins + 1)))
    cx, cy = [], []
    for i in range(len(edges) - 1):
        sel = (lev >= edges[i]) & (lev <= edges[i + 1])
        if sel.sum() >= 6:
            cx.append(float(np.mean(lev[sel])))
            cy.append(float(np.quantile(sig[sel], 0.25) ** 2))
    if len(cx) < 3:
        return _flat(np.sqrt(np.median(cy)) if cy else np.median(sig), sig.size)

    A = np.stack([np.ones(len(cx)), np.asarray(cx)], axis=1)
    coef, *_ = np.linalg.lstsq(A, np.asarray(cy), rcond=None)
    a, b = float(max(coef[0], 0.0)), float(coef[1])
    if b < 0.0:
        # Variance cannot fall as signal rises -- shot noise only adds. A negative
        # slope means the fit was driven by scene content rather than by noise
        # (bright regions here are film base and collimation shutters, which are
        # genuinely flatter than the lung fields), so drop to a level-independent
        # model rather than extrapolate a physically impossible trend into the
        # highlights, where it would predict zero noise and an infinite SNR.
        a, b = float(np.median(cy)), 0.0
    # Never let sigma_v reach zero: the quantiser alone guarantees a quarter-LSB
    # of uncertainty, and a zero here propagates to a zero density floor and a
    # certificate that claims the photograph carries unlimited contrast.
    var_floor = max(float(min(cy)) * 0.25, (0.25 / 255.0) ** 2)
    return {
        "a": a, "b": b, "var_floor": var_floor, "method": "affine",
        "n_blocks": int(sig.size), "n_bins": len(cx),
    }


def noise_sigma(noise_model: dict, pixel_value) -> np.ndarray:
    """Evaluate the fitted sigma_v(v). Kept a plain function of a plain dict so the
    noise model stays JSON-serialisable and survives a round trip through the
    certificate tables."""
    v = np.asarray(pixel_value, dtype=np.float64)
    var = noise_model.get("a", 0.0) + noise_model.get("b", 0.0) * v
    return np.sqrt(np.maximum(var, noise_model.get("var_floor", (1.0 / 255.0) ** 2)))


# --------------------------------------------------------------------------- #
# the inversion
# --------------------------------------------------------------------------- #


def _normalize(photo: np.ndarray) -> np.ndarray:
    x = np.asarray(photo, dtype=np.float64)
    if x.ndim == 3:
        x = x.mean(axis=-1)
    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def _px_per_mm(fid: Fiducials, shape: tuple[int, int], field_fraction: float = 0.88) -> float:
    """Pixels per millimetre, from the detected collimation field.

    Approximate by construction: it assumes the collimated field covers
    `field_fraction` of a standard cassette diagonal. A 20% scale error moves a
    finding's characteristic spatial frequency by 20%, which perturbs the MTF term
    in the floor by a few percent for large findings and more for miliary nodules.
    `px_per_mm_rel_sigma` carries that forward. Pass a measured value instead
    whenever the clinic can supply one -- a ruler in the frame settles it exactly,
    and that is the second item on the deployment checklist.
    """
    h, w = shape
    if fid.field_quad is not None:
        q = fid.field_quad
        diag_px = 0.5 * (np.hypot(*(q[2] - q[0])) + np.hypot(*(q[3] - q[1])))
    else:
        diag_px = float(np.hypot(h, w))
    diag_mm = float(np.hypot(*CASSETTE_MM)) * field_fraction
    return float(diag_px / diag_mm)


def invert(
    photo: np.ndarray,
    film: FilmModel | None = None,
    fid: Fiducials | None = None,
    iterations: int = 2,
    canonical_size: int = 384,
    px_per_mm: float | None = None,
    gamma_prior: float = T.GAMMA_PRIOR,
    gamma_sigma: float = T.GAMMA_PRIOR_SIGMA,
) -> CalibratedFilm:
    """Invert one photo to calibrated optical density. See the module docstring."""
    film = film or FilmModel()
    v = _normalize(photo)
    h, w = v.shape
    fid = fid if fid is not None else detect(v)

    tau_min = float(density_to_transmittance(film.d_min))
    tau_max = float(density_to_transmittance(film.d_max))

    # --- provisional tone: BRIGHT anchors only.
    #
    # The beam-stop anchor must be withheld from the first fit, and this is the
    # subtlest point in the whole inversion. With the veil not yet known, the
    # dark anchor's expected luminance is tau_max ~ 6e-4, i.e. zero, so a
    # two-parameter fit sets the black point c0 to whatever the beam stop
    # measures -- which is the veil. The tone curve then swallows the veil, the
    # linearised beam stop reads zero, the next iteration measures no glare, and
    # the loop sits happily at a completely wrong fixed point of V = 0. Nothing
    # downstream notices, and every certificate comes out veil-free.
    #
    # Starting from the bright anchors with c0 pinned at zero breaks the
    # degeneracy: the beam stop then shows its full pedestal, the veil is
    # measured, and from the second iteration the dark anchor carries a real
    # target luminance and c0 converges to the ISP's actual black level.
    all_anchors = T.anchors_from_fiducials(v, fid, film)
    bright = [a for a in all_anchors if a.name != "direct_exposure"]
    tone = T.fit_tone(bright or all_anchors, film, gamma_prior, gamma_sigma)
    veil = np.zeros((h, w))
    illum = np.ones((h, w))
    psf = P.PSFEstimate(sigma_x=2.0, sigma_y=2.0, method="prior", sigma_uncertainty=1.5)
    glare = G.GlareEstimate(np.zeros((h, w)), 0.0, 0, "none")
    illum_diag: dict = {}

    n_iter = max(1, int(iterations))
    for _ in range(n_iter):
        L = tone.to_luminance(v)
        psf = P.estimate_psf(L, fid)
        illum, illum_diag = T.fit_illumination(L, fid, tone, film, veil=veil)
        glare = G.estimate_veil(
            L, fid, psf_sigma=psf.sigma, tau_max=tau_max, tau_min=tau_min,
            scene_estimate=np.maximum(L - veil, 0.0), illumination=illum,
        )
        veil = glare.veil
        anchors = T.anchors_from_fiducials(v, fid, film, veil=veil, illumination=illum)
        tone = T.fit_tone(anchors, film, gamma_prior, gamma_sigma)

    # --- final densities
    L = tone.to_luminance(v)
    # The veil can never exceed the measurement minus what the film must transmit:
    # even at D_max the film passes tau_max of the box. Clamping here rather than
    # flooring the signal at some epsilon keeps both quantities physical, and it
    # matters a great deal downstream -- sigma_D goes as 1/signal, so a veil
    # estimate that overshoots the measurement sends the density floor to 10^6 and
    # every derived number with it.
    veil = np.minimum(veil, np.maximum(L - illum * tau_max, 0.0))
    signal = np.maximum(L - veil, illum * tau_max)
    tau = np.clip(signal / np.maximum(illum, 1e-9), 1e-9, None)
    density = -np.log10(tau)
    density = np.clip(density, film.d_min - 0.5, film.d_max + 0.5)

    # --- error budget
    noise = estimate_pixel_noise(v, mask=fid.field_mask)
    sigma_v = noise_sigma(noise, v)
    q = 1.0 / 255.0
    sigma_v_total = np.sqrt(sigma_v**2 + (q**2) / 12.0)

    dL_dv = tone.dL_dv(v)
    # dD/dL = -1/(ln10 * L_film); random terms are pixel noise, quantisation and
    # the local scatter of the veil surface.
    inv_ln10 = 1.0 / np.log(10.0)
    dD_dL = inv_ln10 / signal
    sigma_random = dD_dL * np.sqrt((dL_dv * sigma_v_total) ** 2 + float(glare.veil_sigma) ** 2)

    # Systematic: gamma (a scale error on D), gain, illumination interpolation,
    # and the film's own anchor tolerances.
    with np.errstate(divide="ignore", invalid="ignore"):
        d_lnu = np.log(np.maximum((v - tone.c0) / max(tone.c1, 1e-12), 1e-12))
    sys_gamma = np.abs(d_lnu * inv_ln10) * float(tone.gamma_sigma)
    sys_gain = np.full((h, w), float(tone.gamma) * inv_ln10 * (float(tone.residual) / max(tone.c1, 1e-9)))
    illum_rel = float(illum_diag.get("nonuniformity", 0.05))
    sys_illum = np.full((h, w), inv_ln10 * illum_rel * 0.5)
    sys_film = np.full((h, w), float(film.d_min_sigma))
    sigma_systematic = np.sqrt(sys_gamma**2 + sys_gain**2 + sys_illum**2 + sys_film**2)

    sigma_random = np.nan_to_num(sigma_random, nan=1.0, posinf=1.0)
    sigma_systematic = np.nan_to_num(sigma_systematic, nan=1.0, posinf=1.0)

    # --- rectification for downstream comparison
    Hc = None
    if fid.field_quad is not None:
        cs = int(canonical_size)
        dst = np.array([[0, 0], [cs - 1, 0], [cs - 1, cs - 1], [0, cs - 1]], dtype=np.float64)
        try:
            Hc = _ops.estimate_homography(fid.field_quad, dst)
        except (ValueError, np.linalg.LinAlgError):
            Hc = None

    scale = float(px_per_mm) if px_per_mm else _px_per_mm(fid, (h, w))

    return CalibratedFilm(
        density=density,
        sigma_random=sigma_random,
        sigma_systematic=sigma_systematic,
        luminance=L,
        signal=signal,
        veil=veil,
        illumination=illum,
        tone=tone,
        psf=psf,
        glare=glare,
        fiducials=fid,
        film=film,
        noise_model=noise,
        quantization_step=q,
        pixel_values=v,
        px_per_mm=scale,
        px_per_mm_rel_sigma=0.0 if px_per_mm else 0.20,
        canonical_homography=Hc,
        canonical_size=int(canonical_size),
        iterations=n_iter,
        diagnostics={
            "illumination": illum_diag,
            "glare": glare.diagnostics,
            "tone": tone.diagnostics,
            "tau_min": tau_min,
            "tau_max": tau_max,
            "coverage": fid.coverage.value,
        },
    )


def invertible(fid: Fiducials) -> bool:
    """Whether a physics-derived bound is available at all for this image.

    Without a beam stop there is no glare measurement, and without a glare
    measurement the dominant term in the floor is unknown -- not large, *unknown*.
    The right response is for the certificate to abstain, and this is the check
    that makes it do so instead of quietly substituting a prior.
    """
    return fid.coverage in (Coverage.FULL, Coverage.PARTIAL) and fid.has_beamstop
