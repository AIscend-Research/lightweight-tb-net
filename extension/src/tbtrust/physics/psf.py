"""Capture PSF from the collimation border, by ISO 12233 slanted-edge analysis.

The collimation border is a physically hard step: on one side is unexposed film
at base+fog, on the other is film that took the full unattenuated beam. Nothing
about the patient, the film or the processor smooths it. So every bit of blur
observed across it in a phone photo was added by the capture -- hand shake,
defocus, the lens, the demosaic, the JPEG. Its edge-spread function *is* the
capture PSF, measured rather than assumed.

This is the same trick as photometry from a standard star: a known point (or in
our case step) source in the frame hands you the instrument response for free,
under exactly the conditions of the exposure you care about.

Two details that decide whether the number is real
--------------------------------------------------
* **Linearise first.** The ESF must be built from luminance, not from
  tone-mapped pixel values. An S-curved ISP steepens the mid-tones and would
  report an edge sharper than it is. `invert.py` therefore runs a first tone fit,
  linearises, measures the PSF, and only then re-fits -- the loop is not
  decoration.
* **The edge must be slanted.** An edge exactly on the pixel grid samples the ESF
  at one phase only; the few degrees of tilt from a hand-held phone are what
  provide the sub-pixel phase diversity that makes the oversampled ESF possible.
  `fiducials.EdgeFit.usable_for_mtf` enforces the window.

Anisotropy is the payoff
------------------------
Left/right borders measure blur along x; top/bottom measure it along y. Hand
shake is directional, defocus is not. The ratio separates them, which is what
lets `triage.py` tell an operator "hold still" rather than the useless "the image
is blurry".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _ops
from .fiducials import EdgeFit, Fiducials

# Gaussian LSF of width sigma has MTF(f) = exp(-2 pi^2 sigma^2 f^2), so
# MTF(f50) = 0.5 gives sigma = sqrt(ln2 / 2) / (pi * f50).
_SIGMA_FROM_F50 = float(np.sqrt(np.log(2.0) / 2.0) / np.pi)


@dataclass
class PSFEstimate:
    """The measured capture blur, in the two directions the borders can see."""

    sigma_x: float                      # blur along image x, in pixels
    sigma_y: float
    freqs: np.ndarray = field(default_factory=lambda: np.zeros(0))   # cycles/pixel
    mtf: np.ndarray = field(default_factory=lambda: np.zeros(0))     # direction-averaged
    mtf50: float = float("nan")         # cycles/pixel
    method: str = "slanted_edge"        # 'slanted_edge' | 'opportunistic' | 'prior'
    n_edges: int = 0
    sigma_uncertainty: float = 0.0
    per_edge: list[dict] = field(default_factory=list)

    @property
    def sigma(self) -> float:
        """Isotropic-equivalent width: the geometric mean of the two directions."""
        return float(np.sqrt(max(self.sigma_x, 1e-6) * max(self.sigma_y, 1e-6)))

    @property
    def anisotropy(self) -> float:
        """max/min of the two directional widths. > ~1.6 reads as directional smear."""
        a, b = max(self.sigma_x, 1e-6), max(self.sigma_y, 1e-6)
        return float(max(a, b) / min(a, b))

    @property
    def motion_dominant(self) -> bool:
        return self.anisotropy > 1.6

    def mtf_at(self, f) -> np.ndarray:
        """MTF at arbitrary spatial frequencies (cycles/pixel).

        Uses the measured curve where it exists and the Gaussian-equivalent model
        outside its support. The Gaussian is not a cosmetic fallback: it is the
        conservative continuation, because a real PSF's MTF has heavier tails than
        a Gaussian's at high frequency, so extrapolating with the Gaussian
        *under*-states the surviving contrast and therefore *over*-states the
        density floor. Erring toward a stricter certificate is the safe direction.
        """
        f = np.atleast_1d(np.asarray(f, dtype=np.float64))
        model = np.exp(-2.0 * np.pi**2 * self.sigma**2 * f**2)
        if self.freqs.size < 4:
            return model
        out = np.interp(f, self.freqs, self.mtf, left=1.0, right=np.nan)
        return np.where(np.isnan(out), model, np.clip(out, 1e-6, 1.0))


# --------------------------------------------------------------------------- #
# single-edge analysis
# --------------------------------------------------------------------------- #


def edge_spread_function(
    img: np.ndarray,
    edge: EdgeFit,
    half_width: float = 12.0,
    oversample: int = 4,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Oversampled ESF: bin every nearby pixel by its signed distance to the edge.

    Returns (distance_px, esf) or None if the samples are too sparse to bin. The
    sub-pixel edge positions come from the total-least-squares line fitted in
    `fiducials`, so distances carry the phase diversity the oversampling needs.
    """
    a, b, c = edge.line
    h, w = img.shape
    pts = edge.points
    # Restrict to the band the edge samples actually span, so we do not drag in
    # the corners where two borders meet and the step is not a single edge.
    if edge.side in ("left", "right"):
        y0, y1 = pts[:, 1].min(), pts[:, 1].max()
        x0 = max(0, int(pts[:, 0].min() - half_width - 2))
        x1 = min(w, int(pts[:, 0].max() + half_width + 2))
        y0, y1 = max(0, int(y0)), min(h, int(y1) + 1)
    else:
        x0, x1 = max(0, int(pts[:, 0].min())), min(w, int(pts[:, 0].max()) + 1)
        y0 = max(0, int(pts[:, 1].min() - half_width - 2))
        y1 = min(h, int(pts[:, 1].max() + half_width + 2))
    if y1 - y0 < 8 or x1 - x0 < 4:
        return None

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    dist = a * xx + b * yy + c
    vals = img[y0:y1, x0:x1]
    sel = np.abs(dist) <= half_width
    if sel.sum() < 200:
        return None
    d, v = dist[sel], vals[sel]

    nbins = int(2 * half_width * oversample)
    idx = np.clip(((d + half_width) * oversample).astype(int), 0, nbins - 1)
    tot = np.bincount(idx, weights=v, minlength=nbins)
    cnt = np.bincount(idx, minlength=nbins).astype(np.float64)
    occupied = cnt > 0
    if occupied.mean() < 0.75:
        return None                        # too many empty bins to trust the ESF
    centers = (np.arange(nbins) + 0.5) / oversample - half_width
    esf = np.where(occupied, tot / np.maximum(cnt, 1), np.nan)
    esf = np.interp(centers, centers[occupied], esf[occupied])
    return centers, esf


def mtf_from_esf(centers: np.ndarray, esf: np.ndarray, oversample: int = 4):
    """ESF -> LSF -> MTF, with the differentiator's own response divided out.

    The finite difference used to get the LSF is itself a filter with response
    |sinc(f*dx)|. Leaving it in makes the reported MTF droop at high frequency
    and the recovered sigma too large; the correction is one line and it matters
    at the frequencies where small findings live.
    """
    dx = 1.0 / oversample
    lsf = np.gradient(np.asarray(esf, dtype=np.float64), dx)
    if float(np.abs(lsf).sum()) < 1e-12:
        return None
    if lsf.sum() < 0:
        lsf = -lsf                                  # dark->bright or bright->dark
    lsf = lsf - np.median(np.concatenate([lsf[:4], lsf[-4:]]))

    # Centre on the LSF centroid and apply a Hamming window, or the finite record
    # length rings all over the MTF.
    n = len(lsf)
    wgt = np.maximum(lsf, 0.0)
    if wgt.sum() < 1e-12:
        return None
    centroid = float((np.arange(n) * wgt).sum() / wgt.sum())
    shift = round(n / 2 - centroid)
    lsf = np.roll(lsf, shift)
    lsf = lsf * np.hamming(n)
    s = lsf.sum()
    if abs(s) < 1e-12:
        return None
    lsf = lsf / s

    spec = np.abs(np.fft.rfft(lsf))
    freqs = np.fft.rfftfreq(n, d=dx)                # cycles/pixel, up to oversample/2
    if spec[0] < 1e-12:
        return None
    mtf = spec / spec[0]

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.abs(np.sinc(freqs * dx))
    mtf = np.where(corr > 0.2, mtf / np.maximum(corr, 1e-6), mtf)

    keep = freqs <= 1.0                              # nothing above 2x pixel Nyquist is meaningful
    freqs, mtf = freqs[keep], np.clip(mtf[keep], 0.0, 2.0)

    # MTF50 by linear interpolation on the first downward crossing of 0.5.
    below = np.nonzero(mtf < 0.5)[0]
    if below.size == 0 or below[0] == 0:
        f50 = float(freqs[-1])
    else:
        i = int(below[0])
        f0, f1, m0, m1 = freqs[i - 1], freqs[i], mtf[i - 1], mtf[i]
        f50 = float(f0 + (0.5 - m0) * (f1 - f0) / (m1 - m0)) if m1 != m0 else float(f1)
    return freqs, mtf, max(f50, 1e-3)


def analyse_edge(img: np.ndarray, edge: EdgeFit, half_width: float = 12.0) -> dict | None:
    """One edge -> its MTF, MTF50 and Gaussian-equivalent sigma."""
    got = edge_spread_function(img, edge, half_width=half_width)
    if got is None:
        return None
    res = mtf_from_esf(*got)
    if res is None:
        return None
    freqs, mtf, f50 = res
    return {
        "side": edge.side,
        "freqs": freqs,
        "mtf": mtf,
        "mtf50": f50,
        "sigma": float(_SIGMA_FROM_F50 / f50),
        "slant_deg": edge.slant_deg,
        "n_points": len(edge.points),
    }


# --------------------------------------------------------------------------- #
# whole-image estimate
# --------------------------------------------------------------------------- #


def estimate_psf(
    linear_img: np.ndarray,
    fid: Fiducials,
    prior_sigma: float = 2.0,
    prior_uncertainty: float = 1.5,
) -> PSFEstimate:
    """Measure the capture PSF from whatever edges this image actually has.

    `linear_img` must be in luminance, not pixel values (see the module docstring).
    Degrades in three steps: usable collimation edges -> an opportunistic edge of
    the highest-gradient straight boundary available -> a prior. Each step is
    labelled in `.method` and carries a larger `.sigma_uncertainty`, and the
    certificate refuses to make a strong claim on a prior-only PSF.
    """
    img = np.asarray(linear_img, dtype=np.float64)
    results = [r for e in fid.mtf_edges if (r := analyse_edge(img, e)) is not None]

    if not results:
        opp = _opportunistic_edge(img, fid)
        if opp is not None:
            return PSFEstimate(
                sigma_x=opp, sigma_y=opp, method="opportunistic", n_edges=0,
                sigma_uncertainty=max(0.5, 0.5 * opp),
            )
        return PSFEstimate(
            sigma_x=prior_sigma, sigma_y=prior_sigma, method="prior", n_edges=0,
            sigma_uncertainty=prior_uncertainty,
        )

    # A left/right border is crossed horizontally, so it reports blur along x.
    sx = [r["sigma"] for r in results if r["side"] in ("left", "right")]
    sy = [r["sigma"] for r in results if r["side"] in ("top", "bottom")]
    allsig = [r["sigma"] for r in results]
    sigma_x = float(np.median(sx)) if sx else float(np.median(allsig))
    sigma_y = float(np.median(sy)) if sy else float(np.median(allsig))

    grid = np.linspace(0.0, 1.0, 129)
    mtf = np.mean([np.interp(grid, r["freqs"], r["mtf"], left=1.0, right=0.0) for r in results], axis=0)
    f50 = float(np.median([r["mtf50"] for r in results]))
    spread = float(np.std(allsig)) if len(allsig) > 1 else 0.25 * float(np.mean(allsig))

    return PSFEstimate(
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        freqs=grid,
        mtf=np.clip(mtf, 0.0, 1.0),
        mtf50=f50,
        method="slanted_edge",
        n_edges=len(results),
        sigma_uncertainty=max(spread, 0.05),
        per_edge=[{k: v for k, v in r.items() if k not in ("freqs", "mtf")} for r in results],
    )


def _opportunistic_edge(img: np.ndarray, fid: Fiducials) -> float | None:
    """Fallback sharpness from the sharpest boundary in the image.

    Deliberately biased pessimistic. Without a collimation border the sharpest
    thing available is usually the patient's own silhouette against direct
    exposure, which is genuinely soft -- scatter and the finite focal spot blur it
    before the phone ever sees it. So the sigma this returns is an *upper* bound
    on the capture blur, the resulting floor is an upper bound on the floor, and
    the certificate it produces is stricter than the truth rather than looser.
    """
    if fid.beamstop_mask is None or not fid.beamstop_mask.any():
        return None
    gy, gx = np.gradient(_ops.gaussian_blur(img, 0.8))
    grad = np.hypot(gy, gx)
    band = _ops.binary_dilate(fid.beamstop_mask, 6) & ~_ops.binary_erode(fid.beamstop_mask, 2)
    if band.sum() < 50:
        return None
    g = grad[band]
    strong = float(np.quantile(g, 0.98))
    if strong < 1e-9:
        return None
    span = float(np.quantile(img[band], 0.95) - np.quantile(img[band], 0.05))
    if span <= 0:
        return None
    # For a Gaussian edge of contrast `span`, peak gradient = span / (sigma*sqrt(2*pi)).
    sigma = span / (strong * np.sqrt(2 * np.pi))
    return float(np.clip(sigma, 0.4, 25.0))


def wiener_deconvolve(img: np.ndarray, sigma: float, nsr: float = 0.02) -> np.ndarray:
    """Wiener deconvolution against a Gaussian OTF of the measured width.

    Used only to make the recovered density map look right for a human and for
    the matched-filter detectability test. It is deliberately *not* in the path
    that computes the resolution floor: deconvolution amplifies noise by exactly
    the factor it restores contrast by, so it moves signal and noise together and
    cannot create information the channel did not carry. Treating a deconvolved
    image as if it had beaten the floor would be the classic way to fool yourself
    here.
    """
    a = np.asarray(img, dtype=np.float64)
    h, w = a.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    otf = np.exp(-2.0 * np.pi**2 * float(sigma) ** 2 * (fy**2 + fx**2))
    F = np.fft.rfft2(a)
    G = np.conj(otf) / (np.abs(otf) ** 2 + float(nsr))
    return np.fft.irfft2(F * G, s=(h, w))
