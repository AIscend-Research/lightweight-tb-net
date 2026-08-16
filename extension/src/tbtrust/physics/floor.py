"""The density resolution floor: the smallest density step this photograph can carry.

This is the module the whole physics track exists to produce. Given a photo that
has been inverted to calibrated density, it computes -- per pixel, without labels,
without a network, and without any training data -- the smallest optical-density
difference that could still be distinguished there.

The claim it supports is not "the model is unsure". It is stronger and different
in kind: *the information is not in the photograph*. That is a statement about a
measurement channel, it is falsifiable, and `validate.py` falsifies it in
simulation by checking that an optimal detector with full knowledge of the lesion
succeeds above the predicted floor and fails below it.

The bound
---------
For a finding with unit-amplitude spatial profile t(x), density contrast dD, seen
through a capture with modulation transfer MTF(f) and per-pixel differential
density noise sigma_D(x), a matched filter -- the optimal linear detector for a
known signal in additive noise, so no detector can do better -- achieves

    SNR  =  dD * sqrt(E)  /  sigma_D ,        E = sum_f |T(f)|^2 MTF(f)^2

and requiring SNR >= k gives the floor

    dD_floor(x)  =  k * sigma_D(x) / sqrt(E) .

`E` is the template energy surviving the blur. It handles two things at once that
are usually hand-waved separately: the area gain from integrating a large finding
over many pixels, and the contrast loss from blurring a small one. No choice of
"characteristic frequency" is needed -- the whole spectrum is integrated.

`k` is the Rose criterion. k = 5 is the classical threshold for reliable
detection of a known signal by a human observer; k = 3 is "marginal, sometimes
seen". Both are reported.

Where sigma_D comes from, and why the veil dominates
----------------------------------------------------
`invert.py` derives the per-pixel differential density uncertainty. Written out,
with v the pixel value, c0 the black point, gamma the tone exponent, I the film
signal and V the veil:

    sigma_D  =  (gamma / ln10) * sigma_v / (v - c0)  *  (1 + V/I)

Three things to notice. The tone exponent scales it, which is why gamma is worth
measuring. It is inversely proportional to the *signal above the black point*, so
a crushed or underexposed region carries almost no density information no matter
how many bits you spend on it. And the veil enters as `1 + V/I` -- exactly the
reciprocal of the contrast compression `1/(1 + V/I)` -- so a 20% veil costs you
20% of your density resolution, everywhere it lands. That term is usually the
largest one in a real clinic photo, and it is the one nobody measures, because
measuring it needs a beam stop and nobody realised the film already has one.

Only the *random* half of the error budget appears here. A gamma or gain error is
common to the lesion and to the lung field two millimetres away and cancels in
their difference. Folding the systematic terms in would inflate the floor several
fold and turn a useful bound into a useless one. See `invert.py`.

What this bound is not
----------------------
It is a bound on the **measurement channel**, not on diagnostic difficulty. The
dominant obstacle to spotting a real nodule on a real chest radiograph is
anatomical clutter -- ribs, vessels, the heart border -- not photon noise, and a
lesion can sit comfortably above this floor and still be invisible against a rib.
`anatomical_noise` estimates that clutter and `density_floor(...,
include_anatomical=True)` will fold it in, but the certificate deliberately does
not, because the certificate's claim is the narrow, defensible one: *this
photograph destroyed information that the film had*. Conflating the two would
make the bound un-falsifiable, since anatomical clutter is present in the
original film too and is not something a retake can fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _ops
from .findings import FindingSpec
from .invert import CalibratedFilm, noise_sigma

# Rose criterion: SNR needed for reliable detection of a known signal.
ROSE_K = 5.0
ROSE_K_MARGINAL = 3.0


@dataclass
class FloorSpec:
    """Knobs for the bound. Defaults are the conservative, quotable choices."""

    rose_k: float = ROSE_K
    include_anatomical: bool = False
    max_template_px: int = 129        # template canvas cap; keeps FFTs small
    min_energy: float = 1e-6
    correlated_terms: tuple[str, ...] = ("veil_fit",)


@dataclass
class FloorMap:
    """The floor for one finding on one image, plus the terms that built it."""

    finding: FindingSpec
    floor: np.ndarray                 # per-pixel minimum resolvable |delta D|
    sigma_d: np.ndarray               # the differential density noise behind it
    template_energy: float            # E, surviving template energy
    template_energy_unblurred: float  # E with a perfect lens: the blur cost is the ratio
    rose_k: float
    px_per_mm: float
    terms: dict = field(default_factory=dict)     # per-term floor maps, for attribution

    @property
    def blur_penalty(self) -> float:
        """How much larger the floor is because of blur alone (>= 1)."""
        if self.template_energy <= 0:
            return float("inf")
        return float(np.sqrt(self.template_energy_unblurred / self.template_energy))

    def stats(self, mask: np.ndarray | None = None) -> dict:
        m = np.ones(self.floor.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if not m.any():
            return {"n_px": 0}
        f = self.floor[m]
        f = f[np.isfinite(f)]
        if f.size == 0:
            return {"n_px": 0}
        return {
            "n_px": int(f.size),
            "floor_median": float(np.median(f)),
            "floor_p90": float(np.quantile(f, 0.90)),
            "floor_min": float(f.min()),
            "floor_max": float(f.max()),
            "blur_penalty": self.blur_penalty,
        }


# --------------------------------------------------------------------------- #
# template energy
# --------------------------------------------------------------------------- #


def _template(size_px: float, canvas: int) -> np.ndarray:
    """Unit-amplitude Gaussian-profile lesion on a small canvas, energy-normalised
    only by its own shape (the amplitude stays 1, which is what makes E carry the
    area gain)."""
    n = int(canvas) | 1
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    rr = np.hypot(yy - c, xx - c)
    r = max(size_px / 2.0, 0.35)
    return np.exp(-0.5 * (rr / (r / 1.5)) ** 2)


def template_energy(
    finding: FindingSpec,
    px_per_mm: float,
    mtf_at,
    spec: FloorSpec | None = None,
) -> tuple[float, float]:
    """Surviving and unblurred template energy, E and E_0.

    Computed in the Fourier domain against the *measured* MTF rather than a
    Gaussian fit to it, so a lens with heavy tails or a JPEG with ringing is
    scored as it actually behaves. Parseval keeps the unblurred energy equal to
    the sum of t^2 in the spatial domain, which is a cheap correctness check and
    is asserted in the tests.
    """
    spec = spec or FloorSpec()
    size_px = max(finding.size_px(px_per_mm), 0.7)
    canvas = int(np.clip(int(6 * size_px) | 1, 15, spec.max_template_px))
    t = _template(size_px, canvas)
    n = t.shape[0]

    T = np.fft.rfft2(t)
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.rfftfreq(n)[None, :]
    fr = np.hypot(fy, fx)
    mtf = np.asarray(mtf_at(fr.ravel()), dtype=np.float64).reshape(fr.shape)

    # Parseval on a real FFT: the k=0 and (for even n) Nyquist columns are counted
    # once, every other column twice. Getting this wrong is a silent factor-of-two
    # in the floor, so the weights are explicit.
    wgt = np.full(fx.shape, 2.0)
    wgt[0, 0] = 1.0
    if n % 2 == 0:
        wgt[0, -1] = 1.0
    wgt = np.broadcast_to(wgt, T.shape).copy()
    wgt[:, 0] = 1.0
    if n % 2 == 0:
        wgt[:, -1] = 1.0

    e_blur = float(np.sum(wgt * np.abs(T * mtf) ** 2) / (n * n))
    e_open = float(np.sum(wgt * np.abs(T) ** 2) / (n * n))
    return max(e_blur, spec.min_energy), max(e_open, spec.min_energy)


# --------------------------------------------------------------------------- #
# noise decomposition
# --------------------------------------------------------------------------- #


def sigma_d_terms(cal: CalibratedFilm) -> dict[str, np.ndarray]:
    """Rebuild the differential density noise term by term, for attribution.

    Each entry is the sigma_D that *this term alone* would produce. The total is
    their quadrature sum plus the veil amplification, which multiplies all of
    them, so the veil is reported as a multiplicative factor rather than as a
    fourth additive term -- that is what it physically is, and presenting it as
    additive would make the attribution in `limiting_factor` wrong.
    """
    v = cal.pixel_values
    inv_ln10 = 1.0 / np.log(10.0)
    dL_dv = cal.tone.dL_dv(v)
    # dD/dL evaluated on the *film* signal, so the veil amplification is included
    # once, here, and not double counted below.
    dD_dL = inv_ln10 / np.maximum(cal.signal, 1e-9)

    sigma_read = noise_sigma(cal.noise_model, v)
    sigma_quant = np.full_like(v, cal.quantization_step / np.sqrt(12.0))

    out = {
        "sensor_noise": dD_dL * dL_dv * sigma_read,
        "quantization": dD_dL * dL_dv * sigma_quant,
        "veil_fit": dD_dL * np.full_like(v, float(cal.glare.veil_sigma)),
    }
    # The veil's contrast compression, isolated: how much larger every term is
    # than it would be on a veil-free capture of the same film.
    with np.errstate(divide="ignore", invalid="ignore"):
        amp = np.where(cal.signal > 0, (cal.signal + cal.veil) / np.maximum(cal.signal, 1e-9), 1.0)
    out["_veil_amplification"] = np.nan_to_num(amp, nan=1.0, posinf=1.0)
    return {k: np.nan_to_num(val, nan=0.0, posinf=0.0) for k, val in out.items()}


def anatomical_noise(cal: CalibratedFilm, size_px: float, mask: np.ndarray | None = None) -> np.ndarray:
    """Local density clutter at the finding's scale: ribs, vessels, overlapping tissue.

    Estimated as the local standard deviation of the recovered density after
    band-passing to the finding's scale -- structure larger than the finding is
    background a reader subtracts by eye, structure smaller is texture the matched
    filter averages away, and what is left at the finding's own scale is what
    actually masks it.

    Reported alongside the channel bound but excluded from the certificate by
    default; see the module docstring for why that separation matters.
    """
    d = np.asarray(cal.density, dtype=np.float64)
    s = max(size_px / 2.0, 0.6)
    band = _ops.gaussian_blur(d, s) - _ops.gaussian_blur(d, s * 3.0)
    if mask is not None and mask.any():
        band = np.where(mask, band, 0.0)
    local_var = _ops.gaussian_blur(band**2, max(2.0 * s, 2.0))
    return np.sqrt(np.maximum(local_var, 0.0))


# --------------------------------------------------------------------------- #
# the floor
# --------------------------------------------------------------------------- #


def density_floor(
    cal: CalibratedFilm,
    finding: FindingSpec,
    spec: FloorSpec | None = None,
    px_per_mm: float | None = None,
) -> FloorMap:
    """Per-pixel smallest resolvable density step for one finding on one photo."""
    spec = spec or FloorSpec()
    scale = float(px_per_mm if px_per_mm is not None else cal.px_per_mm)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("px_per_mm is unknown; pass it explicitly or supply a field quad")

    e_blur, e_open = template_energy(finding, scale, cal.psf.mtf_at, spec)
    terms = sigma_d_terms(cal)
    amp = terms.pop("_veil_amplification")

    if spec.include_anatomical:
        terms["anatomical"] = anatomical_noise(cal, finding.size_px(scale), cal.analysable_mask())

    # White versus correlated noise, and the distinction is not cosmetic.
    #
    # The matched filter beats down noise by sqrt(E) -- the area gain that lets a
    # large consolidation be seen at a contrast a single pixel could never resolve.
    # That gain is only real for noise that is independent pixel to pixel. The
    # error in the fitted veil surface is not: it varies on the scale of the glare
    # field, which is a sizeable fraction of the frame, so across a 45 mm
    # consolidation it is essentially a constant offset and averaging does nothing
    # to it.
    #
    # Crediting it with the area gain anyway is what made the bound come out
    # several times too optimistic for large findings in the detectability test --
    # the failure mode that matters most, since an optimistic floor certifies a
    # photograph as adequate when it is not.
    white = {k: v for k, v in terms.items() if k not in spec.correlated_terms}
    corr = {k: v for k, v in terms.items() if k in spec.correlated_terms}

    var_white = sum(t**2 for t in white.values()) if white else 0.0
    var_corr = sum(t**2 for t in corr.values()) if corr else 0.0
    sigma_d = np.sqrt(var_white + var_corr)

    gain = spec.rose_k / np.sqrt(e_blur)
    floor = spec.rose_k * np.sqrt(var_white / e_blur + var_corr)

    # Per-term floors: what the floor would be if only that term were present,
    # each with the area gain it actually earns. `veil` is expressed as the
    # difference between the real floor and the floor of an identical but
    # veil-free capture, which is the actionable quantity -- it is what a
    # successful retake would buy you.
    per_term = {name: gain * t for name, t in white.items()}
    per_term.update({name: spec.rose_k * np.asarray(t) for name, t in corr.items()})
    per_term["veil"] = floor * (1.0 - 1.0 / np.maximum(amp, 1e-9))
    per_term["blur"] = floor * (1.0 - np.sqrt(e_blur / max(e_open, 1e-12)))

    return FloorMap(
        finding=finding,
        floor=np.nan_to_num(floor, nan=np.inf, posinf=np.inf),
        sigma_d=sigma_d,
        template_energy=e_blur,
        template_energy_unblurred=e_open,
        rose_k=spec.rose_k,
        px_per_mm=scale,
        terms=per_term,
    )


def limiting_factor(fm: FloorMap, mask: np.ndarray | None = None) -> tuple[str, dict]:
    """Which term is costing the most density resolution, and by how much.

    Attribution is by leave-one-out on the quadrature sum for the additive terms
    and by the direct multiplicative penalty for veil and blur, so the numbers are
    comparable: each is "how much smaller would the floor be without this".
    `triage.py` turns the winner into an instruction, so it has to name the thing
    the operator can actually change.
    """
    m = np.ones(fm.floor.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if not m.any():
        return "unknown", {}
    total = float(np.median(fm.floor[m]))
    contrib = {}
    for name, arr in fm.terms.items():
        val = float(np.median(np.asarray(arr)[m]))
        contrib[name] = val
    if total <= 0 or not np.isfinite(total):
        return "unknown", contrib
    share = {k: v / total for k, v in contrib.items()}
    winner = max(share, key=lambda k: share[k])
    return winner, {"total_floor": total, "contribution": contrib, "share": share}


def floor_vs_severity_row(cal: CalibratedFilm, findings: list[FindingSpec],
                          mask: np.ndarray | None = None, spec: FloorSpec | None = None) -> dict:
    """One flat row of floors for a set of findings. The unit of the results tables."""
    m = mask if mask is not None else cal.lung_field_mask()
    row: dict = {}
    for f in findings:
        fm = density_floor(cal, f, spec)
        st = fm.stats(m)
        row[f"floor_{f.key}_median"] = st.get("floor_median", float("nan"))
        row[f"floor_{f.key}_p90"] = st.get("floor_p90", float("nan"))
        row[f"blurpen_{f.key}"] = fm.blur_penalty
    return row
