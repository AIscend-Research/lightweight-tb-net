"""Forward model: photographing a developed radiograph on a lightbox with a phone.

This is the generative half of the physics track, and it exists for one reason:
the certificate of insufficiency in `certificate.py` is a *falsifiable* claim, and
you can only falsify it where ground truth exists. Here the veiling glare, the
point-spread function, the tone curve and the true optical density map are all
known by construction, so `validate.py` can ask the blind estimator to recover
them and score it, and can ask whether the predicted detectability threshold
actually predicts detectability.

The chain, in the order the photons see it
------------------------------------------
1. **Film.** A density map D(x, y), carrying the three fiducials the inverse path
   needs: a lead side-marker at D_min, a collimation border stepping from D_min
   (outside the field, unexposed) to D_max (inside, direct exposure), and a
   direct-exposure rim between the patient silhouette and that border.
2. **Lightbox.** Luminance L = I(x, y) * 10^-D, where I is the box's own
   illumination field -- never uniform in a rural clinic, often a strip of
   fluorescent tubes or, increasingly, a window.
3. **Geometry.** The phone is not square to the film: a homography.
4. **Lens.** One point-spread function with two components: a narrow diffraction
   plus defocus plus hand-shake core, and a broad low-amplitude halo. That halo
   *is* veiling glare -- glare is not a separate additive phenomenon, it is the
   wings of the PSF, and modelling it as `(1-v)*core + v*halo` keeps the whole
   thing energy-conserving instead of inventing light. Specular reflection of the
   room off the film's surface is separate and genuinely additive, so it is.
5. **Sensor.** Shot noise (Poisson in collected electrons) then read noise.
6. **ISP.** An unknown monotone tone curve: exposure gain, an encoding gamma, a
   black-level lift, and a contrast S-curve.
7. **Quantiser + codec.** 8 bits, then JPEG.

A deliberate asymmetry: the forward tone curve here has four parameters
including an S-curve, and the estimator in `tone.py` fits a two-parameter
power law. That mismatch is on purpose. An estimator tested against its own
generative assumptions measures nothing, and the residual model error it leaves
behind is a real term in the error budget that `validate.py` reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from . import _ops
from .density import D_LUNG_RANGE, FilmModel, density_to_transmittance

# --------------------------------------------------------------------------- #
# fiducial geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FiducialSpec:
    """Where the physical calibration targets sit on the film, in fractions of it.

    Defaults follow ordinary chest-radiography practice: the beam is collimated
    to just inside the cassette leaving an unexposed border, the L/R lead marker
    is placed in an upper outer corner clear of the lung field, and the gap
    between the patient's silhouette and the collimation border receives the
    unattenuated beam.
    """

    collimation_margin: float = 0.06     # unexposed film outside the beam, as a fraction of the short side
    marker_size: float = 0.050           # lead marker height, as a fraction of image height
    marker_corner: Literal["tl", "tr", "bl", "br"] = "tr"
    marker_inset: float = 0.008          # gap between marker and collimation border
    marker_letter: Literal["L", "R"] = "L"
    # Width of the D_max rim inside the border. Wide enough to seat the marker
    # inside it, which is where radiographers actually place it -- a clear glyph on
    # opaque black is legible from across the room, and that same extreme local
    # contrast is the only reliable way to tell a lead marker from bright anatomy.
    # Seating it in the patient silhouette instead, as this originally did, halves
    # the detector's contrast score and costs `coverage: full` on most images.
    direct_exposure_width: float = 0.075


def _letter_mask(letter: str, h: int, w: int) -> np.ndarray:
    """A blocky 'L' or 'R' glyph. Shape only matters for the shape filter in
    `fiducials.py`, so this is a stroke drawing, not a font."""
    m = np.zeros((h, w), dtype=bool)
    t = max(1, round(0.26 * w))       # stroke thickness
    m[:, :t] = True                        # the common vertical stem
    if letter.upper() == "L":
        m[-t:, :] = True
    else:                                  # 'R': bowl on the upper half + a leg
        m[:t, : int(0.85 * w)] = True
        m[: h // 2, -t:] = True
        m[h // 2 - t // 2 : h // 2 + t - t // 2, : int(0.85 * w)] = True
        for i in range(h // 2, h):
            j = int((i - h // 2) / max(h - h // 2 - 1, 1) * (w - t))
            m[i, j : j + t] = True
    return m


def add_fiducials(
    density: np.ndarray,
    film: FilmModel | None = None,
    spec: FiducialSpec | None = None,
) -> tuple[np.ndarray, dict]:
    """Paint collimation border, direct-exposure rim and lead marker onto a density map.

    Returns the map and a dict of ground-truth masks and corner coordinates, so
    the detector in `fiducials.py` can be scored against where things actually are
    rather than only against whether the final numbers look plausible.
    """
    film = film or FilmModel()
    spec = spec or FiducialSpec()
    d = np.array(density, dtype=np.float64, copy=True)
    h, w = d.shape
    short = min(h, w)

    m = round(spec.collimation_margin * short)
    y0, y1, x0, x1 = m, h - m, m, w - m
    if y1 - y0 < 8 or x1 - x0 < 8:
        raise ValueError("collimation_margin leaves no field; use a larger image")

    # Outside the collimation border the film never saw the beam: base+fog, clear.
    outside = np.ones_like(d, dtype=bool)
    outside[y0:y1, x0:x1] = False
    d[outside] = film.d_min

    # Direct-exposure rim: full beam, no patient -> maximum density, optically black.
    rim = round(spec.direct_exposure_width * short)
    de = np.zeros_like(d, dtype=bool)
    de[y0 : y0 + rim, x0:x1] = True
    de[y1 - rim : y1, x0:x1] = True
    de[y0:y1, x0 : x0 + rim] = True
    de[y0:y1, x1 - rim : x1] = True
    d[de] = film.d_max

    # Lead side marker: zero transmission -> unexposed film -> base+fog, the
    # brightest thing in the frame, and the bright densitometry anchor.
    mh = max(6, min(round(spec.marker_size * h), max(6, rim - 2)))
    mw = max(4, round(0.62 * mh))
    inset = round(spec.marker_inset * short)
    # Centre the marker *within* the direct-exposure rim, not inboard of it.
    pad = max(1, (rim - mh) // 2) + inset
    my = y0 + pad if spec.marker_corner[0] == "t" else y1 - pad - mh
    mx = x0 + pad if spec.marker_corner[1] == "l" else x1 - pad - mw
    my = int(np.clip(my, y0, y1 - mh))
    mx = int(np.clip(mx, x0, x1 - mw))
    glyph = _letter_mask(spec.marker_letter, mh, mw)
    marker = np.zeros_like(d, dtype=bool)
    marker[my : my + mh, mx : mx + mw] = glyph
    d[marker] = film.d_min

    truth = {
        "field_rect": (y0, x0, y1, x1),
        "field_quad": np.array([[x0, y0], [x1 - 1, y0], [x1 - 1, y1 - 1], [x0, y1 - 1]], dtype=np.float64),
        "marker_mask": marker,
        "direct_exposure_mask": de,
        "outside_mask": outside,
        "marker_letter": spec.marker_letter,
    }
    return d, truth


# --------------------------------------------------------------------------- #
# a synthetic radiograph, for tests and for the validation harness
# --------------------------------------------------------------------------- #


def synthetic_chest_density(
    size: int = 512,
    rng: np.random.Generator | None = None,
    film: FilmModel | None = None,
    spec: FiducialSpec | None = None,
) -> tuple[np.ndarray, dict]:
    """A crude but density-correct chest radiograph: lung fields, mediastinum, ribs.

    Not anatomically serious. It only has to have the right *density statistics*
    -- a lung field in the D_LUNG_RANGE band with rib-scale structure on top --
    so that lesion contrast, local noise and the resolution floor are computed
    against something with realistic local variance rather than against flat grey.
    """
    rng = rng or np.random.default_rng(0)
    film = film or FilmModel()
    h = w = int(size)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ny, nx = yy / h, xx / w

    lo, hi = D_LUNG_RANGE
    d = np.full((h, w), lo + 0.15 * (hi - lo))                    # soft tissue background

    # Two lung fields: air, so high exposure, so high density (dark on the box).
    for cx in (0.33, 0.67):
        r = np.sqrt(((nx - cx) / 0.19) ** 2 + ((ny - 0.52) / 0.30) ** 2)
        d += (hi - d) * np.clip(1.6 * (1.0 - r), 0.0, 1.0) ** 0.7

    # Mediastinum + spine: dense tissue and bone, low density, bright.
    spine = np.exp(-((nx - 0.5) ** 2) / (2 * 0.045**2))
    d -= (d - lo) * 0.85 * spine
    heart = np.exp(-(((nx - 0.56) / 0.15) ** 2 + ((ny - 0.66) / 0.17) ** 2))
    d -= (d - lo) * 0.55 * heart

    # Ribs: the dominant structured "background" a lesion has to be seen against.
    ribs = 0.16 * np.sin(2 * np.pi * (ny * 9.0 + 0.5 * np.cos(2 * np.pi * (nx - 0.5) * 0.8)))
    d += ribs * (d > lo + 0.25 * (hi - lo))

    d += rng.normal(0.0, 0.012, (h, w))                            # film grain
    d = np.clip(_ops.gaussian_blur(d, 0.8), film.d_min, film.d_max)
    return add_fiducials(d, film=film, spec=spec)


def lesion_profile(
    shape_hw: tuple[int, int],
    center_yx: tuple[float, float],
    diameter_px: float,
    profile: Literal["disc", "gaussian"] = "gaussian",
) -> np.ndarray:
    """Unit-amplitude spatial profile of a lesion. The single source of truth for
    the lesion's shape, shared by the forward model and the matched filter.

    Sharing matters: the matched filter is only optimal for the signal that was
    actually inserted, so if the two ever drift apart the detectability test
    quietly stops measuring what it claims to.
    """
    h, w = shape_hw
    cy, cx = center_yx
    r = float(diameter_px) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    if profile == "disc":
        return _ops.gaussian_blur((rr <= r).astype(np.float64), max(0.6, r * 0.12))
    return np.exp(-0.5 * (rr / max(r / 1.5, 0.5)) ** 2)


def insert_lesion(
    density: np.ndarray,
    center_yx: tuple[float, float],
    diameter_px: float,
    delta_d: float,
    profile: Literal["disc", "gaussian"] = "gaussian",
    clip: tuple[float, float] = (0.0, 6.0),
) -> np.ndarray:
    """Add a lesion of known density contrast. Used by the detectability test.

    A TB lesion is *more* attenuating than the lung it replaces, so it lowers
    exposure and therefore lowers density: `delta_d` is subtracted. Pass a
    negative value for a cavity's air-filled centre.

    The result is clipped to a physical density range. Without it a sweep that
    probes contrasts around a very high floor drives density negative, which means
    transmittance above 1 -- film emitting more light than the lightbox put into
    it -- and the forward model then overflows into infinities and NaNs several
    steps later, at the Poisson draw, where the cause is unrecognisable.
    """
    d = np.array(density, dtype=np.float64, copy=True)
    shape = lesion_profile(d.shape, center_yx, diameter_px, profile)
    return np.clip(d - float(delta_d) * shape, clip[0], clip[1])


def lesion_template(shape_hw: tuple[int, int], center_yx, diameter_px: float,
                    profile: Literal["disc", "gaussian"] = "gaussian") -> np.ndarray:
    """The unit-amplitude spatial profile of `insert_lesion`, for a matched filter."""
    return lesion_profile(shape_hw, center_yx, diameter_px, profile)


# --------------------------------------------------------------------------- #
# capture parameters
# --------------------------------------------------------------------------- #


@dataclass
class CaptureParams:
    """Every knob of the forward model. All lengths in pixels of the output photo."""

    # --- lens: PSF core (defocus + hand shake) and halo (veiling glare) -------
    psf_sigma: float = 1.2               # narrow core, isotropic part
    motion_length: float = 0.0           # linear smear length; 0 disables
    motion_angle: float = 0.0            # radians
    glare_fraction: float = 0.03         # fraction of light redistributed into the halo
    glare_sigma_frac: float = 0.12       # halo width as a fraction of the image short side

    # --- specular reflection off the film surface (genuinely additive) --------
    flare_amplitude: float = 0.0         # in units of mean scene luminance
    flare_center: tuple[float, float] = (0.3, 0.3)   # (fy, fx) in [0,1]
    flare_sigma_frac: float = 0.18

    # --- lightbox illumination non-uniformity ---------------------------------
    illum_depth: float = 0.10            # peak-to-trough fraction
    illum_angle: float = 0.7             # radians, direction of the gradient
    illum_vignette: float = 0.06         # radial falloff at the corners

    # --- geometry --------------------------------------------------------------
    rotation_deg: float = 0.0
    keystone: float = 0.0                # 0 = square-on; ~0.08 is a noticeably tilted phone

    # --- sensor ---------------------------------------------------------------
    full_well: float = 6000.0            # electrons at luminance 1.0; sets shot noise
    read_noise_e: float = 6.0            # electrons RMS

    # --- ISP tone curve: v = black + (1-black) * S( (gain*L)^(1/gamma) ) -------
    tone_gain: float = 1.0
    tone_gamma: float = 2.2
    tone_black: float = 0.02
    tone_scurve: float = 0.25            # 0 = pure power law; the estimator does not model this

    # --- output ----------------------------------------------------------------
    bit_depth: int = 8
    jpeg_quality: int | None = 85        # None disables the codec


@dataclass
class CaptureTruth:
    """Ground truth for one simulated capture. Everything `validate.py` scores against."""

    params: CaptureParams
    density_canonical: np.ndarray        # true D in the rectified field frame
    field_quad_photo: np.ndarray         # true collimation corners in the photo, (x, y) x4
    homography: np.ndarray               # canonical field -> photo
    fiducial_truth: dict = field(default_factory=dict)
    glare_field_true: np.ndarray | None = None   # true veil luminance in the photo frame
    luminance_true: np.ndarray | None = None     # pre-lens scene luminance in the photo frame
    signal_true: np.ndarray | None = None        # post-lens, veil-free signal: (1-v) * (L * core)

    @property
    def veil_fraction_true(self) -> float:
        """Veil over signal in the middle of the frame, defined exactly as the
        estimator defines it (`veil / signal`), so the two are comparable.

        Comparing the veil against the *pre-lens* luminance instead is a trap: the
        veil is a redistribution of that same light, so the ratio can exceed one
        and even go negative once you subtract, which looks like a catastrophic
        estimator failure when it is only a mismatched denominator.
        """
        if self.glare_field_true is None or self.signal_true is None:
            return float("nan")
        h, w = self.glare_field_true.shape
        c = (slice(int(0.3 * h), int(0.7 * h)), slice(int(0.3 * w), int(0.7 * w)))
        return float(np.median(self.glare_field_true[c] / np.maximum(self.signal_true[c], 1e-12)))

    @property
    def psf_sigma_effective(self) -> float:
        """Gaussian-equivalent width of the PSF core, including motion smear.

        A line of length L has variance L^2/12 along its axis; adding it in
        quadrature with the isotropic core is the right scalar summary to compare
        against what the slanted edge recovers *along one direction*, which is
        what `psf.py` actually measures.
        """
        p = self.params
        motion_var = (p.motion_length**2) / 12.0
        return float(np.sqrt(p.psf_sigma**2 + motion_var))


# --------------------------------------------------------------------------- #
# the forward model
# --------------------------------------------------------------------------- #


def _tone_forward(lum: np.ndarray, p: CaptureParams) -> np.ndarray:
    """Scene luminance -> normalised pixel value in [0, 1]. Monotone by construction."""
    u = np.clip(p.tone_gain * np.maximum(lum, 0.0), 0.0, 1.0) ** (1.0 / p.tone_gamma)
    s = float(p.tone_scurve)
    # u + s*u*(1-u)*(2u-1) has derivative >= 1 - s/2, so it stays monotone for s < 2.
    u = np.clip(u + s * u * (1.0 - u) * (2.0 * u - 1.0), 0.0, 1.0)
    return np.clip(p.tone_black + (1.0 - p.tone_black) * u, 0.0, 1.0)


def _illumination_field(shape: tuple[int, int], p: CaptureParams) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    ny, nx = yy / max(h - 1, 1), xx / max(w - 1, 1)
    grad = np.cos(p.illum_angle) * (nx - 0.5) + np.sin(p.illum_angle) * (ny - 0.5)
    field = 1.0 + p.illum_depth * grad
    r2 = (nx - 0.5) ** 2 + (ny - 0.5) ** 2
    return field * (1.0 - p.illum_vignette * (r2 / 0.5))


def _geometry(shape: tuple[int, int], p: CaptureParams) -> np.ndarray:
    """Homography taking the film's own pixel frame into the photo frame."""
    h, w = shape
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
    k = float(p.keystone)
    dst = np.array(
        [
            [0 + k * w, 0 + k * h * 0.5],
            [w - 1 - k * w * 0.35, 0 - k * h * 0.15],
            [w - 1 - k * w * 0.1, h - 1 - k * h * 0.4],
            [0 + k * w * 0.5, h - 1 + k * h * 0.1],
        ],
        dtype=np.float64,
    )
    th = np.deg2rad(p.rotation_deg)
    c, s = np.cos(th), np.sin(th)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    R = np.array([[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy], [0, 0, 1]])
    dsth = np.hstack([dst, np.ones((4, 1))]) @ R.T
    return _ops.estimate_homography(src, dsth[:, :2] / dsth[:, 2:3])


def capture(
    density: np.ndarray,
    params: CaptureParams | None = None,
    fiducial_truth: dict | None = None,
    film: FilmModel | None = None,
    rng: np.random.Generator | None = None,
    canonical_size: int = 384,
) -> tuple[np.ndarray, CaptureTruth]:
    """Photograph a density map. Returns a uint8 photo and its ground truth.

    The returned photo is what a blind estimator is allowed to see. Everything
    else -- the veil, the PSF, the tone parameters, the true density in the
    rectified frame -- goes into `CaptureTruth` and is only for scoring.
    """
    p = params or CaptureParams()
    film = film or FilmModel()
    rng = rng or np.random.default_rng(0)
    d = np.asarray(density, dtype=np.float64)
    h, w = d.shape
    short = min(h, w)

    # 1-2. film on a lightbox
    tau = density_to_transmittance(d)
    lum = tau * _illumination_field((h, w), p)

    # 3. off-axis phone
    H = _geometry((h, w), p)
    lum_photo = _ops.warp_perspective(lum, H, (h, w), fill=0.0)

    # 4. one lens PSF with a narrow core and a broad halo. The halo is the veil.
    core = _ops.gaussian_blur(lum_photo, p.psf_sigma)
    if p.motion_length > 0:
        core = _ops.convolve2d(core, _ops.motion_kernel(p.motion_length, p.motion_angle))
    halo_sigma = max(2.0, p.glare_sigma_frac * short)
    halo = _ops.gaussian_blur(lum_photo, halo_sigma)
    v = float(np.clip(p.glare_fraction, 0.0, 0.9))
    signal_true = (1.0 - v) * core                # the part that carries film detail
    lum_lens = signal_true + v * halo
    veil_true = v * halo                          # what the beam-stop probe should recover

    # specular reflection of the room off the film's glossy surface: additive
    if p.flare_amplitude > 0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        fy, fx = p.flare_center[0] * h, p.flare_center[1] * w
        fs = max(2.0, p.flare_sigma_frac * short)
        blob = np.exp(-((yy - fy) ** 2 + (xx - fx) ** 2) / (2 * fs**2))
        flare = p.flare_amplitude * float(np.mean(lum_photo)) * blob
        lum_lens = lum_lens + flare
        veil_true = veil_true + flare

    # 5. sensor: shot noise in electrons, then read noise.
    # nan_to_num before the Poisson draw: any non-finite luminance that slipped
    # through raises "lam value too large" from deep inside numpy, with nothing in
    # the traceback pointing at the density map that actually caused it.
    electrons = np.nan_to_num(np.maximum(lum_lens, 0.0), nan=0.0, posinf=1e6) * p.full_well
    electrons = rng.poisson(np.clip(electrons, 0, 1e9)).astype(np.float64)
    electrons += rng.normal(0.0, p.read_noise_e, electrons.shape)
    lum_sensor = np.clip(electrons / p.full_well, 0.0, None)

    # 6-7. ISP, quantiser, codec
    vpix = _tone_forward(lum_sensor, p)
    levels = 2**int(p.bit_depth) - 1
    photo = np.clip(np.round(vpix * levels), 0, levels).astype(np.uint8 if p.bit_depth <= 8 else np.uint16)
    if p.jpeg_quality is not None and p.bit_depth <= 8:
        photo = _jpeg_roundtrip(photo, int(p.jpeg_quality))

    # ground truth in the rectified field frame
    ft = fiducial_truth or {}
    quad_film = ft.get("field_quad")
    if quad_film is None:
        quad_film = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
    qh = np.hstack([quad_film, np.ones((4, 1))]) @ H.T
    quad_photo = qh[:, :2] / qh[:, 2:3]

    cs = int(canonical_size)
    dst = np.array([[0, 0], [cs - 1, 0], [cs - 1, cs - 1], [0, cs - 1]], dtype=np.float64)
    H_film_to_canon = _ops.estimate_homography(quad_film, dst)
    d_canon = _ops.warp_perspective(d, H_film_to_canon, (cs, cs), fill=film.d_min)

    truth = CaptureTruth(
        params=p,
        density_canonical=d_canon,
        field_quad_photo=quad_photo,
        homography=H,
        fiducial_truth=ft,
        glare_field_true=veil_true,
        luminance_true=lum_photo,
        signal_true=signal_true,
    )
    return photo, truth


def _jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img, mode="L").save(buf, "JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf))


# --------------------------------------------------------------------------- #
# severity -> parameters, so the physics track can share the project's axis
# --------------------------------------------------------------------------- #


def sample_params(severity: float, rng: np.random.Generator | None = None) -> CaptureParams:
    """Draw capture parameters at a severity in [0, 1], matching the project's sweep axis.

    Ranges are chosen so severity 0 is a careful capture (tripod-steady, square
    on, shaded lightbox) and severity 1 is the worst plausible real-world case
    that a clinician would still consider submitting: several pixels of shake, a
    fifth of the light veiled, a hard specular reflection and a heavily
    compressed JPEG. Nothing here is adversarial -- `eval/adversarial_degradation.py`
    already owns that job.
    """
    rng = rng or np.random.default_rng()
    s = float(np.clip(severity, 0.0, 1.0))

    def lerp(lo, hi):
        return lo + (hi - lo) * s

    return CaptureParams(
        psf_sigma=float(rng.uniform(0.6, 1.1) + lerp(0.0, 3.0)),
        motion_length=float(max(0.0, rng.uniform(-1.0, 1.0) + lerp(0.0, 9.0))),
        motion_angle=float(rng.uniform(0, np.pi)),
        glare_fraction=float(np.clip(rng.uniform(0.8, 1.2) * lerp(0.01, 0.22), 0.0, 0.5)),
        glare_sigma_frac=float(rng.uniform(0.08, 0.18)),
        flare_amplitude=float(max(0.0, rng.uniform(-0.4, 1.0) * lerp(0.0, 0.5))),
        flare_center=(float(rng.uniform(0.15, 0.85)), float(rng.uniform(0.15, 0.85))),
        illum_depth=float(lerp(0.03, 0.45) * rng.uniform(0.7, 1.3)),
        illum_angle=float(rng.uniform(0, 2 * np.pi)),
        # Always at least ~1 degree of tilt, even at severity 0. Nobody holds a
        # phone perfectly square to a lightbox, and the exception matters here:
        # ISO 12233 needs a slanted edge for its sub-pixel phase diversity, so a
        # simulated 0.0-degree capture has no measurable MTF and the PSF silently
        # falls back to the opportunistic estimate. That would be an artefact of
        # the simulator, not a property of the method.
        rotation_deg=float(rng.choice([-1.0, 1.0]) * (1.0 + abs(rng.normal(0, 0.6)) * lerp(1.0, 8.0))),
        keystone=float(abs(rng.normal(0, 0.5)) * lerp(0.005, 0.09)),
        full_well=float(rng.uniform(3000, 9000)),
        read_noise_e=float(rng.uniform(3.0, 8.0) + lerp(0.0, 10.0)),
        tone_gain=float(rng.uniform(0.7, 1.4)),
        tone_gamma=float(rng.uniform(1.8, 2.6)),
        tone_black=float(rng.uniform(0.0, 0.06)),
        tone_scurve=float(rng.uniform(0.0, 0.45)),
        jpeg_quality=round(lerp(95, 30)),
    )


def simulate(
    display_image: np.ndarray | None = None,
    severity: float = 0.5,
    rng: np.random.Generator | None = None,
    size: int = 512,
    film: FilmModel | None = None,
    spec: FiducialSpec | None = None,
    canonical_size: int = 384,
) -> tuple[np.ndarray, CaptureTruth]:
    """One-call convenience: display image (or a synthetic film) -> photo + truth.

    `display_image` is a normalised [0, 1] or uint8 chest radiograph from one of
    the public datasets. It is mapped to density with the archive transfer in
    `density.display_to_density` and then has fiducials painted on, because the
    public archives have usually cropped them off -- which is the load-bearing
    assumption of this whole track and is exactly what `scripts/audit_fiducials.py`
    measures on real data rather than assuming.
    """
    from .density import display_to_density

    rng = rng or np.random.default_rng()
    film = film or FilmModel()
    if display_image is None:
        d, ft = synthetic_chest_density(size=size, rng=rng, film=film, spec=spec)
    else:
        x = np.asarray(display_image, dtype=np.float64)
        if x.ndim == 3:
            x = x.mean(axis=-1)
        if x.max() > 1.5:
            x = x / 255.0
        if x.shape[0] != size or x.shape[1] != size:
            gy, gx = np.mgrid[0:size, 0:size].astype(np.float64)
            x = _ops.bilinear_sample(
                x, gy * (x.shape[0] - 1) / max(size - 1, 1), gx * (x.shape[1] - 1) / max(size - 1, 1)
            )
        d, ft = add_fiducials(display_to_density(x, film), film=film, spec=spec)

    params = sample_params(severity, rng)
    return capture(d, params, fiducial_truth=ft, film=film, rng=rng, canonical_size=canonical_size)
