"""Smartphone re-photography degradation pipeline.

Simulates the artifacts you get when a clinician photographs an analog X-ray film
on a lightbox with a phone: motion/defocus blur, specular glare from the lightbox,
uneven shadow, a small capture angle, JPEG compression, and reduced resolution.

Design choices that matter for the project:

* Each degradation takes a *continuous* severity in [0, 1] (Phase 2 "six-sigma"
  extension: fine-grained severity rather than binary levels).
* `apply_smartphone_capture` returns the degraded image AND a record of exactly
  which ops fired and at what strength. That record is what you turn into the
  weak-supervision uncertainty labels ("heavy degradation -> high uncertainty is
  appropriate") in `manifest.py`.
* Written with numpy + Pillow only, so it runs and can be unit-tested without a
  GPU or torch. The torch `Dataset` in `dataset.py` calls into this.

Everything operates on HxW or HxWxC uint8 numpy arrays and returns the same.

TODO(Phase 2 extension): add a learned/GAN phone-camera model and a
real re-photography capture set, then ablate which best matches real artifacts.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(img)


def _to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img)


def _lerp(lo: float, hi: float, s: float) -> float:
    """Map severity s in [0,1] onto a physical parameter range [lo, hi]."""
    return lo + (hi - lo) * float(np.clip(s, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# individual degradations, each param'd by severity in [0, 1]
# --------------------------------------------------------------------------- #
def motion_blur(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Directional blur from hand shake. Severity -> kernel length (px)."""
    k = round(_lerp(3, 25, severity))
    if k < 3:
        return img
    angle = rng.uniform(0, np.pi)
    kernel = np.zeros((k, k), np.float32)
    cx = (k - 1) / 2.0
    for i in range(k):
        x = round(cx + (i - cx) * np.cos(angle))
        y = round(cx + (i - cx) * np.sin(angle))
        if 0 <= x < k and 0 <= y < k:
            kernel[y, x] = 1.0
    kernel /= max(kernel.sum(), 1.0)
    return _convolve(img, kernel)


def defocus_blur(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Out-of-focus blur. Severity -> gaussian radius."""
    radius = _lerp(0.5, 6.0, severity)
    return _to_np(_to_pil(img).filter(ImageFilter.GaussianBlur(radius)))


def glare(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Specular highlight from the lightbox: a bright soft blob added to the image."""
    h, w = img.shape[:2]
    cy, cx = rng.uniform(0.15, 0.85, size=2) * np.array([h, w])
    radius = _lerp(0.15, 0.5, severity) * min(h, w)
    strength = _lerp(40, 200, severity)
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius**2)))
    add = (strength * blob).astype(np.float32)
    if img.ndim == 3:
        add = add[..., None]
    return np.clip(img.astype(np.float32) + add, 0, 255).astype(np.uint8)


def shadow(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Uneven illumination: a smooth multiplicative gradient darkening one side."""
    h, w = img.shape[:2]
    angle = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = np.cos(angle) * (xx / w) + np.sin(angle) * (yy / h)
    grad = (grad - grad.min()) / (np.ptp(grad) + 1e-6)        # 0..1 across the frame
    depth = _lerp(0.05, 0.6, severity)
    mult = 1.0 - depth * grad                                  # darken toward one edge
    if img.ndim == 3:
        mult = mult[..., None]
    return np.clip(img.astype(np.float32) * mult, 0, 255).astype(np.uint8)


def capture_angle(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Small rotation from an off-axis phone. Severity -> max degrees."""
    max_deg = _lerp(1.0, 12.0, severity)
    deg = rng.uniform(-max_deg, max_deg)
    return _to_np(_to_pil(img).rotate(deg, resample=Image.BILINEAR, fillcolor=0))


def jpeg_compression(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Re-encode as JPEG at a severity-dependent quality (low q = strong artifacts)."""
    quality = round(_lerp(90, 20, severity))
    buf = io.BytesIO()
    _to_pil(img).convert("RGB" if img.ndim == 3 else "L").save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf))


def downscale(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Resolution loss: shrink then restore, so high-freq detail is gone."""
    factor = _lerp(1.0, 4.0, severity)
    h, w = img.shape[:2]
    small = _to_pil(img).resize((max(1, int(w / factor)), max(1, int(h / factor))), Image.BILINEAR)
    return _to_np(small.resize((w, h), Image.BILINEAR))


def _convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Depthwise 2D convolution with 'same' padding, numpy-only."""
    kh, kw = kernel.shape
    pad = (kh // 2, kw // 2)
    chans = [img] if img.ndim == 2 else [img[..., c] for c in range(img.shape[2])]
    out = []
    for ch in chans:
        p = np.pad(ch, ((pad[0], pad[0]), (pad[1], pad[1])), mode="reflect").astype(np.float32)
        acc = np.zeros_like(ch, dtype=np.float32)
        for i in range(kh):
            for j in range(kw):
                if kernel[i, j] != 0:
                    acc += kernel[i, j] * p[i : i + ch.shape[0], j : j + ch.shape[1]]
        out.append(np.clip(acc, 0, 255))
    stacked = np.stack(out, axis=-1) if img.ndim == 3 else out[0]
    return stacked.astype(np.uint8)


# registry: name -> function. Add new degradations here.
DEGRADATIONS = {
    "motion_blur": motion_blur,
    "defocus_blur": defocus_blur,
    "glare": glare,
    "shadow": shadow,
    "capture_angle": capture_angle,
    "jpeg": jpeg_compression,
    "downscale": downscale,
}


@dataclass
class DegradationRecord:
    """What was applied to one image, so it can be turned into an uncertainty label."""

    ops: dict[str, float] = field(default_factory=dict)  # op name -> severity actually used

    @property
    def total_severity(self) -> float:
        """A single scalar summarizing how degraded the image is (0 = clean), in [0, 1].

        Mean over the *possible* op slots, not over the ops that happened to
        fire, so images with more simultaneous degradations score higher.

        Averaging over only the fired ops -- which is what this did previously,
        contradicting this docstring -- makes the score *fall* as more artifacts
        stack up: one op at severity 0.9 scored 0.9 while six ops at 0.5 scored
        0.5. That inverts the quantity for its whole purpose. It is the
        signal-to-noise proxy behind the channel-capacity framing in
        docs/phase1_framing.md and the intended basis for the weak uncertainty
        label, both of which need "more degradation => higher score" to hold.
        """
        if not self.ops:
            return 0.0
        return float(sum(self.ops.values()) / len(DEGRADATIONS))


@dataclass
class SmartphoneDegradation:
    """Composite transform. Samples a subset of degradations per image.

    Parameters
    ----------
    severity : float
        Global degradation strength in [0, 1]. 0 returns the image untouched.
    p_each : float
        Probability each individual degradation is applied.
    ops : list[str] | None
        Which degradations are eligible; defaults to all of them.
    seed : int | None
        For reproducibility; None -> fresh randomness each call.
    """

    severity: float = 0.5
    p_each: float = 0.5
    ops: list[str] | None = None
    seed: int | None = None

    def __post_init__(self):
        self.ops = self.ops or list(DEGRADATIONS.keys())
        bad = set(self.ops) - set(DEGRADATIONS)
        if bad:
            raise ValueError(f"unknown degradations: {sorted(bad)}")

    def __call__(self, img: np.ndarray) -> tuple[np.ndarray, DegradationRecord]:
        rng = np.random.default_rng(self.seed)
        record = DegradationRecord()
        if self.severity <= 0:
            return img, record
        # Blurs are mutually exclusive (you rarely get both defocus and motion at once).
        order = [o for o in self.ops if o not in ("motion_blur", "defocus_blur")]
        blur_choice = None
        eligible_blurs = [o for o in ("motion_blur", "defocus_blur") if o in self.ops]
        if eligible_blurs and rng.random() < self.p_each:
            blur_choice = rng.choice(eligible_blurs)
            order = [blur_choice, *order]
        out = img
        for op in order:
            if op == blur_choice or rng.random() < self.p_each:
                # jitter the per-image severity around the global level
                s = float(np.clip(self.severity * rng.uniform(0.6, 1.2), 0, 1))
                out = DEGRADATIONS[op](out, s, rng)
                record.ops[op] = s
        return out, record


def degrade(img: np.ndarray, severity: float, seed: int | None = None) -> np.ndarray:
    """Convenience one-shot: return only the degraded image."""
    out, _ = SmartphoneDegradation(severity=severity, seed=seed)(img)
    return out
