"""Find the calibration targets a chest radiograph already carries.

Three physical objects of *known* optical density are present in an ordinary
clinical chest film, and each buys one of the three unknowns in the phone-capture
channel:

===============================  ==============  =====================================
object                           known density   what it measures
===============================  ==============  =====================================
lead L/R side marker             D_min           bright densitometry anchor
direct-exposure region           D_max           the veiling-glare beam stop
collimation border               step D_min/D_max  the capture PSF (slanted edge)
===============================  ==============  =====================================

See `density.py` for why the lead marker is the *bright* anchor and the
direct-exposure region the dark one -- the project brief has these the other way
round, which is the X-ray transmission convention rather than the developed-film
one.

The honest caveat, and why this module reports coverage rather than just answers
-------------------------------------------------------------------------------
Public archives are cropped, windowed and rescaled by whoever assembled them, and
they frequently discard exactly these regions -- a tight crop to the lung fields
removes the collimation border, the direct-exposure rim and the marker in one
stroke. This whole track's load-bearing assumption is that enough of them survive.

So `detect` never pretends. It returns a `Coverage` grading and each estimator
downstream degrades explicitly:

* ``FULL``     -- marker + beam stop + >= 2 collimation edges. Everything measurable.
* ``PARTIAL``  -- a beam stop but no marker and/or no usable edge. Glare is still
                  measured directly; the tone curve falls back to a gamma prior and
                  the PSF to an edge-of-opportunity, both with inflated uncertainty.
* ``NONE``     -- not even a dark surround. No physics-derived bound is available
                  and the certificate must abstain rather than guess.

`scripts/audit_fiducials.py` runs this over a whole manifest and reports the
coverage histogram per clinic. Run it before believing any of the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from . import _ops

# Acceptance gate for the lead-marker detector, chosen from its measured operating
# characteristic on the simulated corpus: at 0.6 every true detection is kept and
# roughly a third of the false candidates are cut, versus 0.5. The asymmetry is
# deliberate. A missed marker only downgrades an image to PARTIAL, where the film
# base still anchors the density scale and the beam stop still measures the glare.
# A *false* marker is worse: it feeds `tone.fit_illumination` a bogus interior
# sample of the lightbox, and that error propagates into every density in the frame.
MARKER_ACCEPT = 0.6


class Coverage(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


@dataclass
class EdgeFit:
    """One collimation edge: a fitted line plus the samples behind it."""

    side: str                       # 'left' | 'right' | 'top' | 'bottom'
    line: tuple[float, float, float]   # a*x + b*y + c = 0, normalised so a^2+b^2=1
    points: np.ndarray              # the sub-pixel edge samples used, as (x, y)
    residual_px: float              # RMS distance of samples from the line
    slant_deg: float                # angle off the ideal axis; ISO 12233 wants a few degrees
    contrast: float                 # bright-side minus dark-side level, in [0, 1]

    @property
    def usable_for_mtf(self) -> bool:
        """ISO 12233 needs a straight, slanted, high-contrast edge with many samples.

        A perfectly axis-aligned edge (slant ~ 0) gives no sub-pixel phase
        diversity, so the oversampled ESF has gaps and the recovered MTF is
        garbage; too much slant and the projection mixes genuinely different
        parts of the edge. 1-15 degrees is the usable window. The residual gate
        rejects an edge that is not actually straight -- a torn film corner, or a
        mis-detection that stitched two sides together.
        """
        return (
            len(self.points) >= 24
            and self.residual_px < 1.5
            and 0.7 <= abs(self.slant_deg) <= 15.0
            and self.contrast > 0.10
        )


@dataclass
class Fiducials:
    """What was found in one photo, and how much of the physics it unlocks."""

    shape: tuple[int, int]
    coverage: Coverage = Coverage.NONE
    field_quad: np.ndarray | None = None          # (x, y) x4, TL TR BR BL
    field_mask: np.ndarray | None = None
    edges: list[EdgeFit] = field(default_factory=list)
    marker_mask: np.ndarray | None = None
    marker_confidence: float = 0.0
    marker_bbox: tuple[int, int, int, int] | None = None
    beamstop_mask: np.ndarray | None = None
    beamstop_source: str = "none"                 # 'collimated_rim' | 'dark_surround' | 'none'
    outside_mask: np.ndarray | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def mtf_edges(self) -> list[EdgeFit]:
        return [e for e in self.edges if e.usable_for_mtf]

    @property
    def has_bright_anchor(self) -> bool:
        return self.marker_mask is not None and self.marker_confidence >= MARKER_ACCEPT

    @property
    def has_beamstop(self) -> bool:
        return self.beamstop_mask is not None and int(self.beamstop_mask.sum()) > 0

    def summary(self) -> dict:
        """Flat, JSON-friendly row. This is what the coverage audit tabulates."""
        return {
            "coverage": self.coverage.value,
            "has_marker": bool(self.has_bright_anchor),
            "marker_confidence": float(self.marker_confidence),
            "n_edges": len(self.edges),
            "n_mtf_edges": len(self.mtf_edges),
            "has_beamstop": bool(self.has_beamstop),
            "beamstop_source": self.beamstop_source,
            "beamstop_px": int(self.beamstop_mask.sum()) if self.has_beamstop else 0,
            "has_field_quad": self.field_quad is not None,
            **{f"diag_{k}": v for k, v in self.diagnostics.items()},
        }


# --------------------------------------------------------------------------- #
# sub-pixel edge scanning
# --------------------------------------------------------------------------- #


def _subpixel_crossing(profile: np.ndarray, level: float, from_start: bool) -> float | None:
    """First bright->dark crossing of `level` walking inward, linearly interpolated.

    Sub-pixel matters: the edge angle is fitted from these crossings, and a
    half-pixel bias across a 400-pixel edge is a whole degree of slant error,
    which lands straight in the recovered MTF.

    Only the *first* transition counts, and only its direction is constrained.
    Requiring the far end of the ray to still be dark -- the obvious way to write
    this -- fails on almost every scanline of a real film: the ray leaves the
    bright unexposed margin, crosses the dark direct-exposure rim, and then runs
    into brightly-attenuating anatomy, so it ends where it started. That rejection
    left about a dozen surviving samples per side out of two hundred, and the
    collimation quad fitted through them was off by tens of pixels while still
    looking plausible.
    """
    p = np.asarray(profile, dtype=np.float64)
    if p.size < 2:
        return None
    seq = p if from_start else p[::-1]
    if seq[0] < level:
        return None                                # ray does not begin on the film margin
    below = np.nonzero(seq < level)[0]
    if below.size == 0:
        return None                                # never gets dark: no border on this ray
    idx = int(below[0])
    if idx == 0:
        return None
    a, b = seq[idx - 1], seq[idx]
    if abs(b - a) < 1e-9:
        return None
    pos = (idx - 1) + (level - a) / (b - a)
    return float(pos if from_start else (len(p) - 1 - pos))


def _scan_side(img: np.ndarray, side: str, level: float, span_frac: float, depth_frac: float):
    """Collect sub-pixel bright->dark crossings walking inward from one border."""
    h, w = img.shape
    pts = []
    if side in ("left", "right"):
        lo, hi = int(h * (0.5 - span_frac / 2)), int(h * (0.5 + span_frac / 2))
        depth = max(4, int(w * depth_frac))
        for y in range(lo, hi):
            row = img[y, :depth] if side == "left" else img[y, w - depth :]
            c = _subpixel_crossing(row, level, from_start=(side == "left"))
            if c is not None:
                pts.append((c if side == "left" else (w - depth + c), float(y)))
    else:
        lo, hi = int(w * (0.5 - span_frac / 2)), int(w * (0.5 + span_frac / 2))
        depth = max(4, int(h * depth_frac))
        for x in range(lo, hi):
            col = img[:depth, x] if side == "top" else img[h - depth :, x]
            c = _subpixel_crossing(col, level, from_start=(side == "top"))
            if c is not None:
                pts.append((float(x), c if side == "top" else (h - depth + c)))
    return np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 2))


def _fit_edge(side: str, pts: np.ndarray, contrast: float) -> EdgeFit | None:
    """Total-least-squares line through the crossings, plus its quality gates."""
    if len(pts) < 12:
        return None
    x, y = pts[:, 0], pts[:, 1]
    if side in ("left", "right"):
        # near-vertical: fit x = m*y + c so the parameterisation stays well-posed
        m, c = _ops.fit_line_tls(y, x)
        a, b, cc = 1.0, -m, -c
        slant = np.degrees(np.arctan(m))
    else:
        m, c = _ops.fit_line_tls(x, y)
        a, b, cc = -m, 1.0, -c
        slant = np.degrees(np.arctan(m))
    n = np.hypot(a, b)
    a, b, cc = a / n, b / n, cc / n
    resid = float(np.sqrt(np.mean((a * x + b * y + cc) ** 2)))
    # Reject the outer 10% of samples and refit: a torn corner or a label stuck to
    # the film otherwise drags the whole line and quietly inflates the slant.
    d = np.abs(a * x + b * y + cc)
    keep = d <= np.quantile(d, 0.9)
    if keep.sum() >= 12 and resid > 0.3:
        return _fit_edge(side, pts[keep], contrast)
    return EdgeFit(side=side, line=(a, b, cc), points=pts, residual_px=resid,
                   slant_deg=float(slant), contrast=float(contrast))


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #


def _normalize(photo: np.ndarray) -> np.ndarray:
    x = np.asarray(photo, dtype=np.float64)
    if x.ndim == 3:
        x = x.mean(axis=-1)
    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def detect_collimation(img: np.ndarray, border_frac: float = 0.03) -> tuple[list[EdgeFit], np.ndarray | None, dict]:
    """Locate the collimation border: the step from unexposed film to direct exposure.

    Detected by the one property that distinguishes it from anatomy -- it is the
    boundary of a *bright frame* around a *dark field*, present on all four sides.
    A tight crop to the lung fields destroys it, which is precisely the case the
    coverage grading has to catch.
    """
    h, w = img.shape
    bstrip = np.concatenate(
        [
            img[: max(2, int(h * border_frac)), :].ravel(),
            img[-max(2, int(h * border_frac)) :, :].ravel(),
            img[:, : max(2, int(w * border_frac))].ravel(),
            img[:, -max(2, int(w * border_frac)) :].ravel(),
        ]
    )
    ch, cw = int(h * 0.25), int(w * 0.25)
    centre = img[ch : h - ch, cw : w - cw]
    bright = float(np.quantile(bstrip, 0.75))
    dark = float(np.quantile(centre, 0.05))
    diag = {"border_level": bright, "field_dark_level": dark, "border_contrast": bright - dark}

    # No bright frame -> no collimation border in this image. Say so.
    if bright - dark < 0.12 or bright < 0.25:
        return [], None, diag

    level = 0.5 * (bright + dark)
    edges = []
    for side in ("left", "right", "top", "bottom"):
        pts = _scan_side(img, side, level, span_frac=0.7, depth_frac=0.25)
        e = _fit_edge(side, pts, contrast=bright - dark)
        if e is not None:
            edges.append(e)
    diag["n_sides_found"] = len(edges)
    if len(edges) < 4:
        return edges, None, diag

    by = {e.side: e for e in edges}
    try:
        corners = np.array(
            [
                _ops.line_intersection(by["top"].line, by["left"].line),
                _ops.line_intersection(by["top"].line, by["right"].line),
                _ops.line_intersection(by["bottom"].line, by["right"].line),
                _ops.line_intersection(by["bottom"].line, by["left"].line),
            ],
            dtype=np.float64,
        )
    except ValueError:
        return edges, None, diag

    if not np.all(np.isfinite(corners)):
        return edges, None, diag
    # Sanity: the field must be a plausible fraction of the frame. A degenerate
    # or inverted quad means the four lines did not describe one rectangle.
    quad = _ops.order_quad(corners)
    area = 0.5 * abs(
        np.dot(quad[:, 0], np.roll(quad[:, 1], -1)) - np.dot(quad[:, 1], np.roll(quad[:, 0], -1))
    )
    diag["field_area_frac"] = float(area / (h * w))
    if not 0.25 <= area / (h * w) <= 1.05:
        return edges, None, diag
    return edges, quad, diag


def _quad_mask(quad: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterise a convex quad by half-plane intersection."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    inside = np.ones((h, w), dtype=bool)
    c = quad.mean(axis=0)
    for i in range(4):
        (x0, y0), (x1, y1) = quad[i], quad[(i + 1) % 4]
        a, b = (y1 - y0), -(x1 - x0)
        cc = -(a * x0 + b * y0)
        sign = np.sign(a * c[0] + b * c[1] + cc)
        inside &= (sign * (a * xx + b * yy + cc)) >= 0
    return inside


def detect_beamstop(img: np.ndarray, field_mask: np.ndarray | None, psf_guard_px: int = 6):
    """Find the optical beam stop: film at D_max, which transmits ~6e-4 of the box.

    Preference order:

    1. the **direct-exposure rim** just inside a detected collimation border --
       the ideal probe, because it is an annulus so it samples the glare field at
       many positions and pins its spatial *shape*, not just its level;
    2. failing that, the **dark surround** around the patient silhouette, which
       survives even an aggressive crop and still gives a usable, if less
       well-distributed, set of probes.

    The mask is eroded by `psf_guard_px` so that light which is merely *blurred*
    across the collimation edge is excluded. Blur bleed and veiling glare both
    put photons where there should be none, and attributing the first to the
    second would inflate the measured veil on every sharp image. `glare.py`
    separates whatever is left over properly; this erosion just keeps the gross
    contamination out.
    """
    h, w = img.shape
    if field_mask is not None:
        # Search a generous band inward from the field edge and let the *level*,
        # not the geometry, decide where the direct-exposure region ends. Two
        # details matter, and both were learned the hard way here. A fixed narrow
        # band mostly misses the region, because how much unattenuated beam
        # reaches the film varies with patient size and with how tightly the
        # radiographer collimated. And a percentile threshold speckles -- it picks
        # the darkest scattered pixels rather than the contiguous band, and a
        # speckled mask erodes to nothing the moment `glare.estimate_veil` guards
        # it against blur bleed, silently pushing every image onto the degraded
        # near-edge path. Thresholding against the band's own dark level and then
        # closing the result keeps it solid enough to survive that erosion.
        inner = _ops.binary_erode(field_mask, iterations=2)
        band = inner & ~_ops.binary_erode(inner, iterations=max(4, int(0.12 * min(h, w))))
        if band.sum() > 50:
            q05, q50 = np.quantile(img[band], [0.05, 0.50])
            thr = float(q05 + 0.30 * (q50 - q05))
            mask = _ops.binary_erode(_ops.binary_dilate(band & (img <= thr), 2), 2)
            if mask.sum() > 30:
                return mask, "collimated_rim"

    # Fallback: darkest connected surround touching the frame border.
    thr = float(np.quantile(img, 0.08))
    dark = img <= max(thr, 1e-4)
    dark = _ops.binary_erode(_ops.binary_dilate(dark, 1), max(2, psf_guard_px))
    if dark.sum() < 30:
        return None, "none"
    labels, n = _ops.label_components(dark)
    if n == 0:
        return None, "none"
    border = np.zeros_like(dark)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    best, best_area = None, 0
    for st in _ops.component_stats(labels, n):
        comp = labels == st["label"]
        if not (comp & _ops.binary_dilate(border, 3)).any():
            continue
        if st["area"] > best_area:
            best, best_area = comp, st["area"]
    if best is None or best_area < 30:
        return None, "none"
    return best, "dark_surround"


def detect_marker(img: np.ndarray, field_mask: np.ndarray | None, beamstop_mask: np.ndarray | None,
                  outside_mask: np.ndarray | None = None):
    """Find the lead side marker: the brightest compact object, sitting on darkness.

    Anatomy is never this bright -- lead transmits no X-rays at all, so the film
    beneath it is at base+fog, clearer than any bone. The discriminator that
    actually works is not brightness alone (a blown-out glare highlight is also
    bright) but brightness *plus* a very dark immediate surround: the marker is
    conventionally placed in the direct-exposure region, so it is a clear patch on
    an opaque background, which is a contrast no anatomy or specular blob has.

    Returns (mask, confidence, bbox). Confidence is the product of four soft
    scores, so it degrades gracefully instead of switching off at a hard cut.
    """
    h, w = img.shape
    region = field_mask if field_mask is not None else np.ones((h, w), dtype=bool)
    vals = img[region]
    if vals.size < 100:
        return None, 0.0, None

    # Threshold at the film-base level, not at a percentile of the field.
    #
    # A percentile is the wrong instrument here and fails in a specific way: the
    # marker is a few dozen pixels out of ~10^5, so even the 99.5th percentile of
    # the field is set by the spine and mediastinum, and the candidate mask comes
    # back full of anatomy with the marker buried in it. But we *know* how bright
    # the marker should be -- lead transmits nothing, so the film under it is at
    # base+fog, the same density as the unexposed film outside the collimation
    # border, which is right there in the frame. Anchoring the threshold to that
    # measured level turns a fragile ranking problem into a physical one.
    if outside_mask is not None and outside_mask.any():
        base = float(np.median(img[outside_mask]))
        thr = 0.85 * base
    else:
        thr = float(np.quantile(vals, 0.999))
    if thr <= float(np.quantile(vals, 0.5)) + 0.05:
        return None, 0.0, None                     # no distinct bright population

    cand = region & (img >= thr)
    cand = _ops.binary_erode(_ops.binary_dilate(cand, 1), 1)
    labels, n = _ops.label_components(cand)
    if n == 0:
        return None, 0.0, None

    field_area = float(region.sum())
    best = (0.0, None, None)
    for st in _ops.component_stats(labels, n):
        area = st["area"]
        if not (0.00008 * field_area <= area <= 0.02 * field_area):
            continue
        comp = labels == st["label"]

        # 1. compactness: a letter glyph fills a good part of its bounding box,
        #    an elongated glare streak or a rib edge does not
        aspect = st["width"] / max(st["height"], 1)
        s_shape = float(np.exp(-((np.log(max(aspect, 1e-3) / 0.7)) ** 2) / (2 * 0.55**2)))
        s_fill = float(np.clip((st["fill"] - 0.15) / 0.45, 0.0, 1.0)) * float(
            np.clip((0.98 - st["fill"]) / 0.35, 0.0, 1.0)
        )

        # 2. the decisive one: how dark is the immediate surround
        halo = _ops.binary_dilate(comp, 5) & ~_ops.binary_dilate(comp, 2)
        if not halo.any():
            continue
        surround = float(np.median(img[halo]))
        inner = float(np.median(img[comp]))
        # Normalised against a 0.25 step rather than a full-scale one. A marker
        # seated in the direct-exposure region clears that easily, and archives
        # where it overlaps the shoulder or lung apex -- common enough -- still
        # score usefully instead of being rejected outright.
        s_contrast = float(np.clip((inner - surround) / 0.25, 0.0, 1.0))
        if beamstop_mask is not None and beamstop_mask.any():
            near_stop = float((_ops.binary_dilate(comp, 8) & beamstop_mask).sum()) / max(comp.sum(), 1)
            s_contrast = max(s_contrast, float(np.clip(near_stop, 0.0, 1.0)))

        # 3. position: markers are peripheral, never over the lung fields
        cy, cx = st["centroid"]
        ecc = max(abs(cy / h - 0.5), abs(cx / w - 0.5)) * 2.0
        s_pos = float(np.clip((ecc - 0.35) / 0.35, 0.0, 1.0))

        # Geometric mean, not a product. Four scores each in [0, 1] multiply down
        # to a number that no longer means anything on the same scale -- a genuine
        # marker scoring 0.95/0.57/1.0/1.0 lands at 0.54, near the accept gate,
        # purely because there are four of them. The geometric mean keeps the
        # "every criterion must hold" behaviour (one near-zero term still kills the
        # candidate) while leaving the result comparable to the individual scores,
        # so the 0.5 gate means what it looks like it means.
        parts = (s_shape, max(s_fill, 0.2), s_contrast, max(s_pos, 0.15))
        score = float(np.prod(parts) ** (1.0 / len(parts)))
        if score > best[0]:
            best = (score, comp, st["bbox"])

    if best[1] is None:
        return None, 0.0, None
    return best[1], float(np.clip(best[0], 0.0, 1.0)), best[2]


def detect(photo: np.ndarray, psf_guard_px: int = 6) -> Fiducials:
    """Full fiducial sweep over one photo. Cheap enough to run over a whole manifest."""
    img = _normalize(photo)
    h, w = img.shape
    f = Fiducials(shape=(h, w))

    edges, quad, diag = detect_collimation(img)
    f.edges = edges
    f.field_quad = quad
    f.diagnostics.update(diag)
    if quad is not None:
        f.field_mask = _quad_mask(quad, (h, w))
        f.outside_mask = ~f.field_mask

    f.beamstop_mask, f.beamstop_source = detect_beamstop(img, f.field_mask, psf_guard_px=psf_guard_px)
    f.marker_mask, f.marker_confidence, f.marker_bbox = detect_marker(
        img, f.field_mask, f.beamstop_mask, f.outside_mask
    )

    if f.has_beamstop and f.has_bright_anchor and len(f.mtf_edges) >= 1:
        f.coverage = Coverage.FULL
    elif f.has_beamstop:
        f.coverage = Coverage.PARTIAL
    else:
        f.coverage = Coverage.NONE

    f.diagnostics["beamstop_frac"] = (
        float(f.beamstop_mask.sum()) / (h * w) if f.has_beamstop else 0.0
    )
    return f
