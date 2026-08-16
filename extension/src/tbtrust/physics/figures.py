"""Publication figures for the physics track: diagrams, annotated images, image strips.

Deliberately not a plotting module. The physics here is *spatial* and *visual* --
where the glare is, what a lead marker looks like, whether you can actually see a
lesion once the floor rises above its contrast -- and a line plot of AURC against
coverage communicates none of that. Reviewers reading a paper about photographs
of radiographs should be shown photographs of radiographs.

What is in here
---------------
========================================  ====================================
`capture_chain_diagram`                   schematic: X-ray tube -> patient -> film
                                          -> lightbox -> phone -> JPEG, with the
                                          unknowns marked where they enter
`sign_convention_panel`                   why lead is the *bright* anchor; the
                                          one thing everyone gets backwards
`fiducial_anatomy`                        an actual radiograph with the three
                                          calibration targets called out
`finding_atlas`                           stylised chest with TB findings drawn at
                                          true relative size and typical location,
                                          shaded by whether this photo can carry them
`detectability_strip`                     the same lesion at multiples of the
                                          measured floor. The reader's own eye is
                                          the experiment.
`inversion_panels`                        photo -> measured veil -> recovered density
`certificate_card`                        the verdict as a visual artifact
`retake_instruction`                      glare map with an arrow saying where to move
`radiograph_gallery`                      real dataset images, normal vs TB
========================================  ====================================

Everything uses matplotlib only -- no seaborn, no new dependency -- and works
headless. Each function returns its `Figure` so a notebook can show it and a
script can save it.
"""

from __future__ import annotations

import numpy as np

from .density import FilmModel, density_to_display

# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #

INK = "#1b1b1f"
MUTED = "#8a8f98"
PAPER = "#fbfbfd"
PANEL = "#eef0f4"

# One colour per calibration target, used consistently in every figure so a
# reader who learns the mapping once can read the rest at a glance.
C_MARKER = "#e8a317"      # lead marker -- the bright D_min anchor
C_BEAMSTOP = "#d64545"    # direct-exposure region -- the optical beam stop
C_EDGE = "#2e86ab"        # collimation border -- the slanted edge
C_MARGIN = "#7048e8"      # unexposed film outside the border
C_LUNG = "#2f9e44"        # the analysed region

VERDICT_COLOR = {
    "detectable": "#2f9e44",
    "marginal": "#f08c00",
    "insufficient": "#c92a2a",
    "abstain": "#868e96",
}
ACTION_COLOR = {"report": "#2f9e44", "retake": "#f08c00", "refer": "#4c6ef5"}


def _fig(w, h, facecolor=PAPER):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(w, h), facecolor=facecolor)
    return fig


def _clean(ax, xlim=(0, 1), ylim=(0, 1)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("auto")
    ax.axis("off")
    return ax


def _box(ax, x, y, w, h, label, sub=None, color=PANEL, edge=INK, fontsize=8, lw=1.2):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                facecolor=color, edgecolor=edge, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), label, ha="center", va="center",
            fontsize=fontsize, color=INK, weight="bold", zorder=3)
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=fontsize - 1.5, color=MUTED, zorder=3)


def _arrow(ax, x0, y0, x1, y1, color=INK, lw=1.4, style="-|>", ls="-", zorder=4, shrink=2.0):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11,
                                 color=color, linewidth=lw, linestyle=ls, zorder=zorder,
                                 shrinkA=shrink, shrinkB=shrink))


def _callout(ax, xy, xytext, text, color, fontsize=7.5, ha="left"):
    """Leader line from a point on an image to a labelled box outside it."""
    ax.annotate(
        text, xy=xy, xytext=xytext, textcoords="data", ha=ha, va="center",
        fontsize=fontsize, color=INK, zorder=6,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": color, "linewidth": 1.4},
        arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.4,
                    "connectionstyle": "angle3,angleA=0,angleB=90"},
    )


def _normalize(img):
    x = np.asarray(img, dtype=np.float64)
    if x.ndim == 3:
        x = x.mean(axis=-1)
    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0, 1)


# --------------------------------------------------------------------------- #
# 1. the capture chain
# --------------------------------------------------------------------------- #


def capture_chain_diagram(figsize=(13, 5.2)):
    """Schematic of the whole channel, with the three unknowns marked where they enter.

    The method figure. Reading left to right it is the forward problem -- what
    physically happens between a patient and a JPEG. Reading the lower track right
    to left it is the inverse problem, and the point being made is that each
    unknown is recovered from a *physical object already in the frame*, not from
    a model or a prior.
    """
    import matplotlib.pyplot as plt

    fig = _fig(*figsize)
    ax = _clean(fig.add_axes([0.01, 0.01, 0.98, 0.98]))

    stages = [
        ("X-ray tube", "collimated beam", 0.005),
        ("Patient", "attenuation ∫μ dl", 0.145),
        ("Film + cassette", r"D ∝ ∫μ dl", 0.285),
        ("Developed film", r"$D_\mathrm{min}$ … $D_\mathrm{max}$", 0.425),
        ("Lightbox", r"$L = I\cdot10^{-D}$", 0.565),
        ("Phone lens", "PSF + veil", 0.705),
        ("Sensor + ISP", "noise, tone curve", 0.845),
    ]
    w, h, y = 0.128, 0.19, 0.585
    for i, (name, sub, x) in enumerate(stages):
        shade = PANEL if i < 5 else "#ffe8e8"       # the capture stages are where damage happens
        _box(ax, x, y, w, h, name, sub, color=shade, fontsize=8.5)
        if i:
            _arrow(ax, x - 0.012, y + h / 2, x, y + h / 2)

    ax.text(0.985, y + h / 2, "JPEG", ha="left", va="center", fontsize=9, weight="bold", color=INK)
    _arrow(ax, 0.973, y + h / 2, 0.982, y + h / 2)

    ax.text(0.005, y + h + 0.075, "FORWARD  ·  what physically happens", fontsize=10,
            weight="bold", color=INK)
    ax.text(0.005, y + h + 0.028,
            "Beer–Lambert makes optical density linear in path-integrated attenuation. "
            "Everything after the film is damage.", fontsize=8, color=MUTED)

    # --- the three unknowns, entering at their true stages -------------------
    unknowns = [
        # The two lens unknowns are offset horizontally as well as vertically:
        # stacked on the same x, the lower one's connector is drawn straight
        # through the upper one's text.
        (0.705 + w / 2 - 0.030, "veiling glare  V(x)", C_BEAMSTOP, -0.050),
        (0.705 + w / 2 + 0.034, "point spread  PSF", C_EDGE, -0.100),
        (0.845 + w / 2, r"tone curve  γ, $c_1$", C_MARKER, -0.050),
    ]
    for x, label, color, dy in unknowns:
        ax.text(x, y + dy, label, ha="center", va="center", fontsize=8, color=color, weight="bold")
        _arrow(ax, x, y + dy + 0.020, x, y - 0.006, color=color, lw=1.1, ls=(0, (2, 2)), style="-")

    # --- the inverse track ---------------------------------------------------
    yi, hi = 0.115, 0.175
    recovery = [
        ("Direct-exposure\nregion", r"$D_\mathrm{max}$ → beam stop", C_BEAMSTOP, 0.145),
        ("Collimation\nborder", "hard step → MTF", C_EDGE, 0.355),
        ("Lead marker +\nfilm margin", r"$D_\mathrm{min}$ → density scale", C_MARKER, 0.565),
        ("Calibrated\ndensity map", "ΔD, and its floor", C_LUNG, 0.775),
    ]
    for i, (name, sub, color, x) in enumerate(recovery):
        _box(ax, x, yi, 0.175, hi, name, sub, color="white", edge=color, fontsize=8, lw=1.6)
        if i:
            _arrow(ax, x - 0.017, yi + hi / 2, x, yi + hi / 2, color=MUTED, lw=1.2)

    ax.text(0.005, yi + hi / 2, "INVERSE", fontsize=10, weight="bold", color=INK,
            ha="left", va="center")
    ax.text(0.005, yi + hi / 2 - 0.045, "measured, not\nassumed", fontsize=7.5, color=MUTED,
            ha="left", va="center")

    ax.text(0.5, 0.015,
            "Each unknown is recovered from a physical object the radiograph already carries. "
            "No reference shot, no calibration target added to the scene, no training data.",
            ha="center", va="bottom", fontsize=8.5, color=INK, style="italic")

    # link the unknowns to the objects that measure them
    for x_from, x_to, color in ((0.769, 0.232, C_BEAMSTOP), (0.769, 0.442, C_EDGE),
                                (0.909, 0.652, C_MARKER)):
        _arrow(ax, x_from, y - 0.115, x_to, yi + hi + 0.012, color=color, lw=1.3,
               ls=(0, (4, 3)), zorder=1)

    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# 2. the sign convention
# --------------------------------------------------------------------------- #


def sign_convention_panel(film: FilmModel | None = None, figsize=(11.5, 5.0)):
    """Why the lead marker is the *bright* anchor and the beam stop is the dark one.

    This exists because the inversion is upside down relative to the intuition
    almost everyone starts with, including this project's own brief. Lead blocks
    X-rays, so in *transmission* terms it is "zero signal" -- but the film beneath
    it is therefore barely exposed, develops to base+fog, and ends up the most
    transparent thing on the sheet. On a lightbox it is the brightest object in
    the frame.

    Four objects, four rows, and the flip happens visibly between row 1 and row 2.
    Every cell is positioned explicitly rather than through `barh`, so a value
    label can never drift into the neighbouring column -- which is exactly what
    happened when this was drawn with bar charts.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    film = film or FilmModel()
    cols = [
        ("Lead marker", 0.00, film.d_min, C_MARKER, "blocks the beam entirely"),
        ("Bone / spine", 0.25, 0.55, "#6c7280", "strongly attenuating"),
        ("Lung field", 0.80, 1.90, "#6c7280", "air: little attenuation"),
        ("Direct exposure", 1.00, film.d_max, C_BEAMSTOP, "full beam, no patient"),
    ]
    rows = [
        ("X-ray transmission", lambda xmit, d: (xmit, f"{xmit:.2f}")),
        ("film optical density  D", lambda xmit, d: (d / film.d_max, f"{d:.2f}")),
        ("film transmittance  " + r"$10^{-D}$", lambda xmit, d: (10.0**-d, f"{10.0**-d:.3g}")),
    ]

    fig = _fig(*figsize)
    ax = _clean(fig.add_axes([0.005, 0.0, 0.99, 0.86]))

    col_x = [0.235 + i * 0.185 for i in range(4)]
    col_w = 0.155
    row_y = [0.70, 0.50, 0.30]
    bar_h = 0.085

    for (label, _), y in zip(rows, row_y, strict=True):
        ax.text(0.225, y + bar_h / 2, label, ha="right", va="center", fontsize=8.6, color=INK)
    ax.text(0.225, 0.10, "on the lightbox", ha="right", va="center", fontsize=8.6,
            color=INK, weight="bold")

    for (name, xmit, d, color, note), x in zip(cols, col_x, strict=True):
        ax.text(x + col_w / 2, 0.945, name, ha="center", fontsize=10, weight="bold", color=color)
        ax.text(x + col_w / 2, 0.895, note, ha="center", fontsize=7.4, color=MUTED)

        for (_, fn), y in zip(rows, row_y, strict=True):
            frac, text = fn(xmit, d)
            ax.add_patch(Rectangle((x, y), col_w, bar_h, facecolor="#e9ecef", edgecolor="none"))
            ax.add_patch(Rectangle((x, y), col_w * float(np.clip(frac, 0, 1)), bar_h,
                                   facecolor=color, alpha=0.9, edgecolor="none"))
            # Label inside its own cell, right-aligned. Anything else and a wide
            # bar pushes its number under the next column's heading.
            # Ink unless the bar actually reaches under the label -- keying the
            # colour off the value instead leaves mid-range labels white on the
            # pale unfilled track, which is unreadable.
            ax.text(x + col_w - 0.006, y + bar_h / 2, text, ha="right", va="center",
                    fontsize=7.6, color="white" if frac > 0.80 else INK, weight="bold")

        tau = float(10.0**-d)
        ax.add_patch(Rectangle((x, 0.03), col_w, 0.15,
                               facecolor=str(float(np.clip(tau ** (1 / 2.2), 0, 1))),
                               edgecolor=INK, linewidth=1.1))

    # The flip, as a band across the whole grid rather than an arrow in the
    # margin: the margin is where the row labels live, and any annotation put
    # there collides with the longest of them.
    band_y = (row_y[0] + row_y[1] + bar_h) / 2
    ax.add_patch(Rectangle((col_x[0] - 0.01, band_y - 0.028), col_x[-1] + col_w - col_x[0] + 0.02,
                           0.056, facecolor="#c92a2a", alpha=0.09, edgecolor="none"))
    ax.text((col_x[0] + col_x[-1] + col_w) / 2, band_y,
            "▼   film is a NEGATIVE — the ordering inverts here   ▼",
            ha="center", va="center", fontsize=8.4, weight="bold", color="#c92a2a")

    fig.text(0.5, 0.965,
             "The lead marker is the BRIGHT anchor; the direct-exposure region is the beam stop",
             ha="center", fontsize=12, weight="bold", color=INK)
    fig.text(0.5, 0.915,
             "More X-ray exposure → more developed silver → higher density → darker on the lightbox. "
             "So the object that blocks the beam ends up brightest.",
             ha="center", fontsize=8.6, color=MUTED)
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# 3. annotated fiducials on a real image
# --------------------------------------------------------------------------- #


def fiducial_anatomy(photo, fid=None, figsize=(11.5, 6.4), title=None):
    """An actual radiograph with the three calibration targets called out.

    Pass a real dataset image or a simulated capture. If `fid` is omitted the
    detector is run, so this doubles as a visual check on detection -- if the red
    overlay is not sitting on the black rim outside the patient, nothing
    downstream means anything, and that is far easier to see here than in a
    coverage table.
    """
    import matplotlib.pyplot as plt

    from .fiducials import detect

    img = _normalize(photo)
    fid = fid if fid is not None else detect(img)

    fig = _fig(*figsize)
    ax = fig.add_axes([0.26, 0.05, 0.50, 0.86])
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(MUTED)

    def _overlay(mask, color, alpha=0.42):
        if mask is None or not np.any(mask):
            return False
        rgba = np.zeros((*mask.shape, 4))
        rgba[..., :3] = plt.matplotlib.colors.to_rgb(color)
        rgba[..., 3] = np.where(mask, alpha, 0.0)
        ax.imshow(rgba)
        return True

    has_stop = _overlay(fid.beamstop_mask, C_BEAMSTOP)
    has_margin = _overlay(fid.outside_mask, C_MARGIN, alpha=0.30)
    has_marker = _overlay(fid.marker_mask, C_MARKER, alpha=0.95)

    if fid.field_quad is not None:
        q = np.vstack([fid.field_quad, fid.field_quad[:1]])
        ax.plot(q[:, 0], q[:, 1], color=C_EDGE, lw=2.2)

    # callouts on the left, keyed to what each object measures
    entries = [
        (C_MARKER, "Lead L/R marker", r"$D_\mathrm{min}$ — zero X-ray transmission",
         "→ bright densitometry anchor\n→ the only interior sample\n    of the lightbox",
         has_marker, f"confidence {fid.marker_confidence:.2f}"),
        (C_BEAMSTOP, "Direct-exposure region", r"$D_\mathrm{max}$ — full beam, no patient",
         "→ optically black: the BEAM STOP\n→ any light here is veiling glare,\n    measured with no model",
         has_stop, f"source: {fid.beamstop_source}"),
        (C_EDGE, "Collimation border", r"hard $D_\mathrm{min}\!\to\!D_\mathrm{max}$ step",
         "→ ISO 12233 slanted edge\n→ the capture PSF, measured\n    under the real exposure",
         fid.field_quad is not None, f"{len(fid.mtf_edges)} usable edge(s)"),
        (C_MARGIN, "Unexposed film margin", r"$D_\mathrm{min}$, wrapping the frame",
         "→ full-frame flat field\n→ the illumination surface",
         has_margin, ""),
    ]
    y = 0.885
    for color, name, dens, meaning, found, extra in entries:
        mark = "●" if found else "○"
        fig.text(0.015, y, f"{mark} {name}", fontsize=9.5, weight="bold",
                 color=color if found else MUTED)
        fig.text(0.030, y - 0.036, dens, fontsize=7.8, color=INK, style="italic")
        fig.text(0.030, y - 0.100, meaning, fontsize=7.5, color=MUTED, va="center")
        if extra:
            fig.text(0.030, y - 0.152, extra, fontsize=7.2, color=MUTED)
        y -= 0.215

    # coverage verdict on the right
    cov = fid.coverage.value
    cov_color = {"full": "#2f9e44", "partial": "#f08c00", "none": "#c92a2a"}[cov]
    fig.text(0.785, 0.86, "COVERAGE", fontsize=9, weight="bold", color=INK)
    fig.text(0.785, 0.805, cov.upper(), fontsize=17, weight="bold", color=cov_color)
    explain = {
        "full": (
            "marker + beam stop + a usable\nslanted edge.\n\n"
            "Everything is measurable and the\ncertificate can make its strongest\nstatement."
        ),
        "partial": (
            "a beam stop, but the marker or the\nedge is missing.\n\n"
            "Glare is still measured directly —\nit is the dominant term — but the\n"
            "density scale leans on the γ prior."
        ),
        "none": (
            "no optical beam stop.\n\n"
            "The veil is unmeasured, so no bound\nexists. The certificate must ABSTAIN\n"
            "rather than guess. These images are\noutside the method's reach."
        ),
    }[cov]
    fig.text(0.785, 0.755, explain, fontsize=7.6, color=MUTED, va="top")

    fig.text(0.5, 0.965, title or "The calibration targets a chest radiograph already carries",
             ha="center", fontsize=12, weight="bold", color=INK)
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# 4. the TB finding atlas
# --------------------------------------------------------------------------- #


# Stylised PA chest geometry, in axes fractions on a square canvas.
#
# Orientation follows radiographic convention: a PA film is displayed as if the
# patient faces the viewer, so the patient's LEFT is on the viewer's RIGHT. That
# is why the cardiac silhouette sits right of midline here and why the lead "L"
# marker belongs in the top-right corner -- both are on the patient's left.
_THORAX = [
    (0.50, 0.975), (0.38, 0.962), (0.285, 0.925), (0.208, 0.855), (0.158, 0.745),
    (0.132, 0.605), (0.132, 0.450), (0.158, 0.322), (0.205, 0.222), (0.280, 0.155),
    (0.385, 0.120), (0.50, 0.112), (0.615, 0.120), (0.720, 0.155), (0.795, 0.222),
    (0.842, 0.322), (0.868, 0.450), (0.868, 0.605), (0.842, 0.745), (0.792, 0.855),
    (0.715, 0.925), (0.620, 0.962),
]

# Right lung = viewer's LEFT. Nearly symmetric medially; the heart does not
# encroach on it.
_LUNG_R = [
    (0.415, 0.885), (0.340, 0.865), (0.275, 0.800), (0.230, 0.700), (0.205, 0.580),
    (0.200, 0.460), (0.215, 0.350), (0.245, 0.265), (0.290, 0.215), (0.345, 0.205),
    (0.390, 0.240), (0.415, 0.330), (0.428, 0.450), (0.435, 0.580), (0.435, 0.720),
    (0.428, 0.820),
]

# Left lung = viewer's RIGHT. The medial border is pushed laterally between
# y ~ 0.20 and 0.45 by the heart, which is what gives a real CXR its asymmetry.
_LUNG_L = [
    (0.585, 0.885), (0.660, 0.865), (0.725, 0.800), (0.770, 0.700), (0.795, 0.580),
    (0.800, 0.460), (0.785, 0.350), (0.755, 0.265), (0.710, 0.215), (0.665, 0.225),
    (0.640, 0.290), (0.630, 0.380), (0.612, 0.470), (0.585, 0.580), (0.575, 0.720),
    (0.572, 0.820),
]


def _poly(points, **kw):
    from matplotlib.patches import Polygon

    return Polygon(np.asarray(points), closed=True, **kw)


def _lung_paths():
    """Matplotlib Paths for the two lung fields, for containment tests."""
    from matplotlib.path import Path

    return Path(np.asarray(_LUNG_R)), Path(np.asarray(_LUNG_L))


def _chest_outline(ax):
    """Draw a stylised PA chest: thorax, lungs, mediastinum, heart, diaphragm, ribs.

    Schematic on purpose -- its job is to give the finding markers somewhere
    anatomically meaningful to sit. Two details are worth the effort rather than
    decoration:

    * the ribs and diaphragm are **clipped to the thorax**. Drawn as free arcs they
      sail out past the body outline, which reads as a bug and distracts from the
      markers the figure exists to show;
    * the left lung's medial border is indented for the heart, so the two lungs are
      not mirror images. A symmetric chest looks wrong to anyone who reads films.
    """
    from matplotlib.patches import Arc, Ellipse

    thorax = _poly(_THORAX, facecolor="#252a33", edgecolor="#3d4553", lw=1.5, zorder=1)
    ax.add_patch(thorax)

    lungs = []
    for pts in (_LUNG_R, _LUNG_L):
        patch = _poly(pts, facecolor="#0b0d11", edgecolor="none", zorder=2)
        ax.add_patch(patch)
        lungs.append(patch)

    # Mediastinum + spine: narrow. Drawn wide it dominates the frame and squeezes
    # the lungs into slivers, which is the opposite of a chest radiograph.
    ax.add_patch(_poly([(0.474, 0.905), (0.526, 0.905), (0.542, 0.250),
                        (0.458, 0.250)], facecolor="#39414f", edgecolor="none", zorder=3))
    # Heart: left-sided and apex-down-left, so it reads as a silhouette rather
    # than a balloon parked on the midline.
    ax.add_patch(Ellipse((0.596, 0.318), 0.196, 0.238, angle=-20,
                         facecolor="#38404e", edgecolor="none", zorder=4))

    # hemidiaphragms: the right (viewer's left) sits higher, as the liver pushes it up
    for cx, w_, y_ in ((0.315, 0.30, 0.215), (0.690, 0.30, 0.192)):
        d = Arc((cx, y_), w_, 0.17, theta1=185, theta2=355, color="#59627a", lw=3.0, zorder=5)
        d.set_clip_path(thorax)
        ax.add_patch(d)

    # ribs, clipped to the body outline
    for i in range(7):
        yy = 0.895 - i * 0.098
        for cx in (0.315, 0.690):
            r = Arc((cx, yy), 0.40, 0.22, theta1=200, theta2=340,
                    color="#69728a", lw=1.3, alpha=0.65, zorder=6)
            r.set_clip_path(thorax)
            ax.add_patch(r)

    # clavicles
    for cx, t1, t2 in ((0.345, 200, 300), (0.655, 240, 340)):
        c = Arc((cx, 0.905), 0.30, 0.14, theta1=t1, theta2=t2, color="#7c8598", lw=2.6, zorder=7)
        c.set_clip_path(thorax)
        ax.add_patch(c)
    return thorax, lungs


def finding_atlas(cert=None, px_per_mm=None, figsize=(12.5, 6.8), chest_width_mm=340.0):
    """Stylised chest with TB findings drawn at true relative size, shaded by verdict.

    The figure that connects the finding table to something a clinician
    recognises. Each marker is scaled by its actual size in millimetres against a
    ~34 cm chest, so a 2 mm miliary nodule really is a speck beside a 45 mm
    consolidation -- which is exactly why their density floors differ by more than
    an order of magnitude, and why one survives a bad photograph and the other
    does not.

    Sites follow post-primary TB's upper-lobe predilection rather than being
    spread for visual balance; a figure that puts a cavity in the lung base
    teaches the reader something false.

    Pass a `Certificate` and each finding is shaded green / amber / red by whether
    *this photograph* can carry it. Pass none and it is a plain anatomical legend.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse

    from .findings import TB_FINDINGS

    fig = _fig(*figsize)
    ax = fig.add_axes([0.02, 0.02, 0.46, 0.87])
    _clean(ax)
    ax.set_aspect("equal")
    ax.set_facecolor("#161a20")
    thorax, lung_patches = _chest_outline(ax)
    path_r, path_l = _lung_paths()

    def _in_lung(x, y):
        """Which lung patch to clip a marker to, or the thorax if it straddles."""
        if path_r.contains_point((x, y)):
            return lung_patches[0]
        if path_l.contains_point((x, y)):
            return lung_patches[1]
        return thorax

    def mm(v):
        return v / chest_width_mm          # millimetres -> axes fraction

    # (x, y, how to draw). R = viewer's left = patient's right.
    sites = {
        "fibrotic_band": (0.330, 0.830, "band"),      # right apex
        "infiltrate": (0.290, 0.735, "fuzzy"),        # right upper zone
        "cavity_wall": (0.690, 0.755, "ring"),        # left upper zone
        "small_nodule": (0.720, 0.585, "dot"),        # left mid zone
        "consolidation": (0.300, 0.400, "fuzzy"),     # right lower zone
        "pleural_effusion": (0.730, 0.248, "meniscus"),  # left costophrenic angle
        # Drawn last so the diffuse pattern sits on top of the focal ones rather
        # than being hidden under a consolidation.
        "miliary_nodule": (0.500, 0.560, "scatter"),
    }

    verdicts = {}
    if cert is not None and not cert.abstained:
        verdicts = {fv.finding: fv.verdict.value for fv in cert.findings}

    legend_rows = []
    for key, (x, y, kind) in sites.items():
        f = TB_FINDINGS.get(key)
        if f is None:
            continue
        v = verdicts.get(key)
        col = VERDICT_COLOR.get(v, "#aeb6c2")
        r = max(mm(f.size_mm) / 2, 0.005)

        if kind == "scatter":
            # Rejection-sample inside the lung outlines. Scattering by Gaussian
            # alone sprays nodules over the mediastinum and outside the ribcage.
            rng = np.random.default_rng(4)
            placed = 0
            while placed < 90:
                px, py = rng.uniform(0.18, 0.82), rng.uniform(0.22, 0.88)
                if path_r.contains_point((px, py)) or path_l.contains_point((px, py)):
                    ax.add_patch(Circle((px, py), max(r, 0.0045), facecolor=col,
                                        alpha=0.92, zorder=9))
                    placed += 1
        else:
            # Every focal marker is clipped to the lung it belongs in. A 45 mm
            # consolidation is wider than a lung field at this scale, so drawn
            # free it spills onto the chest wall and stops reading as pathology.
            clip = _in_lung(x, y)
            if kind == "dot":
                marks = [Circle((x, y), r, facecolor=col, alpha=0.95, zorder=9)]
            elif kind == "ring":
                marks = [Circle((x, y), r * 2.4, facecolor="#0b0d11", edgecolor=col,
                                lw=max(1.8, r * 80), zorder=9)]
            elif kind == "fuzzy":
                marks = [Ellipse((x, y), r * 2 * k * 1.3, r * 2 * k, facecolor=col,
                                 alpha=a, edgecolor="none", zorder=9)
                         for k, a in ((2.1, 0.18), (1.5, 0.30), (1.0, 0.60))]
            elif kind == "meniscus":
                marks = [Ellipse((x, y), r * 3.0, r * 1.6, facecolor=col, alpha=0.85, zorder=9)]
            else:  # band
                marks = [Ellipse((x, y), r * 5.5, r * 1.0, angle=-34, facecolor=col,
                                 alpha=0.95, zorder=9)]
            for m in marks:
                m.set_clip_path(clip)
                ax.add_patch(m)
        legend_rows.append((f, v, col))

    ax.text(0.5, 0.012, "markers drawn to scale against a ~34 cm chest    ·    "
            "patient's left is on the viewer's right",
            ha="center", fontsize=7.2, color="#98a0ac", transform=ax.transAxes)

    # --- legend / verdict table ---------------------------------------------
    x0 = 0.515
    fig.text(x0, 0.885, "Finding", fontsize=8.6, weight="bold", color=INK)
    fig.text(x0 + 0.185, 0.885, "size", fontsize=8.6, weight="bold", color=INK)
    fig.text(x0 + 0.265, 0.885, "|ΔD|", fontsize=8.6, weight="bold", color=INK)
    fig.text(x0 + 0.345, 0.885, "this photo", fontsize=8.6, weight="bold", color=INK)
    fig.add_artist(plt.Line2D([x0, 0.985], [0.872, 0.872], color=MUTED, lw=0.8))

    y = 0.828
    for f, v, col in legend_rows:
        fig.text(x0, y, "■", fontsize=11, color=col, va="center")
        fig.text(x0 + 0.022, y, f.name, fontsize=8.4, color=INK, va="center")
        fig.text(x0 + 0.185, y, f"{f.size_mm:g} mm", fontsize=8, color=MUTED, va="center")
        fig.text(x0 + 0.265, y, f"{f.delta_d:.3f}", fontsize=8, color=MUTED, va="center")
        # An unevaluated finding gets an explicit dash. A blank cell reads as a
        # rendering failure rather than as "the certificate did not score this one".
        fig.text(x0 + 0.345, y, v if v else "— not scored", fontsize=8.2,
                 color=col if v else MUTED, va="center", weight="bold" if v else "normal")
        y -= 0.064

    if cert is None:
        note = ("Pass a Certificate to shade each finding by whether a given photograph can\n"
                "carry it. Contrast values are NOMINAL placeholders — see physics/findings.py.")
    elif cert.abstained:
        note = ("This image ABSTAINED: no optical beam stop was found, so the veil is\n"
                "unmeasured and no bound can be stated. Findings are shown unshaded.")
    else:
        note = ("Shading: does the measured density floor of this photograph sit below the\n"
                "finding's contrast? Green — carried. Amber — marginal. Red — the information\n"
                "is not in the image, and no model can recover it.")
    fig.text(x0, y - 0.015, note, fontsize=7.6, color=MUTED, va="top")

    if px_per_mm:
        fig.text(x0, 0.03, f"working scale: {px_per_mm:.2f} px/mm", fontsize=7, color=MUTED)
    fig.text(0.25, 0.945, "Where TB shows, and at what size", ha="center",
             fontsize=12, weight="bold", color=INK)
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# 5. the detectability strip -- the reader's own eye as the experiment
# --------------------------------------------------------------------------- #


def detectability_strip(
    base_density,
    params,
    finding,
    cal=None,
    multiples=(0.25, 0.5, 1.0, 2.0, 4.0),
    fiducial_truth=None,
    film: FilmModel | None = None,
    crop_px=None,
    seed=7,
    figsize=(13.5, 5.6),
):
    """The same lesion at multiples of the measured floor. Can you see it?

    The most persuasive figure the physics track can produce, because it does not
    ask the reader to trust a number. The floor is computed blind from the
    photograph; lesions are then inserted into the *film* at fractions and
    multiples of it and re-photographed through the same capture. Below the floor
    the reader cannot find the lesion in the recovered image — and neither can a
    matched filter, which is the quantitative version of the same statement in
    `validate.detectability_experiment`.

    The top row is the film (ground truth, always visible). The bottom row is what
    the phone delivered. The gap between them is the whole thesis.
    """
    import matplotlib.pyplot as plt

    from .film import capture, insert_lesion
    from .floor import density_floor
    from .invert import invert

    film = film or FilmModel()
    base = np.asarray(base_density, dtype=np.float64)
    h, w = base.shape

    if cal is None:
        photo0, _ = capture(base, params, fiducial_truth=fiducial_truth, film=film,
                            rng=np.random.default_rng(seed))
        cal = invert(photo0, film=film)

    # Site selection: mid lung-density band, clear of the fiducials, and as
    # locally *flat* as possible.
    #
    # Flatness is not cosmetic. A site on a rib edge sits on an anatomical
    # gradient of order 1.0 in density, so the display window has to span that
    # gradient and a lesion of ΔD = 0.07 becomes a few percent of the greyscale --
    # invisible at every multiple, which makes the figure argue the opposite of
    # what it should. The quantitative claim is unaffected either way; it is the
    # *visual* comparison that needs a flat background, exactly as a
    # contrast-detail phantom does.
    from . import _ops

    inner = np.zeros_like(base, dtype=bool)
    q = int(0.32 * min(h, w))
    inner[q:h - q, q:w - q] = True
    # Restrict to the region the certificate actually aggregates over, so the
    # floor quoted for this site is comparable with the one on the certificate
    # rather than being read off some unrepresentative corner.
    inner &= cal.lung_field_mask()
    if not inner.any():
        inner = np.zeros_like(base, dtype=bool)
        inner[q:h - q, q:w - q] = True
    lo, hi = np.quantile(base[inner], [0.40, 0.70])
    cand = inner & (base >= lo) & (base <= min(hi, film.d_max - 0.3))
    if not cand.any():
        cand = inner
    gy, gx = np.gradient(_ops.gaussian_blur(base, 2.0))
    flatness = np.where(cand, _ops.gaussian_blur(np.hypot(gy, gx), 6.0), np.inf)
    cy, cx = (float(v) for v in np.unravel_index(int(np.argmin(flatness)), flatness.shape))

    size_px = max(finding.size_px(cal.px_per_mm), 3.0)
    fm = density_floor(cal, finding)
    disc = np.zeros_like(base, dtype=bool)
    yy, xx = np.mgrid[0:h, 0:w]
    disc[np.hypot(yy - cy, xx - cx) <= max(size_px, 4.0)] = True
    floor = float(np.median(fm.floor[disc]))
    floor_lung = float(np.median(fm.floor[cal.lung_field_mask()]))

    half = int(crop_px or max(24, 2.2 * size_px))
    y0, y1 = int(np.clip(cy - half, 0, h)), int(np.clip(cy + half, 0, h))
    x0, x1 = int(np.clip(cx - half, 0, w)), int(np.clip(cx + half, 0, w))

    # Windowing decides whether this figure works at all.
    #
    # `density_to_display` spreads D over the film's whole 0.2-3.2 range, so a
    # lesion of ΔD = 0.26 occupies under a tenth of the greyscale and is invisible
    # even when it is four times the floor -- which makes the figure say the exact
    # opposite of what it should. Both rows are therefore windowed tightly, on a
    # range fixed by the *lesion-free* crop and shared across every panel in the
    # row, so brightness differences between panels are real contrast differences
    # and not a per-panel autoscale.
    n = len(multiples)
    max_dd = float(floor * max(multiples))
    base_crop = base[y0:y1, x0:x1]
    lo_f = float(np.quantile(base_crop, 0.02)) - 0.6 * max_dd
    hi_f = float(np.quantile(base_crop, 0.98)) + 0.6 * max_dd
    ref_crop = cal.density[y0:y1, x0:x1]
    lo_r = float(np.quantile(ref_crop, 0.02)) - 0.6 * max_dd
    hi_r = float(np.quantile(ref_crop, 0.98)) + 0.6 * max_dd

    cyc, cxc = cy - y0, cx - x0

    fig = _fig(*figsize)
    for i, mlt in enumerate(multiples):
        dd = floor * float(mlt)
        lesioned = insert_lesion(base, (cy, cx), size_px, dd)
        ph, _ = capture(lesioned, params, fiducial_truth=fiducial_truth, film=film,
                        rng=np.random.default_rng(seed + 100 + i))
        rec = invert(ph, film=film, fid=cal.fiducials, iterations=1)

        detectable = mlt >= 1.0
        col = VERDICT_COLOR["detectable" if detectable else "insufficient"]

        for row, (arr, lo, hi) in enumerate((
            (lesioned[y0:y1, x0:x1], lo_f, hi_f),
            (rec.density[y0:y1, x0:x1], lo_r, hi_r),
        )):
            axp = fig.add_axes([0.075 + i * (0.895 / n), 0.50 - row * 0.375,
                                0.895 / n - 0.014, 0.335])
            # Density is inverted for display: a lesion attenuates more, lowers
            # density, and must appear *brighter*, as it does on a lightbox.
            axp.imshow(arr, cmap="gray_r", vmin=lo, vmax=hi)
            axp.set_xticks([])
            axp.set_yticks([])
            # Mark where the lesion is. "Can you see it?" is only a fair question
            # if the reader is told where to look -- otherwise the figure measures
            # visual search, not contrast detection.
            axp.add_patch(plt.Circle((cxc, cyc), size_px * 0.95, fill=False,
                                     ec="#00b3ff", lw=1.3, ls=(0, (4, 3)), alpha=0.85))
            for sp in axp.spines.values():
                sp.set_edgecolor(col if row else MUTED)
                sp.set_linewidth(2.4 if row else 1.0)
            if row == 0:
                axp.set_title(f"{mlt:g}× floor\nΔD = {dd:.3f}", fontsize=8.5, color=INK, pad=5)
            else:
                axp.text(0.5, -0.10, "predicted visible" if detectable else "predicted lost",
                         transform=axp.transAxes, ha="center", fontsize=7.6,
                         color=col, weight="bold")

    fig.text(0.065, 0.665, "FILM\n(ground truth)", ha="right", va="center", fontsize=8.5,
             weight="bold", color=INK)
    fig.text(0.065, 0.293, "RECOVERED\nfrom the photo", ha="right", va="center", fontsize=8.5,
             weight="bold", color=INK)

    verdict = "insufficient" if floor > finding.delta_d else "detectable"
    fig.text(0.5, 0.955,
             f"Can you see it?  {finding.name} at multiples of the measured density floor",
             ha="center", fontsize=12, weight="bold", color=INK)
    fig.text(0.5, 0.915,
             f"floor at this site = {floor:.4f} ΔD   (lung-field median {floor_lung:.4f})"
             f"    ·    this finding's contrast = {finding.delta_d:.3f} ΔD"
             f"    ·    verdict here: {verdict.upper()}",
             ha="center", fontsize=9, color=VERDICT_COLOR[verdict])
    fig.text(0.5, 0.035,
             "The floor was computed blind from the photograph, before any lesion was inserted. "
             "Below it the signal is not merely hard to see — it is not in the image.",
             ha="center", fontsize=8.2, color=MUTED, style="italic")
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# 6. the inversion, side by side
# --------------------------------------------------------------------------- #


def inversion_panels(photo, cal, truth=None, figsize=(14, 4.6)):
    """Photo → measured veil → veil-free signal → recovered density (+ truth if simulated)."""
    import matplotlib.pyplot as plt

    img = _normalize(photo)
    lung = cal.lung_field_mask()
    panels = [
        (img, "gray", "The photograph", "8-bit JPEG, all the estimator sees"),
        (cal.veil, "magma", "Measured veil V(x)",
         f"from the beam stop · {cal.glare.method.replace('_', ' ')}"),
        (np.where(lung, cal.veil / np.maximum(cal.signal, 1e-9), np.nan), "inferno",
         "Veil / signal", "contrast is compressed by 1/(1+V/I)"),
        (density_to_display(cal.density), "gray", "Recovered optical density",
         f"γ={cal.tone.gamma:.2f} · PSF σ={cal.psf.sigma:.2f} px"),
    ]
    if truth is not None and getattr(truth, "glare_field_true", None) is not None:
        panels.insert(2, (truth.glare_field_true, "magma", "True veil", "ground truth (simulation only)"))

    n = len(panels)
    fig = _fig(*figsize)
    for i, (arr, cmap, title, sub) in enumerate(panels):
        ax = fig.add_axes([0.015 + i * (0.97 / n), 0.10, 0.97 / n - 0.02, 0.72])
        finite = np.asarray(arr)[np.isfinite(arr)]
        kw = ({"vmin": float(np.quantile(finite, 0.01)), "vmax": float(np.quantile(finite, 0.99))}
              if finite.size else {})
        im = ax.imshow(arr, cmap=cmap, **kw)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9.5, weight="bold", color=INK, pad=6)
        ax.text(0.5, -0.055, sub, transform=ax.transAxes, ha="center", fontsize=7.5, color=MUTED)
        if cmap != "gray":
            fig.colorbar(im, ax=ax, fraction=0.042, pad=0.02).ax.tick_params(labelsize=6)

    fig.text(0.5, 0.95, "Blind inversion of a single photograph", ha="center",
             fontsize=12, weight="bold", color=INK)
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# 7. the certificate as an artifact
# --------------------------------------------------------------------------- #


def certificate_card(cert, cal, decision=None, figsize=(11.5, 6.2)):
    """The verdict rendered as something a clinician could be handed.

    Deliberately not a table dump. The point of the certificate is that it is an
    *actionable* object -- a verdict, the reason, and the specific thing to do --
    so it is drawn the way a report would be, with the floor map as evidence.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig = _fig(*figsize, facecolor="white")
    ax = _clean(fig.add_axes([0, 0, 1, 1]))

    v = cert.verdict.value
    col = VERDICT_COLOR[v]
    ax.add_patch(FancyBboxPatch((0.02, 0.83), 0.96, 0.14,
                                boxstyle="round,pad=0.004,rounding_size=0.01",
                                facecolor=col, edgecolor="none"))
    ax.text(0.04, 0.90, f"CERTIFICATE: {v.upper()}", fontsize=17, weight="bold",
            color="white", va="center")
    ax.text(0.96, 0.915, f"{cert.margin_db:+.1f} dB", fontsize=15, weight="bold",
            color="white", va="center", ha="right")
    worst_line = cert.line(cert.worst_finding) if cert.worst_finding else None
    worst_name = worst_line.finding_name if worst_line else "—"
    ax.text(0.96, 0.868, f"worst: {worst_name}   ·   limited by: {cert.limiting}",
            fontsize=8, color="white", va="center", ha="right", alpha=0.9)

    # Per-finding margin bars, confined to the left 60% of the card so they can
    # never run under the evidence inset on the right.
    ax.text(0.04, 0.775, "Margin per finding   (0 dB = the floor equals the finding's contrast)",
            fontsize=9, weight="bold", color=INK)
    y = 0.715
    span = 30.0
    for fv in cert.findings:
        m = float(np.clip(fv.margin_db, -span, span))
        c = VERDICT_COLOR[fv.verdict.value]
        ax.text(0.04, y, fv.finding_name, fontsize=8.2, color=INK, va="center")
        x0 = 0.235
        ax.plot([x0, x0 + 0.24], [y, y], color="#e5e7eb", lw=7, solid_capstyle="butt", zorder=1)
        mid = x0 + 0.12
        ax.plot([mid, mid + 0.12 * m / span], [y, y], color=c, lw=7,
                solid_capstyle="butt", zorder=2)
        ax.plot([mid, mid], [y - 0.026, y + 0.026], color=INK, lw=1.2, zorder=3)
        ax.text(0.487, y, f"{fv.margin_db:+.1f} dB", fontsize=8, color=c, va="center",
                weight="bold")
        ax.text(0.548, y, f"floor {fv.floor_median:.3f}", fontsize=7.5, color=MUTED, va="center")
        y -= 0.072

    # provenance
    prov = cert.provenance
    ax.text(0.04, y - 0.02, "How this was measured", fontsize=9, weight="bold", color=INK)
    lines = [
        f"fiducial coverage: {prov.get('coverage', '?')}",
        f"tone: {prov.get('tone_method', '?')}   ·   PSF: {prov.get('psf_method', '?')}"
        f"   ·   glare: {prov.get('glare_method', '?')}",
        f"Rose k = {prov.get('rose_k', '?')}   ·   scale "
        f"{'measured' if prov.get('px_per_mm_measured') else f'inferred ({cal.px_per_mm:.2f} px/mm, ±20%)'}",
        f"finding contrasts: {prov.get('contrast_source', '?')}",
    ]
    for i, ln in enumerate(lines):
        ax.text(0.045, y - 0.06 - i * 0.036, ln, fontsize=7.4, color=MUTED)

    # The floor map as evidence, in its own column.
    if cert.floor_map is not None:
        axm = fig.add_axes([0.665, 0.10, 0.30, 0.58])
        lung = cal.lung_field_mask()
        axm.imshow(_normalize(cal.pixel_values), cmap="gray")
        bad = cert.insufficient_mask if cert.insufficient_mask is not None else np.zeros_like(lung)
        # An explicit RGBA red, not a colormap: `autumn` renders 1.0 as yellow, so
        # the overlay and the caption saying "red" disagreed.
        rgba = np.zeros((*lung.shape, 4))
        rgba[..., 0] = 0.79
        rgba[..., 1] = 0.16
        rgba[..., 2] = 0.16
        rgba[..., 3] = np.where(lung & bad, 0.5, 0.0)
        axm.imshow(rgba)
        axm.set_xticks([])
        axm.set_yticks([])
        axm.set_title(f"red: where the floor beats\na {worst_name.lower()}",
                      fontsize=7.8, color=INK, pad=4)

    # the action
    if decision is not None:
        acol = ACTION_COLOR.get(decision.action.value, MUTED)
        ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.61, 0.16,
                                    boxstyle="round,pad=0.006,rounding_size=0.012",
                                    facecolor="white", edgecolor=acol, linewidth=2.2))
        ax.text(0.04, 0.145, decision.action.value.upper(), fontsize=13, weight="bold", color=acol)
        if decision.expected_gain_db > 0:
            ax.text(0.615, 0.148, f"a good retake should recover ~{decision.expected_gain_db:.0f} dB",
                    fontsize=7.4, color=MUTED, ha="right")
        ax.text(0.04, 0.082, _wrap(decision.instruction, 76), fontsize=8, color=INK, va="center")

    plt.close(fig)
    return fig


def _wrap(text, width):
    import textwrap

    return "\n".join(textwrap.wrap(str(text), width))


# --------------------------------------------------------------------------- #
# 8. the retake instruction, spatially
# --------------------------------------------------------------------------- #


def retake_instruction(cal, decision=None, figsize=(11, 5.2)):
    """Where the glare is, and which way to move -- the figure that justifies "retake".

    A confidence score can say "this image is bad". Only a measured glare *field*
    can say "the reflection is upper-left, step to your right", and that
    difference is what makes a retake worth the patient's time instead of
    reproducing the same photograph.
    """
    import matplotlib.pyplot as plt

    from .glare import hotspot

    hs = hotspot(cal.glare, cal.fiducials.field_mask)
    img = _normalize(cal.pixel_values)

    fig = _fig(*figsize)
    ax = fig.add_axes([0.03, 0.06, 0.44, 0.80])
    ax.imshow(img, cmap="gray")
    lung = cal.lung_field_mask()
    veil = np.where(lung, cal.veil, np.nan)
    if np.isfinite(veil).any():
        ax.imshow(veil, cmap="inferno", alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("measured veil over the lung fields", fontsize=9, color=INK)

    h, w = img.shape
    if hs.centroid_yx is not None:
        cy, cx = hs.centroid_yx
        ax.plot(cx, cy, marker="x", ms=14, mew=3, color="#ffffff")
        # arrow showing which way the operator should move
        dx = -0.22 * w if cx > w / 2 else 0.22 * w
        _arrow(ax, cx, cy, cx + dx, cy, color="#00e5ff", lw=3.0, style="-|>", shrink=6)
        ax.text(cx + dx / 2, cy - 0.06 * h, "move the phone", ha="center", fontsize=8.5,
                color="#00e5ff", weight="bold")

    axr = _clean(fig.add_axes([0.50, 0.06, 0.48, 0.80]))
    axr.text(0, 0.95, "WHAT THE PHYSICS SAYS", fontsize=9.5, weight="bold", color=INK, va="top")
    rows = [
        ("veil pattern", "localized reflection" if hs.localized else "diffuse wash"),
        ("direction", hs.direction),
        ("peak / median veil", f"{hs.peak_over_median:.1f}×"),
        ("field affected", f"{hs.affected_fraction:.0%}"),
        ("blur", f"σ = {cal.psf.sigma:.2f} px"
                 + (f", directional ({cal.psf.anisotropy:.1f}× along one axis)"
                    if cal.psf.motion_dominant else ", symmetric → defocus")),
        ("measured from", f"{cal.psf.method.replace('_', ' ')}, {cal.psf.n_edges} edge(s)"),
    ]
    y = 0.86
    for k, val in rows:
        axr.text(0.0, y, k, fontsize=8, color=MUTED)
        axr.text(0.38, y, val, fontsize=8.2, color=INK, weight="bold")
        y -= 0.075

    axr.text(0, y - 0.03, "INSTRUCTION TO THE OPERATOR", fontsize=9.5, weight="bold", color=INK)
    text = decision.instruction if decision is not None else hs.advice
    acol = ACTION_COLOR.get(getattr(getattr(decision, "action", None), "value", ""), C_EDGE)
    axr.text(0, y - 0.10, _wrap(text, 58), fontsize=8.6, color=INK, va="top",
             bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f6f8fa", "edgecolor": acol,
                   "linewidth": 1.8})

    fig.text(0.5, 0.955, "Retake, and specifically what to change", ha="center",
             fontsize=12, weight="bold", color=INK)
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# 9. real images
# --------------------------------------------------------------------------- #


def radiograph_gallery(paths, labels=None, clinics=None, ncols=6, figsize=None,
                       title="Real chest radiographs", size=384):
    """A contact sheet of actual dataset images, labelled normal / TB.

    Worth including in the paper for the same reason the fiducial overlay is: the
    reader should see what these archives actually look like, including how much
    they differ between clinics, before being shown numbers about them.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    # Coerce to plain lists. Passing pandas Series straight through looks fine
    # until the caller hands one with a non-reset index, at which point `labels[i]`
    # indexes by *label* rather than position and raises KeyError on a sampled
    # frame -- which is exactly how these are usually built.
    paths = list(paths)
    labels = None if labels is None else list(labels)
    clinics = None if clinics is None else list(clinics)
    n = len(paths)
    nrows = int(np.ceil(n / ncols))
    fig = _fig(*(figsize or (2.1 * ncols, 2.35 * nrows + 0.5)))

    for i, p in enumerate(paths):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        try:
            img = np.asarray(Image.open(p).convert("L").resize((size, size), Image.BILINEAR))
        except (OSError, ValueError):
            ax.axis("off")
            continue
        ax.imshow(img, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        bits = []
        if clinics is not None:
            bits.append(str(clinics[i]))
        if labels is not None:
            lab = int(labels[i])
            bits.append("TB" if lab == 1 else "normal")
        ax.set_title(" · ".join(bits), fontsize=7.5,
                     color=("#c92a2a" if labels is not None and int(labels[i]) == 1 else INK))
        for s in ax.spines.values():
            s.set_edgecolor("#c92a2a" if labels is not None and int(labels[i]) == 1 else MUTED)
            s.set_linewidth(1.6 if labels is not None and int(labels[i]) == 1 else 0.8)

    fig.suptitle(title, fontsize=12, weight="bold", color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.close(fig)
    return fig


def degradation_ladder(display_image, severities=(0.0, 0.25, 0.5, 0.75, 1.0),
                       size=512, seed=3, figsize=(14, 3.6)):
    """One radiograph, re-photographed at rising capture severity.

    The visual definition of the severity axis every other figure is plotted
    against. Worth putting early in a paper, because "severity 0.75" means nothing
    to a reader until they have seen one.
    """
    import matplotlib.pyplot as plt

    from .film import simulate

    fig = _fig(*figsize)
    n = len(severities)
    for i, s in enumerate(severities):
        photo, _ = simulate(display_image, severity=float(s),
                            rng=np.random.default_rng(seed), size=size)
        ax = fig.add_axes([0.01 + i * (0.98 / n), 0.08, 0.98 / n - 0.012, 0.74])
        ax.imshow(photo, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"severity {s:g}", fontsize=9, color=INK, pad=5)
    fig.text(0.5, 0.955, "The same film, re-photographed at rising capture severity",
             ha="center", fontsize=12, weight="bold", color=INK)
    fig.text(0.5, 0.015,
             "blur · veiling glare · specular reflection · uneven lightbox · off-axis phone · sensor noise · JPEG",
             ha="center", fontsize=8, color=MUTED)
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# saving
# --------------------------------------------------------------------------- #


def save(fig, path, dpi=200):
    """Save a figure, making parent directories. Returns the path."""
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    return p


def show(fig):
    """Display a figure that was created with `plt.close()` already applied.

    Every factory here closes its figure so a script can generate dozens without
    exhausting matplotlib's window budget. Notebooks need them back, and simply
    returning the object is not enough once it is closed -- this re-attaches it to
    the current backend's manager.
    """
    import matplotlib.pyplot as plt

    manager = plt.figure().canvas.manager
    manager.canvas.figure = fig
    fig.set_canvas(manager.canvas)
    return fig
