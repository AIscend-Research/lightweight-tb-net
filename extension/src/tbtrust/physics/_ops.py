"""Image primitives the physics modules need, in numpy only.

The rest of the package is deliberately numpy + Pillow (see the note in
pyproject.toml), and the physics path has to stay importable with no GPU, no
scipy and no OpenCV -- it runs inside `audit_fiducials.py` over tens of thousands
of images on a Kaggle CPU box, and it is imported by tests that must pass on a
bare CI runner. So the handful of operations we need are implemented here rather
than pulled in as four more dependencies.

Everything works on float arrays and uses reflect padding unless stated.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# filtering
# --------------------------------------------------------------------------- #


def gaussian_kernel1d(sigma: float, radius: int | None = None) -> np.ndarray:
    """Normalized 1-D Gaussian. `radius=None` -> ceil(4 sigma), capped at 128."""
    sigma = float(max(sigma, 1e-6))
    if radius is None:
        radius = int(min(128, max(1, np.ceil(4.0 * sigma))))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _convolve_axis(img: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """1-D 'same' convolution along one axis, reflect-padded, fully vectorized.

    Implemented as a weighted sum of shifted views rather than a Python loop over
    pixels: `len(kernel)` whole-array operations, which is what keeps a 4-sigma
    blur on a 1024x1024 film usable inside a per-image loop.
    """
    r = (len(kernel) - 1) // 2
    if r == 0:
        return img * kernel[0]
    pad = [(0, 0)] * img.ndim
    pad[axis] = (r, r)
    p = np.pad(img, pad, mode="reflect")
    out = np.zeros_like(img, dtype=np.float64)
    n = img.shape[axis]
    for i, w in enumerate(kernel):
        if w == 0.0:
            continue
        sl = [slice(None)] * img.ndim
        sl[axis] = slice(i, i + n)
        out += w * p[tuple(sl)]
    return out


# Above this width the direct kernel needs > ~100 taps per axis and the FFT wins
# outright. The veiling-glare halo is always in that regime (its sigma is a tenth
# of the image), and it is evaluated inside every capture and every inversion, so
# the switch is the difference between a validation sweep taking minutes and hours.
_FFT_SIGMA_THRESHOLD = 12.0


def gaussian_blur(img: np.ndarray, sigma: float | tuple[float, float]) -> np.ndarray:
    """Separable Gaussian blur. `sigma` may be scalar or (sigma_y, sigma_x)."""
    sy, sx = (sigma, sigma) if np.isscalar(sigma) else sigma
    out = np.asarray(img, dtype=np.float64)
    if max(sy, sx) > _FFT_SIGMA_THRESHOLD:
        return _gaussian_blur_fft(out, sy, sx)
    if sy > 1e-6:
        out = _convolve_axis(out, gaussian_kernel1d(sy), axis=0)
    if sx > 1e-6:
        out = _convolve_axis(out, gaussian_kernel1d(sx), axis=1)
    return out


def _gaussian_blur_fft(img: np.ndarray, sy: float, sx: float) -> np.ndarray:
    """Wide Gaussian via FFT, with reflect padding so the frame does not wrap.

    Without the padding a bright lung field would wrap around and deposit a halo
    on the opposite border -- precisely where the beam-stop probes sit, which
    would corrupt the very measurement this blur is used to predict.
    """
    h, w = img.shape
    py = int(min(h, np.ceil(3 * max(sy, 1e-6))))
    px = int(min(w, np.ceil(3 * max(sx, 1e-6))))
    p = np.pad(img, ((py, py), (px, px)), mode="reflect")
    ph, pw = p.shape
    fy = np.fft.fftfreq(ph)[:, None]
    fx = np.fft.rfftfreq(pw)[None, :]
    otf = np.exp(-2.0 * np.pi**2 * ((max(sy, 0.0) ** 2) * fy**2 + (max(sx, 0.0) ** 2) * fx**2))
    out = np.fft.irfft2(np.fft.rfft2(p) * otf, s=(ph, pw))
    return out[py : py + h, px : px + w]


def motion_kernel(length: float, angle_rad: float, size: int | None = None) -> np.ndarray:
    """Normalized 2-D line kernel for linear motion blur, anti-aliased.

    Drawn by sampling the line densely and accumulating with bilinear weights, so
    a 7.3-pixel streak at 23 degrees is not silently rounded to a staircase --
    the whole point of the PSF track is that the recovered sigma is compared
    against the one that was actually applied.
    """
    length = float(max(length, 1e-6))
    if size is None:
        size = int(2 * np.ceil(length / 2.0) + 1)
    size = max(3, size | 1)
    k = np.zeros((size, size), dtype=np.float64)
    c = (size - 1) / 2.0
    n = max(2, int(np.ceil(length * 8)))
    for t in np.linspace(-length / 2.0, length / 2.0, n):
        y = c + t * np.sin(angle_rad)
        x = c + t * np.cos(angle_rad)
        y0, x0 = int(np.floor(y)), int(np.floor(x))
        fy, fx = y - y0, x - x0
        for dy, wy in ((0, 1 - fy), (1, fy)):
            for dx, wx in ((0, 1 - fx), (1, fx)):
                yy, xx = y0 + dy, x0 + dx
                if 0 <= yy < size and 0 <= xx < size:
                    k[yy, xx] += wy * wx
    s = k.sum()
    return k / s if s > 0 else k


def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """'same' 2-D convolution, reflect-padded. For small kernels only."""
    kh, kw = kernel.shape
    ry, rx = kh // 2, kw // 2
    p = np.pad(np.asarray(img, dtype=np.float64), ((ry, ry), (rx, rx)), mode="reflect")
    h, w = img.shape[:2]
    out = np.zeros((h, w), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            if kernel[i, j] != 0.0:
                out += kernel[i, j] * p[i : i + h, j : j + w]
    return out


# --------------------------------------------------------------------------- #
# morphology + connected components on boolean masks
# --------------------------------------------------------------------------- #


def _shift_stack(mask: np.ndarray, fill: bool) -> list[np.ndarray]:
    """The eight neighbours of `mask`, padded with `fill` at the border."""
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            s = np.full_like(mask, fill)
            ys_src = slice(max(0, -dy), mask.shape[0] - max(0, dy))
            ys_dst = slice(max(0, dy), mask.shape[0] - max(0, -dy))
            xs_src = slice(max(0, -dx), mask.shape[1] - max(0, dx))
            xs_dst = slice(max(0, dx), mask.shape[1] - max(0, -dx))
            s[ys_dst, xs_dst] = mask[ys_src, xs_src]
            out.append(s)
    return out


def binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    for _ in range(int(iterations)):
        m = np.logical_or.reduce([m, *_shift_stack(m, False)])
    return m


def binary_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    for _ in range(int(iterations)):
        m = np.logical_and.reduce([m, *_shift_stack(m, False)])
    return m


def label_components(mask: np.ndarray, max_iter: int = 512) -> tuple[np.ndarray, int]:
    """8-connected labelling by iterative max-propagation. Returns (labels, count).

    Label 0 is background. Propagation converges in roughly the geodesic diameter
    of the largest component, which for the compact blobs we look for (lead
    markers) is tens of iterations, not hundreds. `max_iter` is a guard against a
    pathological snake-shaped mask, and hitting it degrades to over-segmentation
    rather than to a hang -- which is the right failure for a detector whose
    output is then filtered by area and shape anyway.
    """
    full = np.asarray(mask, dtype=bool)
    if not full.any():
        return np.zeros(full.shape, dtype=np.int32), 0

    # Work inside the mask's bounding box. Marker candidates occupy a fraction of
    # a percent of the frame, and propagating over the whole image would spend
    # every iteration on empty background.
    ys, xs = np.nonzero(full)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = full[y0:y1, x0:x1]

    lab = np.where(m, np.arange(1, m.size + 1, dtype=np.int64).reshape(m.shape), 0)
    for _ in range(int(max_iter)):
        new = np.where(m, np.maximum.reduce([lab, *_shift_stack(lab, 0)]), 0)
        if np.array_equal(new, lab):
            break
        lab = new
    uniq = np.unique(lab)
    uniq = uniq[uniq != 0]
    remap = np.zeros(int(lab.max()) + 1, dtype=np.int32)
    remap[uniq] = np.arange(1, len(uniq) + 1, dtype=np.int32)

    out = np.zeros(full.shape, dtype=np.int32)
    out[y0:y1, x0:x1] = remap[lab]
    return out, len(uniq)


def component_stats(labels: np.ndarray, count: int) -> list[dict]:
    """Per-component area, centroid, bbox and fill ratio (area / bbox area)."""
    stats = []
    for k in range(1, count + 1):
        ys, xs = np.nonzero(labels == k)
        if ys.size == 0:
            continue
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        bbox_area = (y1 - y0 + 1) * (x1 - x0 + 1)
        stats.append(
            {
                "label": k,
                "area": int(ys.size),
                "centroid": (float(ys.mean()), float(xs.mean())),
                "bbox": (y0, x0, y1, x1),
                "height": y1 - y0 + 1,
                "width": x1 - x0 + 1,
                "fill": float(ys.size / max(bbox_area, 1)),
            }
        )
    return stats


# --------------------------------------------------------------------------- #
# thresholding
# --------------------------------------------------------------------------- #


def otsu_threshold(img: np.ndarray, nbins: int = 256) -> float:
    """Otsu's between-class-variance threshold on a float image."""
    x = np.asarray(img, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return lo
    hist, edges = np.histogram(x, bins=nbins, range=(lo, hi))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    m0 = np.cumsum(p * centers) / np.maximum(w0, 1e-12)
    mt = float((p * centers).sum())
    m1 = (mt - np.cumsum(p * centers)) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(between))])


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


def fit_line_tls(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Total-least-squares line y = m x + c, robust to a near-vertical fit in x.

    Ordinary least squares minimises residuals in y only, which biases the slope
    when the x positions carry their own error -- and the x positions here are
    sub-pixel edge estimates, which certainly do. The edge angle feeds the
    slanted-edge projection, where a slope error smears the oversampled ESF and
    inflates the recovered PSF width, so it is worth doing properly.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm, ym = x.mean(), y.mean()
    dx, dy = x - xm, y - ym
    sxx, syy, sxy = float(dx @ dx), float(dy @ dy), float(dx @ dy)
    if abs(sxy) < 1e-12 and abs(sxx - syy) < 1e-12:
        return 0.0, float(ym)
    theta = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    m = np.tan(theta)
    return float(m), float(ym - m * xm)


def bilinear_sample(img: np.ndarray, ys: np.ndarray, xs: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Bilinear sample `img` at float coordinates; out-of-bounds -> `fill`."""
    a = np.asarray(img, dtype=np.float64)
    h, w = a.shape[:2]
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    fy = ys - y0
    fx = xs - x0
    ok = (y0 >= 0) & (x0 >= 0) & (y0 + 1 < h) & (x0 + 1 < w)
    y0c = np.clip(y0, 0, h - 2)
    x0c = np.clip(x0, 0, w - 2)
    v = (
        a[y0c, x0c] * (1 - fy) * (1 - fx)
        + a[y0c + 1, x0c] * fy * (1 - fx)
        + a[y0c, x0c + 1] * (1 - fy) * fx
        + a[y0c + 1, x0c + 1] * fy * fx
    )
    return np.where(ok, v, fill)


def estimate_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 homography mapping src -> dst from >= 4 point pairs (DLT + SVD).

    Points are (x, y). Coordinates are conditioned by the standard isotropic
    normalisation before the SVD; without it the DLT matrix is badly scaled at
    image resolutions and the recovered corners drift by whole pixels.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 4:
        raise ValueError("need >= 4 matching (x, y) point pairs")

    def _normalize(p):
        c = p.mean(axis=0)
        d = np.sqrt(((p - c) ** 2).sum(axis=1)).mean()
        s = np.sqrt(2) / max(d, 1e-12)
        T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
        ph = np.hstack([p, np.ones((len(p), 1))]) @ T.T
        return ph[:, :2], T

    sn, Ts = _normalize(src)
    dn, Td = _normalize(dst)
    rows = []
    for (x, y), (u, v) in zip(sn, dn, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H


def warp_perspective(img: np.ndarray, H: np.ndarray, out_shape: tuple[int, int], fill: float = 0.0) -> np.ndarray:
    """Warp `img` by homography H (src -> dst) into an out_shape canvas."""
    oh, ow = out_shape
    Hinv = np.linalg.inv(np.asarray(H, dtype=np.float64))
    yy, xx = np.mgrid[0:oh, 0:ow].astype(np.float64)
    ones = np.ones_like(xx)
    p = np.stack([xx, yy, ones], axis=0).reshape(3, -1)
    q = Hinv @ p
    wq = np.where(np.abs(q[2]) < 1e-12, 1e-12, q[2])
    xs = (q[0] / wq).reshape(oh, ow)
    ys = (q[1] / wq).reshape(oh, ow)
    return bilinear_sample(img, ys, xs, fill=fill)


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order four (x, y) corners as [top-left, top-right, bottom-right, bottom-left]."""
    p = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    p = p[np.argsort(ang)]
    start = int(np.argmin(p[:, 0] + p[:, 1]))    # top-left = smallest x+y
    return np.roll(p, -start, axis=0)


def line_intersection(l1: tuple[float, float, float], l2: tuple[float, float, float]) -> tuple[float, float]:
    """Intersect two lines given as (a, b, c) with a*x + b*y + c = 0."""
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    den = a1 * b2 - a2 * b1
    if abs(den) < 1e-12:
        raise ValueError("parallel lines have no unique intersection")
    return ((b1 * c2 - b2 * c1) / den, (a2 * c1 - a1 * c2) / den)


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #


def robust_std(x: np.ndarray) -> float:
    """MAD-based standard deviation; immune to the outliers a lesion would add."""
    a = np.asarray(x, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size < 2:
        return 0.0
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def fit_poly_surface(ys: np.ndarray, xs: np.ndarray, vals: np.ndarray, shape: tuple[int, int], degree: int = 2):
    """Least-squares 2-D polynomial surface through scattered samples.

    Returns (surface_of_shape, coefficients). Coordinates are normalised to
    [-1, 1] so the Vandermonde matrix stays conditioned. The degree is silently
    reduced when there are too few samples to determine it -- a rank-deficient
    fit through five points would otherwise produce a wildly oscillating glare
    field, and a flat one is the honest answer there.
    """
    h, w = shape
    ysn = (np.asarray(ys, dtype=np.float64) / max(h - 1, 1)) * 2 - 1
    xsn = (np.asarray(xs, dtype=np.float64) / max(w - 1, 1)) * 2 - 1
    v = np.asarray(vals, dtype=np.float64)

    def terms(yy, xx, deg):
        return np.stack([yy**i * xx**j for i in range(deg + 1) for j in range(deg + 1 - i)], axis=-1)

    deg = int(degree)
    while deg > 0 and terms(ysn[:1], xsn[:1], deg).shape[-1] > len(v) // 3:
        deg -= 1
    A = terms(ysn, xsn, deg)
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float64)
    gyn = (gy / max(h - 1, 1)) * 2 - 1
    gxn = (gx / max(w - 1, 1)) * 2 - 1
    surface = terms(gyn, gxn, deg) @ coef
    return surface, coef
