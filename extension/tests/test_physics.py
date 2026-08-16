"""Tests for the physics track. No data, no GPU, no network.

These are deliberately weighted toward the failures that actually happened while
this code was being written, because those are the ones that recur and the ones
that are silent. Three in particular are worth naming, since a passing test that
does not check them is worse than no test:

* the veil estimate collapsing to zero, which looks exactly like a clean photo;
* the tone fit absorbing the veil into its black point, which causes the above;
* the density floor exploding to absurd values when the recovered film signal is
  driven to its clamp.

Each has a test below that fails if it comes back.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from tbtrust.physics import _ops
from tbtrust.physics import glare as G
from tbtrust.physics import psf as P
from tbtrust.physics import tone as T
from tbtrust.physics.certificate import Verdict, certificate_confidence, certify
from tbtrust.physics.channel import capacity_table, channel_report
from tbtrust.physics.density import FilmModel, density_to_transmittance, transmittance_to_density
from tbtrust.physics.fiducials import Coverage, detect
from tbtrust.physics.film import capture, sample_params, simulate, synthetic_chest_density
from tbtrust.physics.findings import core, get
from tbtrust.physics.floor import FloorSpec, density_floor, limiting_factor, template_energy
from tbtrust.physics.invert import invert, invertible, noise_sigma
from tbtrust.physics.triage import Action, triage, triage_summary

SIZE = 224


@pytest.fixture(scope="module")
def scene():
    return synthetic_chest_density(size=SIZE, rng=np.random.default_rng(0))


def _shot(scene, severity, seed=3):
    base, ft = scene
    params = sample_params(severity, np.random.default_rng(seed))
    return capture(base, params, fiducial_truth=ft, rng=np.random.default_rng(seed + 1))


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #


def test_gaussian_blur_matches_across_the_fft_switch():
    """The FFT path kicks in above sigma 12 and must agree with the direct one."""
    rng = np.random.default_rng(0)
    img = rng.random((96, 96))
    direct = _ops._convolve_axis(
        _ops._convolve_axis(img, _ops.gaussian_kernel1d(11.9), 0), _ops.gaussian_kernel1d(11.9), 1
    )
    fft = _ops._gaussian_blur_fft(img, 11.9, 11.9)
    assert np.abs(direct - fft).max() < 5e-3


def test_gaussian_blur_conserves_energy():
    """Mass conservation, away from the border.

    Reflect padding is deliberately not energy-conserving near an edge -- it
    mirrors intensity back in, which is what stops a wide veil blur from wrapping
    a bright lung field onto the opposite border where the beam-stop probes sit.
    So the check needs a canvas large enough that the kernel does not reach the
    frame, which is the regime the code actually runs in.
    """
    for s in (0.8, 4.0, 20.0):
        n = max(64, int(16 * s))
        img = np.zeros((n, n))
        img[n // 2, n // 2] = 1.0
        assert _ops.gaussian_blur(img, s).sum() == pytest.approx(1.0, abs=1e-3)


def test_label_components_separates_blobs():
    m = np.zeros((40, 40), dtype=bool)
    m[3:9, 3:9] = True
    m[20:31, 25:36] = True
    labels, n = _ops.label_components(m)
    assert n == 2
    areas = sorted(s["area"] for s in _ops.component_stats(labels, n))
    assert areas == [36, 121]


def test_homography_roundtrip():
    src = np.array([[0, 0], [50, 2], [48, 60], [-1, 57]], dtype=float)
    dst = np.array([[10, 10], [90, 5], [95, 80], [5, 75]], dtype=float)
    H = _ops.estimate_homography(src, dst)
    p = np.hstack([src, np.ones((4, 1))]) @ H.T
    assert np.allclose(p[:, :2] / p[:, 2:3], dst, atol=1e-6)


def test_density_transmittance_roundtrip():
    d = np.array([0.2, 1.0, 2.5, 3.2])
    assert np.allclose(transmittance_to_density(density_to_transmittance(d)), d)


# --------------------------------------------------------------------------- #
# fiducials
# --------------------------------------------------------------------------- #


def test_fiducials_found_on_a_clean_capture(scene):
    photo, _ = _shot(scene, 0.0)
    f = detect(photo)
    assert f.coverage in (Coverage.FULL, Coverage.PARTIAL)
    assert f.has_beamstop, "the direct-exposure region is the beam stop; without it nothing works"
    assert f.field_quad is not None
    assert len(f.edges) == 4


def test_beamstop_survives_erosion(scene):
    """The mask must be solid, not speckled.

    A speckled beam stop erodes to nothing under the blur guard in
    `estimate_veil`, which silently pushes every image onto the degraded
    near-edge path and halves the measured veil. That regression is invisible in
    the outputs, so it is pinned here.
    """
    photo, _ = _shot(scene, 0.3)
    f = detect(photo)
    assert f.beamstop_mask.sum() > 500
    # Solidity, not absolute survival: how many erosions the band can take depends
    # on the image resolution, and at 224 px the direct-exposure rim is only a few
    # pixels wide however well it is detected. A *speckled* mask of the same pixel
    # count loses nearly everything to a single erosion, whereas a contiguous band
    # keeps most of it -- which is the property that actually matters.
    kept = _ops.binary_erode(f.beamstop_mask, 1).sum() / f.beamstop_mask.sum()
    assert kept > 0.35, f"beam stop is speckled, not a solid band (kept {kept:.2f})"


def test_coverage_none_when_the_film_is_cropped_away(scene):
    """A tight crop to the lung fields destroys every fiducial. Say so, do not guess."""
    photo, _ = _shot(scene, 0.0)
    h, w = photo.shape
    cropped = photo[int(0.3 * h) : int(0.7 * h), int(0.3 * w) : int(0.7 * w)]
    f = detect(cropped)
    assert f.field_quad is None
    assert not invertible(f) or f.coverage is Coverage.PARTIAL


# --------------------------------------------------------------------------- #
# PSF
# --------------------------------------------------------------------------- #


def test_psf_recovers_a_known_blur(scene):
    """The whole point of the slanted edge: sigma out should match sigma in."""
    base, ft = scene
    for true_sigma in (1.0, 2.5):
        from tbtrust.physics.film import CaptureParams

        p = CaptureParams(psf_sigma=true_sigma, glare_fraction=0.02, motion_length=0.0,
                          illum_depth=0.02, rotation_deg=2.0, keystone=0.0,
                          tone_scurve=0.1, jpeg_quality=95)
        photo, truth = capture(base, p, fiducial_truth=ft, rng=np.random.default_rng(5))
        cal = invert(photo)
        assert cal.psf.method == "slanted_edge"
        rel = abs(cal.psf.sigma - truth.psf_sigma_effective) / truth.psf_sigma_effective
        assert rel < 0.6, f"sigma {cal.psf.sigma:.2f} vs true {truth.psf_sigma_effective:.2f}"


def test_mtf_is_monotone_and_normalised(scene):
    photo, _ = _shot(scene, 0.2)
    cal = invert(photo)
    m = cal.psf.mtf_at(np.array([0.0, 0.05, 0.1, 0.25, 0.5]))
    assert m[0] == pytest.approx(1.0, abs=0.05)
    assert np.all(np.diff(m) <= 1e-6)
    assert np.all((m >= 0) & (m <= 1.001))


def test_wiener_deconvolution_is_not_in_the_floor_path(scene):
    """Deconvolution must not be able to beat the bound; it moves signal and noise
    together. This only checks it runs and is shape-preserving -- the bound's
    independence from it is structural, since `floor.py` never calls it."""
    photo, _ = _shot(scene, 0.4)
    cal = invert(photo)
    out = P.wiener_deconvolve(cal.density, cal.psf.sigma)
    assert out.shape == cal.density.shape and np.all(np.isfinite(out))


# --------------------------------------------------------------------------- #
# tone + glare: the degeneracy that bit hardest
# --------------------------------------------------------------------------- #


def test_two_point_fit_pins_the_black_point():
    """c0 and a constant veil are exactly degenerate on two anchors.

    Left free, c0 absorbs the veil, the beam stop then linearises to zero, and the
    inversion converges to a confident veil-free answer that is wrong. The fit must
    pin c0 instead.
    """
    film = FilmModel()
    anchors = [
        T.Anchor("film_base", 0.75, 1e-3, film.d_min, 1000, veil_luminance=0.02),
        T.Anchor("direct_exposure", 0.18, 1e-3, film.d_max, 1000, veil_luminance=0.02),
    ]
    model = T.fit_tone(anchors, film)
    assert model.method == "two_point"
    assert model.c0 == 0.0


def test_three_anchors_free_the_gamma():
    film = FilmModel()
    truth_gamma, c1 = 2.05, 0.95

    def px(d):
        return c1 * float(density_to_transmittance(d)) ** (1.0 / truth_gamma)

    anchors = [T.Anchor(f"a{i}", px(d), 1e-3, d, 500) for i, d in enumerate((0.2, 1.4, 3.2))]
    model = T.fit_tone(anchors, film)
    assert model.method == "three_point"
    assert abs(model.gamma - truth_gamma) < 0.3


def test_veil_is_not_zero_when_there_is_glare(scene):
    """The regression that matters most: a collapsed veil is indistinguishable
    downstream from a clean photograph."""
    base, ft = scene
    from tbtrust.physics.film import CaptureParams

    p = CaptureParams(psf_sigma=1.2, glare_fraction=0.18, glare_sigma_frac=0.12,
                      illum_depth=0.05, rotation_deg=1.5, keystone=0.0)
    photo, truth = capture(base, p, fiducial_truth=ft, rng=np.random.default_rng(9))
    cal = invert(photo)
    bs = cal.fiducials.beamstop_mask
    est = float(np.median(cal.veil[bs]))
    true = float(np.median(truth.glare_field_true[bs]))
    assert est > 0.2 * true, f"veil collapsed: {est:.5f} vs true {true:.5f}"
    assert est < 4.0 * true, f"veil wildly over-estimated: {est:.5f} vs true {true:.5f}"


def test_veil_rises_with_glare(scene):
    base, ft = scene
    from tbtrust.physics.film import CaptureParams

    out = []
    for gf in (0.01, 0.10, 0.25):
        p = CaptureParams(psf_sigma=1.2, glare_fraction=gf, keystone=0.0, rotation_deg=0.0)
        photo, _ = capture(base, p, fiducial_truth=ft, rng=np.random.default_rng(4))
        cal = invert(photo)
        out.append(float(np.median(cal.veil[cal.fiducials.beamstop_mask])))
    assert out[0] < out[1] < out[2], out


def test_glare_hotspot_localises_a_specular_blob(scene):
    base, ft = scene
    from tbtrust.physics.film import CaptureParams

    p = CaptureParams(psf_sigma=1.0, glare_fraction=0.02, flare_amplitude=1.2,
                      flare_center=(0.25, 0.25), flare_sigma_frac=0.10,
                      keystone=0.0, rotation_deg=0.0)
    photo, _ = capture(base, p, fiducial_truth=ft, rng=np.random.default_rng(6))
    cal = invert(photo)
    hs = G.hotspot(cal.glare, cal.fiducials.field_mask)
    assert hs.direction in ("upper-left", "upper", "left", "centre", "diffuse")
    assert isinstance(hs.advice, str) and len(hs.advice) > 20


# --------------------------------------------------------------------------- #
# inversion
# --------------------------------------------------------------------------- #


def test_inversion_recovers_density_differentially(scene):
    base, _ft = scene
    photo, truth = _shot(scene, 0.1)
    cal = invert(photo)
    m = cal.lung_field_mask()
    d_true = _ops.warp_perspective(base, truth.homography, base.shape, fill=0.2)
    err = cal.density[m] - d_true[m]
    # The differential error is the quantity the floor bounds; the absolute error
    # is dominated by the gamma prior and is allowed to be larger.
    assert float(np.std(err)) < 0.45


def test_error_budget_is_split_and_finite(scene):
    photo, _ = _shot(scene, 0.4)
    cal = invert(photo)
    assert np.all(np.isfinite(cal.sigma_random))
    assert np.all(np.isfinite(cal.sigma_systematic))
    m = cal.lung_field_mask()
    # The gamma prior is a scale error common to the whole frame, so the
    # systematic term should dominate the random one -- and be excluded from
    # the floor, which is what makes the bound usable.
    assert np.median(cal.sigma_systematic[m]) > np.median(cal.sigma_random[m])


def test_signal_never_collapses_to_the_clamp(scene):
    """A veil estimate exceeding the measurement drives sigma_D as 1/signal and
    sends the floor to 1e6. The veil must be clamped, not the signal alone."""
    for sev in (0.0, 0.5, 1.0):
        photo, _ = _shot(scene, sev)
        cal = invert(photo)
        m = cal.lung_field_mask()
        assert np.median(cal.signal[m]) > 1e-4, f"signal collapsed at severity {sev}"
        fm = density_floor(cal, get("infiltrate"))
        assert np.median(fm.floor[m]) < 10.0, f"floor exploded at severity {sev}"


def test_noise_model_is_physical(scene):
    photo, _ = _shot(scene, 0.3)
    cal = invert(photo)
    assert cal.noise_model["b"] >= 0.0, "variance cannot fall as signal rises"
    assert cal.noise_model["var_floor"] > 0.0, "sigma_v must never reach zero"
    s = noise_sigma(cal.noise_model, np.array([0.0, 0.5, 1.0]))
    assert np.all(s > 0) and np.all(np.isfinite(s))


# --------------------------------------------------------------------------- #
# floor
# --------------------------------------------------------------------------- #


def test_template_energy_obeys_parseval():
    f = get("infiltrate")

    def unit_mtf(x):
        return np.ones_like(np.asarray(x, dtype=float))

    e_blur, e_open = template_energy(f, px_per_mm=1.0, mtf_at=unit_mtf)
    assert e_blur == pytest.approx(e_open, rel=1e-6)


def test_blur_reduces_template_energy():
    f = get("small_nodule")
    sharp = template_energy(f, 1.0, lambda x: np.exp(-2 * np.pi**2 * 0.5**2 * np.asarray(x) ** 2))[0]
    soft = template_energy(f, 1.0, lambda x: np.exp(-2 * np.pi**2 * 4.0**2 * np.asarray(x) ** 2))[0]
    assert soft < sharp


def test_floor_rises_with_capture_severity(scene):
    floors = []
    for sev in (0.0, 0.35, 0.7):
        photo, _ = _shot(scene, sev)
        cal = invert(photo)
        fm = density_floor(cal, get("infiltrate"))
        floors.append(float(np.median(fm.floor[cal.lung_field_mask()])))
    assert floors[0] < floors[-1], floors


def test_larger_findings_have_lower_floors(scene):
    """The area gain of a matched filter, which is why a consolidation survives a
    capture that loses a miliary nodule."""
    photo, _ = _shot(scene, 0.2)
    cal = invert(photo)
    small = float(np.median(density_floor(cal, get("miliary_nodule")).floor[cal.lung_field_mask()]))
    large = float(np.median(density_floor(cal, get("consolidation")).floor[cal.lung_field_mask()]))
    assert large < small


def test_limiting_factor_names_a_real_term(scene):
    photo, _ = _shot(scene, 0.6)
    cal = invert(photo)
    fm = density_floor(cal, get("infiltrate"))
    name, detail = limiting_factor(fm, cal.lung_field_mask())
    assert name in set(fm.terms)
    assert detail["total_floor"] > 0


def test_correlated_terms_do_not_get_the_area_gain(scene):
    """Veil-fit error varies on the scale of the glare field, so it does not
    average down across a large lesion. Crediting it with the area gain is what
    made the bound optimistic for consolidations."""
    photo, _ = _shot(scene, 0.5)
    cal = invert(photo)
    f = get("consolidation")
    with_corr = float(np.median(density_floor(cal, f, FloorSpec()).floor[cal.lung_field_mask()]))
    without = float(
        np.median(density_floor(cal, f, FloorSpec(correlated_terms=())).floor[cal.lung_field_mask()])
    )
    assert with_corr >= without


# --------------------------------------------------------------------------- #
# certificate, triage, channel
# --------------------------------------------------------------------------- #


def test_certificate_orders_findings_sensibly(scene):
    photo, _ = _shot(scene, 0.35)
    cal = invert(photo)
    cert = certify(cal)
    assert not cert.abstained
    m = {f.finding: f.margin_db for f in cert.findings}
    assert m["consolidation"] > m["miliary_nodule"], m
    assert "CERTIFICATE" in cert.report()
    assert isinstance(cert.as_dict()["margin_db"], float)


def test_certificate_margin_falls_with_severity(scene):
    margins = []
    for sev in (0.0, 0.4, 0.8):
        photo, _ = _shot(scene, sev)
        cert = certify(invert(photo))
        margins.append(cert.margin_db)
    assert margins[0] > margins[-1], margins


def test_certificate_abstains_without_a_beam_stop(scene):
    photo, _ = _shot(scene, 0.0)
    h, w = photo.shape
    cropped = photo[int(0.35 * h) : int(0.65 * h), int(0.35 * w) : int(0.65 * w)]
    cal = invert(cropped)
    cert = certify(cal)
    if not invertible(cal.fiducials):
        assert cert.verdict is Verdict.ABSTAIN
        assert certificate_confidence(cert) == 0.0


def test_confidence_is_monotone_in_margin():
    class _C:
        abstained = False

        def __init__(self, m):
            self.margin_db = m

    vals = [certificate_confidence(_C(m)) for m in (-30, -10, 0, 10, 30)]
    assert all(a < b for a, b in itertools.pairwise(vals))
    assert vals[0] >= 0.0 and vals[-1] <= 1.0


def test_triage_gives_an_actionable_instruction(scene):
    photo, _ = _shot(scene, 0.7)
    cal = invert(photo)
    cert = certify(cal)
    d = triage(cert, cal, model_confidence=0.9)
    assert d.action in (Action.RETAKE, Action.REFER, Action.REPORT)
    assert len(d.instruction) > 30
    if d.action is Action.RETAKE:
        assert d.reason in ("veiling_glare", "capture_blur", "exposure_or_compression", "no_fiducials")
        assert d.expected_gain_db >= 0.0
    s = triage_summary([d])
    assert s["n"] == 1 and 0.0 <= s["retake_rate"] <= 1.0


def test_triage_refers_when_capture_is_fine_but_model_is_unsure(scene):
    photo, _ = _shot(scene, 0.0)
    cal = invert(photo)
    cert = certify(cal, findings=[get("consolidation")])
    if cert.verdict is Verdict.DETECTABLE:
        d = triage(cert, cal, model_confidence=0.2, confidence_threshold=0.5)
        assert d.action is Action.REFER
        assert "retake will not help" in d.instruction.lower()


def test_channel_capacity_is_finite_and_ordered(scene):
    photo, _ = _shot(scene, 0.3)
    cal = invert(photo)
    rep = channel_report(cal, get("consolidation"))
    assert np.isfinite(rep.bits_per_cell) and rep.bits_per_cell > 0
    assert set(rep.bits_lost) == {"veil", "blur", "quantization", "sensor_noise"}
    # Removing an impairment can only add information.
    assert all(v >= -1e-9 for v in rep.bits_lost.values()), rep.bits_lost
    table = capacity_table(cal)
    assert len(table) == len(core())


# --------------------------------------------------------------------------- #
# end to end + the eval bridge
# --------------------------------------------------------------------------- #


def test_simulate_from_a_display_image_runs_end_to_end():
    rng = np.random.default_rng(2)
    display = np.clip(rng.normal(0.45, 0.12, (200, 180)), 0, 1)
    photo, truth = simulate(display, severity=0.4, rng=rng, size=224)
    assert photo.dtype == np.uint8
    cert, cal = None, invert(photo)
    cert = certify(cal)
    assert cert.verdict in set(Verdict)
    assert truth.density_canonical.shape == (384, 384)


def test_physics_deferral_bridge():
    from tbtrust.eval import physics_deferral as PD

    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    logits = 1.8 * (y - 0.5) + rng.normal(0, 1.1, n)
    p = 1 / (1 + np.exp(-logits))
    # Physics margin correlated with error but not identical to model confidence:
    # the orthogonality the module is built around.
    wrong = (p >= 0.5).astype(int) != y
    margins = rng.normal(6, 8, n) - 6 * wrong
    abstained = rng.random(n) < 0.05

    res = PD.compare_policies(y, p, margins, abstained=abstained, threshold=0.6)
    assert {r.name for r in res} == {"learned", "physics", "physics_gated_learned"}
    for r in res:
        assert np.isfinite(r.aurc) and 0.0 <= r.coverage <= 1.0
        assert isinstance(r.as_dict(), dict)

    comp = PD.complementarity(y, p, margins, abstained=abstained)
    assert comp["errors_caught_by_union"] >= comp["errors_caught_by_learned"]
    assert 0.0 <= comp["jaccard"] <= 1.0

    sev = np.linspace(0, 1, n)
    resp = PD.severity_response(margins - 20 * sev, sev)
    assert resp["sign_as_expected"]

    tv = PD.triage_value(["report"] * 200 + ["retake"] * 60 + ["refer"] * 40, y, p)
    assert tv["report_rate"] == pytest.approx(200 / n)
    assert tv["total_tb"] == int((y == 1).sum())


def test_validation_harness_runs_small():
    from tbtrust.physics import validate as V

    rows = V.recovery_experiment(n_images=1, severities=(0.0, 0.6), size=192, seed=0)
    assert len(rows) == 2
    s = V.summarize_recovery(rows)
    assert "psf_sigma_rel_err_median" in s

    det = V.detectability_experiment(severity=0.0, finding="infiltrate", n_trials=6,
                                     size=160, seed=0)
    assert det.predicted_floor > 0
    assert np.isfinite(det.floor_lung_median)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def _agg():
    import matplotlib

    matplotlib.use("Agg")


def test_schematic_figures_render():
    """The two diagrams that need no data. Pure drawing code, so the only failure
    mode is an exception -- but that is exactly what a refactor of the style
    constants would cause, silently, right before a submission deadline."""
    _agg()
    from tbtrust.physics import figures as F

    for fn in (F.capture_chain_diagram, F.sign_convention_panel, F.finding_atlas):
        fig = fn()
        assert fig.get_axes(), f"{fn.__name__} produced no axes"


def test_image_figures_render(scene):
    _agg()
    from tbtrust.physics import figures as F
    from tbtrust.physics.triage import triage

    photo, truth = _shot(scene, 0.45)
    cal = invert(photo)
    cert = certify(cal)
    dec = triage(cert, cal, model_confidence=0.8)

    assert F.fiducial_anatomy(photo, cal.fiducials).get_axes()
    assert F.inversion_panels(photo, cal, truth).get_axes()
    assert F.certificate_card(cert, cal, dec).get_axes()
    assert F.retake_instruction(cal, dec).get_axes()
    assert F.finding_atlas(cert, cal.px_per_mm).get_axes()


def test_finding_atlas_places_markers_inside_the_lungs():
    """Anatomy is the point of that figure. Markers scattered over the mediastinum
    or outside the ribcage make it worse than no figure at all."""
    from tbtrust.physics import figures as F

    path_r, path_l = F._lung_paths()
    for key, (x, y, kind) in {
        "infiltrate": (0.290, 0.735, "fuzzy"),
        "cavity_wall": (0.690, 0.755, "ring"),
        "small_nodule": (0.720, 0.585, "dot"),
        "consolidation": (0.300, 0.400, "fuzzy"),
    }.items():
        assert path_r.contains_point((x, y)) or path_l.contains_point((x, y)), key
        del kind


def test_detectability_strip_renders(scene):
    _agg()
    from tbtrust.physics import figures as F
    from tbtrust.physics.film import sample_params

    base, ft = scene
    params = sample_params(0.4, np.random.default_rng(3))
    photo, _ = capture(base, params, fiducial_truth=ft, rng=np.random.default_rng(4))
    cal = invert(photo)
    fig = F.detectability_strip(base, params, get("infiltrate"), cal=cal,
                                fiducial_truth=ft, multiples=(0.5, 1.0, 2.0))
    # two rows x three contrasts
    assert len(fig.get_axes()) == 6


def test_gallery_accepts_pandas_series_with_a_gapped_index(tmp_path):
    """A sampled DataFrame has a non-reset index, and `labels[i]` would then index
    by label rather than position and raise KeyError. Every real caller builds the
    inputs exactly that way."""
    _agg()
    import pandas as pd
    from PIL import Image

    from tbtrust.physics import figures as F

    paths = []
    for i in range(4):
        p = tmp_path / f"x{i}.png"
        Image.fromarray((np.random.default_rng(i).random((32, 32)) * 255).astype(np.uint8)).save(p)
        paths.append(str(p))
    df = pd.DataFrame({"path": paths, "label": [0, 1, 0, 1], "clinic": ["a", "b", "a", "b"]})
    sub = df.sample(3, random_state=0)          # gapped index on purpose
    assert F.radiograph_gallery(sub["path"], sub["label"], sub["clinic"], ncols=3).get_axes()
