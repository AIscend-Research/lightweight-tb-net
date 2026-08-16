#!/usr/bin/env python3
"""End-to-end smoke test on synthetic data. No downloads, no GPU, torch optional.

Run this right after cloning + `pip install -e .` to confirm the plumbing works:

    python scripts/smoke_test.py

It exercises the torch-free core (degradation pipeline, manifest, LOCO splits,
calibration + deferral metrics) on generated images, and if torch is installed it
also does a tiny forward pass + one train step. Prints PASS/FAIL per stage.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# make src/ importable when run from the repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbtrust.data import degradation as deg
from tbtrust.data import manifest as M
from tbtrust.data import splits as S
from tbtrust.eval import calibration as C
from tbtrust.eval import deferral as D
from tbtrust.eval import metrics as MET


def _ok(msg):
    print(f"  PASS  {msg}")


def _fake_xray(rng, size=128):
    """A crude lung-ish grayscale image so degradations have structure to chew on."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = 60 + 40 * np.sin(xx / 12) * np.cos(yy / 15)
    for _ in range(2):  # two bright "lung fields"
        cy, cx = rng.uniform(0.35, 0.65, 2) * size
        base += 80 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (size / 6) ** 2)))
    return np.clip(base + rng.normal(0, 8, (size, size)), 0, 255).astype(np.uint8)


def test_degradation(rng):
    img = _fake_xray(rng)
    for name, fn in deg.DEGRADATIONS.items():
        out = fn(img.copy(), severity=0.7, rng=rng)
        assert out.shape[:2] == img.shape[:2], f"{name} changed shape"
        assert out.dtype == np.uint8, f"{name} wrong dtype"
    comp = deg.SmartphoneDegradation(severity=0.8, seed=0)
    out, rec = comp(img)
    assert out.shape == img.shape
    assert 0.0 <= rec.total_severity <= 1.0
    # severity 0 must be a no-op
    clean, rec0 = deg.SmartphoneDegradation(severity=0.0)(img)
    assert np.array_equal(clean, img) and rec0.total_severity == 0.0
    _ok(f"degradation pipeline ({len(deg.DEGRADATIONS)} ops + composite + severity record)")


def test_manifest_and_splits(rng):
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        rows = []
        # two two-class clinics + two single-class clinics, via filename prefixes
        plan = [("MCUCXR", "montgomery", [0, 1]), ("CHNCXR", "shenzhen", [0, 1]),
                ("niaid", "niaid", [1]), ("rsna", "rsna", [0])]
        for prefix, _clinic, labels in plan:
            for i in range(20):
                lab = rng.choice(labels)
                p = root / f"{prefix}_{i:03d}_{lab}.png"
                Image.fromarray(_fake_xray(rng, 32)).save(p)
                rows.append({"path": str(p), "clinic": M.infer_clinic_from_path(str(p)),
                             "label": int(lab), "degradation_severity": 0.0})
        df = M.build_manifest(rows)
        assert set(df["clinic"]) == {"montgomery", "shenzhen", "niaid", "rsna"}
        rep = M.class_balance_report(df)
        assert rep.loc["montgomery", "total"] == 20
        _ok("manifest build + provenance inference + class-balance report")

        # two-class holdout works; single-class holdout is refused
        split = S.leave_one_clinic_out(df, "montgomery", seed=0)
        assert set(split["split"]) >= {"train", "val", "test"}
        assert (split[split.split == "test"]["clinic"] == "montgomery").all()
        try:
            S.leave_one_clinic_out(df, "niaid", seed=0)
            raise AssertionError("single-class holdout should have raised")
        except ValueError:
            pass
        folds = S.all_loco_folds(df, two_class_only=True)
        assert set(folds) == {"montgomery", "shenzhen"}
        _ok("leave-one-clinic-out split + single-class guard + fold enumeration")


def test_eval_metrics(rng):
    # simulate a decent-but-imperfect model: prob correlates with label + noise
    n = 400
    y = rng.integers(0, 2, n)
    logits = 2.0 * (y - 0.5) + rng.normal(0, 1.2, n)  # overlapping -> realistic
    p = 1 / (1 + np.exp(-logits))

    summ = MET.summary(y, p)
    assert 0 <= summ["accuracy"] <= 1 and 0 <= summ["brier"] <= 1
    ece = C.expected_calibration_error(y, p)
    mce = C.max_calibration_error(y, p)
    assert 0 <= ece <= 1 and 0 <= mce <= 1

    # Temperature scaling must actually fix a *known* miscalibration, not merely
    # return some positive number. Inflate the logits 4x so the model is
    # definitively overconfident; the fit has to soften (T > 1) and lower ECE.
    over = 4.0 * logits
    T = C.fit_temperature(over, y)
    ece_over = C.expected_calibration_error(y, C.apply_temperature(over, 1.0))
    ece_fixed = C.expected_calibration_error(y, C.apply_temperature(over, T))
    assert np.isfinite(T) and T > 1.0, "overconfident logits must be softened"
    assert ece_fixed < ece_over, "temperature scaling must reduce ECE here"
    _ok(f"metrics (acc={summ['accuracy']:.2f}, brier={summ['brier']:.3f}) + ECE={ece:.3f}/MCE={mce:.3f} "
        f"+ temp-scale T={T:.2f} (ECE {ece_over:.3f}->{ece_fixed:.3f} on overconfident logits)")

    curve = D.risk_coverage_curve(y, p)
    assert curve[0].coverage >= curve[-1].coverage  # higher threshold -> less coverage
    aurc = D.area_under_risk_coverage(y, p)
    op = D.tune_threshold(y, p, target="accuracy", target_value=0.9, min_coverage=0.3)
    rescue = D.human_rescue_rate(y, p, threshold=op.threshold)
    # deferring the least-confident cases should not hurt accuracy on what's kept
    assert op.accuracy >= summ["accuracy"] - 1e-6

    # The deferral policy must rank on an explicit uncertainty signal when given
    # one -- that hook is what lets eval/run.py compare MC-dropout / the learned
    # head / ensembles against plain softmax confidence. An oracle signal that
    # knows which cases are wrong has to beat softmax on AURC.
    wrong = (p >= 0.5).astype(int) != y
    oracle_conf = D.confidence_from_uncertainty(wrong.astype(float) + rng.uniform(0, 0.01, n))
    aurc_oracle = D.area_under_risk_coverage(y, p, confidence=oracle_conf)
    assert aurc_oracle < aurc, "supplied confidence is being ignored by the deferral curve"
    _ok(f"safe-deferral (AURC={aurc:.3f} softmax vs {aurc_oracle:.3f} oracle-signal, "
        f"tuned T*={op.threshold:.2f} -> cov={op.coverage:.2f}, acc={op.accuracy:.2f}, "
        f"rescue_frac={rescue['would_correct_frac']:.2f})")


def test_conformal(rng):
    """Split-conformal coverage: the distribution-free backstop on deferral."""
    from tbtrust.eval import conformal as CP

    def draw(n, sharp=1.5, flip=1.0):
        yy = (rng.random(n) < 0.5).astype(int)
        zz = flip * sharp * (2 * yy - 1) + rng.normal(0, 1.0, n)
        return yy, 1 / (1 + np.exp(-zz))

    alpha = 0.1
    y_cal, p_cal = draw(800)
    y_new, p_new = draw(800)
    cal = CP.calibrate(y_cal, p_cal, alpha=alpha)
    ev = CP.evaluate(y_new, p_new, cal)
    assert cal.guarantee_attainable
    assert ev["coverage"] >= (1 - alpha) - 0.05, "conformal undercovers on exchangeable data"

    # Under a shifted "clinic" (model confidently wrong) coverage must drop, and
    # the shortfall is the reported cross-site diagnostic -- not a silent failure.
    y_shift, p_shift = draw(600, flip=-1.0)
    rep = CP.coverage_gap_report(y_cal, p_cal, y_shift, p_shift, alpha=alpha)
    assert rep["coverage_shortfall_vs_target"] > 0
    _ok(f"conformal (q_hat={cal.q_hat:.3f}, coverage={ev['coverage']:.3f} vs target {1 - alpha:.2f}, "
        f"abstain={ev['abstention_rate']:.2f}; shifted-clinic shortfall="
        f"{rep['coverage_shortfall_vs_target']:.3f})")


def test_physics(rng):
    """Photograph a synthetic film, invert it blind, and certify it.

    The physics track has no third-party dependency and no torch requirement, so
    it belongs in the torch-free core of the smoke test. The assertions are the
    ordering properties that must hold whatever the nominal finding contrasts in
    `physics/findings.py` are set to -- a worse capture must produce a higher
    floor and a smaller margin. `scripts/validate_physics.py` is the real
    quantitative check.
    """
    from tbtrust.physics import certify
    from tbtrust.physics.film import capture, sample_params, synthetic_chest_density
    from tbtrust.physics.findings import get
    from tbtrust.physics.floor import density_floor
    from tbtrust.physics.invert import invert
    from tbtrust.physics.triage import triage

    base, ftruth = synthetic_chest_density(size=224, rng=rng)
    results = []
    for severity in (0.0, 0.8):
        # Same parameter-sampler seed for both conditions, so the two captures are
        # paired and differ only through the severity dial. Drawing independently
        # lets an unlucky pair of full-well values invert the comparison, and a
        # smoke test that fails one run in five teaches people to ignore it.
        params = sample_params(severity, np.random.default_rng(17))
        photo, truth = capture(base, params, fiducial_truth=ftruth,
                               rng=np.random.default_rng(23))
        cal = invert(photo)
        cert = certify(cal)
        floor = float(np.median(density_floor(cal, get("infiltrate")).floor[cal.lung_field_mask()]))
        results.append((cal, cert, floor, truth))

    (cal0, cert0, floor0, truth0), (cal1, cert1, floor1, _) = results
    assert cal0.fiducials.has_beamstop, "no optical beam stop found on a clean synthetic film"
    assert cal0.psf.method == "slanted_edge", f"PSF fell back to {cal0.psf.method}"
    rel = abs(cal0.psf.sigma - truth0.psf_sigma_effective) / max(truth0.psf_sigma_effective, 1e-6)
    assert rel < 0.8, f"PSF recovery off by {rel:.0%}"
    assert floor1 > floor0, f"floor did not rise with degradation ({floor0:.4f} -> {floor1:.4f})"
    assert cert1.margin_db < cert0.margin_db, "certificate margin did not fall with degradation"

    decision = triage(cert1, cal1, model_confidence=0.9)
    assert decision.instruction and decision.action.value in ("report", "retake", "refer")
    _ok(f"physics: PSF {cal0.psf.sigma:.2f}px (true {truth0.psf_sigma_effective:.2f}), "
        f"floor dD {floor0:.4f}->{floor1:.4f}, margin {cert0.margin_db:+.1f}->{cert1.margin_db:+.1f} dB, "
        f"triage='{decision.action.value}'")


def test_torch_optional(rng):
    try:
        import torch
    except ImportError:
        print("  SKIP  torch not installed -> skipping model forward/train step "
              "(install with `pip install -e .` to enable)")
        return
    from tbtrust.models.tbnet import TBNet

    model = TBNet(with_uncertainty_head=True)
    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    assert out["logit"].shape == (4,) and out["uncertainty"].shape == (4,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logit"], torch.rand(4))
    loss.backward()
    _ok("torch model forward + backward (TBNet)")


def main() -> int:
    rng = np.random.default_rng(0)
    print("TB-Trust smoke test")
    try:
        test_degradation(rng)
        test_manifest_and_splits(rng)
        test_eval_metrics(rng)
        test_conformal(rng)
        test_physics(rng)
        test_torch_optional(rng)
    except AssertionError as e:
        print(f"  FAIL  {e}")
        return 1
    print("\nAll core stages passed. The scaffold is wired up correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
