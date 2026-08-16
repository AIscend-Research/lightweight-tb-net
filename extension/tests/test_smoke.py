"""Pytest version of the smoke checks, for CI (`pytest -q`)."""

import numpy as np
import pytest

from tbtrust.data import degradation as deg
from tbtrust.data import manifest as M
from tbtrust.data import splits as S
from tbtrust.eval import calibration as C
from tbtrust.eval import deferral as D
from tbtrust.eval import forecast_verification as FV
from tbtrust.eval import metrics as MET


def _img(rng, size=64):
    return (rng.uniform(0, 255, (size, size))).astype(np.uint8)


def test_every_degradation_preserves_shape_and_dtype():
    rng = np.random.default_rng(0)
    img = _img(rng)
    for fn in deg.DEGRADATIONS.values():
        out = fn(img.copy(), 0.6, rng)
        assert out.shape[:2] == img.shape[:2]
        assert out.dtype == np.uint8


def test_zero_severity_is_noop():
    rng = np.random.default_rng(0)
    img = _img(rng)
    out, rec = deg.SmartphoneDegradation(severity=0.0)(img)
    assert np.array_equal(out, img)
    assert rec.total_severity == 0.0


def test_provenance_inference():
    assert M.infer_clinic_from_path("data/raw/MCUCXR_0001_0.png") == "montgomery"
    assert M.infer_clinic_from_path("data/raw/CHNCXR_0002_1.png") == "shenzhen"
    assert M.infer_clinic_from_path("x/rsna/abc.png") == "rsna"


def test_single_class_holdout_is_refused():
    rows = [{"path": f"MCUCXR_{i}_{i%2}.png", "clinic": "montgomery", "label": i % 2,
             "degradation_severity": 0.0} for i in range(10)]
    rows += [{"path": f"niaid_{i}.png", "clinic": "niaid", "label": 1,
              "degradation_severity": 0.0} for i in range(10)]
    df = M.build_manifest(rows)
    S.leave_one_clinic_out(df, "montgomery")  # ok, two classes
    # `assert False` disappears under `python -O`, which would silently turn this
    # into a test that passes whether or not the guard fires.
    with pytest.raises(ValueError, match="only class"):
        S.leave_one_clinic_out(df, "niaid")    # niaid is TB-only


def test_deferral_improves_kept_accuracy():
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    p = 1 / (1 + np.exp(-(2 * (y - 0.5) + rng.normal(0, 1.2, n))))
    base = MET.accuracy(y, p)
    op = D.tune_threshold(y, p, target="accuracy", target_value=0.9, min_coverage=0.3)
    assert op.accuracy >= base - 1e-6
    assert 0 <= C.expected_calibration_error(y, p) <= 1


def test_tbnet_param_count_matches_tbnet_paper():
    """Pins the default TBNet widths to TB-Net's reported ~4.24M params (see
    models/tbnet.py's docstring), so a well-meaning width tweak doesn't silently
    drift the reproduction away from the paper without anyone noticing."""
    pytest.importorskip("torch")
    from tbtrust.models.tbnet import TBNet

    n_params = sum(p.numel() for p in TBNet().parameters())
    assert 4.0e6 <= n_params <= 4.5e6, f"TBNet has {n_params:,} params, expected ~4.24M"


def test_evidential_logit_matches_dirichlet_prob():
    torch = pytest.importorskip("torch")
    from tbtrust.models.evidential import EvidentialClassifier, evidential_loss, evidential_prob

    model = EvidentialClassifier(backbone="resnet50", pretrained=False)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert set(out) == {"logit", "uncertainty", "evidence"}
    assert torch.allclose(torch.sigmoid(out["logit"]), evidential_prob(out["evidence"]), atol=1e-5)
    assert ((out["uncertainty"] >= 0) & (out["uncertainty"] <= 1)).all()
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loss = evidential_loss(out["evidence"], y, epoch=1, annealing_epochs=10)
    assert torch.isfinite(loss)
    loss.backward()  # gradients must actually flow through the digamma/lgamma terms


def test_deep_ensemble_predict_shapes_and_agrees_with_members():
    torch = pytest.importorskip("torch")
    from tbtrust.models.ensemble import DeepEnsemble
    from tbtrust.models.tbnet import TBNet

    members = [TBNet(with_uncertainty_head=False) for _ in range(3)]
    ensemble = DeepEnsemble(members=members)
    x = torch.randn(5, 3, 32, 32)
    mean, std = ensemble.predict(x)
    assert mean.shape == (5,) and std.shape == (5,)
    assert (std >= 0).all()
    # three differently-initialized members must not all agree exactly
    assert std.sum().item() > 0


def test_sequential_mc_dropout_stops_within_budget():
    torch = pytest.importorskip("torch")
    from tbtrust.eval.sequential_deferral import sequential_mc_dropout_decide
    from tbtrust.models.tbnet import TBNet

    model = TBNet(with_uncertainty_head=False, dropout=0.5)
    x = torch.randn(1, 3, 32, 32)
    result = sequential_mc_dropout_decide(model, x, passes_min=2, passes_max=10)
    assert result.decision in {"report", "defer"}
    assert 2 <= result.passes_used <= 10


def test_cusum_monitor_flags_sustained_upward_drift_not_noise():
    from tbtrust.eval.sequential_deferral import CUSUMMonitor

    rng = np.random.default_rng(0)
    mon = CUSUMMonitor(target=0.2, slack=0.05, threshold=0.5)
    # 15 in-control readings should not alarm
    for _ in range(15):
        out = mon.update(float(0.2 + rng.normal(0, 0.02)))
    assert not out["alarm_high"]
    # a sustained upward shift should eventually alarm
    alarmed = False
    for _ in range(20):
        out = mon.update(float(0.35 + rng.normal(0, 0.02)))
        alarmed = alarmed or out["alarm_high"]
    assert alarmed


def test_worst_case_degradation_search_finds_loss_at_least_as_high_as_average():
    pytest.importorskip("torch")
    from tbtrust.eval.adversarial_degradation import worst_case_degradation_search
    from tbtrust.models.tbnet import TBNet

    rng = np.random.default_rng(0)
    model = TBNet(with_uncertainty_head=True)
    img = _img(rng, size=48)
    result = worst_case_degradation_search(model, img, label=1, severity=0.7, n_trials=6, image_size=48)
    assert result.worst_loss >= result.avg_loss - 1e-9
    assert isinstance(result.worst_ops, dict)


def test_murphy_decomposition_reconstructs_brier_score():
    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(0, 2, n)
    p = 1 / (1 + np.exp(-(2 * (y - 0.5) + rng.normal(0, 1.2, n))))
    dec = FV.murphy_decomposition(y, p, n_bins=10)
    assert abs(dec["brier_reconstructed"] - dec["brier_actual"]) < 1e-2
    assert dec["reliability"] >= 0 and dec["resolution"] >= 0


def test_brier_skill_score_zero_for_base_rate_only_forecast():
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    p_base_rate_only = np.full(n, y.mean())
    bss = FV.brier_skill_score(y, p_base_rate_only)
    assert abs(bss) < 1e-9  # a forecast that's just the base rate has zero skill by definition


def test_brier_skill_score_positive_for_informative_forecast():
    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(0, 2, n)
    p = 1 / (1 + np.exp(-(3 * (y - 0.5) + rng.normal(0, 0.8, n))))  # correlated with label
    assert FV.brier_skill_score(y, p) > 0


# --------------------------------------------------------------------------- #
# Temperature scaling.
#
# These pin a real bug: fit_temperature used fixed-step gradient descent with an
# inverted gradient sign, so it *ascended* the NLL. On overconfident logits it
# drove T toward zero and returned NaN once exp overflowed -- the exact case
# temperature scaling exists to fix -- and the only assertion guarding it was
# `T > 0`, which NaN does not even fail loudly on. Compare against a brute-force
# grid optimum so any future reimplementation has to actually minimise the NLL.
# --------------------------------------------------------------------------- #
def _brute_force_temperature(z, y, lo=0.05, hi=20.0, n=4000):
    grid = np.linspace(lo, hi, n)
    return grid[int(np.argmin([C.temperature_nll(z, y, t) for t in grid]))]


def _logits_with_confidence(inflate, n=3000, seed=0):
    """Logits whose miscalibration is known: inflate > 1 => overconfident."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    return y, inflate * (0.8 * (2 * y - 1) + rng.normal(0, 1.0, n))


@pytest.mark.parametrize("inflate", [4.0, 1.0, 0.25])
def test_fit_temperature_matches_brute_force_optimum(inflate):
    y, z = _logits_with_confidence(inflate)
    T = C.fit_temperature(z, y)
    assert np.isfinite(T) and T > 0
    assert abs(T - _brute_force_temperature(z, y)) < 0.05


def test_fit_temperature_softens_overconfident_logits_and_improves_ece():
    y, z = _logits_with_confidence(4.0)   # deliberately overconfident
    T = C.fit_temperature(z, y)
    assert T > 1.0, "overconfident logits must be softened, not sharpened"
    ece_before = C.expected_calibration_error(y, C.apply_temperature(z, 1.0))
    ece_after = C.expected_calibration_error(y, C.apply_temperature(z, T))
    assert ece_after < ece_before


def test_fit_temperature_survives_extreme_logits_without_overflow():
    y, z = _logits_with_confidence(1.0)
    z = z * 500.0                          # would overflow a naive 1/(1+exp(-z/T))
    T = C.fit_temperature(z, y)
    assert np.isfinite(T)
    p = C.apply_temperature(z, T)
    assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))


def test_fit_temperature_returns_one_for_single_class_calibration_data():
    assert C.fit_temperature(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1])) == 1.0


# --------------------------------------------------------------------------- #
# Deferral ranks on an explicit uncertainty signal.
# --------------------------------------------------------------------------- #
def test_deferral_uses_supplied_confidence_not_just_softmax():
    """An oracle uncertainty must beat softmax confidence on AURC."""
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, n)
    # Probabilities carry only weak signal, so max(p, 1-p) ranks errors poorly.
    p = np.clip(0.5 + 0.05 * (2 * y - 1) + rng.normal(0, 0.05, n), 0.01, 0.99)
    wrong = (p >= 0.5).astype(int) != y
    oracle_uncertainty = wrong.astype(float) + rng.uniform(0, 0.01, n)
    oracle_conf = D.confidence_from_uncertainty(oracle_uncertainty)

    aurc_softmax = D.area_under_risk_coverage(y, p)
    aurc_oracle = D.area_under_risk_coverage(y, p, confidence=oracle_conf)
    assert aurc_oracle < aurc_softmax


def test_confidence_from_uncertainty_is_monotone_decreasing_and_bounded():
    u = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    c = D.confidence_from_uncertainty(u)
    assert np.all(np.diff(c) < 0)
    assert c.min() == 0.0 and c.max() == 1.0
    flat = D.confidence_from_uncertainty(np.full(5, 0.3))
    assert np.allclose(flat, 0.5)          # no discrimination available


def test_human_rescue_rate_honours_supplied_confidence():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.9, 0.4, 0.6, 0.1])     # first and last are wrong
    conf = np.array([0.1, 0.9, 0.9, 0.1])  # flags exactly the two wrong ones
    out = D.human_rescue_rate(y, p, threshold=0.5, confidence=conf)
    assert out["deferred"] == 2
    assert out["would_correct_frac"] == 1.0


# --------------------------------------------------------------------------- #
# Split-conformal prediction.
# --------------------------------------------------------------------------- #
def _exchangeable_draw(rng, n=400, sharp=1.5):
    y = (rng.random(n) < 0.5).astype(int)
    z = sharp * (2 * y - 1) + rng.normal(0, 1.0, n)
    return y, 1 / (1 + np.exp(-z))


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_conformal_attains_target_coverage_on_exchangeable_data(alpha):
    from tbtrust.eval import conformal as CP

    rng = np.random.default_rng(0)
    coverages = []
    for _ in range(120):
        y_cal, p_cal = _exchangeable_draw(rng)
        y_test, p_test = _exchangeable_draw(rng)
        cal = CP.calibrate(y_cal, p_cal, alpha=alpha)
        coverages.append(CP.evaluate(y_test, p_test, cal)["coverage"])
    # Marginal guarantee: mean coverage >= 1 - alpha (small Monte-Carlo slack).
    assert np.mean(coverages) >= (1 - alpha) - 0.02


def test_conformal_flags_calibration_set_too_small_for_alpha():
    from tbtrust.eval import conformal as CP

    # n = 5 cannot certify alpha = 0.05: ceil((5+1)*0.95) = 6 > 5.
    cal = CP.calibrate([0, 1, 0, 1, 0], [0.2, 0.8, 0.3, 0.7, 0.1], alpha=0.05)
    assert cal.guarantee_attainable is False
    assert cal.q_hat == 1.0
    # A vacuous q_hat must abstain on everything rather than pretend to decide.
    assert CP.decide([0.2, 0.9], cal.q_hat)["report"].sum() == 0


def test_conformal_set_sizes_map_onto_the_deferral_trichotomy():
    from tbtrust.eval import conformal as CP

    d = CP.decide([0.99, 0.5, 0.01], q_hat=0.2)
    assert list(d["set_size"]) == [1, 0, 1]
    assert list(d["report"]) == [True, False, True]
    assert d["defer_reason"][1] == "no_plausible_label"
    assert list(d["prediction"][[0, 2]]) == [1, 0]

    ambiguous = CP.decide([0.5], q_hat=0.9)
    assert ambiguous["set_size"][0] == 2
    assert ambiguous["defer_reason"][0] == "ambiguous"


def test_conformal_reports_coverage_shortfall_under_shift():
    from tbtrust.eval import conformal as CP

    rng = np.random.default_rng(0)
    y_val, p_val = _exchangeable_draw(rng, n=600, sharp=2.0)
    # A "held-out clinic" the model is confidently wrong about.
    y_test = (rng.random(400) < 0.5).astype(int)
    p_test = 1 / (1 + np.exp(-(-2.0 * (2 * y_test - 1) + rng.normal(0, 1.0, 400))))
    rep = CP.coverage_gap_report(y_val, p_val, y_test, p_test, alpha=0.1)
    assert rep["heldout_clinic"]["coverage"] < rep["in_distribution_reference"]["coverage"]
    assert rep["coverage_shortfall_vs_target"] > 0


# --------------------------------------------------------------------------- #
# Train and eval must derive identical splits.
#
# Pins the leak found during the audit: train/loop.py passed cfg.data.val_frac and
# cfg.seed while eval/run.py and run_experiments.py passed neither, silently using
# the defaults. Any config setting val_frac or a non-zero seed then calibrated the
# temperature, the deferral threshold and the conformal quantile on images the
# model had trained on.
# --------------------------------------------------------------------------- #
def test_loco_split_from_config_is_reproducible_and_honours_config():
    df = M.build_manifest([
        {"path": f"/{c}/{i}.png", "clinic": c, "label": i % 2, "degradation_severity": 0.0}
        for c in ("montgomery", "shenzhen") for i in range(60)
    ])
    cfg = {"seed": 7, "data": {"holdout_clinic": "montgomery", "val_frac": 0.3}}

    a = S.loco_split_from_config(df, cfg)
    b = S.loco_split_from_config(df, cfg)
    assert a["split"].equals(b["split"])                     # deterministic

    # val_frac is actually applied: 30% of the 60 seen-clinic rows, not the 15% default.
    seen = a[a["clinic"] != "montgomery"]
    assert abs((seen["split"] == "val").sum() / len(seen) - 0.3) < 0.05

    # A different seed or val_frac must give a different split -- which is exactly
    # why train and eval have to derive theirs the same way.
    other = S.loco_split_from_config(df, {"seed": 8, "data": cfg["data"]})
    assert not a["split"].equals(other["split"])

    # And the held-out clinic never leaks into train/val.
    assert set(a.loc[a["clinic"] == "montgomery", "split"]) == {"test"}


# --------------------------------------------------------------------------- #
# Degradation severity accounting.
#
# `total_severity` is the SNR proxy behind the channel-capacity framing and the
# intended basis for the weak uncertainty label, so "more degradation => higher
# score" has to hold. It previously averaged over only the ops that fired, which
# inverted it: one op at 0.9 scored 0.9 while six ops at 0.5 scored 0.5.
# --------------------------------------------------------------------------- #
def test_total_severity_increases_with_more_simultaneous_ops():
    one_severe = deg.DegradationRecord(ops={"glare": 0.9})
    many_mild = deg.DegradationRecord(ops=dict.fromkeys(deg.DEGRADATIONS, 0.5))
    assert many_mild.total_severity > one_severe.total_severity


def test_total_severity_bounded_and_zero_when_clean():
    assert deg.DegradationRecord().total_severity == 0.0
    saturated = deg.DegradationRecord(ops=dict.fromkeys(deg.DEGRADATIONS, 1.0))
    assert saturated.total_severity == pytest.approx(1.0)
    for k in range(len(deg.DEGRADATIONS) + 1):
        rec = deg.DegradationRecord(ops=dict.fromkeys(list(deg.DEGRADATIONS)[:k], 1.0))
        assert 0.0 <= rec.total_severity <= 1.0


# --------------------------------------------------------------------------- #
# Evaluation-time degradation must be reproducible AND varied per image.
#
# A fixed dataset seed used to be handed straight to SmartphoneDegradation, which
# builds a fresh Generator(seed) per call -- so every image received an identical
# draw. Leaving it None instead made every fetch different, so a severity sweep
# was unrepeatable and a second pass over the loader saw different images than
# the first. The per-item offset fixes both.
# --------------------------------------------------------------------------- #
def _tiny_image_manifest(tmp_path, n=6):
    from PIL import Image

    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        p = tmp_path / f"MCUCXR_{i:03d}_{i % 2}.png"
        Image.fromarray(np.clip(rng.normal(120, 20, (48, 48)), 0, 255).astype(np.uint8), "L").save(p)
        rows.append({"path": str(p), "clinic": "montgomery", "label": i % 2,
                     "split": "test", "degradation_severity": 0.0})
    return __import__("pandas").DataFrame(rows)


def test_seeded_eval_degradation_is_reproducible_and_varies_across_images(tmp_path):
    torch = pytest.importorskip("torch")  # noqa: F841
    from tbtrust.data.dataset import TBDataset

    df = _tiny_image_manifest(tmp_path)
    a = TBDataset(df, split="test", image_size=48, degradation_severity=0.6, seed=7)
    b = TBDataset(df, split="test", image_size=48, degradation_severity=0.6, seed=7)

    # Same seed, two datasets, repeated fetches -> byte-identical images.
    assert np.allclose(a[0]["image"].numpy(), b[0]["image"].numpy())
    assert np.allclose(a[0]["image"].numpy(), a[0]["image"].numpy())

    # ...but different images must not receive the identical degradation draw.
    assert not np.allclose(a[0]["image"].numpy(), a[1]["image"].numpy())

    # Unseeded stays random per fetch (that is the training-time augmentation).
    u = TBDataset(df, split="test", image_size=48, degradation_severity=0.6)
    assert not np.allclose(u[0]["image"].numpy(), u[0]["image"].numpy())


# --------------------------------------------------------------------------- #
# Config loading and fail-fast guards.
# --------------------------------------------------------------------------- #
def test_load_experiment_accepts_bare_name_and_full_path():
    from tbtrust.config import load_experiment

    by_path = load_experiment("configs/loco_montgomery.yaml", config_dir="configs")
    by_name = load_experiment("loco_montgomery.yaml", config_dir="configs")
    assert by_path == by_name
    assert by_name["data"]["holdout_clinic"] == "montgomery"


def test_load_experiment_reports_a_missing_config_clearly():
    from tbtrust.config import load_experiment

    with pytest.raises(FileNotFoundError, match="not found"):
        load_experiment("does_not_exist.yaml", config_dir="configs")


def test_training_refuses_an_empty_val_split(tmp_path):
    """An empty val split used to train happily, write no checkpoint, and then
    fail much later in eval on a path train() had reported as its result."""
    pytest.importorskip("torch")
    from tbtrust.train.loop import _build_loaders

    df = _tiny_image_manifest(tmp_path, n=8)
    df.loc[:, "clinic"] = ["montgomery"] * 4 + ["shenzhen"] * 4
    manifest = tmp_path / "manifest.csv"
    M.save(M.build_manifest(df.to_dict("records")), manifest)

    cfg = {
        "seed": 0,
        "data": {"manifest": str(manifest), "holdout_clinic": "montgomery",
                 "image_size": 48, "val_frac": 0.0, "require_two_class_test": True},
        "degradation": {"train_low": 0.0, "train_high": 0.5, "val_fixed": 0.25},
        "train": {"batch_size": 2, "num_workers": 0},
    }
    with pytest.raises(ValueError, match="'val' split is empty"):
        _build_loaders(cfg)


def test_seeded_training_dataset_varies_across_epochs_but_repeats_per_epoch(tmp_path):
    """Training augmentation must be reproducible from cfg.seed yet still differ
    epoch to epoch. Seeding without an epoch term would show the model the exact
    same degraded image every epoch; no seed at all draws from OS entropy, which
    np.random.seed cannot control, leaving the whole run irreproducible."""
    pytest.importorskip("torch")
    from tbtrust.data.dataset import TBDataset, uniform_severity

    df = _tiny_image_manifest(tmp_path)
    df.loc[:, "split"] = "train"

    def make():
        return TBDataset(df, split="train", image_size=48, seed=5,
                         severity_sampler=uniform_severity(0.2, 0.9, seed=5))

    a, b = make(), make()
    a.set_epoch(0)
    b.set_epoch(0)
    assert np.allclose(a[0]["image"].numpy(), b[0]["image"].numpy())   # reproducible

    a.set_epoch(1)
    assert not np.allclose(a[0]["image"].numpy(), b[0]["image"].numpy())  # varies by epoch

    a.set_epoch(0)
    assert np.allclose(a[0]["image"].numpy(), b[0]["image"].numpy())   # and is stable


def test_severity_sampler_is_independent_of_fetch_order(tmp_path):
    """A seeded run must not depend on the order the DataLoader happens to fetch
    items in -- that order changes with shuffling and with num_workers."""
    pytest.importorskip("torch")
    from tbtrust.data.dataset import TBDataset, uniform_severity

    df = _tiny_image_manifest(tmp_path)
    df.loc[:, "split"] = "train"

    def sev_for(order):
        ds = TBDataset(df, split="train", image_size=48, seed=3,
                       severity_sampler=uniform_severity(0.1, 0.9, seed=3))
        out = {}
        for i in order:
            out[i] = float(ds[i]["severity"])
        return out

    forward = sev_for([0, 1, 2, 3])
    backward = sev_for([3, 2, 1, 0])
    assert forward == backward


def test_random_split_is_in_distribution_and_partitions_everything():
    """The in-distribution reference split: clinic-blind, so every clinic appears
    in train AND test. That is what makes it the baseline the cross-site gap is
    measured against, rather than another deployment number."""
    df = M.build_manifest([
        {"path": f"/{c}/{i}.png", "clinic": c, "label": i % 2, "degradation_severity": 0.0}
        for c in ("montgomery", "shenzhen") for i in range(60)
    ])
    out = S.random_split(df, val_frac=0.2, test_frac=0.2, seed=0)

    assert len(out) == len(df)
    assert set(out["split"]) == {"train", "val", "test"}
    assert out.groupby("path")["split"].nunique().max() == 1
    for split in ("train", "val", "test"):
        assert out[out.split == split]["clinic"].nunique() == 2, f"{split} is not clinic-blind"
    assert abs((out.split == "test").mean() - 0.2) < 0.05
    assert abs((out.split == "val").mean() - 0.2) < 0.05


def test_split_from_config_dispatches_on_split_mode():
    df = M.build_manifest([
        {"path": f"/{c}/{i}.png", "clinic": c, "label": i % 2, "degradation_severity": 0.0}
        for c in ("montgomery", "shenzhen") for i in range(40)
    ])
    base = {"seed": 0, "data": {"holdout_clinic": "montgomery", "val_frac": 0.2}}

    loco = S.split_from_config(df, {**base, "data": {**base["data"], "split_mode": "loco"}})
    assert set(loco.loc[loco.clinic == "montgomery", "split"]) == {"test"}

    rand = S.split_from_config(df, {**base, "data": {**base["data"],
                                                     "split_mode": "random", "test_frac": 0.2}})
    assert rand[rand.split == "test"]["clinic"].nunique() == 2

    with pytest.raises(ValueError, match=r"unknown data\.split_mode"):
        S.split_from_config(df, {**base, "data": {**base["data"], "split_mode": "nonsense"}})
