"""Tests for the domain-generalization half of the cross-site story.

These cover the pieces ported from the `xctb` prototype: the CORAL/IRM losses,
the gradient-reversal layer, clinic-conditional FiLM, the clinic ids the Dataset
emits, and the degradation-vs-uncertainty correlation check. Everything torch-y
is `importorskip`ed so the torch-free half of the suite still runs anywhere.
"""

import numpy as np
import pytest

from tbtrust.data import manifest as M
from tbtrust.eval import degradation_uncertainty as DU

# --- clinic ids ------------------------------------------------------------


def test_clinic_index_is_stable_and_maps_unknowns_into_range():
    assert M.clinic_index("montgomery") == 0
    assert M.clinic_index("Shenzhen") == M.clinic_index("shenzhen")
    # Anything unrecognised must land in the catch-all slot rather than raise or,
    # worse, index the embedding out of bounds mid-epoch.
    for name in ("unknown", "some_new_hospital", ""):
        idx = M.clinic_index(name)
        assert idx == len(M.CLINICS)
        assert 0 <= idx < M.NUM_CLINIC_SLOTS


# --- DG losses -------------------------------------------------------------


def test_coral_loss_is_zero_for_identical_clinics_and_positive_when_shifted():
    torch = pytest.importorskip("torch")
    from tbtrust.losses.dg import coral_loss

    g = torch.randn(32, 8, generator=torch.Generator().manual_seed(0))
    assert coral_loss([g, g.clone()]).item() == pytest.approx(0.0, abs=1e-6)
    assert coral_loss([g, g + 3.0]).item() > 0.1


def test_coral_loss_skips_clinics_too_small_for_a_covariance():
    torch = pytest.importorskip("torch")
    from tbtrust.losses.dg import coral_loss

    # One image from a clinic has no covariance; with only one usable group left
    # there is nothing to align, so the penalty is 0 rather than a NaN.
    loss = coral_loss([torch.randn(16, 4), torch.randn(1, 4)])
    assert loss.item() == 0.0


def test_irm_penalty_is_small_at_the_optimum_and_large_when_miscalibrated():
    torch = pytest.importorskip("torch")
    from tbtrust.losses.dg import irm_penalty

    y = torch.tensor([1.0, 1.0, 0.0, 0.0])
    confident_and_right = torch.tensor([8.0, 8.0, -8.0, -8.0])
    confident_and_wrong = torch.tensor([-8.0, -8.0, 8.0, 8.0])
    assert irm_penalty(confident_and_right, y).item() < irm_penalty(confident_and_wrong, y).item()


def test_gradient_reversal_is_identity_forward_and_flips_sign_backward():
    torch = pytest.importorskip("torch")
    from tbtrust.models.grl import dann_lambda, grad_reverse

    x = torch.ones(3, requires_grad=True)
    out = grad_reverse(x, 2.0)
    assert torch.allclose(out, x)          # forward: identity
    out.sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -2.0))   # backward: -lambda

    # The adversary is eased in, not switched on at full strength.
    assert dann_lambda(0, 100) == pytest.approx(0.0, abs=1e-6)
    assert dann_lambda(100, 100) > 0.99
    assert dann_lambda(50, 100) > dann_lambda(10, 100)


def test_clinic_film_starts_at_identity_and_unseen_clinic_uses_mean_embedding():
    torch = pytest.importorskip("torch")
    from tbtrust.models.clinic_film import ClinicFiLM

    film = ClinicFiLM(feature_dim=6, num_clinics=M.NUM_CLINIC_SLOTS)
    feats = torch.randn(4, 6)
    idx = torch.tensor([0, 1, 2, 3])
    # Initialised to scale=1 / shift=0, so an untrained FiLM must not perturb the
    # features at all -- otherwise turning it on would change the baseline before
    # any learning has happened, and the ablation would not be measuring FiLM.
    assert torch.allclose(film(feats, idx), feats, atol=1e-6)
    # No clinic label (every LOCO test image): falls back to the mean embedding,
    # which must still produce a value, not an index error.
    assert film(feats, None).shape == feats.shape


# --- model wiring ----------------------------------------------------------


def test_classifier_exposes_features_and_domain_head_only_when_configured():
    torch = pytest.importorskip("torch")
    from tbtrust.models.baseline import TBClassifier

    plain = TBClassifier(backbone="resnet18", pretrained=False, with_uncertainty_head=True)
    out = plain(torch.randn(2, 3, 64, 64))
    assert out["logit"].shape == (2,)
    assert out["features"].dim() == 2 and out["features"].shape[0] == 2
    with pytest.raises(RuntimeError):
        plain.domain_logits(out["features"])          # no domain head was built

    dann = TBClassifier(backbone="resnet18", pretrained=False, with_domain_head=True)
    d_out = dann(torch.randn(2, 3, 64, 64))
    assert dann.domain_logits(d_out["features"]).shape == (2, M.NUM_CLINIC_SLOTS)


def test_build_model_reads_dg_and_film_from_config():
    pytest.importorskip("torch")
    from tbtrust.models.baseline import build_model

    cfg = {"model": {"backbone": "resnet18", "pretrained": False, "clinic_film": True},
           "dg": {"method": "dann"}}
    model = build_model(cfg)
    assert model.domain_head is not None
    assert model.film is not None
    # The default config must stay the plain baseline.
    plain = build_model({"model": {"backbone": "resnet18", "pretrained": False}})
    assert plain.domain_head is None and plain.film is None


def test_unknown_dg_method_is_rejected_with_the_valid_options():
    torch = pytest.importorskip("torch")
    from tbtrust.train.loop import _dg_loss

    out = {"logit": torch.zeros(4), "features": torch.zeros(4, 3)}
    with pytest.raises(ValueError, match="coral"):
        _dg_loss(None, out, torch.zeros(4), torch.zeros(4, dtype=torch.long),
                 {"dg": {"method": "nonsense"}}, 0, 10)


def test_dg_loss_is_zero_when_disabled_or_single_clinic():
    torch = pytest.importorskip("torch")
    from tbtrust.train.loop import _dg_loss

    out = {"logit": torch.zeros(4), "features": torch.randn(4, 3)}
    y = torch.zeros(4)
    one_clinic = torch.zeros(4, dtype=torch.long)
    two_clinics = torch.tensor([0, 0, 1, 1])

    loss, _ = _dg_loss(None, out, y, two_clinics, {"dg": {"method": "none"}}, 0, 10)
    assert loss.item() == 0.0
    # CORAL on a batch that happens to contain one clinic has nothing to align.
    loss, _ = _dg_loss(None, out, y, one_clinic, {"dg": {"method": "coral"}}, 0, 10)
    assert loss.item() == 0.0


# --- degradation vs. uncertainty -------------------------------------------


def test_uncertainty_rising_with_severity_gives_positive_rho():
    rng = np.random.default_rng(0)
    severity = rng.uniform(0, 1, 200)
    honest = severity + rng.normal(0, 0.05, 200)
    blind = rng.uniform(0, 1, 200)

    good = DU.uncertainty_vs_severity(severity, honest)
    bad = DU.uncertainty_vs_severity(severity, blind)
    assert good["spearman_rho"] > 0.9
    assert abs(bad["spearman_rho"]) < 0.3
    assert good["mean_uncertainty_most_degraded_third"] > good["mean_uncertainty_cleanest_third"]
    assert good["n"] == 200


def test_spearman_is_rank_based_not_linear_and_handles_ties():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    # Monotone but very non-linear: Spearman must still read exactly 1.
    assert DU.spearman_correlation(x, np.exp(10 * x)) == pytest.approx(1.0)
    assert DU.spearman_correlation(x, -x) == pytest.approx(-1.0)
    # A constant uncertainty has no ranking, so the correlation is undefined
    # rather than 0 -- reporting 0 would read as "measured, and it's uninformative".
    assert np.isnan(DU.spearman_correlation(x, np.ones(4)))
    assert np.isfinite(DU.spearman_correlation(np.array([1.0, 1.0, 2.0, 3.0]), x))
    with pytest.raises(ValueError):
        DU.spearman_correlation(x, x[:2])
