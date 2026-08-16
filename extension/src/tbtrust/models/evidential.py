"""Evidential deep learning (Phase 3 uncertainty comparison, featured method).

Sensoy, Kaplan & Kandemir, 2018, "Evidential Deep Learning to Quantify
Classification Uncertainty" (NeurIPS). One forward pass, one network -- no
sampling (MC-dropout) and no N-model cost (deep ensembles) -- which is why
`docs/phase1_framing.md` picks this as the featured calibration-focused head
alongside cheap MC-dropout: it fits the low-compute constraint.

The idea: instead of predicting a class probability directly, the network
predicts non-negative *evidence* e = (e0, e1) for "normal" / "TB", read as
observed counts supporting a Dirichlet distribution over the class probability,
alpha = e + 1 (the +1 is the uniform Dirichlet prior -- zero evidence gives
alpha=(1,1), the uninformative prior, not a 50/50 point estimate). From that
Dirichlet:

    S = alpha0 + alpha1          "total evidence" (higher = more confident)
    p_TB = alpha1 / S            the point prediction
    vacuity u = K / S  (K=2)     epistemic uncertainty: no evidence -> u=1

Trained with the type-II maximum-likelihood Bayes-risk loss plus a KL penalty
that shrinks evidence for the *wrong* class specifically (so the network isn't
punished for having evidence for the right class), annealed in over training so
early gradients aren't dominated by the regularizer before the network can
predict anything at all.

Known failure mode worth checking empirically (this is itself a Phase 4 result,
not just a caveat): EDL can still look falsely confident on inputs that are
out-of-distribution but not adversarially far from the training manifold --
compare its vacuity against MC-dropout's std on the held-out clinics rather than
trusting either uncritically.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .baseline import _make_backbone

K_CLASSES = 2  # normal, TB


def evidence_to_alpha(evidence: torch.Tensor) -> torch.Tensor:
    return evidence + 1.0


def evidential_prob(evidence: torch.Tensor) -> torch.Tensor:
    """P(TB) = alpha_TB / S. `evidence` is (B, 2) = (e_normal, e_TB)."""
    alpha = evidence_to_alpha(evidence)
    return alpha[:, 1] / alpha.sum(dim=1)


def evidential_logit(evidence: torch.Tensor) -> torch.Tensor:
    """log(alpha_TB / alpha_normal), so sigmoid(this) == evidential_prob(evidence) exactly.

    Lets EvidentialClassifier's "logit" output be consumed by every existing
    eval/deferral/calibration function unchanged -- they all do
    `torch.sigmoid(out["logit"])` to get P(TB), and get the right number back.
    """
    alpha = evidence_to_alpha(evidence)
    return torch.log(alpha[:, 1]) - torch.log(alpha[:, 0])


def evidential_vacuity(evidence: torch.Tensor) -> torch.Tensor:
    """u = K / S in [0, 1]. Zero evidence (e=(0,0), S=K) -> u=1, maximally uncertain."""
    alpha = evidence_to_alpha(evidence)
    return K_CLASSES / alpha.sum(dim=1)


def _dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL(Dir(alpha) || Dir(1,...,1)), closed form. `alpha` is (B, K)."""
    K = alpha.shape[1]
    S = alpha.sum(dim=1, keepdim=True)
    term1 = torch.lgamma(S.squeeze(1)) - torch.lgamma(alpha).sum(dim=1) - torch.lgamma(torch.tensor(float(K)))
    term2 = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=1)
    return term1 + term2


def evidential_loss(
    evidence: torch.Tensor,
    labels: torch.Tensor,
    epoch: int = 0,
    annealing_epochs: int = 10,
    kl_weight: float = 1.0,
) -> torch.Tensor:
    """Bayes-risk cross-entropy under the Dirichlet + annealed KL-to-uniform penalty.

    `labels` is (B,) with values in {0, 1} (1 = TB), matching the rest of the
    codebase's convention. `evidence` is (B, 2) non-negative.
    """
    y = torch.stack([1.0 - labels.float(), labels.float()], dim=1)  # (B, 2) one-hot
    alpha = evidence_to_alpha(evidence)
    S = alpha.sum(dim=1, keepdim=True)

    # Bayes risk of the expected cross-entropy under Dir(alpha) -- the digamma
    # trick avoids sampling the Dirichlet to estimate the expectation.
    err = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)

    # Only penalize evidence for the *wrong* class: alpha_tilde=1 (no penalty)
    # on the true class, alpha_tilde=alpha (penalized) elsewhere.
    alpha_tilde = y + (1.0 - y) * alpha
    kl = _dirichlet_kl_to_uniform(alpha_tilde)

    lam = min(1.0, epoch / max(annealing_epochs, 1))
    return (err + lam * kl_weight * kl).mean()


class EvidentialHead(nn.Module):
    """features -> non-negative evidence (B, 2), via softplus (smoother than ReLU near 0,
    which matters here since alpha=evidence+1 and small gradients near zero evidence
    still need to flow)."""

    def __init__(self, in_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, K_CLASSES))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(feat))


class EvidentialClassifier(nn.Module):
    """Drop-in replacement for TBClassifier/TBNet: same backbone choices, same
    output-dict shape (`logit`, `uncertainty`), plus `evidence` so train/loop.py
    can detect this model and switch to `evidential_loss` instead of BCE (see
    `train/loop.py`, which dispatches on `"evidence" in out`).
    """

    def __init__(self, backbone: str = "densenet121", pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone, feat = _make_backbone(backbone, pretrained)
        self.dropout = nn.Dropout(dropout)
        self.head = EvidentialHead(feat)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.dropout(self.backbone(x))
        evidence = self.head(feat)
        return {
            "logit": evidential_logit(evidence),
            "uncertainty": evidential_vacuity(evidence),
            "evidence": evidence,
        }


def build_evidential_model(cfg: dict) -> EvidentialClassifier:
    m = cfg.get("model", {})
    return EvidentialClassifier(
        backbone=m.get("backbone", "densenet121"),
        pretrained=m.get("pretrained", True),
        dropout=m.get("dropout", 0.3),
    )
