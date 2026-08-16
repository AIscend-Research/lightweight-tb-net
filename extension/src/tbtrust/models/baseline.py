"""Baseline TB classifier: a standard backbone + a dropout head.

Why a backbone and not TB-Net out of the box? TB-Net's published code is
TensorFlow 1.15 / checkpoint format (see models/tbnet.py). To let the whole team
run end-to-end on day one, the default runnable model is a torchvision/timm CNN
(DenseNet-121 is the CheXNet-style default for chest X-rays). Reproducing TB-Net's
attention-condenser architecture is a first-class task, tracked in tbnet.py, and
drops in behind the same interface.

The head keeps dropout *active at inference* when asked, which is what makes
MC-dropout uncertainty possible (see models/uncertainty.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.manifest import NUM_CLINIC_SLOTS
from .clinic_film import ClinicFiLM
from .grl import grad_reverse


def _make_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Return (feature_extractor, feature_dim). Tries timm, falls back to torchvision."""
    try:
        import timm

        model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
        return model, model.num_features
    except Exception:
        from torchvision import models as tvm

        if name.startswith("densenet"):
            net = tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT if pretrained else None)
            dim = net.classifier.in_features
            net.classifier = nn.Identity()
            return net, dim
        net = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None)
        dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, dim


class TBClassifier(nn.Module):
    """Backbone -> dropout -> logit(s). Optionally a second head for uncertainty,
    a domain-adversarial head (DANN), and clinic-conditional FiLM modulation.

    Outputs a dict:
        logit       : (B,) raw score for the TB class
        features    : (B, D) pooled backbone features, pre-dropout. The
                      domain-generalization losses in losses/dg.py operate on
                      these, so the training loop needs them out of one forward
                      pass rather than a second one.
        uncertainty : (B,) optional predicted 'appropriate uncertainty' in [0,1]
                      (only if with_uncertainty_head=True)

    `clinic_idx` is optional everywhere: the whole eval path calls `model(x)`
    with no clinic label, which is correct under LOCO -- the held-out clinic has
    no learned embedding, and FiLM falls back to the mean training clinic.
    """

    def __init__(
        self,
        backbone: str = "densenet121",
        pretrained: bool = True,
        dropout: float = 0.3,
        with_uncertainty_head: bool = True,
        num_clinics: int = NUM_CLINIC_SLOTS,
        with_domain_head: bool = False,
        with_clinic_film: bool = False,
    ):
        super().__init__()
        self.backbone, feat = _make_backbone(backbone, pretrained)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feat, 1)
        self.with_uncertainty_head = with_uncertainty_head
        if with_uncertainty_head:
            self.uncertainty_head = nn.Sequential(nn.Linear(feat, 64), nn.ReLU(), nn.Linear(64, 1))

        self.film = ClinicFiLM(feat, num_clinics) if with_clinic_film else None
        self.domain_head = None
        if with_domain_head:
            self.domain_head = nn.Sequential(
                nn.Linear(feat, feat // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(feat // 2, num_clinics),
            )

    def forward(self, x: torch.Tensor, clinic_idx: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        if self.film is not None:
            feat = self.film(feat, clinic_idx)
        dropped = self.dropout(feat)
        out = {"logit": self.classifier(dropped).squeeze(-1), "features": feat}
        if self.with_uncertainty_head:
            out["uncertainty"] = torch.sigmoid(self.uncertainty_head(dropped).squeeze(-1))
        return out

    def domain_logits(self, features: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        """Clinic prediction through the gradient-reversal layer (DANN)."""
        if self.domain_head is None:
            raise RuntimeError("Model was built without a domain head (dg.method is not 'dann').")
        return self.domain_head(grad_reverse(features, lambd))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x)["logit"])


def build_model(cfg: dict) -> TBClassifier:
    m = cfg.get("model", {})
    dg = str(cfg.get("dg", {}).get("method", "none")).lower()
    return TBClassifier(
        backbone=m.get("backbone", "densenet121"),
        pretrained=m.get("pretrained", True),
        dropout=m.get("dropout", 0.3),
        with_uncertainty_head=m.get("with_uncertainty_head", True),
        num_clinics=m.get("num_clinics", NUM_CLINIC_SLOTS),
        with_domain_head=(dg == "dann"),
        with_clinic_film=bool(m.get("clinic_film", False)),
    )
