"""Worst-case smartphone degradation (Phase 3 parallel track).

The severity sweep in `eval/run.py` measures *average-case* robustness: one
random draw of blur/glare/shadow/etc. per image per severity level. That's the
right number for "how good is a typical photo," but it says nothing about
whether some *particular* combination of artifacts (say, motion blur stacked
with glare right over the relevant lung field) is quietly much worse than
average -- and, more importantly for this project's actual claim, whether the
uncertainty head notices when that happens.

`data/degradation.py`'s ops are non-differentiable (PIL resize/rotate/JPEG
re-encode), so this is not a gradient-based adversarial attack -- it's a
black-box worst-of-N search: draw N random degradations at a fixed severity,
keep whichever maximizes the model's loss on that image. That is a real,
if weaker, adversary (an "N-query black-box worst-case", not an optimal one);
report it as that, not as a certified robustness bound. The evaluation this
enables is the point: compare average-case vs. worst-of-N both on accuracy
*and* on whether the uncertainty signal rises specifically on the images where
the worst-case search actually found something worse, rather than rising
uniformly (which would mean it isn't tracking per-image risk at all).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.dataset import IMAGENET_MEAN, IMAGENET_STD
from ..data.degradation import SmartphoneDegradation


@dataclass
class WorstCaseResult:
    label: int
    avg_loss: float                # mean loss over the N random draws
    avg_uncertainty: float | None  # mean predicted uncertainty over the N draws
    worst_loss: float              # loss of the single worst-of-N draw
    worst_uncertainty: float | None
    worst_ops: dict = field(default_factory=dict)  # which degradation ops fired, and how strong


def _preprocess(arr: np.ndarray, image_size: int):
    import torch
    from PIL import Image

    img = Image.fromarray(arr).resize((image_size, image_size), Image.BILINEAR)
    x = np.asarray(img).astype(np.float32) / 255.0
    x = np.stack([x, x, x], axis=0)
    t = torch.from_numpy(x)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def worst_case_degradation_search(
    model,
    image: np.ndarray,
    label: int,
    severity: float = 0.7,
    n_trials: int = 20,
    image_size: int = 224,
    device: str = "cpu",
    seed: int = 0,
) -> WorstCaseResult:
    """N random degradation draws at fixed severity; report the average and the worst."""
    import torch
    import torch.nn.functional as F

    model.eval()
    losses, uncertainties, records = [], [], []
    for i in range(n_trials):
        degraded, record = SmartphoneDegradation(severity=severity, seed=seed + i)(image)
        x = _preprocess(degraded, image_size).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            loss = F.binary_cross_entropy_with_logits(out["logit"], torch.tensor([float(label)], device=device))
        losses.append(loss.item())
        uncertainties.append(out["uncertainty"].item() if "uncertainty" in out else None)
        records.append(record.ops)

    worst_idx = int(np.argmax(losses))
    have_u = uncertainties[0] is not None
    return WorstCaseResult(
        label=label,
        avg_loss=float(np.mean(losses)),
        avg_uncertainty=float(np.mean(uncertainties)) if have_u else None,
        worst_loss=losses[worst_idx],
        worst_uncertainty=uncertainties[worst_idx] if have_u else None,
        worst_ops=records[worst_idx],
    )


def evaluate_adversarial_robustness(
    model,
    manifest_test,  # type: pd.DataFrame -- avoid a hard pandas import at module load
    severity: float = 0.7,
    n_trials: int = 20,
    sample_n: int = 50,
    image_size: int = 224,
    device: str = "cpu",
    seed: int = 0,
) -> dict:
    """Aggregate worst-case-search results over a sample of the test set.

    Returns accuracy/loss under average- vs. worst-of-N-case degradation, mean
    uncertainty under each, and the Pearson correlation between per-image
    "how much worse did the search find" (worst_loss - avg_loss) and "how much
    did uncertainty rise" (worst_uncertainty - avg_uncertainty). A positive,
    reasonably strong correlation is the evidence the uncertainty head is
    tracking per-image risk rather than just average severity.
    """
    from PIL import Image

    rows = manifest_test.sample(n=min(sample_n, len(manifest_test)), random_state=seed)
    results = [
        worst_case_degradation_search(
            model,
            np.asarray(Image.open(row["path"]).convert("L")),
            int(row["label"]),
            severity=severity,
            n_trials=n_trials,
            image_size=image_size,
            device=device,
            seed=seed,
        )
        for _, row in rows.iterrows()
    ]

    avg_losses = np.array([r.avg_loss for r in results])
    worst_losses = np.array([r.worst_loss for r in results])
    report = {
        "n_images": len(results),
        "severity": severity,
        "n_trials": n_trials,
        "avg_case_mean_loss": float(avg_losses.mean()),
        "worst_case_mean_loss": float(worst_losses.mean()),
        "loss_increase_from_search": float((worst_losses - avg_losses).mean()),
    }

    if results[0].avg_uncertainty is not None:
        avg_u = np.array([r.avg_uncertainty for r in results])
        worst_u = np.array([r.worst_uncertainty for r in results])
        report["avg_case_mean_uncertainty"] = float(avg_u.mean())
        report["worst_case_mean_uncertainty"] = float(worst_u.mean())
        report["uncertainty_increase_from_search"] = float((worst_u - avg_u).mean())

        loss_delta = worst_losses - avg_losses
        u_delta = worst_u - avg_u
        if loss_delta.std() > 0 and u_delta.std() > 0:
            report["loss_vs_uncertainty_delta_correlation"] = float(np.corrcoef(loss_delta, u_delta)[0, 1])
        else:
            report["loss_vs_uncertainty_delta_correlation"] = float("nan")
        report["interpretation"] = (
            "positive correlation => uncertainty rises specifically on the images "
            "the worst-case search actually made harder, not just on average "
            "severity; near-zero or negative => the uncertainty head isn't "
            "tracking per-image worst-case risk and the deferral policy can't "
            "catch these cases."
        )
    else:
        report["note"] = "model has no 'uncertainty' output; ran the loss-only worst-case comparison."

    return report
