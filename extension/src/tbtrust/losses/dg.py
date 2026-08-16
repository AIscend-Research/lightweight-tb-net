"""Lightweight domain-generalization losses.

The LOCO evaluation measures a cross-site generalization gap; these are the
objectives that try to *close* it, by stopping the backbone from memorizing one
clinic's imaging signature. They are deliberately small reimplementations rather
than a dependency on a full benchmark suite, so the codebase stays edge-focused
and easy to read. DomainBed and DeepDG have the fuller variants worth reading.

DANN is not here because it is not a standalone loss: it is the domain head plus
the gradient-reversal layer, driven from the training loop. See
`tbtrust.models.grl` and `tbtrust.train.loop`.

Torch-only module: nothing in `eval/` imports it, so the torch-free metric code
stays torch-free.
"""

from __future__ import annotations

import torch


def coral_loss(features_by_clinic: list[torch.Tensor]) -> torch.Tensor:
    """Deep CORAL (Sun & Saenko, 2016): align feature mean and covariance across
    clinics. Returns the mean pairwise alignment penalty over all clinic pairs.

    features_by_clinic: list of (n_i, d) tensors, one per clinic present in the
    batch. Clinics with fewer than 2 samples are skipped (covariance needs 2+),
    which matters here because a shuffled batch will regularly contain a single
    image from the smallest clinic.
    """
    feats = [f for f in features_by_clinic if f.size(0) >= 2]
    if len(feats) < 2:
        return torch.zeros((), device=feats[0].device if feats else "cpu")

    def mean_cov(f):
        mu = f.mean(dim=0, keepdim=True)
        fc = f - mu
        cov = (fc.t() @ fc) / (f.size(0) - 1)
        return mu.squeeze(0), cov

    stats = [mean_cov(f) for f in feats]
    d = feats[0].size(1)
    total = 0.0
    pairs = 0
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            mu_i, cov_i = stats[i]
            mu_j, cov_j = stats[j]
            mean_term = torch.sum((mu_i - mu_j) ** 2)
            cov_term = torch.sum((cov_i - cov_j) ** 2) / (4 * d * d)
            total = total + mean_term + cov_term
            pairs += 1
    return total / max(pairs, 1)


def irm_penalty(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """IRM penalty (Arjovsky et al., 2019): squared gradient of the per-clinic
    loss w.r.t. a dummy scale of 1.0. Call once per clinic and average.

    `logits` is the single TB logit per image, `targets` the float 0/1 label, to
    match this repo's binary head (the original formulation uses cross-entropy
    over classes; binary cross-entropy is the same idea on one logit).
    """
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits * scale, targets)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)
