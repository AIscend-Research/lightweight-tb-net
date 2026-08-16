"""Physics-gated deferral: the certificate and the learned signal, on one axis.

`eval/deferral.py` already compares uncertainty methods head to head by holding
the calibrated probabilities fixed and swapping the score the policy ranks on.
The physics certificate slots straight into that hook, which makes the comparison
the paper needs almost free:

    confidence     max(p, 1-p)                       the baseline
    mc_dropout     predictive spread                 learned, epistemic
    head           the learned uncertainty head      learned, trained on severity
    ensemble       member disagreement               learned, epistemic
    physics        certificate margin                measured, no labels, no network

The interesting result is not that one wins. It is that the physics signal is
*orthogonal* to all the learned ones, because it answers a different question.
The learned scores rank cases by how likely the classifier is to be wrong. The
certificate ranks them by whether the photograph could carry the finding at all.
A case can be easy for the classifier and uninformative in the photo -- that is a
confident wrong answer waiting to happen, and only the physics sees it coming.

Hence the three policies compared here:

* **learned only** -- rank on the learned score, defer the tail.
* **physics only** -- rank on the certificate margin.
* **gated** -- defer every image the certificate calls INSUFFICIENT first, then
  rank whatever survives by the learned score. This is the one to deploy, and the
  one that matches the clinical story: first ask whether the image is usable, then
  ask whether the finding is ambiguous.

`complementarity` measures whether the orthogonality claim actually holds on your
data rather than assuming it, and `triage_value` reports the split of the deferred
set into retakes and referrals -- the number that decides whether a clinic can
live with the policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import deferral as D
from .metrics import accuracy, sensitivity, specificity


@dataclass
class PolicyResult:
    """One deferral policy, evaluated at its operating point plus its AURC."""

    name: str
    aurc: float
    coverage: float
    accuracy: float
    sensitivity: float
    specificity: float
    n_deferred: int
    deferred_wrong_frac: float          # of what we deferred, how much was actually wrong
    kept_wrong_frac: float              # of what we answered, how much we got wrong
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "extra"} | self.extra


def _at_operating_point(y, p, keep: np.ndarray, name: str, aurc: float, extra: dict | None = None) -> PolicyResult:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= 0.5).astype(int)
    n = len(y)
    n_def = int((~keep).sum())
    def_wrong = float((pred[~keep] != y[~keep]).mean()) if n_def else float("nan")
    if not keep.any():
        return PolicyResult(name, aurc, 0.0, float("nan"), float("nan"), float("nan"),
                            n_def, def_wrong, float("nan"), extra or {})
    return PolicyResult(
        name=name,
        aurc=aurc,
        coverage=float(keep.sum() / n),
        accuracy=accuracy(y[keep], p[keep]),
        sensitivity=sensitivity(y[keep], p[keep]),
        specificity=specificity(y[keep], p[keep]),
        n_deferred=n_def,
        deferred_wrong_frac=def_wrong,
        kept_wrong_frac=float((pred[keep] != y[keep]).mean()),
        extra=extra or {},
    )


def physics_confidence(margins_db, abstained=None, scale_db: float = 12.0) -> np.ndarray:
    """Certificate margins -> a confidence in [0, 1] the deferral curve can rank on.

    Vectorised counterpart of `physics.certificate.certificate_confidence`, kept
    here so the eval path does not need a `Certificate` object per row -- the
    certificates are computed once by `scripts/physics_certificates.py` and joined
    to the manifest as plain columns.

    Abstentions map to 0.0, not to the middle. An image whose glare could not be
    measured is the last one to trust, and scoring "unmeasured" as "average" is
    how a safety valve quietly stops working on exactly the images that broke it.
    """
    m = np.asarray(margins_db, dtype=float)
    conf = 1.0 / (1.0 + np.exp(-m / max(scale_db, 1e-6)))
    bad = ~np.isfinite(m)
    if abstained is not None:
        bad = bad | np.asarray(abstained, dtype=bool)
    return np.where(bad, 0.0, conf)


def compare_policies(
    labels,
    probs,
    margins_db,
    learned_confidence=None,
    abstained=None,
    insufficient=None,
    threshold: float | None = None,
    margin_gate_db: float = 0.0,
    scale_db: float = 12.0,
) -> list[PolicyResult]:
    """Learned-only vs physics-only vs physics-gated, on one fixed set of probabilities.

    Parameters
    ----------
    threshold
        Confidence cutoff for the learned policy, **tuned on validation** and passed
        in here. Re-tuning it on the data being reported would inflate every number
        in the table; `eval/run.py` enforces the same discipline for the other
        uncertainty methods and this follows it.
    insufficient
        Boolean per image from the certificate. Defaults to margin <= `margin_gate_db`,
        which is the same rule `certificate.certify` applies.
    """
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs, dtype=float)
    n = len(y)

    learned = D._confidence(p, learned_confidence)
    phys = physics_confidence(margins_db, abstained, scale_db)
    if insufficient is None:
        m = np.asarray(margins_db, dtype=float)
        insufficient = ~np.isfinite(m) | (m <= margin_gate_db)
        if abstained is not None:
            insufficient = insufficient | np.asarray(abstained, dtype=bool)
    insufficient = np.asarray(insufficient, dtype=bool)

    t = float(threshold) if threshold is not None else float(np.quantile(learned, 0.2))

    out = [
        _at_operating_point(
            y, p, learned >= t, "learned",
            D.area_under_risk_coverage(y, p, confidence=learned),
            {"threshold": t},
        ),
        _at_operating_point(
            y, p, ~insufficient, "physics",
            D.area_under_risk_coverage(y, p, confidence=phys),
            {"gate_db": margin_gate_db},
        ),
    ]

    # Gated: the certificate vetoes first, the learned score ranks the remainder.
    # Implemented as a *product* of confidences rather than a two-stage filter so
    # that a single ranking exists and AURC stays well defined -- a hard veto has
    # no risk-coverage curve of its own, because it fixes the coverage.
    gated_conf = np.where(insufficient, 0.0, learned)
    out.append(
        _at_operating_point(
            y, p, (~insufficient) & (learned >= t), "physics_gated_learned",
            D.area_under_risk_coverage(y, p, confidence=gated_conf),
            {"threshold": t, "vetoed_by_physics": int(insufficient.sum()),
             "vetoed_frac": float(insufficient.mean()) if n else float("nan")},
        )
    )
    return out


def complementarity(labels, probs, margins_db, learned_confidence=None,
                    abstained=None, quantile: float = 0.2) -> dict:
    """Do the physics and the learned signal flag the *same* images?

    Takes the worst `quantile` of each signal and reports the overlap, plus how
    many errors each catches that the other misses. This is the evidence for or
    against the orthogonality claim in the module docstring, and it is worth
    reporting even when the physics policy loses on AURC: a signal that catches a
    small set of errors nothing else catches is still worth having in a screening
    system, because those are the confident-and-wrong cases.
    """
    y = np.asarray(labels).astype(int)
    p = np.asarray(probs, dtype=float)
    pred = (p >= 0.5).astype(int)
    wrong = pred != y

    learned = D._confidence(p, learned_confidence)
    phys = physics_confidence(margins_db, abstained)
    n = len(y)
    k = max(1, round(quantile * n))

    l_flag = np.zeros(n, dtype=bool)
    l_flag[np.argsort(learned)[:k]] = True
    p_flag = np.zeros(n, dtype=bool)
    p_flag[np.argsort(phys)[:k]] = True

    both = l_flag & p_flag
    union = l_flag | p_flag
    valid = np.isfinite(learned) & np.isfinite(phys)
    corr = (
        float(np.corrcoef(learned[valid], phys[valid])[0, 1])
        if valid.sum() > 2 and np.std(learned[valid]) > 0 and np.std(phys[valid]) > 0
        else float("nan")
    )
    return {
        "n": n,
        "flagged_each": k,
        "jaccard": float(both.sum() / max(union.sum(), 1)),
        "pearson_r": corr,
        "errors_total": int(wrong.sum()),
        "errors_caught_by_learned": int((wrong & l_flag).sum()),
        "errors_caught_by_physics": int((wrong & p_flag).sum()),
        "errors_only_physics": int((wrong & p_flag & ~l_flag).sum()),
        "errors_only_learned": int((wrong & l_flag & ~p_flag).sum()),
        "errors_caught_by_union": int((wrong & union).sum()),
    }


def triage_value(actions, labels, probs) -> dict:
    """Split the deferred set into retakes and referrals and price each one.

    `actions` is a sequence of `physics.triage.Action` values (or their strings).
    The point of the table: a retake costs half a minute at the lightbox, a
    referral costs the patient a journey. A policy that defers 30% of images is
    tolerable if most of those are retakes and intolerable if most are referrals,
    and no scalar coverage number distinguishes the two.
    """
    a = np.asarray([getattr(x, "value", x) for x in actions], dtype=object)
    y = np.asarray(labels).astype(int)
    pred = (np.asarray(probs, dtype=float) >= 0.5).astype(int)
    wrong = pred != y
    n = len(y)
    out: dict = {"n": n}
    for act in ("report", "retake", "refer"):
        sel = a == act
        out[f"{act}_rate"] = float(sel.mean()) if n else float("nan")
        out[f"{act}_error_rate"] = float(wrong[sel].mean()) if sel.any() else float("nan")
        # TB+ cases sent down each path: the sensitivity-relevant view, and the one
        # a screening programme is actually judged on.
        out[f"{act}_tb_positive"] = int((sel & (y == 1)).sum())
    reported = a == "report"
    out["reported_missed_tb"] = int((reported & (y == 1) & (pred == 0)).sum())
    out["total_tb"] = int((y == 1).sum())
    return out


def severity_response(margins_db, severities) -> dict:
    """Does the certificate margin actually fall as capture quality falls?

    The physics analogue of `eval/degradation_uncertainty.py`, and the same
    premise check: the "retake the photo" message is only justified for a signal
    that responds to how bad the photo is. The certificate should pass this by
    construction -- it is computed from the capture -- so a weak correlation here
    means something is broken upstream, most likely the fiducial detector failing
    on the degraded images and silently sending them to ABSTAIN.
    """
    m = np.asarray(margins_db, dtype=float)
    s = np.asarray(severities, dtype=float)
    ok = np.isfinite(m) & np.isfinite(s)
    if ok.sum() < 3:
        return {"n": int(ok.sum()), "spearman": float("nan"), "pearson": float("nan")}

    def _rank(x):
        order = np.argsort(x)
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(len(x), dtype=float)
        return r

    mm, ss = m[ok], s[ok]
    pear = float(np.corrcoef(mm, ss)[0, 1]) if np.std(mm) > 0 and np.std(ss) > 0 else float("nan")
    rm, rs = _rank(mm), _rank(ss)
    spear = float(np.corrcoef(rm, rs)[0, 1]) if np.std(rm) > 0 and np.std(rs) > 0 else float("nan")
    return {
        "n": int(ok.sum()),
        "spearman": spear,
        "pearson": pear,
        "abstain_frac": float((~np.isfinite(m)).mean()),
        # Negative is the expected sign: worse capture -> smaller margin.
        "sign_as_expected": bool(np.isfinite(spear) and spear < 0),
    }
