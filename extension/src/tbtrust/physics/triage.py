"""Retake, refer, or report -- and the physics decides which.

Existing selective-classification work produces one number and one action: below
threshold, defer. In a rural clinic that is not enough, because "defer" collapses
two completely different situations:

* **The photograph is bad.** The film in front of the operator still holds the
  information; the capture threw it away. The right action is thirty seconds and
  another photo -- and, crucially, an instruction saying *what to change*.
* **The photograph is fine and the case is hard.** Retaking will produce an
  identical image and waste the visit. The right action is a referral.

Nothing in a learned uncertainty score distinguishes these. A confidence of 0.6
looks the same either way. The measured channel does distinguish them, because it
separates the two causes physically: the density floor is a property of the
capture, and the model's residual uncertainty given an adequate capture is a
property of the case.

    floor too high, cause is operator-fixable  ->  RETAKE (with the specific fix)
    floor too high, cause is not fixable       ->  REFER  (the film itself is poor)
    floor fine, model uncertain                ->  REFER  (genuine clinical ambiguity)
    floor fine, model confident                ->  REPORT
    no fiducials, so no bound available        ->  RETAKE, framed as a framing problem

The instruction is the deliverable. `glare.hotspot` already knows whether the veil
is a specular blob that moves when the phone moves or a diffuse wash that does
not, and `psf.PSFEstimate.anisotropy` already knows whether the blur is
directional shake or symmetric defocus. Those distinctions turn "the image is bad"
into "step to your left" or "hold still" or "shade the lightbox", which is the
difference between a retake that helps and one that reproduces the same photo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .certificate import Certificate, Verdict
from .glare import hotspot
from .invert import CalibratedFilm


class Action(str, Enum):
    REPORT = "report"
    RETAKE = "retake"
    REFER = "refer"


# Map the floor's limiting term onto the thing an operator could actually change.
# `veil` and `veil_fit` are both glare -- the first is the contrast compression the
# veil imposes, the second the uncertainty in measuring it, and that uncertainty is
# large precisely when there is a lot of veil. Routing `veil_fit` anywhere else
# tells an operator with a fixable reflection that nothing can be done.
FIXABLE_CATEGORY = {
    "veil": "glare",
    "veil_fit": "glare",
    "blur": "blur",
    "quantization": "exposure",
    "sensor_noise": "exposure",
}
OPERATOR_FIXABLE = set(FIXABLE_CATEGORY)


@dataclass
class TriageDecision:
    action: Action
    reason: str                     # short machine-readable cause
    instruction: str                # what to tell the operator or clinician, in words
    certificate_verdict: Verdict
    model_confident: bool | None = None
    expected_gain_db: float = 0.0   # margin a successful retake should recover
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "instruction": self.instruction,
            "certificate": self.certificate_verdict.value,
            "model_confident": self.model_confident,
            "expected_gain_db": self.expected_gain_db,
            **{f"ev_{k}": v for k, v in self.evidence.items()},
        }


def _blur_instruction(cal: CalibratedFilm) -> str:
    psf = cal.psf
    if psf.method == "prior":
        return ("Sharpness could not be measured because no collimation border is visible. "
                "Re-frame to include the whole film, including its unexposed margin, and retake.")
    if psf.motion_dominant:
        axis = "horizontally" if psf.sigma_x > psf.sigma_y else "vertically"
        return (f"Camera shake, smeared {axis} (blur is {psf.anisotropy:.1f}x worse along one axis). "
                "Brace the phone against something solid or rest your elbows on the lightbox, "
                "tap to focus, and retake.")
    return (f"Image is out of focus (equivalent blur {psf.sigma:.1f} pixels). "
            "Move about a hand's width further from the film, tap the screen on the lung fields "
            "to refocus, wait for the focus to settle, and retake.")


def _exposure_instruction(cal: CalibratedFilm, mask: np.ndarray) -> str:
    v = cal.pixel_values[mask] if mask.any() else cal.pixel_values.ravel()
    lo = float(np.quantile(v, 0.05))
    hi = float(np.quantile(v, 0.95))
    if hi < 0.45:
        return ("The lung fields are recorded too dark to carry density detail. Raise the "
                "lightbox brightness or increase the phone's exposure (tap and drag up on the "
                "lung fields), and retake.")
    if lo > 0.85:
        return ("The lung fields are blown out. Lower the phone's exposure (tap and drag down) "
                "or dim the lightbox so the darkest part of the film is not clipped, then retake.")
    return ("Too little tonal range survives in the lung fields. Fill more of the frame with the "
            "film, avoid digital zoom, and retake at the highest quality setting available.")


def triage(
    cert: Certificate,
    cal: CalibratedFilm,
    model_confidence: float | None = None,
    confidence_threshold: float = 0.5,
    mask: np.ndarray | None = None,
) -> TriageDecision:
    """Turn a certificate plus (optionally) a model confidence into an action.

    `model_confidence` is whatever the deferral policy already ranks on -- softmax
    confidence, the learned head, MC-dropout spread mapped through
    `deferral.confidence_from_uncertainty`. It is optional: the physics alone
    produces a sound RETAKE/REFER split, and the model only breaks the tie in the
    case where the capture is adequate.
    """
    m = cal.lung_field_mask() if mask is None else np.asarray(mask, dtype=bool)
    confident = None if model_confidence is None else bool(model_confidence >= confidence_threshold)
    hs = hotspot(cal.glare, cal.fiducials.field_mask)

    ev = {
        "margin_db": cert.margin_db,
        "limiting": cert.limiting,
        "coverage": cert.coverage.value,
        "psf_sigma_px": float(cal.psf.sigma),
        "psf_anisotropy": float(cal.psf.anisotropy),
        "glare_localized": bool(hs.localized),
        "glare_direction": hs.direction,
        "glare_affected_fraction": float(hs.affected_fraction),
        "model_confidence": model_confidence,
    }

    # --- no bound available at all -------------------------------------------
    if cert.abstained:
        return TriageDecision(
            action=Action.RETAKE,
            reason="no_fiducials",
            instruction=(
                "This photo does not show enough of the film to check whether it is good enough. "
                "Photograph the whole sheet: include the pale unexposed margin all the way round "
                "and the L or R lead marker, square on, filling the frame. Then retake."
            ),
            certificate_verdict=cert.verdict,
            model_confident=confident,
            evidence=ev,
        )

    # --- the capture destroyed the signal -------------------------------------
    if cert.verdict in (Verdict.INSUFFICIENT, Verdict.MARGINAL):
        line = cert.line(cert.worst_finding) if cert.worst_finding else None
        share = (line.limiting_detail.get("share", {}) if line else {})
        # How much margin a perfect retake would recover: remove the operator-fixable
        # terms and see where the floor lands. Reported so a clinic can decide
        # whether a retake is worth it -- a 1 dB gain is not.
        # Shares are attributions, not a partition -- the terms interact, so they
        # can sum past one. Clamp before turning it into a decibel figure, or a
        # single dominant term produces an absurd promised gain.
        removable = float(np.clip(sum(v for k, v in share.items() if k in OPERATOR_FIXABLE), 0.0, 0.95))
        gain_db = float(-20.0 * np.log10(max(1.0 - removable, 0.05)))
        ev["recoverable_share"] = removable

        category = FIXABLE_CATEGORY.get(cert.limiting)
        if category == "glare":
            return TriageDecision(
                Action.RETAKE, "veiling_glare", hs.advice, cert.verdict, confident, gain_db, ev
            )
        if category == "blur":
            return TriageDecision(
                Action.RETAKE, "capture_blur", _blur_instruction(cal), cert.verdict, confident, gain_db, ev
            )
        if category == "exposure":
            return TriageDecision(
                Action.RETAKE,
                "exposure_or_compression",
                _exposure_instruction(cal, m),
                cert.verdict, confident, gain_db, ev,
            )
        # Not fixable by the operator: the film itself is fogged, underexposed or
        # degraded. Another photo of the same sheet will look identical.
        return TriageDecision(
            Action.REFER,
            f"unfixable_{cert.limiting}",
            ("The film itself carries too little contrast for a reliable screening read, and "
             "another photograph will not change that. Refer for specialist review or repeat "
             "the radiograph."),
            cert.verdict, confident, 0.0, ev,
        )

    # --- capture is adequate; the question is now clinical ---------------------
    if confident is False:
        return TriageDecision(
            Action.REFER,
            "model_uncertain_adequate_capture",
            ("Image quality is adequate -- the photograph carries the density detail a screening "
             "read needs -- but the finding is ambiguous. A retake will not help. Refer for "
             "specialist review."),
            cert.verdict, confident, 0.0, ev,
        )

    return TriageDecision(
        Action.REPORT,
        "adequate_and_confident",
        "Image quality is adequate and the prediction is confident. Report the result.",
        cert.verdict, confident, 0.0, ev,
    )


def triage_summary(decisions: list[TriageDecision]) -> dict:
    """Aggregate a batch. The table the deployment section of the paper needs.

    `retake_rate` is the operationally load-bearing number: it is how much extra
    work the policy creates for a clinic, and a policy that flags 60% of images
    for retake will be switched off in a week however good its physics is.
    """
    if not decisions:
        return {"n": 0}
    n = len(decisions)
    by_action: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for d in decisions:
        by_action[d.action.value] = by_action.get(d.action.value, 0) + 1
        by_reason[d.reason] = by_reason.get(d.reason, 0) + 1
    retakes = [d for d in decisions if d.action is Action.RETAKE]
    return {
        "n": n,
        "report_rate": by_action.get("report", 0) / n,
        "retake_rate": by_action.get("retake", 0) / n,
        "refer_rate": by_action.get("refer", 0) / n,
        "mean_expected_gain_db": float(np.mean([d.expected_gain_db for d in retakes])) if retakes else 0.0,
        "by_action": by_action,
        "by_reason": by_reason,
    }
