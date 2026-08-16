"""The certificate of insufficiency: a falsifiable, label-free verdict on a photo.

For each finding, compare the measured density resolution floor against the
finding's characteristic density contrast. Three outcomes:

    contrast comfortably above the floor  ->  DETECTABLE
    contrast straddling the floor         ->  MARGINAL
    contrast below the floor              ->  INSUFFICIENT

and a fourth, `ABSTAIN`, for when the image lacks the fiducials to measure
anything -- which the code must say out loud rather than papering over with a
prior.

What makes this different from an uncertainty estimate
------------------------------------------------------
An INSUFFICIENT verdict is not "the network is unsure". It is: *the photograph
does not contain the information*. No model, however good, and no amount of
training data, can recover a density step smaller than the channel can carry. The
statement is about the measurement, is computed with no labels and no network,
and is wrong in a checkable way -- `validate.py` puts lesions of known contrast
through the simulated channel and confirms that an optimal detector's empirical
threshold lands where the certificate said it would.

That is also its main defence against the standard critique of learned
uncertainty in this setting -- "you have trained a blur detector, not an
uncertainty model". A blur detector has no units. This has units of optical
density, and it can be wrong.

Reading the margin
------------------
The scalar to quote is the margin in dB:

    margin_db  =  20 * log10( delta_D_finding / floor )

positive means the finding's contrast clears the floor. Reported with an
uncertainty that propagates the spread of the finding's contrast, so a verdict
never looks sharper than the table behind it. The thresholds are on the margin,
not on a bare ratio, so a single number orders every image and every finding on a
common scale -- which is what `eval/physics_deferral.py` ranks on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .fiducials import Coverage
from .findings import FindingSpec, core
from .floor import FloorSpec, density_floor, limiting_factor
from .invert import CalibratedFilm, invertible

# Margin thresholds in dB. A finding needs to clear the floor by ~3 dB (a factor
# of 1.4) before we call it carried, and falls below zero to be called lost. The
# band between is where a retake is worth the patient's time.
MARGIN_DETECTABLE_DB = 3.0
MARGIN_INSUFFICIENT_DB = 0.0


class Verdict(str, Enum):
    DETECTABLE = "detectable"
    MARGINAL = "marginal"
    INSUFFICIENT = "insufficient"
    ABSTAIN = "abstain"


@dataclass
class FindingVerdict:
    """The certificate's line item for one finding."""

    finding: str
    finding_name: str
    verdict: Verdict
    delta_d: float
    delta_d_sigma: float
    floor_median: float
    floor_p90: float
    margin_db: float
    margin_db_sigma: float
    insufficient_area_fraction: float     # fraction of the analysed region where the floor wins
    limiting: str
    limiting_detail: dict = field(default_factory=dict)
    blur_penalty: float = 1.0
    contrast_source: str = "NOMINAL"

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "limiting_detail"}
        d["verdict"] = self.verdict.value
        d["limiting_share"] = self.limiting_detail.get("share", {})
        return d


@dataclass
class Certificate:
    """The whole verdict for one photograph."""

    verdict: Verdict
    findings: list[FindingVerdict]
    coverage: Coverage
    worst_finding: str | None
    limiting: str
    margin_db: float                      # the worst line item's margin: the scalar to rank on
    floor_map: np.ndarray | None = None   # per-pixel floor for the worst finding
    insufficient_mask: np.ndarray | None = None
    provenance: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        return self.verdict is Verdict.ABSTAIN

    def line(self, key: str) -> FindingVerdict | None:
        return next((f for f in self.findings if f.finding == key), None)

    def as_dict(self) -> dict:
        """Flat, JSON-serialisable. One row per image in the certificate table."""
        row = {
            "certificate": self.verdict.value,
            "coverage": self.coverage.value,
            "worst_finding": self.worst_finding,
            "limiting_factor": self.limiting,
            "margin_db": self.margin_db,
            **{f"prov_{k}": v for k, v in self.provenance.items()},
        }
        for fv in self.findings:
            row[f"verdict_{fv.finding}"] = fv.verdict.value
            row[f"margin_db_{fv.finding}"] = fv.margin_db
            row[f"floor_{fv.finding}"] = fv.floor_median
            row[f"insuff_area_{fv.finding}"] = fv.insufficient_area_fraction
        return row

    def report(self) -> str:
        """Human-readable certificate. What a reviewer should be shown, not a dict."""
        lines = [
            f"CERTIFICATE: {self.verdict.value.upper()}   (fiducial coverage: {self.coverage.value})",
            f"  limiting factor : {self.limiting}",
            f"  worst margin    : {self.margin_db:+.1f} dB  ({self.worst_finding})",
            "",
            f"  {'finding':<22} {'ΔD':>7} {'floor':>8} {'margin':>10}  verdict",
            f"  {'-' * 22} {'-' * 7} {'-' * 8} {'-' * 10}  {'-' * 12}",
        ]
        lines.extend(
            f"  {fv.finding_name:<22} {fv.delta_d:7.3f} {fv.floor_median:8.3f} "
            f"{fv.margin_db:+7.1f}±{fv.margin_db_sigma:<4.1f} {fv.verdict.value}"
            for fv in self.findings
        )
        if self.provenance.get("contrast_source") == "NOMINAL":
            lines += ["", "  NOTE: finding contrasts are NOMINAL placeholders "
                      "(see physics/findings.py). Relative comparisons hold; absolute "
                      "verdicts inherit their uncertainty."]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #


def _margin_db(delta_d: float, floor: float) -> float:
    if not np.isfinite(floor) or floor <= 0:
        return float("-inf") if not np.isfinite(floor) else float("inf")
    return float(20.0 * np.log10(max(delta_d, 1e-12) / floor))


def certify(
    cal: CalibratedFilm,
    findings: list[FindingSpec] | None = None,
    mask: np.ndarray | None = None,
    spec: FloorSpec | None = None,
    detectable_db: float = MARGIN_DETECTABLE_DB,
    insufficient_db: float = MARGIN_INSUFFICIENT_DB,
) -> Certificate:
    """Compute the certificate for one inverted photo.

    `mask` defaults to the lung-field heuristic in `CalibratedFilm`. Pass a real
    segmentation if you have one; the per-pixel floor does not change, only which
    pixels are aggregated over.
    """
    findings = findings or core()
    spec = spec or FloorSpec()

    prov = {
        "coverage": cal.fiducials.coverage.value,
        "tone_method": cal.tone.method,
        "psf_method": cal.psf.method,
        "glare_method": cal.glare.method,
        "px_per_mm_measured": bool(cal.px_per_mm_rel_sigma == 0.0),
        "rose_k": spec.rose_k,
        "contrast_source": findings[0].source if findings else "unknown",
        "include_anatomical": spec.include_anatomical,
    }

    if not invertible(cal.fiducials):
        return Certificate(
            verdict=Verdict.ABSTAIN, findings=[], coverage=cal.fiducials.coverage,
            worst_finding=None, limiting="no_fiducials", margin_db=float("nan"),
            provenance=prov,
            diagnostics={"reason": "no optical beam stop in this image, so the veil is "
                                   "unmeasured and no bound can be stated"},
        )

    m = cal.lung_field_mask() if mask is None else np.asarray(mask, dtype=bool)
    if not m.any():
        return Certificate(
            verdict=Verdict.ABSTAIN, findings=[], coverage=cal.fiducials.coverage,
            worst_finding=None, limiting="no_analysable_region", margin_db=float("nan"),
            provenance=prov, diagnostics={"reason": "no analysable region after masking"},
        )

    lines: list[FindingVerdict] = []
    worst: tuple[float, FindingVerdict, np.ndarray] | None = None

    for f in findings:
        fm = density_floor(cal, f, spec)
        st = fm.stats(m)
        if not st.get("n_px"):
            continue
        floor_med = st["floor_median"]
        margin = _margin_db(f.delta_d, floor_med)
        # Propagate the contrast spread only. The floor's own uncertainty is
        # dominated by the systematic terms deliberately excluded from it (see
        # floor.py), so quoting a floor error bar here would double count them.
        rel = f.delta_d_sigma / max(f.delta_d, 1e-12)
        margin_sigma = float(20.0 / np.log(10.0) * rel)

        insuff = np.zeros(fm.floor.shape, dtype=bool)
        insuff[m] = fm.floor[m] > f.delta_d
        area = float(insuff[m].mean())

        if margin >= detectable_db:
            v = Verdict.DETECTABLE
        elif margin <= insufficient_db:
            v = Verdict.INSUFFICIENT
        else:
            v = Verdict.MARGINAL

        lim, detail = limiting_factor(fm, m)
        fv = FindingVerdict(
            finding=f.key, finding_name=f.name, verdict=v,
            delta_d=f.delta_d, delta_d_sigma=f.delta_d_sigma,
            floor_median=floor_med, floor_p90=st["floor_p90"],
            margin_db=margin, margin_db_sigma=margin_sigma,
            insufficient_area_fraction=area, limiting=lim, limiting_detail=detail,
            blur_penalty=fm.blur_penalty, contrast_source=f.source,
        )
        lines.append(fv)
        if worst is None or margin < worst[0]:
            worst = (margin, fv, fm.floor)

    if worst is None:
        return Certificate(
            verdict=Verdict.ABSTAIN, findings=[], coverage=cal.fiducials.coverage,
            worst_finding=None, limiting="no_findings_evaluated", margin_db=float("nan"),
            provenance=prov,
        )

    # The image's overall verdict is the worst line item. Screening is a
    # sensitivity problem: an image that can carry a lobar consolidation but not a
    # miliary pattern has still lost the finding you most need not to miss, and
    # averaging across findings would hide exactly that.
    _, wv, wfloor = worst
    overall = wv.verdict
    insuff_mask = np.zeros(wfloor.shape, dtype=bool)
    insuff_mask[m] = wfloor[m] > wv.delta_d

    return Certificate(
        verdict=overall,
        findings=lines,
        coverage=cal.fiducials.coverage,
        worst_finding=wv.finding,
        limiting=wv.limiting,
        margin_db=wv.margin_db,
        floor_map=wfloor,
        insufficient_mask=insuff_mask,
        provenance=prov,
        diagnostics={
            "n_findings": len(lines),
            "analysed_px": int(m.sum()),
            "analysed_fraction": float(m.mean()),
            "veil_amplification_median": float(
                np.median((cal.signal[m] + cal.veil[m]) / np.maximum(cal.signal[m], 1e-9))
            ),
        },
    )


def certify_photo(photo, findings: list[FindingSpec] | None = None,
                  **invert_kwargs) -> tuple[Certificate, CalibratedFilm]:
    """Convenience: photo in, certificate out. Returns the inversion too, since
    every caller that wants a certificate also wants the diagnostics behind it."""
    from .invert import invert

    cal = invert(photo, **invert_kwargs)
    return certify(cal, findings=findings), cal


def certificate_confidence(cert: Certificate, scale_db: float = 12.0) -> float:
    """Map a certificate onto [0, 1] so the deferral policy can rank on it.

    A logistic in the margin, so it is monotone in the physics and saturates
    rather than letting one spectacular image dominate a threshold. Abstentions
    get 0.0 -- an image whose glare could not be measured is the *last* one to
    trust, not a neutral case, and treating "unmeasured" as "average" is how a
    safety valve silently stops working.

    `eval/physics_deferral.py` feeds this straight into
    `deferral.risk_coverage_curve(confidence=...)`, which is what puts the physics
    bound and the learned uncertainty on the same axis and lets the paper compare
    them as ranking signals.
    """
    if cert.abstained or not np.isfinite(cert.margin_db):
        return 0.0
    return float(1.0 / (1.0 + np.exp(-cert.margin_db / max(scale_db, 1e-6))))
