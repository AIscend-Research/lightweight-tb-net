"""Smartphone capture as a communication channel, with the capacity actually computed.

Phase 1 of the project reached for an information-theory framing: treat the phone
capture as a noisy channel carrying the diagnostic signal, and let the
signal-to-noise ratio say when deferral is necessary. As written there it was a
metaphor -- there was no measured signal, no measured noise and therefore no
capacity, just a suggestive analogy.

Once the veil, the PSF and the tone curve are measured, the metaphor becomes
arithmetic. The signal is the density contrast of a finding. The noise is the
differential density noise `floor.py` derives. The bandwidth limit is the measured
MTF. So the Gaussian-channel capacity of one resolution cell is

    C  =  0.5 * log2( 1 + (dD * sqrt(E) / sigma_D)^2 )   bits

with E the blur-surviving template energy, and the whole lung field carries
`C * (number of independent cells)` bits about that finding class.

Why bother, when the certificate already answers the clinical question
----------------------------------------------------------------------
Three reasons, in increasing order of usefulness.

* It puts a *scale* on degradation that a pass/fail verdict cannot. Two images
  can both be INSUFFICIENT while one is 2 dB short and the other 20; bits
  separate them, which matters when you are deciding what to spend on a retake.
* It is additive, so the cost of each degradation can be reported as bits
  destroyed: `bits_lost` ablates the veil, the blur and the quantiser one at a
  time and reports what each one took. That is the honest version of the
  degradation ablation the project already runs on accuracy, and it needs no
  model and no labels.
* It gives an upper bound on *any* decision rule. If the channel carries 0.3 bits
  about whether a miliary pattern is present, no classifier -- yours, TB-Net's, or
  a radiologist's -- can do better than that allows. That is a much stronger
  statement than an accuracy drop on a test set.

The usual caveat applies: this is capacity for the *measurement* channel, given a
known signal template. It says what the photograph carries, not how hard the
diagnosis is.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .findings import FindingSpec, core
from .floor import FloorSpec, density_floor, template_energy
from .invert import CalibratedFilm


@dataclass
class ChannelReport:
    """Capacity of one photograph for one finding class."""

    finding: str
    snr_median: float
    snr_db_median: float
    bits_per_cell: float
    n_cells: float
    total_bits: float
    bits_lost: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "finding": self.finding,
            "snr_db": self.snr_db_median,
            "bits_per_cell": self.bits_per_cell,
            "n_cells": self.n_cells,
            "total_bits": self.total_bits,
            **{f"bits_lost_{k}": v for k, v in self.bits_lost.items()},
        }


def _capacity_bits(snr: np.ndarray) -> np.ndarray:
    """Shannon capacity of a real Gaussian channel: 0.5 log2(1 + SNR^2)."""
    return 0.5 * np.log2(1.0 + np.maximum(np.asarray(snr, dtype=np.float64), 0.0) ** 2)


def channel_report(
    cal: CalibratedFilm,
    finding: FindingSpec,
    mask: np.ndarray | None = None,
    spec: FloorSpec | None = None,
) -> ChannelReport:
    """Capacity, and what each degradation took, for one finding on one photo.

    The cell count is the analysed area divided by the finding's own footprint:
    independent trials of "is there one here", which is the right granularity for
    a detection channel. Counting pixels instead would inflate the total by the
    correlation length and mean nothing.
    """
    spec = spec or FloorSpec()
    m = cal.lung_field_mask() if mask is None else np.asarray(mask, dtype=bool)
    fm = density_floor(cal, finding, spec)
    if not m.any():
        return ChannelReport(finding.key, 0.0, -np.inf, 0.0, 0.0, 0.0)

    # SNR at the Rose-free scale: floor = k * sigma / sqrt(E), so
    # dD * sqrt(E) / sigma = k * dD / floor.
    snr = fm.rose_k * finding.delta_d / np.maximum(fm.floor[m], 1e-12)
    bits = _capacity_bits(snr)

    size_px = max(finding.size_px(fm.px_per_mm), 1.0)
    cell_area = np.pi * (size_px / 2.0) ** 2
    n_cells = float(m.sum()) / max(cell_area, 1.0)

    rep = ChannelReport(
        finding=finding.key,
        snr_median=float(np.median(snr)),
        snr_db_median=float(20.0 * np.log10(max(np.median(snr), 1e-12))),
        bits_per_cell=float(np.median(bits)),
        n_cells=n_cells,
        total_bits=float(np.median(bits) * n_cells),
        bits_lost=bits_lost(cal, finding, m, spec),
    )
    return rep


def bits_lost(
    cal: CalibratedFilm,
    finding: FindingSpec,
    mask: np.ndarray,
    spec: FloorSpec | None = None,
) -> dict[str, float]:
    """Bits per cell destroyed by each degradation, by counterfactual removal.

    Each entry answers: how many more bits per cell would this photograph carry if
    this one impairment were absent and everything else identical? Computed by
    rebuilding the floor with that term removed, which is exact rather than a
    linearisation, and is why these can legitimately be compared against each
    other even though the channel is nonlinear in each of them.

    They do not sum to the total loss, and should not be presented as if they did:
    the terms interact (blur and veil both reduce SNR multiplicatively). Report
    them as attributions, not as a decomposition.
    """
    spec = spec or FloorSpec()
    fm = density_floor(cal, finding, spec)
    base_snr = fm.rose_k * finding.delta_d / np.maximum(fm.floor[mask], 1e-12)
    base = float(np.median(_capacity_bits(base_snr)))
    out: dict[str, float] = {}

    def _gain(better_floor: np.ndarray) -> float:
        snr = fm.rose_k * finding.delta_d / np.maximum(better_floor[mask], 1e-12)
        return float(np.median(_capacity_bits(snr)) - base)

    # veil: the floor without the contrast compression the veil imposes
    amp = np.where(cal.signal > 0, (cal.signal + cal.veil) / np.maximum(cal.signal, 1e-9), 1.0)
    out["veil"] = _gain(fm.floor / np.maximum(np.nan_to_num(amp, nan=1.0, posinf=1.0), 1e-9))

    # blur: the floor with a perfect lens, i.e. the unblurred template energy
    out["blur"] = _gain(fm.floor * np.sqrt(fm.template_energy / max(fm.template_energy_unblurred, 1e-12)))

    # quantisation: the floor with the quantiser's contribution removed in quadrature
    q = np.asarray(fm.terms.get("quantization", np.zeros_like(fm.floor)))
    out["quantization"] = _gain(np.sqrt(np.maximum(fm.floor**2 - q**2, 1e-24)))

    # sensor noise, likewise
    s = np.asarray(fm.terms.get("sensor_noise", np.zeros_like(fm.floor)))
    out["sensor_noise"] = _gain(np.sqrt(np.maximum(fm.floor**2 - s**2, 1e-24)))

    return {k: float(v) for k, v in out.items()}


def capacity_table(
    cal: CalibratedFilm,
    findings: list[FindingSpec] | None = None,
    mask: np.ndarray | None = None,
    spec: FloorSpec | None = None,
) -> list[dict]:
    """One channel report per finding, flattened for a DataFrame."""
    findings = findings or core()
    return [channel_report(cal, f, mask, spec).as_dict() for f in findings]


def reference_capacity(
    cal: CalibratedFilm,
    finding: FindingSpec,
    mask: np.ndarray | None = None,
    spec: FloorSpec | None = None,
) -> dict:
    """Capacity of this capture against an idealised one of the same film.

    The reference is the same photograph with no veil, no blur and a 12-bit
    quantiser -- i.e. what a careful capture with decent equipment would have
    preserved. The ratio is the *fraction of the film's information the phone
    delivered*, which is the single number that best summarises what this whole
    track measures, and the one to put on a slide.
    """
    spec = spec or FloorSpec()
    m = cal.lung_field_mask() if mask is None else np.asarray(mask, dtype=bool)
    if not m.any():
        return {"delivered_fraction": float("nan")}

    fm = density_floor(cal, finding, spec)
    actual = float(np.median(_capacity_bits(fm.rose_k * finding.delta_d / np.maximum(fm.floor[m], 1e-12))))

    ideal_cal = replace(cal, veil=np.zeros_like(cal.veil), signal=cal.luminance,
                        quantization_step=1.0 / 4095.0)
    e_blur, e_open = template_energy(finding, fm.px_per_mm, cal.psf.mtf_at, spec)
    fm_ideal = density_floor(ideal_cal, finding, spec)
    ideal_floor = fm_ideal.floor * np.sqrt(e_blur / max(e_open, 1e-12))
    ideal = float(np.median(_capacity_bits(fm.rose_k * finding.delta_d / np.maximum(ideal_floor[m], 1e-12))))

    return {
        "finding": finding.key,
        "bits_actual": actual,
        "bits_ideal": ideal,
        "delivered_fraction": float(actual / ideal) if ideal > 0 else float("nan"),
    }
