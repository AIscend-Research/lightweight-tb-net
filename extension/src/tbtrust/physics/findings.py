"""The signal side of the bound: what density contrast a TB finding actually presents.

The resolution floor in `floor.py` says what the photograph *can* carry. That is
only half a certificate. The other half is what the finding *is*: a lesion has a
characteristic optical-density contrast against the lung field around it and a
characteristic size, and the certificate is the comparison of the two.

Read this section before quoting a number
-----------------------------------------
The values below are **nominal defaults**, chosen to be physically sensible for
screen-film chest radiography and internally consistent with `D_LUNG_RANGE`. They
are *not* measured, and they are not taken from a specific published table. Every
entry carries `source="NOMINAL"` for exactly that reason.

They are structured so that replacing them is a one-line change, and any
publication using this code must replace them, by either:

* transcribing contrast and size ranges from a radiographic physics reference or
  a contrast-detail study, and recording the citation in `source`; or
* measuring them directly -- expose a contrast-detail phantom (a CDRAD-style plate
  or, more cheaply, a step wedge with drilled cavities) on the same film/processor
  combination the clinic uses, and read the densities off with a densitometer.

`load_findings` reads a YAML or CSV table so the swap needs no code change at all.
**`findings_template.yaml`, beside this file, is that table with the values
blanked and the transcription recipe written out** -- what quantity to take, how
to convert a published *subject* contrast into optical density (multiply by the
film's average gradient; forgetting it costs a factor of two to three and does
not fail safe), the traps that make a transcription wrong, and the ordering and
magnitude checks to run afterwards. Start there rather than editing the dict
below, so the numbers arrive with their citations attached.

Until that happens, the certificate's *relative* statements -- this photo carries
less density resolution than that one, this glare hotspot costs you a factor of
three, the floor rose above the contrast when the veil passed 15% -- are sound,
because they depend only on the floor. The *absolute* verdicts (`DETECTABLE` vs
`INSUFFICIENT` for a named finding) inherit the uncertainty of these numbers, and
`FindingSpec.delta_d_sigma` is what carries it into the reported margin.

Sign convention: `delta_d` is a magnitude. A consolidation is more attenuating
than the lung it replaces, so it *lowers* density; a cavity's air-filled lumen
raises it. Detection depends on the magnitude of the step, so the table stores
magnitudes and `film.insert_lesion` owns the sign.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FindingSpec:
    """One radiographic finding, as a detectability target."""

    key: str
    name: str
    delta_d: float              # |density contrast| against adjacent lung, magnitude
    delta_d_sigma: float        # spread across presentations; enters the margin
    size_mm: float              # characteristic diameter or wall thickness
    size_sigma_mm: float
    note: str = ""
    source: str = "NOMINAL"

    def size_px(self, px_per_mm: float) -> float:
        return float(self.size_mm * px_per_mm)

    def char_frequency(self, px_per_mm: float) -> float:
        """Characteristic spatial frequency in cycles/pixel.

        A feature of width d contributes most of its energy near f = 1/(2d) --
        the fundamental of a bar pair at that width. Used only for the
        interpretable single-number MTF diagnostic; the floor itself integrates
        the whole template spectrum in `floor.template_energy`, which needs no
        such choice.
        """
        d = max(self.size_px(px_per_mm), 1e-6)
        return float(1.0 / (2.0 * d))

    def rescaled(self, delta_d: float | None = None, size_mm: float | None = None) -> FindingSpec:
        return replace(self,
                       delta_d=self.delta_d if delta_d is None else float(delta_d),
                       size_mm=self.size_mm if size_mm is None else float(size_mm))


# --------------------------------------------------------------------------- #
# nominal table -- READ THE MODULE DOCSTRING BEFORE PUBLISHING THESE
# --------------------------------------------------------------------------- #
TB_FINDINGS: dict[str, FindingSpec] = {
    f.key: f
    for f in (
        FindingSpec(
            "miliary_nodule", "Miliary nodule", 0.030, 0.012, 2.0, 0.8,
            note="The hardest target by a wide margin: small and low contrast at once. "
                 "If any finding is going to fall below the floor of a phone photo, it is this one.",
        ),
        FindingSpec(
            "small_nodule", "Small nodule", 0.060, 0.025, 6.0, 3.0,
            note="Individual granuloma or a small focus of disease.",
        ),
        FindingSpec(
            "infiltrate", "Patchy infiltrate", 0.090, 0.040, 15.0, 7.0,
            note="Ill-defined airspace opacity; the low-contrast edge is what makes it hard, "
                 "not the size.",
        ),
        FindingSpec(
            "cavity_wall", "Cavity wall", 0.220, 0.080, 4.0, 2.0,
            note="High contrast but thin. Blur-limited rather than contrast-limited, which is "
                 "why it behaves quite differently from an infiltrate under motion.",
        ),
        FindingSpec(
            "consolidation", "Lobar consolidation", 0.350, 0.120, 45.0, 20.0,
            note="Large and high contrast. Survives almost any capture; useful as the control "
                 "that shows the certificate is not simply failing everything.",
        ),
        FindingSpec(
            "pleural_effusion", "Pleural effusion", 0.400, 0.150, 30.0, 15.0,
            note="Blunted costophrenic angle. Peripheral, so it is the finding most often lost "
                 "to a crop or to vignetting rather than to noise.",
        ),
        FindingSpec(
            "fibrotic_band", "Fibrotic band / scarring", 0.080, 0.035, 5.0, 2.5,
            note="Post-primary sequela; thin and moderate contrast.",
        ),
    )
}

# The subset worth reporting when you only have room for a few rows: the hardest
# target, a mid target, and a control that should always pass.
CORE_FINDINGS = ("miliary_nodule", "infiltrate", "cavity_wall", "consolidation")


def get(key: str) -> FindingSpec:
    if key not in TB_FINDINGS:
        raise KeyError(f"unknown finding {key!r}; known: {sorted(TB_FINDINGS)}")
    return TB_FINDINGS[key]


def core() -> list[FindingSpec]:
    return [TB_FINDINGS[k] for k in CORE_FINDINGS]


def all_findings() -> list[FindingSpec]:
    return list(TB_FINDINGS.values())


def load_findings(path: str | Path) -> dict[str, FindingSpec]:
    """Replace the nominal table from a YAML or CSV file.

    Expected columns / keys: key, name, delta_d, delta_d_sigma, size_mm,
    size_sigma_mm, and optionally note and source. `source` should say where the
    number came from -- it is printed in the certificate provenance block, so a
    reader can see at a glance whether the verdict rests on a measurement or on
    the placeholder.
    """
    p = Path(path)
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml

        rows = yaml.safe_load(p.read_text())
        rows = rows["findings"] if isinstance(rows, dict) and "findings" in rows else rows
    else:
        import csv

        with open(p, newline="") as fh:
            rows = list(csv.DictReader(fh))

    out: dict[str, FindingSpec] = {}
    for r in rows:
        out[str(r["key"])] = FindingSpec(
            key=str(r["key"]),
            name=str(r.get("name", r["key"])),
            delta_d=float(r["delta_d"]),
            delta_d_sigma=float(r.get("delta_d_sigma", 0.0)),
            size_mm=float(r["size_mm"]),
            size_sigma_mm=float(r.get("size_sigma_mm", 0.0)),
            note=str(r.get("note", "")),
            source=str(r.get("source", "user-supplied")),
        )
    if not out:
        raise ValueError(f"no findings parsed from {path}")
    return out


def install(findings: dict[str, FindingSpec]) -> None:
    """Swap the module-level table in place, so every consumer picks it up."""
    TB_FINDINGS.clear()
    TB_FINDINGS.update(findings)


def contrast_detail_curve(findings: list[FindingSpec] | None = None) -> np.ndarray:
    """The (size_mm, delta_d) points as a contrast-detail plot, for figures.

    Overlaying the measured floor on this is the single clearest picture the
    physics track produces: findings above the floor curve are carried by the
    photograph, findings below it are not, and the gap between a clean capture's
    floor and a glared one's is visible at a glance.
    """
    fs = findings or all_findings()
    return np.array([[f.size_mm, f.delta_d] for f in fs], dtype=np.float64)
