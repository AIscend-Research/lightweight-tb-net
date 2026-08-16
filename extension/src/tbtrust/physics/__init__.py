"""Physics-derived uncertainty: measure the capture channel instead of learning it.

Every chest radiograph carries its own calibration targets. A lead L/R marker is
an object of known zero X-ray transmission; the collimation border is a physically
hard step edge; the direct-exposure region is film at maximum density, which is
optically black. Between them they pin the three unknowns in a phone photograph of
that film -- its tone curve, its point-spread function, and its veiling glare --
without any reference shot, any calibration target added to the scene, or any
training data.

Once those are measured rather than guessed, a per-pixel **density resolution
floor** follows: the smallest optical-density difference the photograph could
still carry, given the glare sitting on that pixel, the blur, the sensor noise and
the quantiser. Compare it against the characteristic contrast of a TB finding and
you get a **certificate of insufficiency** -- a falsifiable, label-free, network-free
statement that the information is not in the image. Not "the model is unsure":
*the measurement channel cannot carry that signal.*

Three things this buys the project
----------------------------------
1. **Aleatoric uncertainty from first principles.** Immune to the "you have just
   trained a blur detector" critique, because there is nothing trained in it and
   the output has units of optical density.
2. **A principled retake/refer split.** A high floor caused by localized glare
   means the operator should move the phone, and the glare field says which way. A
   fine floor with a still-uncertain model means real clinical ambiguity, so
   refer. Learned confidence cannot tell these apart; the physics can.
3. **A Shannon framing that is arithmetic rather than analogy.** `channel.py`
   computes bits, and attributes the ones that were lost to the veil, the blur and
   the quantiser separately.

Module map
----------
=================  ==========================================================
`density`          optical density conventions -- **read this first**, the
                   film sign convention is inverted from the project brief
`film`             forward model: photograph a film, with ground truth
`fiducials`        find marker, collimation border and beam stop; grade coverage
`psf`              ISO 12233 slanted-edge MTF from the collimation border
`glare`            veiling glare from the beam stop, blur bleed separated out
`tone`             two-point densitometry; illumination field
`invert`           the fixed-point inversion, and the split error budget
`findings`         TB finding contrasts -- **nominal placeholders**, replace them
`floor`            the density resolution floor
`certificate`      the verdict
`triage`           retake vs refer vs report, with the operator instruction
`channel`          capacity in bits, and where the lost bits went
`validate`         the two experiments that could falsify all of the above
=================  ==========================================================

The load-bearing assumption is that these fiducials survive in the public
archives. Run `scripts/audit_fiducials.py` before believing anything here.
"""

from __future__ import annotations

from .certificate import Certificate, Verdict, certificate_confidence, certify, certify_photo
from .density import FilmModel, density_to_transmittance, transmittance_to_density
from .fiducials import Coverage, Fiducials, detect
from .findings import TB_FINDINGS, FindingSpec, core, load_findings
from .floor import FloorSpec, density_floor, limiting_factor
from .invert import CalibratedFilm, invert, invertible
from .triage import Action, TriageDecision, triage, triage_summary

__all__ = [
    "TB_FINDINGS",
    "Action",
    "CalibratedFilm",
    "Certificate",
    "Coverage",
    "Fiducials",
    "FilmModel",
    "FindingSpec",
    "FloorSpec",
    "TriageDecision",
    "Verdict",
    "certificate_confidence",
    "certify",
    "certify_photo",
    "core",
    "density_floor",
    "density_to_transmittance",
    "detect",
    "invert",
    "invertible",
    "limiting_factor",
    "load_findings",
    "transmittance_to_density",
    "triage",
    "triage_summary",
]
