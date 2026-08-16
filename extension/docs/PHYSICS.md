# Physics-derived uncertainty: measuring the capture channel instead of learning it

Every chest radiograph carries its own calibration targets. Measure them and you can invert the
phone capture and bound what the photograph could possibly have carried — no labels, no
network, no training data.

This document is the reference for `src/tbtrust/physics/`. Read
[`density.py`](../src/tbtrust/physics/density.py) first: the sign convention is inverted from
the project brief and everything else depends on getting it right.

---

## 1. The three free calibration targets

| object on the film | known optical density | what it measures | module |
|---|---|---|---|
| lead L/R side marker | `D_min` ≈ 0.20 (base+fog) | bright densitometry anchor; the only *interior* sample of the lightbox | `tone.py` |
| direct-exposure region | `D_max` ≈ 3.20 | the optical **beam stop** → veiling glare | `glare.py` |
| collimation border | a hard `D_min`→`D_max` step | the capture **PSF**, by ISO 12233 slanted edge | `psf.py` |
| film outside the border | `D_min` | full-frame flat field → illumination surface | `tone.py` |

### The sign convention, which the brief has backwards

Two opposite conventions are in play and they must not be mixed.

- **X-ray transmission.** Lead transmits nothing, so in beam-stop terms the marker is "zero
  signal". This is the convention the project brief uses.
- **Developed film.** Film is a negative: more exposure → more silver → higher optical density
  → darker on a lightbox. Lead *blocks* the beam, so the film beneath it is barely exposed, so
  it sits at base+fog, so it is **transparent — the brightest thing in the frame**. The
  direct-exposure region took the full unattenuated beam, developed to maximum density, and is
  **the darkest**.

Everything downstream of the film — which is everything here, since we only ever see
photographs of developed film — uses the second convention. So the roles swap relative to the
brief:

```
optical beam stop / coronagraph  =  direct-exposure region   (D_max, near-opaque)
bright densitometry anchor       =  lead marker              (D_min, near-clear)
```

The idea is unaffected. There are still two anchors of known density and one of them is
optically black; it is simply the other one.

---

## 2. The pipeline

```
photo (8-bit JPEG)
  └─ fiducials.detect        marker, collimation quad, beam stop  →  Coverage grade
  └─ invert.invert           fixed-point loop, 2 iterations:
        tone (bright anchors, black point pinned)
        → linearise → psf.estimate_psf     (slanted edge)
        → tone.fit_illumination            (D_min regions)
        → glare.estimate_veil              (beam stop)
        → re-fit tone with corrected anchors
     ⇒ CalibratedFilm: density map + a split error budget
  └─ floor.density_floor     per-pixel minimum resolvable ΔD, per finding
  └─ certificate.certify     DETECTABLE / MARGINAL / INSUFFICIENT / ABSTAIN
  └─ triage.triage           REPORT / RETAKE (+ instruction) / REFER
  └─ channel.channel_report  capacity in bits, and where the lost bits went
```

### The bound

For a finding with unit-amplitude profile `t(x)` and density contrast `ΔD`, a matched filter —
optimal for a known signal in additive noise, so nothing can beat it — achieves

```
SNR = ΔD · √E / σ_D ,      E = Σ_f |T(f)|² · MTF(f)²
```

and requiring `SNR ≥ k` (Rose criterion, `k = 5`) gives

```
ΔD_floor(x) = k · σ_D(x) / √E
```

`E` is the template energy surviving the blur. It handles the area gain of a large finding and
the contrast loss of a small one in one quantity, with no need to pick a "characteristic
frequency".

Written out, the per-pixel differential density noise is

```
σ_D = (γ / ln10) · σ_v / (v − c₀) · (1 + V/I)
```

Three things to notice. The tone exponent scales it, which is why γ is worth measuring. It goes
inversely with the *signal above the black point*, so a crushed region carries almost no density
information however many bits you spend. And the veil enters as `1 + V/I`, exactly the
reciprocal of the contrast compression it imposes — a 20% veil costs 20% of your density
resolution wherever it lands. That term usually dominates in a real clinic photo, and nobody
measures it, because measuring it needs a beam stop.

### Random versus systematic, and why the split is load-bearing

`invert.py` reports two error budgets:

- **`sigma_random`** — pixel noise, quantisation, the scatter of the veil surface fit. Differs
  between two nearby pixels.
- **`sigma_systematic`** — the γ prior, the gain, the illumination interpolation, film D_min /
  D_max tolerances. Shared across the frame.

**Only the random terms enter the floor.** The floor bounds a *difference* of densities measured
millimetres apart, and a common scale or offset error cancels in a difference. Merging them
would inflate the floor several-fold and turn a usable bound into a useless one. This is the
same reason the γ prior is tolerable at all: absolute density is uncertain at the few-percent
level, density *differences* are not, and differences are what radiology reads.

`floor.py` makes one further distinction. Veil-fit error is spatially **correlated** on the
scale of the glare field, so it does not average down across a large lesion the way white noise
does. Crediting it with the matched filter's area gain made the bound several times too
optimistic for consolidations in the detectability test — the worst possible failure direction —
so `FloorSpec.correlated_terms` excludes it from that gain.

---

## 3. What has been measured

From `python scripts/validate_physics.py` on the synthetic film (see §5 for how to reproduce):

| quantity | result |
|---|---|
| collimation corners | ~0.15 px error on a clean capture, ~2.7 px at severity 1.0 |
| PSF σ from the slanted edge | within ~15% of truth up to severity 0.75 |
| veil at the beam stop | recovered at 0.8–1.5× truth across the sweep |
| lead marker | found reliably on clean captures; often lost above moderate severity |
| **detectability calibration** | **median predicted/empirical ≈ 1.7** |

The last row is the one that matters. `ratio > 1` means the bound is **conservative**: it
declares information lost slightly before an optimal detector actually loses it. That is the
safe direction for a safety valve. Quote this number with any result — a bound reported without
its measured calibration is an assertion.

### The falsification test

`validate.detectability_experiment` inserts lesions of known contrast, pushes them through the
simulated capture, and lets a matched filter that already knows the lesion's position, size and
shape try to tell present from absent. It then compares the empirical d′ threshold against the
blindly-computed floor.

This is not circular, and the objection is worth answering directly. The floor is computed from
quantities the estimator had to *measure blind* — σ_D from a noise model fitted to the image, E
from an MTF read off the collimation border, the veil amplification from the beam stop. The
empirical threshold comes from the detector's behaviour on actual noisy captures. Nothing forces
them to agree: under-measure the veil and the floor comes out optimistic; over-smooth the MTF
and it comes out pessimistic. The ratio is a score on the estimator, and it is free to be wrong.

---

## 4. Known limitations, in order of how much they should worry you

1. **The veil is measured at the periphery and interpolated across the middle.** The beam stop
   is an annulus. A specular reflection sitting in the centre of the film is caught only when it
   is bright enough to push a pixel above base+fog — `glare._add_impossible_brightness` exploits
   the fact that nothing on a developed sheet is clearer than base+fog, so any excess is
   unambiguously stray light. A *dimmer* central reflection is under-reported and the
   certificate is optimistic there. **This is the leading known bias.**
2. **The finding contrasts are nominal placeholders.** `findings.py` ships physically sensible
   values marked `source="NOMINAL"`, not numbers from a published table. Relative statements
   depend only on the floor and are sound; absolute verdicts inherit the table's uncertainty.
   Replace it with `--findings table.yaml` before publishing.
3. **γ is a prior, not a measurement**, unless a third distinct density is present. See §6.
4. **The bound is about the measurement channel, not diagnostic difficulty.** The dominant
   obstacle to spotting a real nodule is anatomical clutter — ribs, vessels, the heart border —
   not photon noise. A lesion can clear this floor and still be invisible against a rib.
   `floor.anatomical_noise` estimates that clutter and `include_anatomical=True` folds it in,
   but the certificate deliberately does not: its claim is the narrow, falsifiable one, *this
   photograph destroyed information the film had*. Clutter is in the original film too, and no
   retake fixes it.
5. **`px_per_mm` is inferred** from the detected field against a standard cassette diagonal,
   ±20%. Pass a measured value when you can.
6. **The lead marker detector is the weakest component.** It is tuned to reject rather than
   guess (`MARKER_ACCEPT = 0.6`), because a false marker feeds the illumination fit a bogus
   interior sample. Losing it costs less than it sounds: the unexposed film outside the
   collimation border is the same `D_min` density and wraps the whole frame.
7. **All in silico.** Simulated degradation, no real phone recaptures. Same caveat as the rest
   of the project — see [`LIMITATIONS.md`](LIMITATIONS.md).

### Coverage is the load-bearing assumption

None of this works on an image whose fiducials were cropped away, and public archives crop.
`scripts/audit_fiducials.py` measures how often they survived, per clinic. **Run it first.** A
low certifiable rate is a result to report with its number, not a bug to work around — and the
simulated re-photography path in `film.py` remains fully available, since it paints the
fiducials back on and gives a controlled experiment with ground truth.

---

## 4b. Figures

`physics/figures.py` produces the paper's visuals — diagrams and annotated images, not plots.
`python scripts/make_figures.py --out outputs/figures [--manifest ...]` renders all of them.

| figure | what it is for |
|---|---|
| `capture_chain_diagram` | the method figure: the channel, and where each unknown enters |
| `sign_convention_panel` | why lead is the *bright* anchor. Put it early — the method reads backwards without it |
| `fiducial_anatomy` | a real radiograph with the three targets called out, plus the coverage grade |
| `finding_atlas` | stylised chest, findings at true relative size and typical location, shaded by verdict |
| `detectability_strip` | **the one that matters**: the same lesion at multiples of the measured floor |
| `inversion_panels` | photo → measured veil → veil/signal → recovered density |
| `certificate_card` | the verdict as an object a clinician could be handed |
| `retake_instruction` | the glare field with an arrow saying which way to move |
| `radiograph_gallery`, `degradation_ladder` | real dataset images; what "severity 0.75" actually looks like |

Two are worth putting earliest. `sign_convention_panel`, because everything downstream depends
on the reader having the film's negative sense straight. And `detectability_strip`, because it
is the only figure that lets someone check the central claim with their own eyes: the floor is
computed blind from the photograph, lesions are then inserted at fractions and multiples of it,
and below 1× the lesion is invisible in the recovered image — not merely hard to see.

One honest note on that strip. The lesion site is chosen to be locally *flat*, the way a
contrast-detail phantom is. A site on a rib edge sits on an anatomical gradient an order of
magnitude larger than the lesion, so any display window wide enough to show the anatomy hides
the lesion at every contrast. That choice affects only the visual comparison; the quantitative
version in `validate.detectability_experiment` measures d′ against the noise and is unaffected.

## 5. Running it

```bash
# 0. does the corpus even carry the fiducials?  (seconds; no model, no GPU)
python scripts/audit_fiducials.py --manifest data/processed/manifest.csv --out outputs/fiducial_audit

# 1. the experiments that could falsify the bound
python scripts/validate_physics.py --quick          # ~1 min, CI gate
python scripts/validate_physics.py                  # ~10 min, publication run

# 2. a certificate per image, joinable to the model's predictions by `path`
python scripts/physics_certificates.py --manifest data/processed/manifest.csv \
    --out outputs/certificates.csv --severities 0,0.25,0.5,0.75,1.0
```

**Working resolution matters more than any other flag.** The floor depends on how many pixels a
finding spans. A phone photographing a 35 cm film at 3000 px gets ~8 px/mm; at 320 px it gets
0.8, so a 2 mm miliary nodule is sub-pixel and the certificate correctly but uselessly calls
every image insufficient. 1024 is the smallest size at which the severity sweep separates
properly (miliary marginal on a clean capture, insufficient once degraded, larger findings still
carried). §5b turns that threshold into a measured curve.

Notebooks [`05`](../notebooks/05_fiducial_coverage_audit.ipynb) through
[`08`](../notebooks/08_physics_deferral_and_triage.ipynb) walk the whole track on Kaggle.

### Measured cost

From `outputs/resolution_sweep/summary.json` (median per image, single CPU core,
Windows / Python 3.13 / numpy 2.4, no BLAS-heavy path involved — the inversion is
FFT- and stencil-bound):

| working size | px/mm | capture | invert | certify | total |
|---|---|---|---|---|---|
| 256 | 0.65 | 0.05 s | 0.08 s | 0.03 s | **0.16 s** |
| 384 | 0.97 | 0.09 s | 0.25 s | 0.05 s | **0.44 s** |
| 512 | 1.29 | 0.15 s | 0.43 s | 0.11 s | **0.67 s** |
| 768 | 1.94 | 0.49 s | 1.66 s | 0.27 s | **2.55 s** |
| 1024 | 2.58 | 1.29 s | 6.82 s | 0.72 s | **9.40 s** |
| 1536 | 3.87 | 3.03 s | 33.25 s | 1.74 s | **39.36 s** |

Cost grows faster than pixel count: 6× the linear size cost 246×, i.e. roughly the **cube of
the linear size** (N^1.5 in pixels, not N^1.0), because the wide-sigma glare and PSF blurs are
themselves size-dependent. Two consequences: 1024 is not "about 2 s per image" on every machine
— it was ~9 s here, and the earlier ~1.9 s figure should be treated as machine-specific — and a
full-corpus run at 1536 is an overnight job rather than a coffee break. Time it on your own
hardware before planning a sweep, and read the per-size totals above as the shape rather than
the absolute.

## 5b. Resolution dose-response: the threshold as a curve

`scripts/resolution_sweep.py` sweeps capture resolution on the synthetic film and reports the
px/mm at which each finding's contrast clears the measured floor. It needs no data and no GPU
(~15 min for the default 108-cell run).

**"More megapixels" is two different physical claims**, so the script runs both and the answer
is the bracket, not one end of it. Every length in `CaptureParams` is in pixels of the output
photo, so changing `--size` alone silently asserts that the lens, the hand shake and the photon
budget per pixel all improve in lockstep with the sensor:

- **sampling-limited** (`--mode sampling`) — blur fixed *in pixels*, each pixel keeps its own
  full photon well. This is what `--size` does today, and it is the optimistic end.
- **optics-limited** (`--mode optics`) — blur fixed *in millimetres of film* (the lens and the
  operator's hand do not improve because the sensor got denser, so the PSF and motion smear
  scale with px/mm) and the photon budget conserved across the frame rather than per pixel
  (more, smaller wells means more per-pixel shot noise; read noise stays per-pixel). The
  realistic end, and the only one that can show saturation.

### What it measures (3 films × 3 severities × 6 sizes × 2 modes)

Median margin in dB for the miliary nodule, the finding that decides every image's verdict:

| mode | severity | 0.65 | 0.97 | 1.29 | 1.94 | 2.58 | 3.87 px/mm |
|---|---|---|---|---|---|---|---|
| sampling | 0.00 | −17.0 | −11.0 | −6.3 | −0.9 | **+1.5** | **+4.0** |
| sampling | 0.25 | −22.0 | −12.9 | −8.7 | −4.1 | −2.3 | −0.8 |
| sampling | 0.50 | −22.7 | −14.4 | −8.5 | −4.0 | −4.5 | −4.5 |
| optics | 0.00 | −8.9 | −6.7 | **−6.3** | −8.3 | −9.8 | −11.7 |
| optics | 0.25 | −15.8 | −11.4 | **−8.7** | −11.9 | −16.1 | −21.2 |
| optics | 0.50 | −18.5 | −13.2 | **−8.5** | −12.6 | −18.2 | −25.7 |

Four results, in order of how much they should change what you do:

1. **The optics-limited curve is non-monotone and peaks at 1.29 px/mm** (working size 512) at
   every severity, declining on both sides — +3.7 dB/octave below the peak and −3.2 dB/octave
   above it on a clean capture, steepening to +13.9 / −12.7 at severity 0.5. If the lens and the
   light are fixed, resolution past ~1.3 px/mm does not merely stop helping, it **costs measured
   density resolution**, because the added pixels divide the same photons while the blur covers
   proportionally more of them. "Photograph it at the highest resolution the phone offers" is
   not the right instruction.
2. **On a clean capture in the optimistic model, a miliary nodule needs 2.15 px/mm to clear
   INSUFFICIENT and 3.30 px/mm to reach DETECTABLE** — 0.71 and 1.67 megapixels over a
   35.5 × 43.2 cm film. That is a far more modest sensor requirement than "8 px/mm", and it is
   the first number in this repo that answers the deployment question directly.
3. **Resolution does not rescue a degraded capture.** At severity 0.25 and above the miliary
   nodule never clears the floor at any resolution tested, in either mode. Framing, glare and
   steadiness dominate; megapixels do not substitute for them.
4. **Everything larger than a miliary nodule is carried at every resolution tested** — infiltrate,
   cavity wall and consolidation all clear the floor at 0.65 px/mm on a clean capture, so the
   whole resolution question is a question about miliary TB specifically.

### What this does *not* establish

- It is the synthetic film, not real radiographs, and the contrasts are the `NOMINAL` table
  (§4.2), so every absolute px/mm figure inherits that table's uncertainty. The *shape* of the
  curve — the peak's existence and its location — depends only on the floor and is sound.
- The px/mm axis carries the ±20% `px_per_mm` inference error (§4.5), which is roughly ±40% on
  every megapixel figure. A ruler in the frame removes it.
- The decline rate above the peak depends on the modelling choice that read noise stays
  per-pixel while well capacity scales with pixel area. That is right for a fixed sensor
  divided more finely, but a genuinely better sensor is a different experiment.
- 50% of severity-0.5 cells abstained: the fiducial detector loses the beam stop on heavily
  degraded captures, so the certificate reports "cannot measure" rather than a margin. Under the
  optics-limited model, raising resolution can *induce* abstention (one film measured fine at
  768 and abstained at 1024 and above), which is the safety valve switching itself off in
  exactly the direction that matters. That behaviour is safe by design — an abstention maps to
  confidence 0.0 and is deferred first — but it means the high-severity rows above are computed
  on the half of the captures that could still be measured.

---

## 6. The one hardware ask

`γ` is the weakest assumption in the pipeline: with two anchors the fit pins the black point and
takes γ from an sRGB prior, propagating its uncertainty into absolute density. A **third
distinct density** breaks the degeneracy and `fit_tone` switches to fitting γ properly, reading
its uncertainty off the χ² curvature.

The cheapest way to get one in the field is to **tape a small step wedge to the lightbox beside
the film** — a few pence of exposed, processed film with two or three known densities. It turns
the weakest assumption in this pipeline into a measurement. A **ruler in the frame** likewise
settles `px_per_mm` exactly. Both are in
[`DEPLOYMENT_CHECKLIST.md`](DEPLOYMENT_CHECKLIST.md).

---

## 7. Why this is worth having alongside the learned uncertainty

Three things it buys that a confidence score cannot.

1. **Aleatoric uncertainty from first principles.** Immune to the standard critique — "you have
   just trained a blur detector" — because nothing is trained and the output has units of
   optical density. It can be checked, and it can be wrong.
2. **A principled retake/refer split.** "Defer" collapses two situations a clinic must
   distinguish: *the photo is bad* (thirty seconds and another shot — and the glare field says
   which way to move the phone, the PSF anisotropy says whether it was shake or defocus) versus
   *the photo is fine and the case is hard* (retaking reproduces the same image; refer). Only
   the physics separates them, because the floor is a property of the capture and the residual
   uncertainty is a property of the case.
3. **A Shannon framing that is arithmetic rather than analogy.** Phase 1 reached for channel
   capacity and could only gesture at it. `channel.py` computes bits and attributes the lost
   ones to the veil, the blur and the quantiser separately.

`eval/physics_deferral.py` puts the certificate margin on the same axis as the learned signals
so they can be compared head to head, and `complementarity()` measures whether the orthogonality
claim actually holds on your data. Errors the physics catches that nothing else does are the
confident-and-wrong cases — the ones a screening system most needs to catch.
