# Measuring the Capture Channel: A Falsifiable Certificate of Insufficiency for Smartphone-Photographed Chest Radiographs, and What It Adds to Learned Uncertainty

**Status: draft prose, unmeasured numbers.** Every `{{PLACEHOLDER}}` is a value
that must come from a committed artifact; the bracketed path after it names the
file that produces it. Nothing in this document has been measured except where a
number appears without braces, and every such number says where it came from.
The discipline from `docs/PAPER_OUTLINE.md` carries over: a placeholder that
survives into a submission is a made-up number, so the pre-submission check is
"grep for `{{`".

Target venue: ML4H 2026, deadline 10 Sept 2026 (re-check the CFP —
`docs/phase1_framing.md` §4). Backup: MIRASOL @ MICCAI 2026, deadline
unconfirmed and possibly passed.

Two titles are in play and the choice depends on how the results land:

* **A (physics-forward, above).** Use if the certificate is the strongest
  result — if it wins or ties on AURC, or if the complementarity numbers hold.
* **B (deployment-forward).** *Trustworthy Tuberculosis Screening Under
  Simultaneous Domain Shift and Capture Degradation: Leave-One-Clinic-Out
  Evaluation with Physics-Gated Deferral.* Use if the learned track carries the
  paper and the certificate is a component.

---

## Abstract

*(≈200 words. Written last; this is the shape, with the placeholders that fill
it.)*

Low-compute tuberculosis screeners report near-perfect accuracy on clean chest
radiographs from dedicated hardware — TB-Net reports 99.86% on a random split of
pooled public data. Rural screening does not look like that: the input is a
smartphone photograph of an analog film on a lightbox, and the clinic
contributed no training data. We evaluate leave-one-clinic-out under simulated
smartphone re-photography and find accuracy falls to {{ACC_LOCO}} from
{{ACC_ID}} in-distribution [`outputs/loco_sweep/`]. We then ask whether the
image-quality half of that uncertainty is a *measurement* problem rather than a
learning problem. A chest film carries its own calibration targets — a lead
marker at known base+fog density, a direct-exposure region that is optically
black, and a collimation border that is a physically hard step edge — which pin
the phone's tone curve, point-spread function and veiling glare with no
reference shot and no labels, yielding a per-pixel density resolution floor and
a falsifiable *certificate of insufficiency*: the statement that a named finding's
contrast is or is not above what the photograph can carry, in units of optical
density. On synthetic film an optimal matched filter's empirical detectability
threshold lands within a factor of {{CAL_RATIO}} of the predicted floor
[`outputs/physics_validation/summary.json`], conservative on the median. Used as
a deferral signal the certificate achieves AURC {{AURC_PHYS}} against
{{AURC_CONF}} for softmax confidence, and catches {{COMPLEMENT_N}} errors that
no learned signal ranks in its own deferral budget
[`notebooks/08_physics_deferral_and_triage.ipynb`]. All degradation is
synthetic, two clinics serve as held-out folds, and the finding-contrast table is
nominal; we state precisely what that does and does not license.

---

## 1. Introduction

Reported accuracy for automated TB screening is measured under conditions that
do not obtain at deployment, and in two ways at once.

**The image is not the image the model was trained on.** In a rural,
cold-chain-free screening setting there is no PACS and no DICOM export. There is
a developed sheet of film, a lightbox of uncertain age, and a phone. What reaches
the classifier is a photograph: blurred by hand shake and defocus, veiled by
glare off the film's surface and off the room, geometrically warped by an
off-axis capture, tone-mapped by an unknown ISP, and JPEG-compressed.

**The site is not the site the model was trained on.** A screening programme
deploys to clinics that contributed nothing to training. Cross-site shift in
chest radiography is well documented and is not removed by pooling more sources.

Prior work varies one of these at a time, at most. The gap this paper addresses
is the conjunction, and — more specifically — what a system should *do* about it.
A screening tool that returns a wrong answer confidently is worse than one that
declines, but "decline" is not a single action. A bad photograph and a hard case
are different problems with different costs: the first is fixed in thirty
seconds by a retake, the second needs a specialist who may be a day's travel
away. Collapsing them into one "defer" bucket wastes the cheap fix and hides the
expensive one.

Our central claim is that the first of those two — image-quality uncertainty — is
a **measurement problem rather than a learning problem**. The film's own
fiducials are calibration targets of known optical density. Measuring them pins
the capture channel well enough to bound detectability in units of optical
density, without labels and without a network. That bound is orthogonal in kind
to a learned confidence score: it is a statement about what the photograph
contains, not about what the model believes, so it can catch the
confident-and-wrong cases a learned signal structurally cannot see.

**Contributions**, stated as what is measured rather than what is built:

1. **A measured capture channel.** A blind inversion of a phone photograph of a
   film using only the film's own fiducials, recovering tone curve, PSF and
   veiling glare, validated against ground truth in simulation
   (§3.5, `physics/validate.py`).
2. **A falsifiable certificate of insufficiency.** A per-pixel density
   resolution floor and a per-finding verdict in units of optical density, with
   a measured calibration against an optimal detector's empirical threshold —
   the number that makes it a bound rather than an assertion (§3.6).
3. **A leave-one-clinic-out evaluation** of low-compute TB screening under
   simulated smartphone re-photography, isolating cross-site shift and capture
   degradation and reporting them jointly (§4).
4. **A head-to-head comparison of five deferral signals** — softmax confidence,
   MC-dropout spread, a degradation-supervised uncertainty head, deep-ensemble
   disagreement, and the certificate margin — under a *fixed* point predictor, so
   the comparison measures ranking quality and not confounded prediction changes.
5. **A retake/refer split driven by the measured channel**, with an actionable
   instruction, plus the complementarity analysis that is the argument for the
   method even where it loses on AURC.
6. **An honest negative-space account**: a fiducial-coverage audit that decides
   what the physics track is allowed to be a claim about, and a limitations
   section written before the results existed.

## 2. Related work

**Low-compute TB CXR screening.** TB-Net (Wong et al., 2022; arXiv:2104.03165),
LightTBNet (Capellán-Martín et al., 2023), Pasa et al. (2019). All report
random-split or within-cohort *k*-fold accuracy on pooled public data. → *No
leave-one-clinic-out, so no measurement of what happens at an unseen site.*

**Phone-captured medical imaging.** CheXphoto (Phadke et al., 2020;
arXiv:2007.06199) contributes 10,000+ real smartphone photographs and synthetic
photographic transforms of CheXpert radiographs; CheXphotogenic
(arXiv:2011.06129) and the *npj Digital Medicine* recalibration follow-up show
that simple recalibration partially recovers the loss. → *CheXpert findings
rather than TB, a single source so no cross-site axis, and no deferral
mechanism.*

**Calibrated uncertainty and selective classification.** Guo et al. (2017) on
temperature scaling; Gal & Ghahramani (2016) on MC-dropout; Lakshminarayanan et
al. (2017) on deep ensembles; Sensoy et al. (2018) on evidential deep learning;
Sadinle et al. (2019) and Angelopoulos & Bates (2023) on conformal sets. →
*Rarely evaluated under simultaneous covariate shift and input corruption in a
deployment-shaped task, and evaluated as ranking scores rather than as inputs to
an action.*

**Cross-site TB testing.** The closest precedent (Tianjin Haihe ensemble,
PMC11301748) reports the cross-site drop and stops: no uncertainty, no deferral,
no recovery measurement. That is the sharpest contrast to draw.

**Physical image-quality measurement.** The components we use are standard
radiographic-physics and imaging-metrology tools applied to an unusual object.
The Rose criterion (Rose, 1948) fixes the SNR required for reliable detection of
a known signal; slanted-edge MTF estimation follows ISO 12233; beam-stop
measurement of veiling glare is the standard optical technique. → *These are
applied to dedicated calibration hardware, not to fiducials that happen to
already be in the clinical image. The contribution here is the observation that
the film is its own phantom.*

**What nobody does, and this paper does.** Treat the image-quality half of
deployment uncertainty as a measurement with units and an error bar rather than
as something to be learned from labels, and then measure whether that
measurement adds anything to a learned signal on the same axis.

## 3. Method

### 3.1 Problem setup

Binary TB screening from a chest radiograph. Evaluation is leave-one-clinic-out
(LOCO): all data from one source is held out entirely, and every fitted quantity
— network weights, temperature, uncertainty normalisation, deferral threshold,
conformal quantile — is fitted on the remaining sources' *validation* split and
applied unchanged to the held-out clinic.

**On "four clinics", precisely.** The manifest draws on four public sources
(Montgomery, Shenzhen, NIAID, RSNA), but only Montgomery (n = 138) and Shenzhen
(n = 662) contain both classes. Sensitivity or specificity is undefined on a
single-class test set, and there the clinic label is a near-perfect proxy for the
diagnosis. So: **four sources contribute training data; the rotation reports two
two-class holdout folds.** `data/splits.py` enforces this by default. Per-fold
*n* is reported alongside every accuracy, with intervals — Montgomery's 138 test
images do not support a bare point estimate.

### 3.2 Smartphone degradation model

`data/degradation.py` applies seven operations with continuous severity in
[0, 1]: motion blur, defocus, lightbox glare, uneven shadow, capture angle, JPEG
artifacts and resolution loss. Every image carries a record of what fired, which
yields the weak uncertainty-supervision target
(`manifest.uncertainty_target_from_severity`). `data/degradation_learned.py`
provides an unpaired adversarial generator as an alternative strategy, and
`scripts/ablate_degradation.py` scores strategies for realism against real phone
recaptures. **No real recaptures exist in this project**, so that script can
currently only score strategies against each other, which establishes
disagreement and not realism (§6.2).

### 3.3 Models

A DenseNet-121 (CheXNet-style) baseline; a TB-Net *reimplementation* tuned to the
reported ~4.24M parameters, which costs about 2× TB-Net's 0.42 GMACs at that
parameter count (`scripts/benchmark_efficiency.py` — the one axis on which the
comparison is measured rather than assumed); and an evidential variant. All three
share one training and evaluation path. Domain-generalization objectives (Deep
CORAL, IRM, DANN behind a gradient-reversal layer, clinic-conditional FiLM) are
selectable per config, giving a four-way ablation on one fold.

### 3.4 Learned uncertainty signals

Four signals rank the *same* temperature-scaled probabilities and differ only in
the ranking: `confidence` = max(p, 1−p); `mc_dropout` = predictive spread over T
stochastic passes; `head` = the degradation-supervised uncertainty head, or
evidential vacuity; `ensemble` = member disagreement. Holding the point
prediction fixed is deliberate — it isolates ranking quality instead of
confounding it with a change in the classifier's own predictions.

### 3.5 The measured capture channel

Every chest film carries three calibration targets of known optical density.
Because developed film is a negative, the roles are the opposite of the X-ray
convention and the method reads backwards if this is not stated first
(`physics/density.py`; figure `02_sign_convention`):

| object on the film | optical density | what it measures |
|---|---|---|
| lead L/R side marker | D_min ≈ 0.20 (base+fog) | **bright** anchor; the only interior sample of the lightbox |
| direct-exposure region | D_max ≈ 3.20 | optical **beam stop** → veiling glare |
| collimation border | hard D_min→D_max step | capture **PSF**, by slanted edge (ISO 12233) |
| film outside the border | D_min | full-frame flat field → illumination surface |

A fixed-point loop (`physics/invert.py`, two iterations) fits the tone curve on
the bright anchors with the black point pinned, linearises, estimates the PSF
from the slanted edge, fits the illumination surface on the D_min regions,
estimates the veil at the beam stop, and re-fits the tone curve with corrected
anchors. It returns a calibrated density map and a **split error budget**:
`sigma_random` (pixel noise, quantisation, veil-fit scatter) and
`sigma_systematic` (the γ prior, gain, illumination interpolation, film
tolerances).

Only the random terms enter the bound. The floor bounds a *difference* of
densities measured millimetres apart, and a common scale or offset error cancels
in that difference; merging the budgets would inflate the floor several-fold and
turn a usable bound into a useless one. This is also why a γ prior is tolerable
at all: absolute density is uncertain at the few-percent level, density
*differences* are not, and differences are what radiology reads.

### 3.6 The density resolution floor and the certificate

For a finding with unit-amplitude spatial profile *t*(x) and density contrast
ΔD, seen through a capture with modulation transfer MTF(*f*) and per-pixel
differential density noise σ_D(x), a matched filter — the optimal linear
detector for a known signal in additive noise, so no detector can do better —
achieves

    SNR = ΔD · √E / σ_D ,      E = Σ_f |T(f)|² · MTF(f)²

and requiring SNR ≥ *k* (Rose criterion, *k* = 5; *k* = 3 reported as marginal)
gives the per-pixel floor

    ΔD_floor(x) = k · σ_D(x) / √E .

E is the template energy surviving the blur, and it handles in one quantity the
area gain of a large finding and the contrast loss of a small one, with no need
to choose a "characteristic frequency". Written out,

    σ_D = (γ / ln10) · σ_v / (v − c₀) · (1 + V/I) ,

so the tone exponent scales the floor (which is why γ is worth measuring), the
floor goes inversely with the signal *above the black point* (a crushed region
carries almost no density information however many bits are spent on it), and
the veil enters as (1 + V/I) — exactly the reciprocal of the contrast
compression it imposes. A 20% veil costs 20% of the density resolution wherever
it lands. That term usually dominates in a real clinic photograph and is the one
nobody measures, because measuring it needs a beam stop.

One further distinction is load-bearing: veil-fit error is spatially
**correlated** on the scale of the glare field, so it does not average down
across a large lesion the way white noise does. Crediting it with the matched
filter's area gain made the bound several times too optimistic for
consolidations — the worst possible failure direction — so it is excluded from
that gain (`FloorSpec.correlated_terms`).

Comparing the floor against a finding's characteristic contrast gives the
certificate, reported as a margin in dB, `margin_db = 20·log₁₀(ΔD_finding /
floor)`, with thresholds at +3 dB (DETECTABLE) and 0 dB (INSUFFICIENT), the band
between being MARGINAL, and a fourth verdict ABSTAIN for images whose fiducials
were cropped away. The image's verdict is its **worst** line item: screening is a
sensitivity problem, and an image that can carry a lobar consolidation but not a
miliary pattern has lost the finding one most needs not to miss.

An INSUFFICIENT verdict is not "the network is unsure". It is: *the photograph
does not contain the information*. No model and no amount of training data can
recover a density step smaller than the channel can carry.

### 3.7 Falsification

`physics/validate.py` inserts lesions of known contrast into the synthetic film,
pushes them through the simulated capture, and lets a matched filter that already
knows the lesion's position, size and shape try to tell present from absent. The
empirical d′ threshold is then compared against the blindly-computed floor.

This is not circular, and the objection deserves the direct answer: the floor is
computed from quantities the estimator had to *measure blind* — σ_D from a noise
model fitted to the image, E from an MTF read off the collimation border, the
veil amplification from the beam stop — while the empirical threshold comes from
the detector's behaviour on actual noisy captures. Under-measure the veil and
the floor comes out optimistic; over-smooth the MTF and it comes out
pessimistic. The ratio is a score on the estimator and is free to be wrong. We
report it: **ratio = predicted floor / empirical threshold = {{CAL_RATIO}}**
(median; IQR {{CAL_IQR}}) [`outputs/physics_validation/summary.json`]. Above 1
is the conservative and safe direction — the certificate declares information
lost slightly before an optimal detector actually loses it. A bound quoted
without its measured calibration is an assertion, and below 1 the certificate
must not ship.

### 3.8 Triage: retake versus refer

The certificate and the model's residual uncertainty answer different questions,
which is what lets `physics/triage.py` split "defer":

* floor high, and the limiting factor is operator-fixable (glare hotspot,
  motion anisotropy, crushed exposure) → **RETAKE**, with the specific
  instruction — which way the glare is, whether the blur is shake or defocus —
  and the expected gain in dB, so a retake is only requested when it would help;
* floor adequate, model still uncertain → **REFER**;
* floor adequate, model confident → **REPORT**.

`eval/physics_deferral.py` maps the certificate margin through a logistic to
[0, 1] so it can be ranked on the same axis as the learned signals, with
abstentions at 0.0 — an image whose glare could not be measured is the *last*
one to trust, and treating "unmeasured" as "average" is how a safety valve
silently stops working.

### 3.9 Protocol discipline

One paragraph, because it is a differentiator. The temperature, the
uncertainty→confidence normalisation, the deferral threshold and the conformal
quantile are all fitted on the validation split of *seen* clinics and applied,
never re-fitted, on the held-out clinic; train and eval derive the identical
split by construction (`data.splits.split_from_config`). `metrics.json` flags
`temperature_at_bound` when the fitted temperature lands on a search bound, and
no run so flagged is reported.

## 4. Experiments

| # | Question | Produced by | Placeholder |
|---|---|---|---|
| E0 | Do the archives retain the fiducials, per clinic? | `scripts/audit_fiducials.py` | `{{COVERAGE_*}}` |
| E1 | In-distribution reference accuracy | random-split run | `{{ACC_ID}}` |
| E2 | LOCO accuracy per held-out clinic | `run_experiments.py` | `{{ACC_LOCO_*}}` |
| E3 | Accuracy vs. degradation severity | `robustness_sweep` | `{{SWEEP}}` |
| E4 | AURC per uncertainty method per severity | `uncertainty_methods` | `{{AURC_*}}` |
| E5 | Calibration: ECE/MCE, reliability diagram | `eval/calibration.py` | `{{ECE}}`, `{{MCE}}` |
| E6 | Safe deferral: coverage/accuracy at tuned T | `operating_point` | `{{DEFER_RATE}}` |
| E7 | Human-rescue rate | `human_rescue_rate` | `{{RESCUE}}` |
| E8 | Conformal coverage shortfall on held-out clinic | `eval/conformal.py` | `{{SHORTFALL}}` |
| E9 | Per-clinic calibration heatmap (fairness audit) | `eval/crosssite.py` | figure |
| E10 | Brier / BSS / Murphy decomposition | `eval/forecast_verification.py` | `{{BSS}}` |
| E11 | Params / MACs / CPU latency per arch | `benchmark_efficiency.py` | `{{EFFICIENCY}}` |
| E12 | Worst-of-N adversarial degradation | `eval/adversarial_degradation.py` | `{{ADV}}` |
| E13 | Channel recovery vs. ground truth | `validate_physics.py` | `{{VEIL_ERR}}`, `{{PSF_ERR}}` |
| E14 | **Detectability calibration** (falsification) | `validate_physics.py` | `{{CAL_RATIO}}` |
| E15 | Certificate as a deferral signal, vs. learned | `eval/physics_deferral.py` | `{{AURC_PHYS}}` |
| E16 | Complementarity: errors only the physics catches | `physics_deferral.complementarity` | `{{COMPLEMENT_N}}` |
| E17 | Retake/refer split and retake rate | `physics/triage.py` | `{{RETAKE_RATE}}` |
| E18 | Resolution dose-response of the floor | `scripts/resolution_sweep.py` | **measured — §5.8** |

Ablations worth their space: degradation-augmented training vs. not; the
four-way DG ablation (none/CORAL/DANN/FiLM) on one fold; MC-dropout passes *T*
vs. AURC (the latency/quality trade-off that matters on-device); and the floor
with and without the anatomical-clutter term.

**E0 gates everything downstream of it.** If the archives cropped the fiducials
away, the real-photo path is not available and the physics track is a claim about
*simulated re-photography* — images that are re-fiducialised and photographed
through the forward model — rather than about archived radiographs. Both are
legitimate experiments and they license different sentences. The paper states
which one it ran, with the coverage number that decided it.

### 4.1 Pre-registered reading of the results

Written before the runs so the interpretation cannot be fitted to the outcome.

* If the certificate **wins** on AURC: report it as a deferral signal that needs
  no labels and no training, with the calibration ratio attached.
* If it **ties or loses** on AURC but the complementarity numbers hold: that is
  still the paper's argument — a signal that catches a *disjoint* set of errors is
  worth having next to a better-ranking one, and the confident-and-wrong cases are
  the ones a screening system most needs to catch. Report the union.
* If it loses on AURC **and** catches nothing the learned signals miss: report
  that as a negative result, with the coverage and calibration numbers that make
  it interpretable. The repo already carries one honest negative (TB-Net
  unreproduced); a second costs nothing and buys the rest of the paper its
  credibility.
* If the calibration ratio comes out **below 1**: the certificate is optimistic —
  it would pass photographs an optimal detector cannot read — and it is reported
  as a bound that does not yet hold, not shipped as a safety valve.

## 5. Results

*(Lead with the gap, then the recovery, then the method comparison — that
ordering matches the argument rather than the pipeline.)*

**5.1 The gap.** {{ACC_ID}} in-distribution → {{ACC_LOCO}} leave-one-clinic-out
at severity 0, falling to {{ACC_LOCO_SEV1}} at severity 1. Per-fold *n* and
intervals in Table {{T_GAP}}.

**5.2 Fiducial coverage.** {{COVERAGE_MONTGOMERY}} / {{COVERAGE_SHENZHEN}} /
{{COVERAGE_NIAID}} / {{COVERAGE_RSNA}} of images retain a usable beam stop.
This decides §4's path question and is stated before any physics result.

**5.3 The channel is recoverable.** Veil fraction recovered to {{VEIL_ERR}},
PSF σ to {{PSF_ERR}}, differential density RMSE {{DENS_RMSE}} across the severity
sweep, against ground truth by construction.

**5.4 The bound holds.** Median predicted/empirical ratio {{CAL_RATIO}}
(IQR {{CAL_IQR}}), conservative on the median; {{PASS_FRAC}} of conditions pass
individually at a factor-of-two tolerance; certificate margin falls
monotonically with severity ({{MONOTONE}}).

**5.5 Deferral, head to head.** AURC by signal at each severity, Table
{{T_AURC}}. Certificate margin {{AURC_PHYS}} vs. confidence {{AURC_CONF}},
MC-dropout {{AURC_MC}}, head {{AURC_HEAD}}, ensemble {{AURC_ENS}}.

**5.6 Complementarity.** {{COMPLEMENT_N}} errors are caught by the certificate
and by no learned signal within the same deferral budget; {{COMPLEMENT_REV}} the
other way. The union defers {{UNION_RATE}} for {{UNION_ACC}} retained accuracy.

**5.7 Retake versus refer.** {{RETAKE_RATE}} of deferrals are operator-fixable
with an expected gain of {{RETAKE_GAIN}} dB; the rest are referrals. This is the
number that converts a deferral rate into a staffing requirement.

**5.8 Resolution dose-response.** *(Measured on synthetic film; the only §5 entry
that is not a placeholder. `outputs/resolution_sweep/summary.json`, 108 cells,
15 min on one CPU core.)* The floor scales with px/mm, so the certificate has a
resolution at which each finding becomes recoverable, and it is reported as a
bracket between two modelling assumptions — blur fixed in pixels (optimistic)
and blur fixed in mm of film with the photon budget conserved (realistic) —
because "more megapixels" is two different physical claims.

On a clean capture in the optimistic model a miliary nodule clears INSUFFICIENT
at 2.15 px/mm and reaches DETECTABLE at 3.30 px/mm — 0.71 and 1.67 megapixels
over a 35.5 × 43.2 cm film, a far more modest sensor requirement than the ~8
px/mm a full-resolution phone capture provides. In the realistic model the curve
is **non-monotone**: it peaks at 1.29 px/mm at every severity and falls by
3.2 dB/octave above the peak on a clean capture (12.7 dB/octave at severity 0.5),
because added pixels divide the same photons while the blur covers
proportionally more of them. Resolution does not rescue a degraded capture — at
severity ≥ 0.25 the miliary nodule clears the floor at no resolution tested in
either model — and every finding larger than a miliary nodule is carried at every
resolution tested, so the resolution question is specifically a question about
miliary TB. Absolute px/mm figures inherit the NOMINAL contrast table (§6.2) and
the ±20% `px_per_mm` inference (§6.5); the shape of the curve depends only on the
floor.

**Figures.** (a) the capture chain and where each unknown enters; (b) the sign
convention; (c) the detectability strip — the same lesion at multiples of the
measured floor, the only figure that lets a reader check the central claim by
eye; (d) risk-coverage by signal; (e) reliability before/after temperature
scaling; (f) per-clinic calibration heatmap; (g) the certificate card as an
object a clinician could be handed.

## 6. Limitations

Imported from `docs/LIMITATIONS.md` and `docs/PHYSICS.md` §4, written before
results existed specifically so they cannot be trimmed to fit the findings.
Stated plainly, in the order a reviewer should weigh them:

1. **All in silico. There are no real phone recaptures.** Every degradation is
   simulated, and the physics track is validated against a forward model we wrote
   ourselves. An estimator tested against its own generative assumptions measures
   less than it appears to; the deliberate mismatch between the forward tone curve
   (four parameters, with an S-curve) and the fitted one (a two-parameter power
   law) narrows that gap but does not close it. A model robust to *our* blur and
   glare is not thereby robust to a specific phone in a specific clinic. This is
   the single largest threat to external validity. `data/real_recapture/` holds
   the collection protocol and no data; a few dozen real captures would convert
   "the estimator recovers the parameters we gave it" into "the estimator
   recovers a real phone."
2. **The finding contrasts are placeholders.** `physics/findings.py` ships
   physically sensible values marked `source="NOMINAL"`, not numbers from a
   published table or a contrast-detail measurement. Every *relative* statement —
   this photograph carries less density resolution than that one, this glare
   hotspot costs a factor of three, the floor rose above the contrast when the
   veil passed 15% — depends only on the floor and is sound. Every *absolute*
   verdict (DETECTABLE vs. INSUFFICIENT for a named finding) inherits this
   table's uncertainty, which `FindingSpec.delta_d_sigma` propagates into the
   reported margin. The certificate's provenance block prints the source, so no
   verdict can be read without seeing it.
3. **The veil is measured at the periphery and interpolated across the field
   interior.** The beam stop is an annulus. A specular reflection sitting in the
   middle of the film is caught only when it is bright enough to push a pixel
   above base+fog — nothing on a developed sheet is clearer than base+fog, so any
   excess there is unambiguously stray light (`glare._add_impossible_brightness`).
   A *dimmer* central reflection is under-reported, and **the certificate is
   optimistic there** — the dangerous direction. This is the leading known bias.
4. **γ is a prior, not a measurement.** With only the two anchors the film
   provides, the fit pins the black point and takes the tone exponent from an
   sRGB prior, propagating its uncertainty into every absolute density. A third
   distinct density — a step wedge taped beside the film, costing pennies —
   breaks the degeneracy and lets `tone.fit_tone` fit γ properly with an error
   bar read off the χ² curvature.
5. **`px_per_mm` is inferred at ±20%**, from the detected collimation field
   against a standard cassette diagonal. That error propagates into every
   finding's spatial frequency and so into the floor, and into every megapixel
   figure in §5.8 at roughly ±40% in area. A ruler in the frame settles it
   exactly.
6. **The bound is about the measurement channel, not diagnostic difficulty.**
   The dominant obstacle to spotting a real nodule on a real chest radiograph is
   anatomical clutter — ribs, vessels, the heart border — not photon noise. A
   lesion can clear this floor and still be invisible against a rib.
   `floor.anatomical_noise` estimates the clutter and can be folded in, but the
   certificate deliberately does not, because its claim is the narrow, defensible
   one: *this photograph destroyed information the film had*. Clutter is present
   in the original film too and no retake fixes it; conflating the two would make
   the bound unfalsifiable.
7. **Coverage is the load-bearing assumption.** None of the physics works on an
   image whose fiducials were cropped away, and public archives crop. §5.2 is the
   measured rate, and it decides which of the two experiments (§4) this paper
   reports.
8. **No result here comes from a real training run** unless §5 says otherwise
   with a committed artifact path.
9. **Two held-out clinics, not four**, with Montgomery's fold at 138 test images
   and correspondingly wide intervals.
10. **The uncertainty target proxies correctness, not radiologist agreement.**
    The weak label is derived monotonically from applied severity, which encodes
    "a degraded image deserves low confidence". That is a reasonable prior and
    what the channel framing argues for, but it is not "a radiologist would want a
    second read here". No clinician has labelled anything in this project, so the
    second-reader framing is a motivating analogy, not a measured quantity.
11. **The conformal guarantee does not formally transfer across clinics.**
    Calibration and test are not exchangeable under LOCO by construction. The
    shortfall is reported as a distribution-free readout of domain shift; a 1−α
    guarantee is never claimed on a new site.
12. **TB-Net is reimplemented, not reproduced**, and every comparison says so.
13. **Retrospective, curated, already-diagnosed cohorts.** They
    under-represent the ambiguous, early and comorbid presentations that dominate
    real screening, and their normals are not drawn from a screening population.
    Reported sensitivity and specificity are optimistic relative to field
    deployment regardless of how carefully the splits are done.
14. **Deferral assumes a human to defer to.** The safety argument transfers
    workload to a person who must actually exist, be reachable offline, and have
    time. A deferral rate is a staffing decision before it is a hyperparameter.

## 7. Deployment considerations

Condensed from `docs/DEPLOYMENT_CHECKLIST.md`. The point a reviewer should take:
the deferral mechanism transfers workload to a human who must actually exist, and
the two cheap fiducial additions (a step wedge and a ruler, both costing pennies)
remove the two weakest assumptions in the physics track — which is a better
return than any modelling change available. Prevalence, calibration measured on
site-local data, a named recipient for deferred cases, a cap on retakes, and a
pre-agreed trigger for taking the model offline are all prerequisites rather than
future work. Fairness is audited per clinic on *calibration*, not only on
accuracy: a model can hold accuracy at a site while its probabilities drift, and
that is where hidden domain shift shows up (`docs/FAIRNESS_AUDIT.md`).

## 8. Reproducibility

Code, degradation pipeline, physics track and evaluation tools released under
MIT. Data is not redistributable (NIAID DUA; the aggregated Kaggle mirror does
not visibly carry it forward, so no NIAID-attributed image appears in any figure).
Every table regenerates from `scripts/run_experiments.py`,
`scripts/validate_physics.py` and `scripts/physics_certificates.py` plus the
config files; seeds fixed in config; CI runs the smoke test and unit tests on
every push. The physics track depends only on numpy and Pillow, runs on a single
CPU core, and needs no labels — measured cost is in
`outputs/resolution_sweep/summary.json`.

---

## Appendix A — placeholder index

Grep target: `{{`. Every entry must be replaced by a value traceable to a
committed artifact before submission.

| placeholder | artifact |
|---|---|
| `{{ACC_ID}}`, `{{ACC_LOCO*}}`, `{{SWEEP}}` | `outputs/loco_sweep/` |
| `{{AURC_*}}`, `{{DEFER_RATE}}`, `{{RESCUE}}`, `{{ECE}}`, `{{MCE}}`, `{{BSS}}` | `outputs/<fold>/metrics.json` |
| `{{SHORTFALL}}` | `eval/conformal.py` output in `metrics.json` |
| `{{COVERAGE_*}}` | `outputs/fiducial_audit/` |
| `{{CAL_RATIO}}`, `{{CAL_IQR}}`, `{{PASS_FRAC}}`, `{{MONOTONE}}`, `{{VEIL_ERR}}`, `{{PSF_ERR}}`, `{{DENS_RMSE}}` | `outputs/physics_validation/summary.json` |
| `{{AURC_PHYS}}`, `{{COMPLEMENT_N}}`, `{{COMPLEMENT_REV}}`, `{{UNION_*}}`, `{{RETAKE_*}}` | `notebooks/08_physics_deferral_and_triage.ipynb` |
| `{{RES_SPEC}}` | *filled — `outputs/resolution_sweep/summary.json`* |
| `{{EFFICIENCY}}` | `outputs/efficiency_benchmark.json` |

## Appendix B — pre-submission checks

- [ ] No `{{PLACEHOLDER}}` survives anywhere in the manuscript.
- [ ] Every reported number traceable to a committed `metrics.json` / sweep report.
- [ ] "Four clinics" never used to describe the *holdout* rotation.
- [ ] TB-Net comparisons say "reimplementation" every time.
- [ ] Conformal coverage never stated as a guarantee on a held-out clinic.
- [ ] `temperature_at_bound` false in every run whose numbers are reported.
- [ ] Limitations section not shortened relative to `docs/LIMITATIONS.md`.
- [ ] The certificate's contrast-table source is stated wherever an absolute
      verdict appears; if it still reads NOMINAL, §6.2 says so in the same
      paragraph as the result.
- [ ] The calibration ratio appears in the abstract, not only in §5.
- [ ] Every citation in §2 re-verified against the source; the venue CFP
      re-checked for track and page limit.
