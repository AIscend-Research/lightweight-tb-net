# Paper outline (Phase 5)

Target: **ML4H 2026**, submission deadline **10 Sept 2026** (re-check the CFP for
track and page limit — it was "opening soon" when checked). Backup: MIRASOL @
MICCAI 2026, deadline unconfirmed and possibly already passed — see
`docs/phase1_framing.md` §4.

This is a skeleton, not a draft. Every numeric claim is a named placeholder tied
to the artifact that produces it, so filling the paper in is a mechanical step
once the GPU runs finish — and so that no placeholder can accidentally survive
into a submission as a made-up number. **Nothing here has been measured yet.**

Notation: `{{...}}` marks a value that must come from a real run. The bracketed
path after it says which file produces it.

---

## Title

Trustworthy Tuberculosis Detection Under Image Degradation: Uncertainty
Quantification for Cold-Chain-Free Smartphone Screening in Resource-Constrained
Rural Clinics

## Abstract (≈200 words, write last)

Structure: lab-vs-deployment gap → what we do → what we measure → headline result
→ the honest caveat.

> Low-compute TB screeners report near-perfect accuracy on clean chest X-rays
> from dedicated imaging hardware — TB-Net reports 99.86% on a random split of
> pooled public data. Rural screening does not look like that: the input is a
> smartphone photograph of an analog film on a lightbox, and the clinic
> contributed no training data. We evaluate leave-one-clinic-out under simulated
> smartphone-capture degradation and find accuracy falls to {{ACC_LOCO}} from
> {{ACC_ID}} in-distribution [robustness_sweep + reference run]. We add a
> calibrated uncertainty signal and a deferral policy tuned on validation only,
> and show that deferring {{DEFER_RATE}} of cases recovers {{RECOVERED}} of that
> drop [robustness_sweep -> uncertainty_methods -> operating_point]. We compare four uncertainty signals under an
> identical point predictor and find {{BEST_METHOD}} ranks errors best
> (AURC {{AURC_BEST}} vs. {{AURC_CONF}} for softmax confidence). All degradation
> is synthetic and two clinics serve as held-out folds; we report what that does
> and does not license.

## 1. Introduction

The gap to open with: **reported accuracy is measured under conditions that do
not obtain at deployment**, in two simultaneous ways — the image is degraded, and
the site is unseen. Prior work varies one at a time, at most.

Contributions, stated as what is measured rather than what is built:

1. A leave-one-clinic-out evaluation of low-compute TB screening under simulated
   smartphone re-photography, isolating cross-site shift and capture degradation
   and reporting them jointly.
2. A head-to-head comparison of four uncertainty signals — softmax confidence,
   MC-dropout spread, a degradation-supervised uncertainty head, and deep-ensemble
   disagreement — under a *fixed* point predictor, so the comparison measures
   ranking quality rather than confounded prediction changes.
3. A deferral policy with everything fitted on validation only, plus a
   split-conformal layer whose coverage shortfall on a held-out clinic is itself
   a distribution-free measurement of domain shift.
4. An honest negative-space account: what synthetic degradation cannot establish,
   and a deployment checklist for what a clinic would still need.

## 2. Related work

Three threads, each ending in the specific thing it does not do. Full notes and
citations already in `docs/phase1_framing.md` §1 and §3.

- **Low-compute TB CXR screening.** TB-Net (Wong et al. 2022), LightTBNet
  (Capellán-Martín et al. 2023), Pasa et al. (2019). All report random-split or
  within-cohort k-fold. → *No leave-one-clinic-out.*
- **Phone-captured medical imaging.** CheXphoto (Phadke et al. 2020),
  CheXphotogenic, the npj Digital Medicine recalibration follow-up. Establishes
  the degradation drop and that recalibration partly recovers it. → *CheXpert
  findings, not TB; single source, so no cross-site axis; no deferral.*
- **Calibrated uncertainty and selective classification.** Guo et al. 2017;
  Gal & Ghahramani 2016; Lakshminarayanan et al. 2017; Sensoy et al. 2018;
  Sadinle et al. 2019; Angelopoulos & Bates 2023. → *Rarely evaluated under
  simultaneous covariate shift and input corruption in a deployment-shaped task.*

The one paper doing cross-site TB testing (Tianjin Haihe ensemble, PMC11301748)
reports the drop and stops — no uncertainty, no deferral, no recovery measurement.
That is the closest precedent and the sharpest contrast.

## 3. Method

3.1 **Problem setup.** Binary TB screening; LOCO protocol; why only two folds are
two-class (`docs/LIMITATIONS.md` §3).

3.2 **Smartphone degradation model.** Seven ops with continuous severity in [0,1]
(`data/degradation.py`); the record-what-fired design that yields weak
uncertainty labels; the learned-generator alternative and the ablation protocol
(`data/degradation_learned.py`, `scripts/ablate_degradation.py`). State plainly
that realism is unvalidated against real recaptures.

3.3 **Model.** DenseNet-121 baseline; TB-Net reimplementation at matched
parameter count with its ~2× MACs gap disclosed (`models/tbnet.py`); the
evidential variant (`models/evidential.py`).

3.4 **Uncertainty signals.** The four compared, and the design decision that
makes the comparison meaningful: identical temperature-scaled probabilities,
varying only the ranking signal (`eval/run.py`).

3.5 **Deferral policy and its guarantees.** Threshold tuning on validation;
the report/retake/refer trichotomy; split-conformal (LAC) and the
exchangeability caveat under LOCO (`eval/conformal.py`).

3.6 **Protocol discipline.** One paragraph, because it is a differentiator: the
temperature, the uncertainty→confidence normalisation, the deferral threshold and
the conformal quantile are all fitted on the validation split of *seen* clinics
and applied, never re-fitted, on the held-out clinic; train and eval derive the
identical split by construction (`data.splits.split_from_config`).

## 4. Experiments

| # | Question | Produced by | Placeholder |
|---|---|---|---|
| E1 | In-distribution reference accuracy | random-split run | `{{ACC_ID}}` |
| E2 | LOCO accuracy per held-out clinic | `run_experiments.py` | `{{ACC_LOCO_*}}` |
| E3 | Accuracy vs. degradation severity | `robustness_sweep` | `{{SWEEP}}` |
| E4 | AURC per uncertainty method per severity | `uncertainty_methods` | `{{AURC_*}}` |
| E5 | Calibration: ECE/MCE, reliability diagram | `eval/calibration.py` | `{{ECE}}`, `{{MCE}}` |
| E6 | Safe deferral: coverage/accuracy at tuned T |  `uncertainty_methods.operating_point` | `{{DEFER_RATE}}` |
| E7 | Human-rescue rate | `human_rescue_rate` | `{{RESCUE}}` |
| E8 | Conformal coverage shortfall on held-out clinic | `eval/conformal.py` | `{{SHORTFALL}}` |
| E9 | Per-clinic calibration heatmap (fairness audit) | `eval/crosssite.py` | figure |
| E10 | Brier / BSS / Murphy decomposition | `eval/forecast_verification.py` | `{{BSS}}` |
| E11 | Params / MACs / CPU latency per arch | `benchmark_efficiency.py` | `{{EFFICIENCY}}` |
| E12 | Worst-of-N adversarial degradation | `eval/adversarial_degradation.py` | `{{ADV}}` |

Ablations worth their space: degradation-augmented training vs. not (E3 with and
without); uncertainty-head loss weight; MC-dropout passes T vs. AURC (the
latency/quality trade-off that matters on-device).

## 5. Results

Lead with the gap (E1 vs. E2), then the recovery (E6), then the method comparison
(E4) — that ordering matches the argument rather than the pipeline.

Figures: (a) accuracy vs. severity with per-method deferral curves;
(b) reliability diagram before/after temperature scaling; (c) risk-coverage by
uncertainty method; (d) per-clinic calibration heatmap; (e) generalization-gap
bars. (a)–(c) come out of `eval/run.py`; (d)–(e) out of `run_experiments.py`.

Report per-fold n alongside every accuracy. Montgomery's fold is 138 images;
point estimates there carry wide intervals and should be shown with them.

## 6. Limitations

Import `docs/LIMITATIONS.md` — nine items, written before results exist
specifically so they cannot be trimmed to fit the findings. The three that most
constrain the claim: synthetic degradation only, two held-out folds, and an
uncertainty target that proxies correctness rather than radiologist agreement.

## 7. Deployment considerations

Condense `docs/DEPLOYMENT_CHECKLIST.md`. The point reviewers should take: the
deferral mechanism transfers workload to a human who must actually exist, and a
deferral rate is a staffing decision before it is a hyperparameter.

## 8. Reproducibility

Code, degradation pipeline, and calibration/deferral/conformal tools released
under MIT (`LICENSE`); data not redistributable (NIAID DUA). Every table
regenerable from `scripts/run_experiments.py` plus the config files; seeds fixed
in config; CI runs the smoke test and unit tests on every push.

---

## Pre-submission checks

- [ ] No `{{PLACEHOLDER}}` survives anywhere in the manuscript.
- [ ] Every reported number traceable to a committed `metrics.json` / sweep report.
- [ ] "Four clinics" never used to describe the *holdout* rotation.
- [ ] TB-Net comparisons say "reimplementation" every time.
- [ ] Conformal coverage never stated as a guarantee on a held-out clinic.
- [ ] `temperature_at_bound` false in every run whose numbers are reported.
- [ ] Limitations section not shortened relative to `docs/LIMITATIONS.md`.
