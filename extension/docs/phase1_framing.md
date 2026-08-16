# Phase 1: Background, Framing, and the Delta

This is the Phase 1 deliverable: no code, but every claim here is either cited or
marked as an assumption the team should verify. It sets the target the rest of the
repo is built against. (The paper's own intro/related-work prose is a Phase 5 task
and will read differently from this working doc, but should not contradict it.)

## 1. TB-Net: what it is, and the exact gap we're closing

**TB-Net** (Wong, Lee, Rahmat-Khah, Sabri & Alaref, 2022, *Frontiers in Artificial
Intelligence*; also arXiv:2104.03165) is a self-attention CNN for TB screening from
chest X-rays, produced by a machine-driven "generative synthesis" design search
rather than hand design. Its defining component is the **visual attention
condenser**: a block that condenses an activation map into a compact joint
spatial/cross-channel embedding, derives a self-attention map from that embedding,
and uses it to selectively amplify informative regions of the feature map (the
attention-condenser idea itself traces to the same group's AttendNets/TinySpeech
line of work, Wong et al. 2020). The design search produces a network with high
architectural heterogeneity and light-weight micro-architecture patterns, reported
at roughly **4.24M parameters / 0.42 GMACs at 224x224**, reaching **99.86% test
accuracy** on the (then-)public aggregated Kaggle TB benchmark, with the authors
also releasing GSInquire explainability maps to sanity-check that the network
attends to clinically plausible lung regions rather than shortcuts. The official
code and weights are TensorFlow 1.15 checkpoints
(github.com/darwinai/TuberculosisNet); see the two-path reproduction plan already
written into `src/tbtrust/models/tbnet.py`.

**The number is real, and the setting it was measured in is the whole point of this
project.** 99.86% is a random-split, single-imaging-pipeline number: same
acquisition hardware distribution as training, no cross-site holdout, no
smartphone re-photography artifacts. Nothing about that number tells you what
happens when the same architecture (or any architecture) sees a phone photo of a
film shot on a lightbox in a clinic that contributed zero training images. That
is exactly the setting rural cold-chain-free screening runs in, and it is exactly
what this project measures instead.

**Delta statement** (the one-sentence version to keep every design decision
honest): *TB-Net-style low-compute models report near-perfect accuracy on clean,
single-source X-rays; we evaluate the same class of model leave-one-clinic-out
across four public sources under synthetic smartphone-capture degradation, add a
calibrated uncertainty head trained to recognize when degradation makes a
prediction untrustworthy, and show that a tuned deferral policy on that signal
recovers most of the accuracy lost to domain shift and image quality — instead of
silently returning a wrong answer.*

## 2. Uncertainty quantification survey: what's usable in a small model

The requirement is trustworthy, calibration-focused uncertainty in a model small
enough to run on-device in a clinic with no reliable power or connectivity — so
inference cost is a hard constraint, not a nice-to-have.

| Method | Inference cost | What it estimates | Verdict for this project |
|---|---|---|---|
| **MC-dropout** (Gal & Ghahramani, 2016) | T stochastic forward passes (T~20) through one network | Epistemic, via predictive spread | **Use it.** Free — the baseline already trains dropout, `models/uncertainty.py` implements it. Cheapest possible signal, good default. |
| **Deep ensembles** (Lakshminarayanan et al., 2017) | N full forward passes through N independently-trained networks | Epistemic, generally better calibrated than MC-dropout | **Use as the second point of comparison**, not the deployed model. N models = N x storage/inference cost, which fights the low-compute claim; useful as the "gold standard" baseline the cheap methods are judged against. Packed-ensembles (fewer independent models packed into one wider network) are the practical middle ground if the calibration gap turns out to matter. |
| **Evidential deep learning** (Sensoy, Kaplan & Kandemir, 2018; survey: Gao et al., 2025, arXiv:2409.04720) | One forward pass, one network | A Dirichlet distribution over class probs in a single shot; separates aleatoric/epistemic without sampling | **Use it as the featured method.** One forward pass matches the compute budget, and framing "how much evidence supports this prediction" maps directly onto "how degraded/out-of-distribution is this photo," which is the exact question the deferral policy needs answered. Known failure mode (EDL undertrained on OOD-ish inputs can still look confident) is worth checking empirically against MC-dropout on held-out clinics — that comparison is itself a result. |
| **Conformal prediction** (Vovk et al.; Angelopoulos & Bates 2023 tutorial) | Free at inference; needs a held-out calibration set | Distribution-free coverage guarantees on a prediction set, not a single "confidence" number | **Use for validation, not as the deployed head.** Doesn't require touching the model at all, so it's a cheap way to get a formally guaranteed backstop on the deferral threshold ("with probability >= 1-alpha the kept predictions are correct") on top of whichever score (MC-dropout, evidential) drives the day-to-day decision. Good fit for the "human-rescue rate" analysis in Phase 4. |

Featured pair for the paper: **MC-dropout (cheap reference) + evidential deep
learning (the calibration-focused head)**, cross-checked against a deep-ensemble
upper bound, with conformal prediction used post-hoc to put a coverage guarantee
on the tuned deferral threshold.

**Status: all four are now implemented and wired into evaluation.** MC-dropout
and the learned head are in `models/uncertainty.py`; deep ensembles in
`models/ensemble.py`; evidential deep learning in `models/evidential.py`;
temperature scaling and ECE/MCE in `eval/calibration.py`; split-conformal (LAC)
coverage in `eval/conformal.py`. `eval/run.py` compares whichever signals a given
checkpoint can produce, head to head on one fixed set of temperature-scaled
probabilities, and reports AURC per method at every degradation severity — so
"which to feature" is now an empirical question the pipeline answers rather than
a choice argued on paper. `torch-uncertainty` covers the same ground and is
installable as the optional `[uq]` extra to cross-check these implementations,
but nothing in the project imports it.

Note on the conformal guarantee under LOCO: calibration and test are *not*
exchangeable across clinics, so the 1-alpha guarantee does not formally transfer
to a held-out clinic. `eval/conformal.py` turns that into the measurement rather
than papering over it — it reports guaranteed coverage against coverage actually
achieved on the held-out clinic, and the shortfall is a distribution-free readout
of the domain shift. See that module's docstring.

### Extension: the "second reader" framing

In clinical radiology workflow, a reader who is uncertain about a film calls a
senior colleague for a consensus read rather than signing off alone — a real,
already-normalized escalation path, not a hypothetical one. That maps directly
onto this project's uncertainty head: it is learning **"would a second reader be
called on this case?"** rather than an abstract "how sure am I" score. This gives
an extra evaluation angle beyond calibration curves: do the images the model
defers on look like the images a radiologist would also flag for a second
opinion (heavy artifact, low contrast, ambiguous infiltrate) rather than random
noise? That's exactly what the qualitative error analysis and the human-rescue
rate in Phase 4 are testing, reframed with a clinical vocabulary a reviewer will
recognize immediately.

### Extension: channel-capacity framing

Treat the clean film as the "signal" and smartphone capture (blur, glare, shadow,
angle, compression, downscale) as a noisy channel it passes through. Each
degradation op in `data/degradation.py` already has a continuous severity in
[0, 1], and `DegradationRecord.total_severity` is a scalar summary of how much
noise was injected — a stand-in signal-to-noise proxy per image. Framed this way,
the model's job is to extract the TB signal at whatever SNR the channel handed it,
and the uncertainty head's job is to estimate the *residual channel capacity* — how
much reliable signal is actually left to decide from. This isn't a literal
information-theoretic bound (no attempt is made to compute mutual information
here — that would need paired clean/degraded pairs across a calibrated noise
model, which is out of scope), but it's a useful frame for *why* deferral should
correlate with severity, and it justifies the weak-supervision label in
`manifest.uncertainty_target_from_severity` (monotonic in severity, i.e. monotonic
in "channel SNR") on principled rather than ad hoc grounds. Worth one figure: mean
predicted uncertainty vs. `total_severity`, as a sanity check that the head learned
the intended monotone relationship rather than something spurious.

## 3. Does the degradation gap already have a solved answer?

It's documented, but not solved for this setting. Closest prior work:

- **CheXphoto** (Phadke, Chen et al., Stanford ML Group, arXiv:2007.06199) — 10,000+
  real smartphone photos and synthetic photographic transforms of CheXpert X-rays,
  built specifically to benchmark CXR classifiers under phone-capture artifacts
  (glare, off-axis angle, moire). It shows several CheXpert-trained models take a
  real accuracy hit on photos, though some remain comparable to radiologists. It is
  the closest precedent for the synthetic-degradation pipeline design here.
- **CheXphotogenic** (arXiv:2011.06129) and the follow-up recalibration study in
  *npj Digital Medicine* (Rajpurkar et al. lineage, 2021) go further and show that
  simple recalibration (not full retraining) can partially recover performance on
  photographed CXRs.
- A public **smartphone-captured CXR photograph set** exists on PhysioNet
  (`cxr-phone`), useful as a possible source of *real* re-photography data instead
  of only synthetic degradation, if licensing/access allows pulling a small
  validation slice.

**What none of this does:** none of it targets TB screening specifically (they
evaluate the 14-label CheXpert findings, not a low-compute TB screener), none of it
combines the degradation robustness question with **cross-site domain shift**
(all CheXphoto evaluation is within one source), and — the actual gap this project
fills — **none of it ships a calibrated confidence signal that triggers a
retake-or-refer decision**. They measure "how much does accuracy drop," this
project additionally asks "does the model know when to say so, and does deferring
on that signal actually recover the drop." That combination (degradation +
cross-site + calibrated deferral, specifically for TB) is the delta, and it
survives this literature check.

## 4. Target venue and dataset licensing

### Venue

- **Primary target: ML4H 2026** (Machine Learning for Health Symposium, 6th
  edition), Sydney, Dec 6–7, 2026. Submission deadline **Sept 10, 2026** (per
  ml4h.ahli.cc, confirmed by direct fetch on 2026-07-30; the submission portal was
  listed as "opening soon," so re-check the CFP for track/format specifics — full
  paper vs. findings/workshop track — once it opens). This fits the roadmap's
  Phase 5 finish date of Aug 22, 2026 with about 3 weeks of buffer for review and
  formatting.
- **Backup / secondary target: MIRASOL** (Medical Image Computing in
  Resource-Constrained Settings) workshop at **MICCAI 2026**, Strasbourg. Directly
  on-topic (resource-constrained medical imaging is the workshop's whole premise),
  but its CFP page did not resolve at the URL checked (returned 404) and MICCAI
  workshop deadlines typically land mid-year (roughly June–July in past cycles) —
  **this may already have passed by the time Phase 5 wraps; confirm the actual 2026
  deadline before counting on it as a viable target**, or treat it as a venue for a
  follow-up/extended version instead of the first submission.
- Do not assume NeurIPS-affiliated workshops as a fallback without checking: ML4H
  itself split off from NeurIPS as of recent cycles (now a standalone symposium
  held near, not inside, NeurIPS), so "ML4H is a NeurIPS workshop" is stale
  information as of this writing.

### Dataset licensing (verify before any redistribution, not just before training)

- **Montgomery County + Shenzhen (NLM)**: released by the U.S. National Library of
  Medicine for computer-aided-screening research (Jaeger, Candemir et al., 2014,
  "Two public chest X-ray datasets for computer-aided screening of pulmonary
  diseases," *Quant Imaging Med Surg*). These are the two clean two-class LOCO
  holdouts the whole split design depends on. No machine-readable license file was
  found attached to the raw image download during this check — **the team should
  pull the current terms from the NLM download page directly
  (lhncbc.nlm.nih.gov) before publishing derivative artifacts**, and cite the
  Jaeger et al. paper regardless, since that's the community norm for this dataset
  and is expected by reviewers.
- **RSNA Pneumonia Detection Challenge** (source of the `rsna` normals): hosted on
  Kaggle; RSNA-published challenge data has generally been released under CC BY 4.0
  in past challenges, but the competition's own **Rules** page (not fetched
  successfully during this check — confirm at
  kaggle.com/c/rsna-pneumonia-detection-challenge/rules) may add participation-
  specific obligations beyond the base image license. Confirm before using RSNA
  images in any publicly released derivative (e.g. the degraded-image ablation
  set), not just before training.
- **NIAID TB Portals** (source of `niaid` TB positives): **not freely
  redistributable** — access requires registering and signing a Data Use Agreement
  (DUA) at data.tbportals.niaid.nih.gov / datasharing.tbportals.niaid.nih.gov.
  `scripts/download_data.py` already treats this correctly as a manual,
  credentialed step rather than something to script around.
- **The aggregated Kaggle mirror** (`tawsifurrahman/tuberculosis-tb-chest-xray-
  dataset`, used by `download_data.py --kaggle-aggregated` as the fast start) folds
  NIAID-sourced TB-positive images into a single convenience download. **This is a
  real compliance gap worth flagging rather than quietly relying on**: it is not
  obvious that redistribution through that convenience mirror carries forward the
  NIAID DUA's terms. Treat the aggregated Kaggle set as fine for fast local
  iteration and the Montgomery/Shenzhen-derived LOCO folds (whose provenance is
  clean either way), but do **not** treat images inferred as `niaid`-sourced within
  that mirror as cleared for redistribution in a public artifact (paper figures,
  released degraded-image samples, a public demo) without going through the actual
  TB Portals DUA. This has no effect on the two-class LOCO folds this project
  actually reports (Montgomery/Shenzhen only), since NIAID is excluded from them by
  the `require_two_class_test` guard already — but it does affect what you may
  legally include as a NIAID-attributed example figure in the paper.

## 5. What this means for the rest of the repo

Nothing above required a design change to what's already built — it validates the
choices already made (MC-dropout + evidential head over conformal-as-deployed-
method; Montgomery/Shenzhen as the only reported LOCO folds; degradation severity
as a continuous SNR-like proxy) and gives the paper's related-work section its
citations. The concrete new code this phase implies lives in Phase 3
(`models/ensemble.py`, `models/evidential.py`) and is tracked there, not here.
