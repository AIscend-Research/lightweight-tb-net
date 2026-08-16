# Deployment checklist

What a clinic would actually need before using this model on a real patient.
Borrowed in structure from clinical implementation science (and from the
reporting discipline of CONSORT-AI / DECIDE-AI / TRIPOD+AI), narrowed to the
specific failure modes this project studies.

**Current status: nothing below is satisfied. This is a research tool, not a
clinical device, and it is not for diagnosis.** The list is here so the gap is
explicit rather than implied.

---

## A. Before the model is allowed to output anything

- [ ] **Reported on a site that contributed no training data.** A LOCO fold, not
      a random split. Random-split accuracy on pooled data (TB-Net's 99.86%) does
      not predict behaviour at a new clinic and must not be quoted as if it does.
- [ ] **Sensitivity target set by public-health need, not by F1.** A missed TB
      case seeds onward transmission; a false positive costs one confirmatory
      test. Fix the operating point on sensitivity, then report the specificity
      that buys it — `eval.target: sensitivity` in the config.
- [ ] **Calibration measured, not assumed.** ECE/MCE and a reliability diagram on
      the deployment site's own data. An overconfident model makes deferral
      actively dangerous, because the cases it should escalate are exactly the
      ones it will report confidently.
- [ ] **Temperature fitted on site-local validation data**, not inherited from the
      development site. If the fitted temperature lands on a search bound,
      `metrics.json` flags `temperature_at_bound` — stop and investigate.
- [ ] **Prevalence documented.** Every probability output is conditioned on the
      training prevalence. A screening population at 0.5% TB is not the ~50% of
      Montgomery/Shenzhen, and predictive values shift accordingly even with
      identical sensitivity and specificity.

## B. The deferral pathway must exist before deferral is switched on

- [ ] **A named person receives deferred cases**, with a defined response time.
      "Refer to specialist" is not a feature unless there is a specialist.
- [ ] **Deferral rate budgeted against staffing.** Decide the maximum fraction of
      cases the site can absorb, then set the threshold to that coverage — not the
      reverse. `eval.min_coverage` encodes this constraint.
- [ ] **"Retake the photo" and "refer to a specialist" are distinguished.** They
      are different actions with different costs. Degradation-driven uncertainty
      should trigger a retake; genuine diagnostic ambiguity should trigger a
      referral. `eval/conformal.py` separates these as `no_plausible_label` vs.
      `ambiguous`, and `physics/triage.py` decides between them from the measured
      capture channel rather than from a confidence score — a high density floor
      caused by localized glare is operator-fixable, an adequate floor with a
      still-uncertain model is not.
- [ ] **The retake instruction says what to change.** "The image is bad" wastes
      the retake. `physics/triage.py` emits the specific fix — which way the glare
      is, whether the blur is shake or defocus, whether the exposure is crushed —
      from `glare.hotspot` and `psf.PSFEstimate.anisotropy`.
- [ ] **A retake actually helps.** If the failure is a cracked lightbox or a
      broken camera, retaking produces another bad photo. Cap retakes per image,
      escalate on the cap, and log the repeat-retake rate — it is the leading
      indicator of a hardware failure at the site, and it is also a fairness
      number (`docs/FAIRNESS_AUDIT.md` §A3).
- [ ] **Deferral is not silently disabled** when the queue gets long. This is the
      predictable failure mode under workload pressure, and it converts the system
      from safe to dangerous without any code change. Whatever the mechanism —
      config flag, threshold override, a "skip" button — it must be logged and
      visible to someone other than the operator.
- [ ] **The deferral budget is agreed with the people who absorb it**, and the
      per-site burden is measured rather than assumed equal. A site with a dimmer
      lightbox absorbs more retakes for the same case mix.

## B2. Three cheap changes at the lightbox, worth more than any model change

The first two turn an assumption in `physics/` into a measurement; the third
protects the measurements already available. None costs more than pennies and a
morning's training, and together they remove the two weakest assumptions in the
physics track — which is a better return than any modelling change on offer.

- [ ] **Tape a small step wedge beside the film.** A few pence of exposed,
      processed film with two or three known optical densities. With only the two
      anchors the film itself provides, the tone curve's exponent γ has to be taken
      from an sRGB prior and its uncertainty propagated into every absolute density.
      A third distinct density breaks that degeneracy and `physics/tone.fit_tone`
      fits γ properly. This is the single weakest assumption in the physics track
      and the cheapest one to remove.
- [ ] **Put a ruler, or any object of known length, in the frame.** `px_per_mm` is
      otherwise inferred from the detected collimation field against a standard
      cassette diagonal, at roughly ±20%. That error propagates into every
      finding's spatial frequency and so into the density floor. A ruler settles it
      exactly; pass the measured value to `physics.invert(px_per_mm=...)`.
- [ ] **Train operators to photograph the *whole sheet*.** Including the pale
      unexposed margin, the L/R lead marker and the dark direct-exposure rim. Those
      three regions are the entire basis of the physics track, and a tight crop to
      the lung fields destroys all of them — after which the certificate can only
      abstain. `scripts/audit_fiducials.py` measures how often this is being done.

## B3. Before the certificate is allowed to gate anything

The physics track has its own prerequisites, and they are separate from the
model's because it fails in a different way: it does not become wrong, it becomes
silent (`ABSTAIN`) or optimistic.

- [ ] **Fiducial coverage measured at this site**, not inherited from the
      development corpus. `scripts/audit_fiducials.py` on a sample of the site's
      own captures. An abstain rate above the level agreed here means the
      certificate is not protecting this site and the deferral policy must not be
      described to its staff as if it were.
- [ ] **The calibration ratio is above 1** in the most recent
      `scripts/validate_physics.py` run whose numbers are being relied on.
      Predicted floor over empirical threshold below 1 means the certificate
      passes photographs an optimal detector cannot read; it does not ship in that
      state, and CI already exits non-zero when it drifts outside the tolerance
      band.
- [ ] **The finding-contrast table's provenance is stated wherever a verdict is
      shown.** While `physics/findings.py` still reads `source="NOMINAL"`, every
      absolute DETECTABLE / INSUFFICIENT verdict inherits a placeholder's
      uncertainty. Relative comparisons between two photographs are unaffected.
      The certificate's provenance block prints this; do not suppress it in a UI.
- [ ] **The certificate is not the only gate.** It bounds the *measurement
      channel*, not diagnostic difficulty — anatomical clutter is in the original
      film too and no retake fixes it. A photograph can clear the floor and still
      be a case that needs a specialist.
- [ ] **Interior glare is understood to be under-measured.** The beam stop is an
      annulus, so the veil is measured at the periphery and interpolated across
      the middle; a dim central reflection is under-reported and the certificate
      is optimistic there. Until a second probe exists, a curtain or a shaded
      lightbox is the mitigation, and it is free.

## C. Monitoring, once live

- [ ] **Per-site drift monitoring running.** `eval/sequential_deferral.CUSUMMonitor`
      over the stream of per-image uncertainty, with a defined alarm response.
      Cameras degrade, lightboxes get replaced, staff change.
- [ ] **Input-quality distribution logged** and compared against the development
      distribution. If real capture quality falls outside the severity range the
      model trained on, the uncertainty estimates are extrapolating.
- [ ] **Deferred-case outcomes fed back.** Without knowing what the human decided,
      `human_rescue_rate` cannot be measured in production and the deferral
      threshold cannot be re-tuned.
- [ ] **A defined trigger for taking the model offline**, agreed in advance and
      owned by someone other than the person operating it.
- [ ] **The fairness audit re-run on live data at a defined interval**, not only
      once at development time. `docs/FAIRNESS_AUDIT.md` §6 is the checklist; the
      per-clinic calibration comparison is the item that catches hidden domain
      shift before point accuracy does.

## D. Governance and data

- [ ] Regulatory pathway identified for the jurisdiction of use. Nothing in this
      repository has regulatory clearance anywhere.
- [ ] Local ethics/IRB approval for prospective use, and a patient-consent
      position for images captured on personal phones.
- [ ] Image handling defined: a chest X-ray photograph on a clinician's phone is
      identifiable health data. Storage, transmission, retention, and deletion
      need answers before the first capture, not after.
- [ ] Dataset licences honoured for anything published — in particular the NIAID
      TB Portals DUA, which the aggregated Kaggle mirror does not visibly carry
      forward. See `LICENSE` and `docs/phase1_framing.md` section 4.
- [ ] **Failure modes disclosed to users in the interface**, including that the
      model was developed on synthetic degradation and validated on two held-out
      clinics of 138 and 662 images.
- [ ] **Equity of service audited, not just average accuracy.** Per-clinic
      calibration, sensitivity parity, deferral burden and certificate
      measurability — `docs/FAIRNESS_AUDIT.md`. A system that performs equally on
      average while sending twice the retake requests to the poorest-equipped site
      is not delivering the same service there.
- [ ] **No demographic fairness claim is made** while the manifest carries no
      demographic attribute. Sex and age are recoverable for Montgomery and
      Shenzhen from the NLM clinical readings and that join is worth doing; HIV
      status, which changes TB's radiographic presentation, is not available in
      any of these cohorts and its absence is disclosed rather than glossed.

## E. Evidence that does not exist yet

Listed separately because these are not checkboxes a careful engineer can tick —
they require studies nobody has run:

- Prospective evaluation at the deployment site.
- Agreement between model uncertainty and radiologist-flagged ambiguity
  (`docs/LIMITATIONS.md` §4).
- Evidence that the deferral workflow improves patient outcomes rather than
  merely improving retained-set accuracy. A model can post excellent selective
  accuracy while making a clinic slower and no more accurate overall.
- Any validation against a **real phone photograph of a real film**. Everything
  in the physics track is currently validated against a forward model written by
  the same authors, which is a weaker claim than it looks
  (`data/real_recapture/README.md` has the protocol and no data).
- Performance in a high-HIV-prevalence screening population, where TB's
  radiographic presentation differs and where the burden is highest
  (`docs/FAIRNESS_AUDIT.md` §5).
