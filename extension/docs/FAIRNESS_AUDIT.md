# Fairness and audit

The fairness question this project can actually answer is not the one the word
usually points at. There are no protected-attribute labels in the manifest, so
"is the model fair across demographic groups" is, today, unanswerable here — and
saying so is part of the audit rather than an excuse for skipping it (§5).

What *is* answerable, and what matters most for a screening tool deployed across
sites, is whether the system delivers the same quality of service to every
clinic. Point accuracy hides most of the ways it does not. This document defines
the audit, the protocol for running it, and — written before the numbers exist —
what counts as a finding.

Companion to `docs/DEPLOYMENT_CHECKLIST.md` (what must be true before use) and
`docs/LIMITATIONS.md` (what the evidence does not support). The code is
`eval/crosssite.py`, `eval/calibration.py` and `eval/physics_deferral.py`.

---

## 1. The five axes, in order of how much they matter

| # | Axis | Question | Produced by |
|---|---|---|---|
| A1 | **Per-clinic calibration** | Do the probabilities mean the same thing at every site? | `eval/crosssite.calibration_heatmap_data` |
| A2 | **Sensitivity parity** | Does any site get systematically more missed TB? | `eval/crosssite.per_clinic_table` |
| A3 | **Deferral burden** | Does one site absorb the retakes and referrals? | `physics/triage.py` joined per clinic |
| A4 | **Measurability** | Does the safety valve silently switch off at one site? | `scripts/audit_fiducials.py` |
| A5 | **Demographics** | Everything the word "fairness" usually means | *not available — see §5* |

A1 is the load-bearing one and the rest of this document leads with it.

## 2. A1 — the per-clinic calibration comparison

### Why calibration and not accuracy

A model can hold accuracy at a site while its probabilities drift. That is the
failure mode this whole project is built around, because **every downstream
safety mechanism consumes a probability, not a label**: the deferral threshold,
the conformal quantile, the retake/refer split and the operating point are all
tuned on one distribution of probabilities and applied to another. A site where
the model is 3% less accurate but well calibrated is fine — deferral will catch
the extra errors. A site where accuracy is identical but the model is
systematically overconfident is dangerous, because the cases that should escalate
are exactly the ones it will report confidently, and the threshold tuned
elsewhere will not fire.

So: **accuracy parity is the wrong headline and calibration parity is the right
one.** Hidden domain shift shows up here first.

### The protocol

1. Run the LOCO sweep so that every clinic has served as a held-out fold
   (`scripts/run_experiments.py`). Two folds today — Montgomery and Shenzhen —
   because they are the only two-class sources (`docs/LIMITATIONS.md` §3).
2. For each fold, apply the temperature fitted on the *seen* clinics'
   validation split. Never re-fit on the held-out clinic: re-fitting per site
   measures a different, easier question ("could this be calibrated?") than the
   deployment one ("is it calibrated when it arrives?"). Both are worth
   reporting, and the **gap between them is the per-site recalibration debt** —
   the amount of local data a new clinic would have to contribute before its
   probabilities could be trusted.
3. Compute per clinic: n, accuracy, sensitivity, specificity, ECE, MCE, Brier,
   and the Murphy decomposition (`eval/forecast_verification.py`) — reliability
   is the component that carries the calibration signal, and reporting it
   separately from resolution keeps a well-separated but miscalibrated model
   from hiding behind a good Brier score.
4. Also compute them **per (clinic × severity)**. A model can be calibrated at
   severity 0 and overconfident at severity 0.75, and a site whose photographs
   are systematically worse then inherits a miscalibration the pooled number
   never shows.
5. Render `calibration_heatmap_data` as the figure. Rows = held-out clinic,
   columns = severity, cell = ECE.

### Reading it, with the statistics stated honestly

ECE on 138 images is a noisy statistic. With ten equal-width bins the tail bins
hold single-digit counts and the estimate moves several points on resampling, so:

- use **equal-mass bins**, not equal-width, and report the bin count;
- report a **bootstrap CI** on every per-clinic ECE (1000 resamples, stratified
  by label), never a bare point estimate;
- compare clinics with a **paired bootstrap on the difference**, not by eyeballing
  two intervals — overlapping CIs do not mean the difference is not significant;
- state n next to every cell. A heatmap that puts n = 138 and n = 662 in
  identically-sized squares invites a comparison the data does not support.

### Pre-registered thresholds (decided before the numbers)

- A per-clinic ECE difference whose paired-bootstrap 95% CI excludes zero is
  **reported as a finding**, whatever its sign or size.
- A per-clinic ECE above 0.10 after temperature scaling is reported as
  **"probabilities not usable for deferral at this site"**, and the deferral
  results for that site are reported with that caveat attached in the same
  paragraph, not in a footnote.
- If the recalibration debt (step 2) is large at every site, the honest
  conclusion is that the deployment protocol requires site-local calibration
  data, which is a real operational cost and belongs in §7 of the paper rather
  than in future work.

## 3. A2 — sensitivity parity, not accuracy parity

Screening is asymmetric: a missed TB case seeds onward transmission, a false
positive costs one confirmatory test. So the parity that matters is
**sensitivity across sites at the deployed operating point**, with specificity
reported as what it cost.

- Fix the operating point on sensitivity (`eval.target: sensitivity`), tune it
  on validation, then read specificity per clinic.
- Report sensitivity per clinic with Wilson intervals — on Montgomery's 58
  positives, a point sensitivity is nearly uninformative on its own.
- A site whose sensitivity interval sits below the target band is a finding even
  if its accuracy is the highest in the table, because accuracy there is being
  carried by specificity on the normals.

## 4. A3 and A4 — the two harms that only exist because of deferral

These two have no analogue in a system that always answers, and they are the
audit axes this project adds.

### A3 — deferral burden

Deferral is a workload transfer. If one clinic has an older lightbox, a dimmer
room, or a cheaper phone, its images carry a higher density floor, its
certificates come back INSUFFICIENT more often, and it absorbs more retakes and
more referrals — for the same underlying case mix. The model looks equally good
everywhere; the *service* is worse at the poorer site, and the extra work lands
on the staff least able to absorb it.

Measure it: join `triage_action` from `scripts/physics_certificates.py` to the
manifest and report, per clinic, the retake rate, the refer rate, the median
expected retake gain in dB, and the fraction of retakes that are *repeat*
retakes on the same image. Report it next to the accuracy table, because a
deferral rate is a staffing decision (`docs/DEPLOYMENT_CHECKLIST.md` §B).

Pre-registered: a between-clinic difference in retake rate of more than 10
percentage points is reported as a finding, with the limiting factor
(`limiting_factor` from `floor.py`) that drives it — because a difference
attributable to glare is fixable with a curtain and a difference attributable to
the phone's sensor is not.

### A4 — measurability

`ABSTAIN` is not a neutral verdict. It means the image lacked the fiducials to
measure anything, so the physics safety valve is switched off for that image —
and if abstention concentrates at one clinic (a tighter crop convention, a
different cassette, an operator trained to frame the lung fields), that clinic
is running without the protection the paper claims for the method.

Measure it: per-clinic coverage grade and abstain rate from
`scripts/audit_fiducials.py` and the certificate table. Report it as a fairness
number, not only as the methods gate it also is. A method that protects the
well-equipped sites and abstains at the under-equipped ones has an equity
problem regardless of its average performance, and the mitigation is cheap and
known — train operators to photograph the whole sheet, including the pale
margin, the lead marker and the dark rim (`docs/DEPLOYMENT_CHECKLIST.md` §B2).

## 5. A5 — the demographic axes, and which of them are reachable

**Not in the manifest today.** `build_manifest.py` records `path`, `clinic`,
`label` and `split`. No demographic attribute reaches the model or the
evaluation, so no demographic fairness claim can be made from what is currently
committed. Any statement of the form "the model is fair across X" would be
unsupported.

What is reachable with work, and what is not:

- **Sex and age** are recoverable for Montgomery and Shenzhen: the NLM releases
  ship per-image clinical readings alongside the images, carrying patient sex and
  age. Joining them into the manifest is a contained task and would enable a
  genuine subgroup analysis on the two folds that actually report. **This is the
  single highest-value fairness item available and it needs no new data.**
- **HIV status is not available and matters more than either.** HIV-associated
  pulmonary TB frequently presents atypically on chest radiography — lower rates
  of cavitation, more lymphadenopathy and lower-zone or diffuse patterns, and a
  substantial fraction of radiographs read as normal. A screener developed on
  cohorts that are predominantly HIV-negative can therefore underperform exactly
  where TB burden is highest, and nothing in this evaluation would detect it.
  This belongs in the paper's limitations as a named, directional risk rather
  than as a generic "future work on fairness" sentence. *(Verify and cite before
  publication; stated here from the clinical literature, not measured in this
  project.)*
- **Race, ethnicity, comorbidity, nutritional status, prior TB**: absent, and not
  recoverable from these cohorts.
- **Device and site are the proxies we do have.** Clinic is confounded with
  machine, population and capture convention all at once, so a per-clinic
  difference cannot be attributed to any one of them. `scripts/clinic_stats.py`
  measures the machine axis (resolution, brightness, contrast) so at least that
  component is a number rather than an assumption.

## 6. The audit, as a checklist

Run after every LOCO sweep; nothing here needs a GPU once the sweep exists.

- [ ] Per-clinic table (n, accuracy, sensitivity, specificity, ECE, MCE, Brier,
      reliability) at every severity, with bootstrap CIs.
- [ ] Calibration heatmap, clinic × severity, with n printed in every cell.
- [ ] Recalibration debt: ECE with the transferred temperature minus ECE with a
      site-local temperature.
- [ ] Sensitivity per clinic at the deployed operating point, with Wilson
      intervals against the target.
- [ ] Retake rate, refer rate and median expected gain per clinic, with the
      limiting factor behind any difference.
- [ ] Certificate abstain rate and fiducial coverage grade per clinic.
- [ ] Conformal coverage shortfall per clinic (`eval/conformal.py`) — under LOCO
      this is a measurement of shift, never a guarantee.
- [ ] Sex and age subgroup analysis if the clinical-readings join has been done;
      an explicit "not available" line if it has not.
- [ ] Every finding above reported whatever its sign, per §2's pre-registration.

## 7. What this audit cannot establish

- That the system is fair to any demographic group (§5).
- That equal calibration across two clinics implies equal calibration at a third.
  Two folds do not characterise a distribution over sites.
- That a site with a good audit is safe to deploy in. The audit is retrospective
  on curated data; deployment adds prospective case mix, prevalence, and an
  operator whose behaviour changes once the tool is in the room.
- That deferral improves outcomes. It measures who bears the deferral cost, not
  whether bearing it helps the patient.
