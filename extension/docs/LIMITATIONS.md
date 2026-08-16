# Limitations

Written proactively, per the Phase 5 plan, so the paper's limitations section is
drafted before results exist and cannot be quietly trimmed to fit a nicer story.
Each item says what the limitation is, why it matters, and what would close it.

## 1. No result in this repository has been produced by a real training run

Every number currently reachable — accuracy, ECE, AURC, coverage — comes from
synthetic data in `scripts/smoke_test.py` or from untrained/2-epoch checkpoints.
The pipeline is verified end to end; the *findings* are not. Nothing in this repo
should be quoted as an empirical claim until `scripts/run_experiments.py` has been
run on the real cohorts on a GPU.

**Closes when:** the LOCO sweep runs on real data and `outputs/loco_sweep/` is
populated.

## 2. All degradation is synthetic; there are no real phone recaptures

`data/degradation.py` is a physics-style simulation and
`data/degradation_learned.py` is an unpaired learned generator. Neither has been
validated against actual photographs of actual films, because no such set exists
here. `scripts/ablate_degradation.py` is built to score them against real
recaptures and currently can only score them against each other, which cannot
establish realism — only disagreement.

This is the single largest threat to external validity. A model robust to *our*
blur/glare model is not thereby robust to a Redmi 9A in a Ugandan clinic.

**Closes when:** the pilot protocol in `data/real_recapture/README.md` is
executed (20–30 images is enough), or a public real-recapture set (CheXphoto real
subset, PhysioNet `cxr-phone`) is licensed and used as the reference distribution.

## 3. Two held-out clinics, not four

Four sources contribute data, but NIAID and Belarus are TB-only and RSNA is
normal-only, so only Montgomery (138 images) and Shenzhen (662) can serve as
two-class holdout folds. `data/splits.py` refuses the others by default.

Consequences: the cross-site claim rests on two rotations, Montgomery's fold has
138 test images so its confidence intervals are wide, and "leave-one-clinic-out
across four clinics" would be a misstatement. Report per-fold n and interval
estimates, not point accuracies alone.

**Closes when:** balanced multi-class cohorts are constructed (mixing RSNA
normals with NIAID positives under the DUA), or additional two-class sources are
added.

## 4. The uncertainty target is a proxy for correctness, not radiologist agreement

`manifest.uncertainty_target_from_severity` derives the weak supervision label
monotonically from applied degradation severity. That encodes "a degraded image
deserves low confidence," which is a reasonable prior and is what the
channel-capacity framing argues for — but it is not the same as "a radiologist
would want a second read here." The second-reader framing in
`docs/phase1_framing.md` is a motivating analogy, not a measured quantity. No
clinician has labelled anything in this project.

**Closes when:** a radiologist rates a sample of held-out images for
"would you seek a second opinion," and that is correlated against predicted
uncertainty.

## 5. Deferral assumes a human is available to defer *to*

The entire safety argument is that low-confidence cases go to a person. In the
rural, cold-chain-free settings this project targets, that person may not exist,
may not be reachable offline, or may be the same overloaded health worker who
took the photo. A high deferral rate is not free: it is a workload transfer, and
at some rate it makes the system useless rather than safe.

`human_rescue_rate` measures whether deferred cases were *worth* deferring, which
is the right question, but it cannot measure whether the referral pathway exists.

**Closes when:** deferral rate is budgeted against a stated staffing assumption —
see `docs/DEPLOYMENT_CHECKLIST.md`.

## 6. The conformal guarantee does not formally transfer across clinics

Split-conformal coverage assumes calibration and test data are exchangeable.
Under LOCO they are not, by construction. `eval/conformal.py` reports the
shortfall between guaranteed and achieved coverage as a diagnostic rather than
claiming the guarantee holds on the held-out clinic. Do not state a 1-α guarantee
on a new site.

**Closes when:** it does not — this is inherent. It can only be reported honestly.

## 7. TB-Net is reimplemented, not reproduced

`models/tbnet.py` is reproduction path B: an attention-condenser-flavoured CNN
tuned to TB-Net's reported ~4.24M parameters, not a port of the released
TensorFlow 1.15 checkpoint. Its MACs are roughly 2× TB-Net's at matched
parameters, documented in the module. Any comparison to TB-Net's published 99.86%
is a comparison to a *reimplementation*, and must say so.

**Closes when:** path A (porting the TF checkpoint layer by layer and verifying
activations on a fixed input) is done, or the comparison is dropped.

## 8. Retrospective, public, curated data

All cohorts are curated research datasets of already-diagnosed patients. They
under-represent the ambiguous, early-stage, and comorbid presentations that
dominate real screening, and their normals are not drawn from a screening
population. Reported sensitivity/specificity will be optimistic relative to
field deployment regardless of how carefully the splits are done.

## 9. Not a clinical device

This is a research tool. It is not validated, not regulated, not approved, and
not for diagnosis. See `docs/DEPLOYMENT_CHECKLIST.md` for what would have to be
true before any clinical use.
