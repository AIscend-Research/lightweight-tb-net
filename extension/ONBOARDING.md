# Onboarding: where the work lives

Read this after you've run `python scripts/smoke_test.py` and seen it pass. It
maps the project roadmap onto files and suggests how to divide the work so people
aren't editing the same modules.

## First 30 minutes for everyone

1. `pip install -e .` and run `python scripts/smoke_test.py` (all PASS).
2. Skim `data/degradation.py`, `data/splits.py`, and `eval/deferral.py`. These
   three are the spine of the project; almost everything plugs into them.
3. Read the two "facts to read first" in the README (TB-Net is TF1; the dataset
   has single-class sources). They shape most design decisions.
4. Pick a track below.

## Phase -> files

**Phase 1, framing and the delta.** Done — see `docs/phase1_framing.md` for the
TB-Net summary, the UQ method survey (MC-dropout + evidential deep learning
featured, deep ensembles as the calibration upper bound, conformal prediction as
a post-hoc coverage guarantee on the deferral threshold), the second-reader and
channel-capacity framings, the phone-captured-CXR literature check (CheXphoto et
al. — the gap this project fills is real), and the target-venue/dataset-licensing
notes (ML4H 2026, Sept 10 deadline; NIAID DUA compliance caveat on the aggregated
Kaggle mirror). The code already encodes the claim: robustness + calibrated
uncertainty + safe deferral, measured leave-one-clinic-out. All of the featured
UQ machinery is implemented here — MC-dropout (`models/uncertainty.py`), deep
ensembles (`models/ensemble.py`), evidential deep learning
(`models/evidential.py`), temperature scaling and ECE/MCE (`eval/calibration.py`),
split-conformal coverage (`eval/conformal.py`), and the risk-coverage/AURC
selective-classification metrics (`eval/deferral.py`). `torch-uncertainty` covers
the same ground and is available as the optional `[uq]` extra for cross-checking,
but nothing in the project imports it.

**Phase 2, data and preprocessing.** `scripts/download_data.py`, `data/manifest.py`,
`data/splits.py`, `data/degradation.py`. The degradation pipeline and the manifest
+ split logic are already working. Still open: point the download script and
provenance rules at however you actually lay out `data/raw` and confirm the
per-clinic class balance with `build_manifest.py` once real images are in place --
that step needs the actual data and can't be done from this scaffold alone. The
Phase 2 extension is done in code: `data/degradation_learned.py` is an unpaired
adversarial (small generator + PatchGAN discriminator) learned degrader behind the
same call convention as the physics pipeline, and `scripts/ablate_degradation.py`
scores any degradation strategy against real recaptures (or against each other, if
you don't have real recaptures yet) using no-reference image features and a
classifier-separability score. What's still missing is the real recapture set
itself to make that ablation meaningful rather than just wired-up --
`data/real_recapture/README.md` has the pilot collection protocol and a
stopgap (existing public phone-photo CXR datasets) if physical capture isn't
feasible. The severity is already continuous, which is the "fine-grained
severity" extension.

`scripts/clinic_stats.py` (from the `xctb` prototype) is the other Phase 2
deliverable worth running the day real images land: it samples each clinic and
reports median resolution, mean brightness and mean contrast. That table is the
evidence that the clinics really are different imaging conditions, and it belongs
in the paper next to the per-clinic generalization gaps -- otherwise "cross-site
domain shift" is an assumption rather than a measurement.

**Phase 3, model development.** `models/baseline.py`, `models/tbnet.py`,
`models/uncertainty.py`, `models/evidential.py`, `models/ensemble.py`,
`train/loop.py`. The baseline trains today, and `arch: baseline | tbnet |
evidential` all drop into the same `tbtrust-train`/`tbtrust-eval` path (the
evidential loss is dispatched automatically off the model's output dict, see
`train/loop.py`). What's done: (a) TB-Net's default widths are now tuned to its
reported ~4.24M params (pinned by a test); `scripts/benchmark_efficiency.py`
measured the honest gap that's left -- MACs are ~2x TB-Net's at matched params,
a real architecture-efficiency difference documented in `tbnet.py`, not yet
closed. (b) MC-dropout (`uncertainty.py`), evidential deep learning
(`evidential.py`, the featured calibration-focused head per
`docs/phase1_framing.md`), and deep ensembles (`ensemble.py`, via
`scripts/train_ensemble.py`) are all implemented and comparable head-to-head once
training runs happen. (c) the deferral threshold is tuned via
`eval/deferral.tune_threshold`; `eval/sequential_deferral.py` adds two
surveillance-literature extensions on top of the point threshold -- adaptive-
stopping MC-dropout (spend fewer passes on easy cases) and a CUSUM chart for
detecting sustained per-clinic drift in confidence over a stream of images, not
just per-image. `eval/adversarial_degradation.py` (run via
`scripts/evaluate_adversarial_robustness.py`) is the parallel-track extension:
worst-of-N-query black-box degradation search, checking whether uncertainty
actually spikes on the cases the search makes harder.

(d) **Domain generalization**, merged in from the parallel `xctb` prototype:
`losses/dg.py` (Deep CORAL, IRM), `models/grl.py` (gradient reversal for DANN)
and `models/clinic_film.py` (clinic-conditional FiLM, the cross-field extension).
Set `dg.method: coral | dann | irm` or `model.clinic_film: true` in any config;
`configs/loco_montgomery_{coral,dann,film}.yaml` are the ready-made ablations
against `configs/loco_montgomery.yaml`. This is the other half of the cross-site
story and it is orthogonal to everything else: the DG objectives change what the
backbone learns, the uncertainty methods change how confidence is read off it, so
any `dg.method` combines with any uncertainty method in `eval/run.py`. The one
constraint is `model.arch: baseline` -- TBNet and the evidential head expose no
pooled features or domain head, and `train/loop.py` says so rather than failing
on a missing dict key. The open problem worth a contribution slot is in
`clinic_film.py`: under LOCO the held-out clinic has no learned embedding, so
every test image currently falls back to the mean training clinic.

While validating this phase: a pre-existing bug in `data/dataset.py`
(`uniform_severity`/`constant_severity` returned unpicklable lambdas, silently
fine on Linux/fork but broken with `num_workers > 0` under spawn-based
multiprocessing -- macOS/Windows always, POSIX too as of Python 3.14) is now
fixed. Worth knowing if a training run mysteriously worked on Kaggle but not on
someone's laptop.

**Phase 4, evaluation and extensions.** `eval/calibration.py`, `eval/deferral.py`,
`eval/crosssite.py`, `eval/run.py`, `eval/forecast_verification.py`,
`scripts/run_experiments.py`. The two core novel metrics are implemented: Safe
Deferral Rate (risk-coverage) and Uncertainty Calibration (reliability diagrams,
ECE/MCE). `eval/run.py` sweeps accuracy across degradation severity, and now
also: tunes the deferral threshold on the **val** split and only applies it to
test (a real leak was found and fixed here -- it used to tune and report on the
same test split), reports `human_rescue_rate` and a Murphy decomposition at the
primary severity, and adds Brier Skill Score to every severity in the sweep.
`eval/forecast_verification.py` is the meteorology extension: Murphy (1973)
decomposition (BS = reliability - resolution + uncertainty, reconstruction
pinned by a test) and Brier Skill Score against the base-rate reference forecast
-- Ranked Probability Score is deliberately *not* separately implemented, since
it's mathematically identical to Brier score for a two-class problem and adding
it would misrepresent that as a new metric. `scripts/run_experiments.py` is the
piece that was missing before: it actually runs (or aggregates already-trained)
the LOCO folds into `eval/crosssite.py`'s generalization-gap table and per-clinic
calibration heatmap, plus forecast-verification metrics per clinic -- the manual
bash loop below only trains+evaluates each fold in isolation, this ties them
together into the actual cross-site comparison.

**The physics track, cutting across Phases 2-4.** `physics/`, `docs/PHYSICS.md`,
`eval/physics_deferral.py`, notebooks `05`-`08`. This is the second, independent
uncertainty track and it shares almost no code with the learned one, so it can be
picked up in parallel by someone who wants a self-contained piece.

The idea in one line: a chest film carries its own calibration targets -- a lead
marker at known base+fog density, a direct-exposure region that is optically black,
a collimation border that is a hard step edge -- so the phone's tone curve, PSF and
veiling glare can all be *measured* from a single photograph rather than learned.
That gives a per-pixel **density resolution floor**, and comparing it to a TB
finding's contrast gives a certificate that the information is or is not in the
image, in units of optical density, with no labels and no network.

Read the modules in dependency order: `density.py` (the sign convention, which the
project brief has inverted -- start here), then `film.py` (the forward model, which
is what makes everything checkable), `fiducials.py`, then `psf.py` / `glare.py` /
`tone.py`, then `invert.py`, `floor.py`, `certificate.py`.

Three things a newcomer should know before changing anything:

* **Run `scripts/audit_fiducials.py` first.** The whole track assumes the public
  archives kept the fiducials. That is an empirical question, it is cheap to
  answer, and the answer bounds every claim.
* **`scripts/validate_physics.py` is the arbiter, not the tests.** The tests pin
  ordering properties and known regressions; the validation script measures whether
  the bound actually predicts an optimal detector's threshold. If you change an
  estimator, that ratio is the number that says whether you improved it.
* **The finding contrasts in `findings.py` are nominal placeholders** marked
  `source="NOMINAL"`. Relative results are sound; absolute verdicts inherit their
  uncertainty. Replacing that table is a genuine, self-contained contribution —
  start from `physics/findings_template.yaml`, which is the same table with the
  values blanked, the transcription recipe (including the film-gradient
  conversion that is easy to drop) and the checks to run afterwards.

**Phase 5, writing.** The results tables and figures come straight out of
`eval/run.py` (metrics.json + the reliability/deferral figure) and
`scripts/make_figures.py`. `docs/PAPER_DRAFT.md` is the draft: full prose with
every unmeasured value as a `{{PLACEHOLDER}}` naming the artifact that must
produce it, so filling it in is mechanical and no invented number can survive.
`docs/FAIRNESS_AUDIT.md` is the audit that goes with it — per-clinic
calibration, sensitivity parity, deferral burden and certificate measurability —
and `docs/DEPLOYMENT_CHECKLIST.md` is what would have to be true before any
clinical use. Write the limitations honestly: everything is synthetic
degradation with no real phone captures yet, the uncertainty target is a proxy
for correctness rather than measured radiologist agreement, and deferral assumes
trained staff are on hand.

## A way to split it across ~4 people

The modules are deliberately decoupled so these tracks rarely touch the same file.

- **Data owner.** Phase 2 end to end. Owns `data/`. Deliverable: a committed
  `manifest.csv` with clean provenance and a class-balance report, plus the
  degradation ablation. Everyone else is blocked on the manifest, so this goes first.
- **Model / TB-Net owner.** `models/tbnet.py` reproduction and the baseline vs.
  TB-Net comparison. Owns `models/` and `train/loop.py`.
- **Uncertainty owner.** MC-dropout, ensembles, evidential, calibration, and the
  deferral policy. Owns the uncertainty parts of `models/` and `eval/calibration.py`
  + `eval/deferral.py`. Coordinates with the model owner on the shared model interface.
- **Evaluation / analysis owner.** Cross-site analysis, the figures, and the
  results tables. Owns `eval/crosssite.py` and `eval/run.py`, and drives the
  leave-one-clinic-out rotation (run each fold, collect metrics.json, build the
  clinic-pair heatmaps).

- **Domain-generalization owner** (the fifth track, if there is a fifth person;
  otherwise it folds into the model owner). CORAL / DANN / IRM / FiLM, the
  ablation against the plain baseline, and the `clinic_film.py` unseen-clinic
  problem. Owns `losses/dg.py`, `models/grl.py`, `models/clinic_film.py` and the
  `_dg_loss` function in `train/loop.py`.

The shared contract that keeps everyone unblocked: models return
`{"logit": ..., "features": ..., "uncertainty": ...}`, the manifest columns are
fixed, the Dataset yields `image / label / uncertainty_target / severity /
clinic_idx`, and the `split` column is `train`/`val`/`test`. Change those only by
agreement.

## Running the full leave-one-clinic-out sweep

```bash
for clinic in montgomery shenzhen; do
  tbtrust-train --config configs/loco_${clinic}.yaml
  tbtrust-eval  --config configs/loco_${clinic}.yaml \
                --checkpoint outputs/${clinic}/best.ckpt
done
```

Each run writes `outputs/<clinic>/metrics.json` and a reliability + deferral
figure. `eval/crosssite.py` turns the collected results into the generalization-gap
bars and the per-clinic calibration matrix.

To measure how much of that gap domain generalization closes, run the same fold
with a DG config and diff the gaps — same data, same split, same seed, so the
difference is the method's doing:

```bash
tbtrust-train --config configs/loco_montgomery_coral.yaml train.output_dir=outputs/coral
tbtrust-eval  --config configs/loco_montgomery_coral.yaml \
              --checkpoint outputs/coral/montgomery/best.ckpt
```

## Handy commands

```bash
python scripts/smoke_test.py            # verify the wiring
pytest -q                               # the CI checks (install dev extras first)
tbtrust-train --config <cfg> [k=v ...]  # train
tbtrust-eval  --config <cfg> --checkpoint <ckpt>
```
