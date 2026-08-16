# Degradation pipeline: design and open questions

Phase 2 asked for a synthetic smartphone-degradation pipeline, a tag on every
image recording clinic / TB-status / degradation type and severity, and
weak-supervision labels for an uncertainty head. This is what got built, what
changed from the original plan, and what is still open.

## The pipeline

`src/tbtrust/data/degradation.py` implements the capture artifacts at
continuous severity (`severity` in `[0, 1]`, `0` is always a no-op):

- `motion_blur`, `defocus_blur` — loss of sharpness, two different causes
- `glare` — a bright specular blob, like a lightbox reflection
- `shadow` — a smooth directional darkening, non-uniform lighting
- `capture_angle` — a handheld off-angle shot
- `jpeg_compression` — re-encoding artifacts
- `downscale` — a low-resolution capture of a large film

It is PIL + numpy only (no scipy/cv2), so it stays importable with no GPU and no
compiled CV stack — that constraint is why `pyproject.toml` does not depend on
albumentations or opencv. `SmartphoneDegradation` composes the kinds at an
overall severity, jitters each kind around the target so a composed image is not
identically scaled everywhere, and returns a `DegradationRecord` of exactly which
per-kind severities were used. Nothing is a black box.

Degradation is applied **on the fly at load time** by `data/dataset.py`, not
written to disk: no doubled dataset, and the severity sweep in `eval/run.py` is
just the same loader re-instantiated at each severity. `TBDataset` derives its
per-image RNG from `(seed, epoch, index)`, so training re-randomises artifacts
every epoch while an eval run is reproducible.

## The strategy-comparison ablation

The roadmap's extension idea was to compare degradation-simulation strategies
(hand-specified blur+noise vs. a learned phone-camera model vs. actual
re-photography of printed films) and treat the comparison itself as a
contribution. All three slots exist:

- **physics** — `data/degradation.py`, above. Implemented.
- **learned** — `data/degradation_learned.py`, a small unpaired adversarial
  generator (`Generator` + `PatchDiscriminator`, `train_learned_degradation`).
  Implemented, untrained: it needs a set of real phone photos to be adversarial
  *against*.
- **real** — actual re-photographed films. Not collected.
  `data/real_recapture/README.md` and `manifest_template.csv` specify exactly
  what to capture and how to record it.

`scripts/ablate_degradation.py` is the ablation. It does not ask which strategy
looks nicer; it trains a small classifier on no-reference quality features (blur,
brightness, contrast, edge density) to tell each synthetic source apart from
`real`. Accuracy near 0.5 means the synthetic images are statistically close to
real captures; near 1.0 means they are easy to spot as fake. Run without
`--real-dir` it still reports how separable `physics` and `learned` are from each
other, which confirms they produce meaningfully different artifacts and tells you
where to point once real recaptures exist.

**Getting even 20–30 real re-photographed films is the highest-value open item in
Phase 2.** It converts the ablation from a synthetic-vs-synthetic comparison into
the realism claim the paper actually wants to make.

## The weak uncertainty label, and where it is used twice

`manifest.uncertainty_target_from_severity` maps severity to a soft "being unsure
here is correct" target. It is used two ways, deliberately:

1. **As a training target.** The baseline classifier's uncertainty head regresses
   to it (`train/loop.py`, `train.uncertainty_loss_weight`). This is the roadmap's
   original weak-supervision plan.
2. **As a validation check.** `eval/degradation_uncertainty.py` correlates
   (Spearman) degradation severity against whatever uncertainty a model actually
   produces, and `eval/run.py` reports this per uncertainty method across the
   severity sweep as `uncertainty_vs_severity`.

The second use is what makes the comparison fair. MC-dropout spread, deep-ensemble
disagreement and evidential vacuity are all label-free — none of them is trained
on the weak label — so scoring every method with the *same* check is the only way
to ask "does this uncertainty respond to capture quality?" without privileging the
one method that was trained to.

That question is not the same as good AURC. A model can rank ambiguous *labels*
well while being blind to how bad the photo is; it would defer the right images
for the wrong reason, and "retake the photo" would be useless advice to its user.
A near-zero rho for a method means its deferral story is not supported yet,
whatever its AURC says.

## What is still open

- No real images are in this repo, so nothing above has run against an actual
  smartphone re-photograph. Everything is exercised on synthetic images in
  `tests/` and `scripts/smoke_test.py`.
- The learned degrader is implemented but untrained, and `real` is uncollected —
  so the ablation currently compares physics against learned only.
- `uncertainty_vs_severity` has been exercised against fabricated honest/dishonest
  uncertainty arrays in `tests/test_domain_generalization.py`, not against a
  trained model. It becomes a real result the first time `tbtrust-eval` runs on a
  real checkpoint.
