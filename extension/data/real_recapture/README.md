# Real re-photography set (pilot protocol)

`data/degradation.py` (physics-based) and `data/degradation_learned.py` (learned
generator) are both stand-ins for the real thing: a clinician photographing an
analog film on a lightbox with a phone. `scripts/ablate_degradation.py` is the
tool that scores each stand-in against real captures -- but it needs some real
captures to score against. There are none in this repo yet (no physical access to
a lightbox + printed films during Phase 2 of this project). This is the honest
limitation flagged in `docs/phase1_framing.md` and it stays in the paper's
limitations section (Phase 5) if it isn't closed by then.

Two ways to close it:

## Option A: collect a small pilot set

You need a printed film or a monitor displaying a clean digital X-ray at
realistic size, a lightbox or bright monitor as the illumination source, and 2-3
different phones. This does not need to be large -- even 20-30 source images x a
few capture conditions each is enough for `ablate_degradation.py`'s classifier
comparison to be meaningful (it's already 5-fold cross-validated for small n).

1. Pick ~20-30 already-public Montgomery/Shenzhen images (both classes,
   pre-approved for reuse -- see the licensing note in `docs/phase1_framing.md`)
   and either print them or display them full-screen on a second monitor.
2. For each source image, capture 3-6 phone photos varying what the synthetic
   pipeline models: capture angle (near head-on vs. ~10-15 degrees off), distance/
   framing, ambient lighting (room light on/off, near a window), and phone (at
   least 2 different devices/cameras if available -- this is the artifact
   diversity a single hand-tuned physics model can't capture).
3. Save under `data/real_recapture/<clinic>/<orig_stem>__<phone_model>__<condition>.jpg`,
   e.g. `data/real_recapture/montgomery/MCUCXR_0003_0__pixel7__angle15_lowlight.jpg`.
   Keep the original filename stem so the pair is recoverable later for a *paired*
   ablation (not just the current unpaired distributional one).
4. Log each capture as a row in `manifest_template.csv` (copy it, don't edit the
   template in place) so `orig_path`, `clinic`, `phone_model`, and `condition` are
   queryable without parsing filenames.
5. Run the ablation:
   ```bash
   python scripts/ablate_degradation.py --real-dir data/real_recapture/montgomery \
       --source data/raw/montgomery --out outputs/degradation_ablation.json
   ```

This directory is gitignored except this README and the manifest template --
recaptured images should not be committed (same reasoning as `data/raw/`: it's
imagery derived from datasets with their own licensing terms, see
`docs/phase1_framing.md` section 4).

## Option B: use an existing real-recapture dataset as a stopgap

The PhysioNet `cxr-phone` set (physionet.org/content/cxr-phone) and Stanford's
CheXphoto (real-photo subset, not just its synthetic transforms) are both
existing real smartphone-photo-of-CXR collections, cited in
`docs/phase1_framing.md` section 3. Neither is TB-labeled or drawn from this
project's four clinics, so they can't replace the LOCO evaluation data, but
either is usable purely as the "real" reference distribution for
`ablate_degradation.py` (does physics/learned degradation of *our* clean images
land in the same feature distribution as real phone photos of *any* CXR film?).
Check each dataset's access terms before downloading; PhysioNet credentialing in
particular may require a signed data use agreement similar to the NIAID one.

## What "done" looks like

A real-vs-synthetic ablation report from `ablate_degradation.py` with a
`real` group populated, i.e. a `*_vs_real` entry in the `separability` section of
its JSON output, plus the interpretation: whichever of `physics`/`learned` has
lower classifier accuracy against `real` is the better stand-in, and that's the
strategy the main training/eval pipeline should default to (or blend, if neither
dominates on all five features).
