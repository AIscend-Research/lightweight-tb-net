# Lightweight TB-Net: a reproducibility study

An attempt to reproduce TB-Net (Wong et al., 2022, *Frontiers in AI*,
[arXiv:2104.03165](https://arxiv.org/abs/2104.03165)) in PyTorch, plus a
compression and robustness study built on top of it.

Everything here runs from a single notebook on a free Kaggle T4 in about
four hours, or from the command line on any machine with a GPU.

## Status

**Results are being regenerated. Do not cite any numbers from this repo's
history.**

An earlier version of this study reported roughly 99% accuracy. Those numbers
were invalid. The train/val/test splits contained 1,400 tuberculosis entries
drawn from only 700 distinct images, present twice under two naming schemes
(`Tuberculosis-N.png` and `TB-N.png`). Because the split was taken over rows
rather than over distinct images, 222 of the 700 TB images appeared in both
the training set and a held-out set, and 115 of the 140 TB images in the test
set were duplicates of training images.

`src/make_splits.py` now keys on a canonical image identity and refuses to
write splits where any image crosses a split boundary. The corrected dataset
is 4,200 distinct images (3,500 normal, 700 TB, a 5:1 ratio rather than the
2.5:1 previously assumed).

Two further findings shape what this study can claim:

- **The original architecture is not recoverable from the released code.**
  The TF1 training script loads its graph with
  `tf.train.import_meta_graph(model_train.meta)`, so the topology lives inside
  a checkpoint whose download link is dead. A layer-for-layer port is not
  possible from published materials. What this repo calls the `full` model is
  a capacity-matched reconstruction (~4.2M parameters, matching the reported
  budget and the described motifs), not a faithful port.
- **The dataset has changed since publication.** The original split files
  reference TB cases numbered up to 3460; the current public Kaggle release
  contains 700.

## Layout

```
reproduce.ipynb        driver notebook: runs every stage end to end
requirements.txt
src/
  common.py            seeding, cached dataset, metrics with bootstrap CIs,
                       phone-capture perturbations
  make_splits.py       leak-checked stratified splits
  build_cache.py       decode images once into a memmapped uint8 array
  preprocessing.py     the paper's B-channel + auto-crop pipeline
  model.py             the 0.27M-parameter re-implementation
  models_repro.py      model registry: compact and full
  experiments.py       every experiment, as resumable stages
  smoke_test.py        full pipeline on synthetic images, about one minute
  _tf1_reference/      original DarwinAI TF1 code, kept as evidence (not runnable)
```

Generated directories (`cache/`, `data_splits/`, `checkpoints/`, `results/`)
are created at run time and are not tracked.

## The two datasets

Both are public on Kaggle and both are needed: the first is what the models
are trained and tested on, the second is what makes a comparison with the
original paper meaningful.

### 1. Working cohort: `tawsifurrahman/tuberculosis-tb-chest-xray-dataset`

The Rahman et al. tuberculosis chest X-ray database. This is the cohort every
model here is trained, validated and tested on.

```
TB_Chest_Radiography_Database/
  Normal/         Normal-1.png ... Normal-3500.png
  Tuberculosis/   Tuberculosis-1.png ... Tuberculosis-700.png
```

4,200 PNG images: 3,500 normal and 700 TB, a 5:1 class imbalance that matters
for how the results should be read. Sensitivity, not accuracy, is the metric
to watch, since a model that predicts "normal" for everything already scores
83% accuracy. Training here selects checkpoints on validation sensitivity for
that reason.

Labels come from the directory name. The notebook locates this dataset by
searching for a directory containing both `Normal/` and `Tuberculosis/`, so
it does not matter where Kaggle mounts it.

Note that this is a *later* release than the one the original paper used: the
original split files reference TB cases numbered up to 3460, and this release
contains 700. The cohort is not the same data, which is one reason a direct
metric comparison with the paper is not available on this dataset alone.

### 2. External validation: `kmader/pulmonary-chest-xray-abnormalities`

The two National Library of Medicine cohorts that Wong et al. actually
reported on, which is why this is the only dataset here that supports a
delta-against-the-original table.

- **Montgomery County, Maryland, USA**: 138 radiographs, files named
  `MCUCXR_####_N.png`
- **Shenzhen No. 3 People's Hospital, China**: 662 radiographs, files named
  `CHNCXR_####_N.png`

Labels are encoded in the filename suffix: `_0` is normal, `_1` is TB. The
`external` stage parses that suffix and reports Montgomery and Shenzhen both
separately and pooled, since the two cohorts differ in equipment and
population.

**This dataset also ships lung segmentation masks whose filenames match the
radiographs.** Counting naively gives 414 Montgomery "images" instead of 138,
because each radiograph is accompanied by left and right lung masks. Both the
notebook's file counts and `stage_external` filter out any path containing
"mask". If your run prints anything other than `Montgomery=138 Shenzhen=662`,
stop and check the layout before trusting that stage.

### A caveat about calling cohort 2 "external"

The Rahman database is itself an aggregation, and the NLM Montgomery and
Shenzhen sets are among its stated sources. Some of the images used for
external validation may therefore also sit in the training split, which would
make that evaluation optimistic rather than genuinely held out.

This repo does not currently detect that overlap. A perceptual-hash
comparison between the two cohorts would settle it, and the result belongs in
any write-up regardless of which way it comes out. Until then, treat the
external numbers as an upper bound.

## Running it

### On Kaggle

Create a notebook with **GPU T4 x1** and **Internet on**, attach both datasets
above, and run `reproduce.ipynb`.

The notebook clones this repo and shells out to `src/`, so no code is pasted
into cells and no private dataset is required. One T4 is enough; nothing here
is multi-GPU, so a second card only burns quota.

Check the dataset pages for their current license and citation terms before
redistributing anything derived from them.

### Locally

```
pip install -r requirements.txt
# extract the Kaggle dataset so that data/Normal/*.png and
# data/Tuberculosis/*.png exist

python src/smoke_test.py                                    # ~1 min, no data needed
python src/make_splits.py --data-path data/                 # leak-checked splits
python src/build_cache.py --data-path data/ --variant both  # ~4 min, one time

python src/experiments.py --stage baseline --arch compact full \
    --caches faithful simple --seeds 0 1 2 3 4
python src/experiments.py --stage prune    --seeds 0 1 2 3 4
python src/experiments.py --stage distill  --seeds 0 1 2 3 4
python src/experiments.py --stage quantize --seeds 0 1 2 3 4
python src/experiments.py --stage external --external-path /path/to/mc_sz
python src/experiments.py --stage phone    --seeds 0 1 2 3 4 --phone-finetune
python src/experiments.py --stage summary
```

Each stage writes `results/<stage>.csv` and skips itself if that file exists,
so an interrupted run resumes cheaply. Pass `--force` to recompute.

## What each stage measures

| Stage | Question | T4 time |
|---|---|---|
| baseline | 2 architectures x 2 preprocessing pipelines x 5 seeds | 45 min |
| prune | unstructured vs structured vs iterative, at 25/50/75% | 60 min |
| distill | MobileNetV3-Small student, against a from-scratch control | 70 min |
| quantize | FP16, INT8 dynamic, pruned+FP16, ONNX CPU latency | 15 min |
| external | Montgomery and Shenzhen, the original paper's cohorts | 5 min |
| phone | per-distortion ablation, plus phone-augmented fine-tuning | 45 min |
| summary | mean and standard deviation across seeds | 1 min |

Every metric carries a percentile bootstrap 95% confidence interval, and
per-sample probabilities are written to `results/probs/` so ROC curves,
operating-point selection and paired model comparisons need no re-inference.

The preprocessing comparison is itself an experiment: `faithful` applies the
paper's B-channel split, padding-aware auto-crop, data-interface crop and
corner masking, while `simple` is the grayscale-and-resize pipeline the
earlier version of this study actually used.

## Reproducibility notes

- All randomness routes through `set_seed()`, including DataLoader workers.
- `results/environment.json` records the exact library and driver versions.
- `src/build_cache.py` documents one deliberate deviation: `dsi.py` calls
  `tf.image.crop_to_bounding_box(img, 11, 11, 168, 202)`, whose last two
  arguments are target height and width, while `preprocessing.py` slices
  `[11:168, 11:202]` as if they were end indices. The cache builder follows
  the TensorFlow semantics; a flag switches to the other reading.
- Latency is a median of 100 CPU runs reported with its interquartile range.
  Measured on shared cloud vCPUs it is indicative, not device latency.

## Known gaps

- Real smartphone photographs of films. The phone-capture study uses
  simulated distortions.
- On-device Android latency.
- Structured pruning zeroes whole filters but no compaction pass physically
  removes them, so the FLOP saving is real while the file-size saving is not.

## License

See [LICENSE.md](LICENSE.md). The original TB-Net code is GPL-3.0 (DarwinAI);
the PyTorch port and the work here follow the same license.
