# Defects found and what was done about them

A running log of every defect found in this codebase during the reproducibility
work, kept because several of them affected published numbers and a reader is
entitled to know which. Ordered by how much they changed the results.

Status is one of **fixed**, **documented** (a deliberate deviation that is
recorded rather than silently resolved), or **open**.

---

## 1. Duplicate images split across train and test

**Status: fixed** | data integrity | `src/make_splits.py`

The split files listed 1,400 tuberculosis entries drawn from only 700 distinct
images. Each radiograph appeared twice, once as `Tuberculosis-N.png` and once
as `TB-N.png`. The generator split over rows rather than distinct images and
labelled anything not starting with `Normal` as positive, so both copies were
treated as independent samples.

Consequence: 222 of the 700 TB images occurred in both the training set and a
held-out set, and 115 of the 140 TB images in the old test split were
duplicates of training images.

Fix: `make_splits.py` now maps filenames onto a canonical image identity,
deduplicates before splitting, and refuses to write splits where any image
crosses a split boundary. `--verify-only` re-checks an existing set.

**Measured impact: smaller than expected.** On clean splits the compact model
scores 98.81 +/- 0.81 accuracy and 95.71 +/- 1.43 sensitivity, against 98.57
and 95.71 on the contaminated splits. The contamination was real and had to be
removed, but it was not what produced the high scores. The corrected cohort is
4,200 distinct images at a 5:1 normal-to-TB ratio, not the 2.5:1 previously
assumed.

**Provenance: settled, and the answer narrows the finding.** The duplicates
were introduced locally. They were not inherited from the original TB-Net
splits.

This repository's git history contains the original DarwinAI repository, whose
`train_split.csv` was committed on 2021-03-08. Its tuberculosis entries are
formatted `TB (188).png`, with a space and parentheses, and its indices run to
roughly 3500 across 6,927 rows in the three splits. The duplicates here are
formatted `TB-699.png`, hyphenated, and stop at exactly 700, which is the size
of the current public Kaggle release. They are therefore renamed copies of
`Tuberculosis-1..700`, most plausibly created to match the original naming
convention and then left in `data/` beside the originals, where the old
generator globbed both.

So this is a defect in this pipeline, not in the published TB-Net splits, and
the write-up should say so plainly rather than implying a wider problem.

**A separate finding does survive, and it is evidence rather than inference.**
The original splits reference 6,927 images with TB indices to about 3500,
while the current Kaggle release contains 700 TB images. The cohort the
original work used is not the cohort available today, and the original repo's
own files demonstrate it.

---

## 2. CSV rows written under the wrong column headers

**Status: fixed** | silent data corruption | `src/common.py`

`append_result` wrote the header from the first row it saw and then appended
subsequent rows positionally. Stages whose rows carry different keys therefore
landed under the wrong headers. One-shot pruning reports `acc_pre_ft` and
`sens_pre_ft`; iterative pruning does not, so its 10 rows had 20 fields against
a 22-field header and every column from the eighth onward was shifted by two.

Consequence: iterative pruning appeared to score 1.00 sensitivity at 99.94
accuracy, which is arithmetically impossible on a 420-image split containing 70
positives. Reading the columns correctly gives 94.86 +/- 0.78 sensitivity, the
single strongest result in the study: gradual pruning removes the sensitivity
cliff that one-shot pruning at the same 75% sparsity produces (7.14 +/- 15.97).

Fix: `append_result` reads back the existing header and aligns each row to it,
widening the file if genuinely new keys appear. The affected `prune.csv` was
repaired in place by reinserting the two missing fields.

---

## 3. The paper's preprocessing pipeline was never used

**Status: fixed** | methodology | `src/build_cache.py`

`src/preprocessing.py` implements the pipeline described in the original work:
blue-channel split, padding-aware auto-crop, resize, the data interface's crop,
and corner masking to hide burnt-in metadata. Nothing imported it. The training
scripts opened images with `PIL.Image.convert("L")` and resized, so every
reported result came from a pipeline the write-up did not describe.

Fix: the cache builder implements both, as the `faithful` and `simple`
variants, and the choice is a first-class experimental axis. The two now differ
by -0.33 +/- 0.29 percentage points of AUC for the compact model, a gap whose
confidence interval includes zero.

---

## 4. Montgomery lung masks counted as radiographs

**Status: fixed** | silent | `src/experiments.py`

The `kmader/pulmonary-chest-xray-abnormalities` dataset ships left and right
lung segmentation masks whose filenames match the radiographs they belong to.
The external-validation stage globbed for the filename pattern and picked up
414 Montgomery "images" instead of 138, scoring binary masks as chest X-rays.

This failure was silent: the stage completed and produced plausible-looking
numbers. Fix: paths containing `mask` are excluded, and the notebook prints
cohort counts so the expected 138 and 662 can be checked before the stage runs.

---

## 5. Distillation crashed on the first batch

**Status: fixed** | crash | `src/common.py`, `src/experiments.py`

The distillation stage built three-channel loaders for the MobileNetV3 student,
but the TBNet teacher takes one channel, so `teacher(x)` raised immediately.

Fix: `CachedTBDataset` gained a `dual` mode yielding
`(teacher_input, student_input, label)`, and `fit()` unpacks either form.

---

## 6. FP16 evaluation crashed on dtype

**Status: fixed** | crash | `src/common.py`

The quantization stage called `.half()` on the model while the loader continued
to yield float32, so the first convolution raised a dtype mismatch.

Fix: a `model_dtype()` helper reads the dtype off the model's parameters, and
every inference path casts inputs to match. Dynamically quantized models keep
float32 parameters and pass through unchanged.

---

## 7. No seeding anywhere

**Status: fixed** | reproducibility | `src/common.py`

No script set any random seed. Results were single unseeded runs and could not
be re-derived. Fix: `set_seed()` covers `random`, `numpy`, `torch`, CUDA and
cuDNN determinism, DataLoader workers are seeded through `worker_init_fn`, and
every experiment runs across five seeds with bootstrap confidence intervals.

---

## 8. Training script referenced files that do not exist

**Status: fixed by removal** | `src/train.py` (deleted)

`train.py` read `train_split_new.csv` and similar from the repository root
while the splits lived in `data_splits/` under different names, so the script
could not run as committed. It has been replaced by `src/experiments.py`.

---

## 9. The write-up cited code that was never committed

**Status: fixed by removal** | documentation

The paper draft described results from `combined_compress.py` and
`train_pytorch.py`. Neither file existed in the repository. The draft also
presented the outcome of a class-weighting change as an expectation
("sensitivity approaching 99-100%") in the voice of a measured result. The
experiment has since been run, and class weighting *reduced* AUC by 0.99 +/-
1.19 percentage points, with all five seeds favouring plain cross-entropy.

---

## 10. Two published result tables disagreed

**Status: superseded** | documentation

The paper's section 4 and `extensions/results_clean.csv` reported different
numbers for the same models, most starkly for the distilled student: 98.57
against 78.78 accuracy. Both predate the corrected splits and the control
experiments, so both have been withdrawn rather than reconciled. All current
numbers come from `results/` and are regenerated by `src/analyze.py`.

---

## 11. Crop semantics differ between the reference implementations

**Status: documented** | `src/build_cache.py`

`_tf1_reference/dsi.py` calls
`tf.image.crop_to_bounding_box(img, 11, 11, 168, 202)`, whose last two
arguments are target height and width, giving rows 11:179 and columns 11:213.
`preprocessing.py` slices `[11:168, 11:202]`, treating the same numbers as end
indices. These are different crops and only one can match the original.

The cache builder follows the TensorFlow semantics and exposes a
`tf_semantics` flag for the other reading. Which the original authors intended
cannot be settled from the released code.

---

## 12. The original architecture cannot be recovered

**Status: open, and unfixable from released materials**

`_tf1_reference/train_tbnet.py` loads its graph with
`tf.train.import_meta_graph(model_train.meta)`. The architecture is stored
inside the checkpoint, not in the source, and the checkpoint download is dead.
A layer-for-layer port is therefore impossible from published materials.

What this repository calls the `full` model is a capacity-matched
reconstruction at roughly 4.2M parameters using the described motifs. It is not
a port, and no claim of architectural fidelity should be made for it.

---

## 13. External validation scores below chance

**Status: open, two causes eliminated**

On Montgomery and Shenzhen the compact model reaches AUC 0.32 +/- 0.02 overall
and 0.29 +/- 0.03 on Shenzhen alone, well below the 0.5 expected from random
ranking, with 2.18% sensitivity. A systematically inverted ranking usually
indicates a label or preprocessing defect rather than a domain-shift finding.

Three candidates, each now testable:

1. **Label convention.** The external stage prints the parsed class balance
   against the published one (Montgomery 80 normal / 58 TB, Shenzhen 326 / 336)
   and writes `results/external_labels.csv` with an explicit verdict. Swapped
   counts would explain every sub-chance AUC at a stroke.
2. **Preprocessing.** The padding-aware auto-crop assumes black borders that
   NLM scans may not have. Compare rendered external images against rendered
   training images before drawing conclusions.
3. **Cohort overlap.** `--stage overlap` perceptually hashes every external
   image against the cached training cohort and reports matches. This does not
   explain a sub-chance AUC, but it decides whether the stage is external at
   all.

**Verdict on item 2: preprocessing is not the cause either.** Evaluating the
`simple`-trained models on `simple`-processed external images does not rescue
the AUC; it lowers it, to 0.264 +/- 0.023 for compact against 0.323 +/- 0.017
under `faithful`. The failure is also cohort-specific: on Montgomery both
pipelines sit near chance (0.47 to 0.52) while on Shenzhen the ranking is
strongly inverted (0.211 to 0.294). Whatever is happening is specific to the
Shenzhen cohort and survives a complete change of preprocessing.

**Verdict on item 1: labels are correct.** The parsed balance is exactly the
published one, Montgomery 80 normal / 58 TB and Shenzhen 326 / 336, so the
`_0`/`_1` convention is being read the right way round and an inverted label
does not explain the sub-chance AUC. Suspicion moves to preprocessing
(`figures/external_preprocessing_check.png` renders external and training
images side by side through the same pipeline) or to something genuinely odd
about how these models rank NLM scans. Still unresolved, still unpublishable.

---

## 14. ONNX export failed, so there is no latency measurement

**Status: root cause found, not yet fixed**

The export is wrapped in a try/except so a failure does not abort the run. It
failed on every model under torch 2.10 and produced no `.onnx` files, so
`latency.csv` was never written. Every deployment-latency statement is
currently unsupported by measurement.

Once the error was actually recorded it turned out to be mundane:
`No module named 'onnxruntime'`. The package is imported at the top of
`onnx_latency`, before any export is attempted, and `reproduce.ipynb`
pip-installs it while `finish_run.ipynb` does not. Nothing was ever wrong with
the exporter.

This is worth recording as a process failure rather than a coding one. The
`except Exception` around the export turned a missing dependency into what
looked like a torch-version incompatibility, and a whole hypothesis about
opset versions and the dynamo exporter was built on top of a message nobody
had read. Broad exception handlers that print only `str(exc)` cost more than
they save.

There is a second missing package behind the first. A later run that did have
`onnxruntime` failed with `No module named 'onnxscript'`: torch 2.6 and later
route `torch.onnx.export` through the dynamo exporter by default, and that
path needs `onnxscript`. So the dependency list is `onnx`, `onnxruntime` and
`onnxscript`, and the multi-configuration export path added earlier does earn
its keep after all, because it tries `dynamo=False` first and sidesteps the
dynamo requirement entirely.

Fix: install all three in `finish_results.ipynb` and rerun the quantize stage.

---

## 15. The overlap check matches far too loosely

**Status: open, results not usable**

`--stage overlap` reported that 641 of 800 external images match a
training-cohort image at dHash Hamming distance 5 or less. That is not
credible: 302 of those matches pair a TB-positive external image with a file
named `Normal-*`, and an 80% duplicate rate between two separately published
cohorts is implausible on its face.

The cause is the hash, not the data. A 64-bit difference hash of a chest
radiograph downsampled to 9x8 encodes gross thoracic structure, which every
chest X-ray shares, so a Hamming radius of 5 matches almost anything. The
first dictionary hit is also returned arbitrarily among near-matches, so
`matched_training_file` is not meaningful.

Rerun with exact matching and a full distance histogram, the answer is that
the hash cannot settle the question. Nearest-neighbour distances form one
smooth unimodal distribution peaking at Hamming 3 (33 images at distance 0,
75 at 1, 130 at 2, 157 at 3, then a steady decline to 4 at distance 13). Real
duplicates would appear as a spike at or near zero separated by a gap; there
is no gap. The 33 exact collisions are consistent with the left tail of that
distribution rather than with genuine repeats.

The honest write-up is that cohort overlap remains untested. Settling it needs
an embedding distance or pixel-level comparison, not a 64-bit perceptual hash
of a downsampled radiograph.


## 16. A forced re-run silently dropped an experimental arm

**Status: fixed by merge, worth remembering**

`finish_run.ipynb` deletes a stage CSV before re-running it, so that forced
rows replace rather than append. Task D re-ran the baseline stage without the
class-weighted condition, so `reset("baseline.csv")` removed the five weighted
rows and nothing put them back. The finding they support, that inverse
frequency weighting reduces AUC with all five seeds agreeing, would have
vanished from the analysis without any error.

The rows were recovered from the repository copy and merged, giving 30 rows:
25 from the new run plus the 5 weighted. The general lesson is that
delete-then-regenerate is only safe when the regenerating command covers every
condition the file already held.


## 17. The quantization stage silently ignored all but the first architecture

**Status: fixed** | silent | `src/experiments.py`

`stage_quantize` opened with `arch = args.arch[0]`, so `--arch compact full`
quantized only `compact` and no error was raised. The consequence is that
`quantize.csv` and `latency.csv` cover the 0.27M model alone; there are no
FP16, INT8 or latency numbers for the 4.2M reconstruction, and nothing in the
output said so.

Fix: the stage now loops over every requested architecture. Recovering the
missing rows needs a rerun of the quantize stage, roughly ten minutes.


## 18. Stale repository URL

**Status: fixed**

The paper draft pointed at a personal GitHub account rather than the
organisation that now hosts the work.
