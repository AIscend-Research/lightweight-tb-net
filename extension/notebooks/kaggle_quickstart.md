# Running on Kaggle (free GPU)

The roadmap flags Kaggle-compute feasibility, and the datasets already live there,
so Kaggle is the path of least resistance for training without a local GPU.

**Five ready-made notebooks now cover this end to end** — upload them to Kaggle
directly rather than copying cells by hand:

1. `00_setup_and_smoke_test.ipynb` — install + smoke test + pytest, no data needed.
2. `01_data_and_degradation_ablation.ipynb` — attach the dataset, build the
   manifest, check class balance, run the Phase 2 degradation ablation.
3. `02_train_models.ipynb` — in-distribution reference, both LOCO folds, TBNet,
   the efficiency benchmark.
4. `03_uncertainty_and_deferral.ipynb` — evidential + ensemble uncertainty,
   worst-case adversarial degradation search, sequential/CUSUM deferral.
5. `04_full_evaluation_and_results.ipynb` — the LOCO sweep aggregation
   (generalization gap, calibration heatmap, forecast-verification metrics),
   per-fold Safe Deferral numbers, and figures.

**These five are executed end to end in CI** (`scripts/test_notebooks.py`, run by
the `notebooks` job) against a synthetic dataset and shrunk configs, so every cell
is known to run rather than only to have been written. They ship without stored
outputs, so a clean checkout has no stale numbers in it.

**Paths are parameterised, not hard-coded.** Each notebook opens with a
configuration cell reading `TBTRUST_REPO`, `TBTRUST_DATA`, `TBTRUST_WORK` and
`TBTRUST_REPO_URL` from the environment, defaulting to the Kaggle locations. On
Kaggle you can run them unmodified; anywhere else, set those four variables (which
is exactly what the CI job does).

Read through the markdown cells before running; each says what it assumes from the
notebook before it. The manual cell-by-cell instructions below are the same idea
distilled into copy-paste form, kept for reference / for a single minimal notebook.

1. Attach the data. In the notebook sidebar, "Add Input" ->
   `tawsifurrahman/tuberculosis-tb-chest-xray-dataset`. It mounts read-only at
   `/kaggle/input/tuberculosis-tb-chest-xray-dataset`.

2. Install the package from your GitHub fork and check the wiring:

```python
!pip -q install git+https://github.com/<you>/tb-trust.git
!git clone -q https://github.com/<you>/tb-trust.git /kaggle/working/tb-trust
%cd /kaggle/working/tb-trust
!python scripts/smoke_test.py
```

3. Build a manifest from the mounted (read-only) input, writing to working dir:

```python
!python scripts/build_manifest.py \
    --raw /kaggle/input/tuberculosis-tb-chest-xray-dataset \
    --out /kaggle/working/manifest.csv
```

4. Point a config at that manifest and train (GPU on in notebook settings):

```python
!tbtrust-train --config configs/loco_montgomery.yaml \
    data.manifest=/kaggle/working/manifest.csv \
    train.output_dir=/kaggle/working/outputs
```

5. Evaluate:

```python
!tbtrust-eval --config configs/loco_montgomery.yaml \
    data.manifest=/kaggle/working/manifest.csv \
    --checkpoint /kaggle/working/outputs/montgomery/best.ckpt
```

Notes:
- The aggregated Kaggle set only gives clean per-clinic provenance for Montgomery
  and Shenzhen via filename prefixes. For NIAID/RSNA provenance and for the raw NLM
  sets, see `scripts/download_data.py`.
- Save `/kaggle/working/outputs` as a notebook output (or push to a Kaggle Dataset)
  so checkpoints and metrics survive the session.
