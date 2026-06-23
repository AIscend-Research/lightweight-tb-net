# Lightweight TB-Net — a reproducibility study

A PyTorch re-implementation of TB-Net (Wong et al., 2022, *Frontiers in AI*),
together with quantization, pruning, distillation, and a phone-capture
robustness extension. All compressed models stay well under 6 MB.

The original paper: [TB-Net: A Tailored, Self-Attention Deep Convolutional Neural Network Design for Detection of Tuberculosis Cases from Chest X-Ray Images](paper/TBNet_original.pdf) ([arXiv:2104.03165](https://arxiv.org/abs/2104.03165)).

---

## Repository layout

```
.
├── paper/                     ← paper draft + original PDF
├── src/                       ← PyTorch reimplementation
│   ├── model.py                  TBNet architecture
│   ├── preprocessing.py          B-channel + auto-crop pipeline from the paper
│   ├── train.py                  trains TBNet from scratch
│   ├── prune.py                  L1 unstructured pruning at 25/50/75 %
│   ├── quantize.py               INT8 / FP16 post-training quantization
│   ├── distill.py                MobileNetV3-Small student via KL distillation
│   ├── deploy.py                 ONNX export + desktop-CPU latency benchmark
│   ├── make_splits.py            70/15/15 train/val/test split generation
│   ├── visualize_results.py      regenerates figures/
│   └── _tf1_reference/           original TF1 code (not runnable; for reference)
├── notebooks/
│   └── reproduce_extensions.ipynb   75% pruning + phone-capture eval (Kaggle)
├── data_splits/                ← train/val/test CSVs
├── models/                     ← trained .pth checkpoints
├── deploy/                     ← exported .onnx files (deployment artifacts)
├── figures/                    ← every figure referenced by the paper
├── extensions/                 ← Kaggle run outputs (results CSVs + figures)
└── example_inputs/             ← sample CXRs for inference demos
```

---

## Results

| Model              | Size     | Accuracy | Sensitivity | Specificity | AUC    |
|--------------------|----------|----------|-------------|-------------|--------|
| TB-Net FP32        | 1.07 MB  | 98.57 %  | 95.71 %     | 99.71 %     | 0.9975 |
| TB-Net FP16        | 0.55 MB  | 98.57 %  | 95.71 %     | 99.71 %     | 0.9975 |
| TB-Net 25 % pruned | 1.07 MB  | 98.78 %  | 96.43 %     | 99.71 %     | 0.9984 |
| TB-Net 50 % pruned | 1.07 MB  | 98.57 %  | 95.71 %     | 99.71 %     | 0.9959 |
| TB-Net 75 % pruned | 1.07 MB  | 96.94 %  | 94.29 %     | 98.00 %     | 0.9918 |
| MobileNetV3 dist.  | 5.91 MB  | 78.78 %  | 73.57 %     | 80.86 %     | 0.8350 |
| ONNX INT8          | 0.30 MB  | (deployment artifact) | | | |

Full table: [extensions/results_clean.csv](extensions/results_clean.csv).
Phone-capture metrics: [extensions/results_phone.csv](extensions/results_phone.csv).

---

## Reproducing the results

### 1. Install

```
conda create -n tbnet python=3.10
conda activate tbnet
pip install -r requirements.txt
```

### 2. Download the dataset

Tawsifur Rahman's TB chest X-ray dataset: [kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset).

Extract so that `data/Normal/Normal-*.png` and `data/Tuberculosis/Tuberculosis-*.png` exist.

### 3. Train + compress

```
python src/train.py        # ~12 min on a T4 GPU
python src/quantize.py     # INT8 + FP16 export
python src/prune.py        # L1 pruning at 25 / 50 / 75 %
python src/distill.py      # MobileNetV3-Small student
python src/deploy.py       # ONNX export + latency benchmark
```

Checkpoints land in `models/`, ONNX exports in `deploy/`.

### 4. Reproduce the extensions (75 % pruning + phone-capture)

Open `notebooks/reproduce_extensions.ipynb` on Kaggle with a free T4 GPU. Attach two datasets:

- `tawsifurrahman/tuberculosis-tb-chest-xray-dataset` (public)
- a private dataset containing the contents of `src/`, `models/`, and `data_splits/` (the notebook auto-detects the layout)

Run all cells (~45 min). Outputs land in `/kaggle/working/`.

---

## Phone-capture extension

The original paper assumes digital X-rays. In rural clinics across Africa and South Asia, clinicians often photograph X-ray films with smartphones. We re-evaluate every compressed model on a CheXphoto-style perturbation of the test set (brightness shift, gaussian blur, additive moiré stripes, small rotation, glare patch).

**Key finding:** under phone-capture distortion, all TB-Net variants collapse to ~30 % accuracy and 1–6 % specificity (they over-predict TB on everything). The distilled MobileNetV3 student is paradoxically more robust, holding 47 % accuracy — suggesting attention-based architectures pay a robustness tax for their efficient capacity. See §7 of [paper/tbnet_paper.md](paper/tbnet_paper.md).

---

## License

See [LICENSE.md](LICENSE.md). The original TB-Net code is GPL-3.0 (DarwinAI). The PyTorch port and extensions in this repo follow the same license.
