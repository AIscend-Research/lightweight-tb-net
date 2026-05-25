\# Lightweight TB-Net



> Compressed deep learning for tuberculosis detection on Android smartphones



A PyTorch reimplementation of TB-Net (Wong et al., 2022) with quantization, pruning, and knowledge distillation. All models run under 6 MB — well below the 50 MB mobile deployment target.



\---



\## Why this exists



Rural clinics across Africa and South Asia often lack digital X-ray machines. Clinicians photograph X-ray films with smartphones instead. Existing AI models are too large and require internet connectivity. This project produces a model that runs entirely on-device, no internet required, on mid-range Android hardware.



\---



\## Results



| Model | Size | Accuracy | Sensitivity | Specificity | AUC |

|---|---|---|---|---|---|

| TB-Net FP32 (baseline) | 1.07 MB | 99.39% | 97.86% | 100.00% | 0.9987 |

| TB-Net FP16 | 0.55 MB | 99.39% | 97.86% | 100.00% | 0.9986 |

| TB-Net INT8 | 0.82 MB | 99.39% | 97.86% | 100.00% | 0.9987 |

| TB-Net 25% pruned | 1.07 MB | 99.59% | 98.57% | 100.00% | 0.9996 |

| TB-Net 50% pruned | 1.07 MB | 98.57% | 97.14% | 99.14% | 0.9987 |

| MobileNetV3 student | 5.91 MB | 98.57% | 96.43% | 99.43% | 0.9973 |

| ONNX FP32 | 1.04 MB | 99.39% | 97.86% | 100.00% | 0.9987 |

| ONNX INT8 | 0.30 MB | 99.39% | 97.86% | 100.00% | 0.9987 |



ONNX CPU inference latency: 0.85 ms average. Estimated Android latency: 50-200 ms.

Recommended deployment model: deploy/tbnet\_int8.onnx (0.30 MB, Android ONNX Runtime Mobile).



\---



\## Quickstart



\### 1. Install dependencies



conda create -n tb\_detect python=3.10

conda activate tb\_detect

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install numpy opencv-python scikit-learn matplotlib Pillow tqdm pandas onnx onnxruntime



\### 2. Download the dataset



Download from Kaggle: https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset



Extract it, then run:



python fix\_dataset.py --datapath "path/to/TB\_Chest\_Radiography\_Database"

python fix\_tb.py

python make\_splits.py



\### 3. Train



python train\_pytorch.py



Trains for 10 epochs. Best checkpoint saved to models/tbnet\_best.pth.



\### 4. Compress



python quantize.py      # FP16 and INT8 post-training quantization

python prune.py         # L1 unstructured pruning at 25%, 50%, 75% sparsity

python distill.py       # Knowledge distillation into MobileNetV3-Small



\### 5. Export for Android



python android\_deploy.py



Outputs deploy/tbnet.onnx and deploy/tbnet\_int8.onnx.

Use with ONNX Runtime Mobile: https://onnxruntime.ai/docs/tutorials/mobile/



\---



\## Repository structure



lightweight-tb-net/

├── tbnet\_pytorch.py        # Model architecture (self-attention CNN)

├── train\_pytorch.py        # Training loop

├── quantize.py             # Quantization experiments

├── prune.py                # Pruning experiments

├── distill.py              # Knowledge distillation

├── android\_deploy.py       # ONNX export + benchmarking

├── fix\_dataset.py          # Windows-compatible preprocessing

├── fix\_tb.py               # TB image renaming (Tuberculosis- to TB-)

├── make\_splits.py          # Stratified train/val/test split generation

├── train\_split\_new.csv     # 3,920 training images

├── val\_split\_new.csv       # 490 validation images

├── test\_split\_new.csv      # 490 test images

├── deploy/

│   ├── tbnet.onnx          # ONNX FP32 (1.04 MB)

│   └── tbnet\_int8.onnx     # ONNX INT8 (0.30 MB) — recommended for Android

└── tbnet\_paper.md          # Full research paper



\---



\## Key findings



\- FP16 quantization halves model size with zero accuracy loss

\- 25% pruning improves sensitivity from 97.86% to 98.57% — moderate pruning acts as a regularizer

\- 75% pruning drops sensitivity to 79.29% — below the WHO 90% minimum threshold for TB screening

\- MobileNetV3 student matches teacher sensitivity within 1.43% (target: within 2%)

\- All models are 8-90x under the 50 MB Android deployment target



\---



\## Paper



See tbnet\_paper.md for the full write-up including methodology, compression analysis, confusion matrices, and deployment feasibility assessment.



\---



\## Citation



If you use this work, please also cite the original TB-Net paper:



@misc{wong2021tbnet,

&#x20; title={TB-Net: A Tailored, Self-Attention Deep Convolutional Neural Network

&#x20;        Design for Detection of Tuberculosis Cases from Chest X-ray Images},

&#x20; author={Alexander Wong and James Ren Hou Lee and Hadi Rahmat-Khah

&#x20;         and Ali Sabri and Amer Alaref},

&#x20; year={2021},

&#x20; eprint={2104.03165},

&#x20; archivePrefix={arXiv}

}



\---



\## License



MIT. Model weights are for research purposes only.

Not intended for clinical diagnosis.

