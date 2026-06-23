# TensorFlow 1 reference (original DarwinAI implementation)

These three files are the original TensorFlow 1 implementation from
https://github.com/darwinai/TuberculosisNet, kept here for reference only:

- `train_tbnet.py` — original training script
- `eval.py` — original evaluation script
- `dsi.py` — original data-interface

**They are not runnable in this repository.** TensorFlow 1 is not in
`requirements.txt`, and the reproduction work in this repo (`src/model.py`,
`src/train.py`, etc.) is a PyTorch re-implementation of the same architecture.

Kept because the PyTorch port follows their preprocessing pipeline and
training procedure, and reviewers may want to compare against the original.
