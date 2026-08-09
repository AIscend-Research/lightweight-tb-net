"""
Shared utilities for the reproducibility study: seeding, cached datasets,
metrics with bootstrap confidence intervals, phone-capture perturbations.

Everything that touches randomness routes through set_seed() so that a run is
reproducible from (script, seed) alone.
"""

import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset

RESIZE_SHAPE = (224, 224)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
CACHE_DIR = os.path.join(REPO_ROOT, "cache")


# ── Determinism ──────────────────────────────────────────────────────────────

def set_seed(seed):
    """Seed every RNG we touch and put cuDNN in deterministic mode."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id):
    """DataLoader worker seeding, so num_workers>0 stays reproducible."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def loader_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ── Cached dataset ───────────────────────────────────────────────────────────

class CachedTBDataset(Dataset):
    """
    Reads 224x224 uint8 images from a memmapped .npy built by build_cache.py.

    Decoding every PNG through the paper's preprocessing pipeline costs ~1.2
    min/epoch; reading from the cache costs seconds. The cache is built once
    and reused by every experiment.

    perturb: optional callable(uint8 HxW array, np.random.Generator) -> uint8,
             used for the phone-capture evaluation and augmentation.
    """

    def __init__(self, split_csv, cache_name, three_channel=False,
                 perturb=None, perturb_seed=0, augment=False):
        self.arr = np.load(os.path.join(CACHE_DIR, f"{cache_name}.npy"),
                           mmap_mode="r")
        index = pd.read_csv(os.path.join(CACHE_DIR, f"{cache_name}_index.csv"))
        self.row_of = dict(zip(index["filename"], index["row"]))

        df = pd.read_csv(split_csv, header=None, names=["filename", "label"])
        df["filename"] = df["filename"].map(os.path.basename)
        missing = [f for f in df["filename"] if f not in self.row_of]
        if missing:
            raise KeyError(
                f"{len(missing)} files in {split_csv} are absent from cache "
                f"'{cache_name}' (first few: {missing[:5]})"
            )
        self.files = df["filename"].tolist()
        self.labels = df["label"].astype(int).tolist()

        self.three_channel = three_channel
        self.perturb = perturb
        self.perturb_seed = perturb_seed
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.array(self.arr[self.row_of[self.files[idx]]])  # uint8 HxW

        if self.perturb is not None:
            # Seeded per (dataset, index) so the perturbed test set is a fixed
            # artifact rather than something that changes every epoch.
            rng = np.random.default_rng(self.perturb_seed * 100003 + idx)
            img = self.perturb(img, rng)

        if self.augment:
            rng = np.random.default_rng(
                torch.randint(0, 2 ** 31 - 1, (1,)).item())
            if rng.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])

        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        x = (x - 0.5) / 0.5

        if self.three_channel:
            # MobileNetV3 expects 3 channels with ImageNet statistics.
            x = x * 0.5 + 0.5
            x = x.repeat(3, 1, 1)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            x = (x - mean) / std

        return x, self.labels[idx]


# ── Phone-capture perturbation ───────────────────────────────────────────────

def phone_perturb(img, rng, components=("brightness", "blur", "moire",
                                         "rotation", "glare")):
    """
    CheXphoto-style simulation of photographing an X-ray film with a phone.

    `components` selects which distortions to apply, so each one can be
    ablated independently.
    """
    out = img.astype(np.float32)
    h, w = out.shape

    if "brightness" in components:
        gain = rng.uniform(0.7, 1.3)
        bias = rng.uniform(-25, 25)
        out = out * gain + bias

    if "blur" in components:
        k = int(rng.choice([3, 5, 7]))
        out = cv2.GaussianBlur(out, (k, k), sigmaX=rng.uniform(0.5, 2.0))

    if "moire" in components:
        freq = rng.uniform(0.15, 0.45)
        phase = rng.uniform(0, 2 * np.pi)
        angle = rng.uniform(0, np.pi)
        yy, xx = np.mgrid[0:h, 0:w]
        proj = xx * np.cos(angle) + yy * np.sin(angle)
        out = out + 12.0 * np.sin(2 * np.pi * freq * proj + phase)

    if "rotation" in components:
        angle_deg = rng.uniform(-5, 5)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if "glare" in components:
        cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        radius = rng.uniform(0.15, 0.35) * w
        yy, xx = np.mgrid[0:h, 0:w]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        out = out + 70.0 * np.exp(-d2 / (2 * radius ** 2))

    return np.clip(out, 0, 255).astype(np.uint8)


# ── Metrics ──────────────────────────────────────────────────────────────────

def _rates(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = 100.0 * (tp + tn) / max(tp + tn + fp + fn, 1)
    sens = 100.0 * tp / max(tp + fn, 1)
    spec = 100.0 * tn / max(tn + fp, 1)
    return acc, sens, spec


def compute_metrics(y_true, y_prob, threshold=0.5, n_boot=1000, seed=0):
    """Point estimates plus percentile bootstrap 95% CIs."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    acc, sens, spec = _rates(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")

    out = {"acc": acc, "sens": sens, "spec": spec, "auc": auc}

    if n_boot:
        rng = np.random.default_rng(seed)
        boots = {"acc": [], "sens": [], "spec": [], "auc": []}
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            a, s, p = _rates(y_true[idx], y_pred[idx])
            boots["acc"].append(a)
            boots["sens"].append(s)
            boots["spec"].append(p)
            boots["auc"].append(roc_auc_score(y_true[idx], y_prob[idx]))
        for k, v in boots.items():
            if v:
                out[f"{k}_lo"] = float(np.percentile(v, 2.5))
                out[f"{k}_hi"] = float(np.percentile(v, 97.5))
    return out


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, n_boot=1000):
    model.eval()
    labels, probs = [], []
    for x, y in loader:
        logits = model(x.to(device))
        p = torch.softmax(logits.float(), dim=1)[:, 1]
        probs.extend(p.cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())
    return compute_metrics(labels, probs, threshold=threshold, n_boot=n_boot)


# ── Bookkeeping ──────────────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def state_dict_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)


def append_result(csv_name, row):
    """Append one result row to results/<csv_name>, creating it if needed."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, csv_name)
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    return path


def stage_done(csv_name):
    return os.path.exists(os.path.join(RESULTS_DIR, csv_name))
