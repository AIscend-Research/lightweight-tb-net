"""
Every experiment in the reproducibility study, as resumable stages.

  python src/experiments.py --stage audit
  python src/experiments.py --stage baseline --arch compact full --seeds 0 1 2 3 4
  python src/experiments.py --stage prune    --seeds 0 1 2 3 4
  python src/experiments.py --stage distill  --seeds 0 1 2 3 4
  python src/experiments.py --stage quantize
  python src/experiments.py --stage external --external-path /path/to/mc_sz
  python src/experiments.py --stage phone    --seeds 0 1 2 3 4

Each stage writes results/<stage>.csv and is skipped if that file already
exists (override with --force), so a killed Kaggle session resumes cheaply.
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

from common import (CachedTBDataset, RESULTS_DIR, append_result, count_params,
                    evaluate, loader_generator, phone_perturb, seed_worker,
                    set_seed, stage_done)
from models_repro import build_model, build_student

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = {s: os.path.join(REPO, "data_splits", f"{s}.csv")
          for s in ("train", "val", "test")}
CKPT_DIR = os.path.join(REPO, "checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data ─────────────────────────────────────────────────────────────────────

def make_loaders(cache, seed, batch_size=32, three_channel=False,
                 augment_train=False, train_perturb=None):
    kw = dict(cache_name=cache, three_channel=three_channel)
    train = CachedTBDataset(SPLITS["train"], augment=augment_train,
                            perturb=train_perturb, perturb_seed=seed, **kw)
    val = CachedTBDataset(SPLITS["val"], **kw)
    test = CachedTBDataset(SPLITS["test"], **kw)
    common = dict(num_workers=2, worker_init_fn=seed_worker,
                  generator=loader_generator(seed), pin_memory=True)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, **common),
        DataLoader(val, batch_size=batch_size, shuffle=False, **common),
        DataLoader(test, batch_size=batch_size, shuffle=False, **common),
    )


def class_weight_tensor():
    df = pd.read_csv(SPLITS["train"], header=None, names=["f", "y"])
    counts = df["y"].value_counts().sort_index().values
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


# ── Generic training loop ────────────────────────────────────────────────────

def fit(model, train_loader, val_loader, epochs, lr=1e-4, weights=None,
        teacher=None, temperature=4.0, alpha=0.7, cosine=False, tag=""):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if cosine else None)
    ce = nn.CrossEntropyLoss(weight=weights)
    if teacher is not None:
        teacher.to(DEVICE).eval()

    best_state, best_sens = copy.deepcopy(model.state_dict()), -1.0
    for ep in range(epochs):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(x)
                soft = F.kl_div(
                    F.log_softmax(logits / temperature, dim=1),
                    F.softmax(t_logits / temperature, dim=1),
                    reduction="batchmean") * (temperature ** 2)
                loss = alpha * soft + (1 - alpha) * loss
            loss.backward()
            opt.step()
            total += loss.item()
        if sched:
            sched.step()

        m = evaluate(model, val_loader, DEVICE, n_boot=0)
        # Select on sensitivity: for TB screening a missed case is the costly
        # error, and accuracy hides majority-class collapse.
        if m["sens"] > best_sens:
            best_sens, best_state = m["sens"], copy.deepcopy(model.state_dict())
        print(f"    [{tag}] ep {ep + 1}/{epochs} loss {total / max(len(train_loader), 1):.4f} "
              f"val acc {m['acc']:.2f} sens {m['sens']:.2f}")

    model.load_state_dict(best_state)
    return model


def save_ckpt(model, name):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = os.path.join(CKPT_DIR, f"{name}.pth")
    torch.save(model.state_dict(), path)
    return path


def size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)


# ── Stage: audit existing checkpoints ────────────────────────────────────────

def stage_audit(args):
    """
    Re-evaluate the checkpoints already committed in models/ to settle which
    of the two conflicting result tables (paper section 4 vs
    extensions/results_clean.csv) reflects reality.
    """
    legacy = os.path.join(REPO, "models")
    if not os.path.isdir(legacy):
        print("no models/ directory; skipping audit")
        return
    for cache in args.caches:
        _, _, test = make_loaders(cache, seed=0)
        _, _, test3 = make_loaders(cache, seed=0, three_channel=True)
        for fname in sorted(os.listdir(legacy)):
            if not fname.endswith(".pth"):
                continue
            path = os.path.join(legacy, fname)
            is_student = "student" in fname or "mobilenet" in fname
            model = build_student(pretrained=False) if is_student \
                else build_model("compact")
            try:
                sd = torch.load(path, map_location="cpu")
                model.load_state_dict(sd)
            except Exception as exc:                       # noqa: BLE001
                print(f"  ! {fname}: cannot load ({type(exc).__name__}: {exc})")
                append_result("audit.csv", {"checkpoint": fname, "cache": cache,
                                            "error": str(exc)[:200]})
                continue
            model.to(DEVICE)
            if "fp16" in fname and DEVICE.type == "cuda":
                model.half()
            m = evaluate(model, test3 if is_student else test, DEVICE)
            append_result("audit.csv", {"checkpoint": fname, "cache": cache,
                                        "size_mb": size_mb(path),
                                        "params": count_params(model), **m})
            print(f"  {fname} [{cache}] acc {m['acc']:.2f} sens {m['sens']:.2f}")


# ── Stage: baselines ─────────────────────────────────────────────────────────

def stage_baseline(args):
    for arch in args.arch:
        for cache in args.caches:
            for weighted in args.weighted:
                for seed in args.seeds:
                    set_seed(seed)
                    tr, va, te = make_loaders(cache, seed,
                                              augment_train=args.augment)
                    model = build_model(arch)
                    w = class_weight_tensor() if weighted else None
                    t0 = time.time()
                    model = fit(model, tr, va, args.epochs, lr=args.lr,
                                weights=w, cosine=args.cosine,
                                tag=f"{arch}/{cache}/s{seed}")
                    name = f"{arch}_{cache}{'_w' if weighted else ''}_s{seed}"
                    path = save_ckpt(model, name)
                    m = evaluate(model, te, DEVICE)
                    append_result("baseline.csv", {
                        "arch": arch, "cache": cache, "weighted": weighted,
                        "seed": seed, "params": count_params(model),
                        "size_mb": size_mb(path), "epochs": args.epochs,
                        "train_s": round(time.time() - t0, 1),
                        "checkpoint": name, **m})
                    print(f"  -> {name}: acc {m['acc']:.2f} sens {m['sens']:.2f}")


def load_baseline(arch, cache, seed, weighted=False):
    name = f"{arch}_{cache}{'_w' if weighted else ''}_s{seed}"
    path = os.path.join(CKPT_DIR, f"{name}.pth")
    model = build_model(arch)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(DEVICE), name


# ── Stage: pruning ───────────────────────────────────────────────────────────

def prunable(model):
    return [(m, "weight") for m in model.modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))]


def apply_prune(model, amount):
    prune.global_unstructured(prunable(model),
                              pruning_method=prune.L1Unstructured,
                              amount=amount)


def strip_masks(model):
    for m, n in prunable(model):
        try:
            prune.remove(m, n)
        except ValueError:
            pass


def sparsity(model):
    tot = nz = 0
    for m, _ in prunable(model):
        w = m.weight.detach()
        tot += w.numel()
        nz += int((w != 0).sum())
    return 100.0 * (1 - nz / max(tot, 1))


def stage_prune(args):
    for arch in args.arch:
        cache = args.caches[0]
        for seed in args.seeds:
            tr, va, te = make_loaders(cache, seed)

            # One-shot pruning at each sparsity level.
            for amount in (0.25, 0.50, 0.75):
                set_seed(seed)
                model, _ = load_baseline(arch, cache, seed)
                apply_prune(model, amount)
                pre = evaluate(model, te, DEVICE, n_boot=0)
                model = fit(model, tr, va, args.ft_epochs, lr=args.lr,
                            tag=f"prune{int(amount * 100)}/{arch}/s{seed}")
                strip_masks(model)
                path = save_ckpt(model, f"{arch}_prune{int(amount * 100)}_s{seed}")
                m = evaluate(model, te, DEVICE)
                append_result("prune.csv", {
                    "arch": arch, "seed": seed, "schedule": "one-shot",
                    "target_sparsity": amount, "actual_sparsity": sparsity(model),
                    "acc_pre_ft": pre["acc"], "sens_pre_ft": pre["sens"],
                    "size_mb": size_mb(path), **m})

            # Iterative pruning to 75%: the paper speculates this recovers
            # better than one-shot; here it is actually measured.
            set_seed(seed)
            model, _ = load_baseline(arch, cache, seed)
            for step in range(args.iter_steps):
                apply_prune(model, args.iter_amount)
                model = fit(model, tr, va, args.iter_ft_epochs, lr=args.lr,
                            tag=f"iter{step + 1}/{arch}/s{seed}")
            strip_masks(model)
            path = save_ckpt(model, f"{arch}_pruneiter_s{seed}")
            m = evaluate(model, te, DEVICE)
            append_result("prune.csv", {
                "arch": arch, "seed": seed, "schedule": "iterative",
                "target_sparsity": 1 - (1 - args.iter_amount) ** args.iter_steps,
                "actual_sparsity": sparsity(model),
                "size_mb": size_mb(path), **m})


# ── Stage: distillation (with a from-scratch control) ────────────────────────

def stage_distill(args):
    cache = args.caches[0]
    teacher_arch = args.arch[0]
    for seed in args.seeds:
        tr, va, te = make_loaders(cache, seed, three_channel=True)
        teacher, tname = load_baseline(teacher_arch, cache, seed)

        for mode in ("distilled", "scratch"):
            set_seed(seed)
            student = build_student(pretrained=args.pretrained_student)
            t0 = time.time()
            student = fit(student, tr, va, args.distill_epochs, lr=args.lr,
                          teacher=teacher if mode == "distilled" else None,
                          temperature=args.temperature, alpha=args.alpha,
                          cosine=True, tag=f"{mode}/s{seed}")
            path = save_ckpt(student, f"student_{mode}_s{seed}")
            m = evaluate(student, te, DEVICE)
            append_result("distill.csv", {
                "mode": mode, "seed": seed, "teacher": tname,
                "pretrained": args.pretrained_student,
                "params": count_params(student), "size_mb": size_mb(path),
                "train_s": round(time.time() - t0, 1), **m})
            print(f"  -> student/{mode}/s{seed}: acc {m['acc']:.2f} sens {m['sens']:.2f}")


# ── Stage: quantization, combined compression, ONNX latency ──────────────────

def onnx_latency(model, path, n=100, three_channel=False):
    import onnxruntime as ort
    ch = 3 if three_channel else 1
    dummy = torch.randn(1, ch, 224, 224)
    torch.onnx.export(model.cpu().eval(), dummy, path,
                      input_names=["input"], output_names=["logits"],
                      opset_version=13, dynamic_axes=None)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = dummy.numpy()
    for _ in range(10):
        sess.run(None, {"input": x})
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 25)), \
        float(np.percentile(times, 75))


def stage_quantize(args):
    cache = args.caches[0]
    arch = args.arch[0]
    os.makedirs(os.path.join(REPO, "deploy_repro"), exist_ok=True)
    for seed in args.seeds:
        _, _, te = make_loaders(cache, seed)

        variants = {"fp32": load_baseline(arch, cache, seed)[0]}
        p25 = os.path.join(CKPT_DIR, f"{arch}_prune25_s{seed}.pth")
        if os.path.exists(p25):
            m = build_model(arch)
            m.load_state_dict(torch.load(p25, map_location="cpu"))
            variants["pruned25"] = m.to(DEVICE)

        for base_name, base in list(variants.items()):
            # FP16 (GPU only: half-precision conv is not supported on CPU).
            if DEVICE.type == "cuda":
                half = copy.deepcopy(base).half()
                path = save_ckpt(half, f"{arch}_{base_name}_fp16_s{seed}")
                r = evaluate(half, te, DEVICE)
                append_result("quantize.csv", {
                    "arch": arch, "seed": seed, "base": base_name,
                    "quant": "fp16", "size_mb": size_mb(path), **r})

            # INT8 dynamic: torch only quantizes Linear layers dynamically,
            # which is why the size reduction is far short of 4x.
            q = torch.quantization.quantize_dynamic(
                copy.deepcopy(base).cpu().eval(), {nn.Linear}, dtype=torch.qint8)
            path = save_ckpt(q, f"{arch}_{base_name}_int8_s{seed}")
            r = evaluate(q, te, torch.device("cpu"))
            append_result("quantize.csv", {
                "arch": arch, "seed": seed, "base": base_name,
                "quant": "int8_dynamic", "size_mb": size_mb(path), **r})

            # ONNX export + CPU latency.
            opath = os.path.join(REPO, "deploy_repro",
                                 f"{arch}_{base_name}_s{seed}.onnx")
            try:
                med, q1, q3 = onnx_latency(copy.deepcopy(base), opath)
                append_result("latency.csv", {
                    "arch": arch, "seed": seed, "base": base_name,
                    "onnx_mb": size_mb(opath), "lat_ms_median": med,
                    "lat_ms_p25": q1, "lat_ms_p75": q3,
                    "device": "cpu", "runs": 100})
            except Exception as exc:                        # noqa: BLE001
                print(f"  ! ONNX/latency failed for {base_name}: {exc}")
            base.to(DEVICE)


# ── Stage: external validation on Montgomery + Shenzhen ──────────────────────

def stage_external(args):
    """
    Evaluate on the cohorts the original paper actually used. Labels are
    encoded in the filename suffix: *_0.png normal, *_1.png TB.
    """
    import glob

    import cv2
    if not args.external_path:
        print("--external-path not given; skipping")
        return
    paths = sorted(glob.glob(os.path.join(args.external_path, "**", "*.png"),
                             recursive=True))
    paths = [p for p in paths if os.path.basename(p)[:-4].endswith(("_0", "_1"))]
    if not paths:
        print(f"no MC/SZ-style images under {args.external_path}; skipping")
        return

    from build_cache import process_faithful
    imgs, labels, cohort = [], [], []
    for p in paths:
        try:
            imgs.append(process_faithful(p))
        except Exception:                                    # noqa: BLE001
            imgs.append(cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (224, 224)))
        labels.append(int(os.path.basename(p)[:-4][-1]))
        cohort.append("montgomery" if "MCUCXR" in os.path.basename(p) else "shenzhen")
    x = torch.from_numpy(np.stack(imgs).astype(np.float32) / 255.0).unsqueeze(1)
    x = (x - 0.5) / 0.5
    y = np.array(labels)
    print(f"external set: {len(y)} images ({y.sum()} TB)")

    from common import compute_metrics
    for arch in args.arch:
        for seed in args.seeds:
            try:
                model, name = load_baseline(arch, args.caches[0], seed)
            except FileNotFoundError:
                continue
            model.eval()
            probs = []
            with torch.no_grad():
                for i in range(0, len(x), 64):
                    logits = model(x[i:i + 64].to(DEVICE))
                    probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            probs = np.array(probs)
            for subset in ("all", "montgomery", "shenzhen"):
                mask = np.ones(len(y), bool) if subset == "all" \
                    else np.array([c == subset for c in cohort])
                if mask.sum() == 0:
                    continue
                m = compute_metrics(y[mask], probs[mask])
                append_result("external.csv", {
                    "arch": arch, "seed": seed, "checkpoint": name,
                    "cohort": subset, "n": int(mask.sum()), **m})


# ── Stage: phone-capture robustness ──────────────────────────────────────────

PHONE_COMPONENTS = ["brightness", "blur", "moire", "rotation", "glare"]


def stage_phone(args):
    cache = args.caches[0]

    def perturb_factory(components):
        return lambda img, rng: phone_perturb(img, rng, components=tuple(components))

    conditions = {"clean": None, "phone_all": PHONE_COMPONENTS}
    for c in PHONE_COMPONENTS:                    # single-distortion ablation
        conditions[f"only_{c}"] = [c]

    for arch in args.arch:
        for seed in args.seeds:
            try:
                model, name = load_baseline(arch, cache, seed)
            except FileNotFoundError:
                continue
            for cond, comps in conditions.items():
                ds = CachedTBDataset(SPLITS["test"], cache_name=cache,
                                     perturb=None if comps is None
                                     else perturb_factory(comps),
                                     perturb_seed=1234)
                dl = DataLoader(ds, batch_size=64, num_workers=2)
                m = evaluate(model, dl, DEVICE)
                append_result("phone.csv", {
                    "arch": arch, "seed": seed, "checkpoint": name,
                    "training": "clean", "condition": cond, **m})

            # Does training with the distortions on recover the collapse?
            if args.phone_finetune:
                set_seed(seed)
                tr, va, _ = make_loaders(cache, seed,
                                         train_perturb=perturb_factory(PHONE_COMPONENTS))
                robust, _ = load_baseline(arch, cache, seed)
                robust = fit(robust, tr, va, args.ft_epochs, lr=args.lr,
                             tag=f"phoneft/{arch}/s{seed}")
                save_ckpt(robust, f"{arch}_phoneft_s{seed}")
                for cond, comps in (("clean", None), ("phone_all", PHONE_COMPONENTS)):
                    ds = CachedTBDataset(SPLITS["test"], cache_name=cache,
                                         perturb=None if comps is None
                                         else perturb_factory(comps),
                                         perturb_seed=1234)
                    m = evaluate(robust, DataLoader(ds, batch_size=64,
                                                    num_workers=2), DEVICE)
                    append_result("phone.csv", {
                        "arch": arch, "seed": seed,
                        "checkpoint": f"{arch}_phoneft_s{seed}",
                        "training": "phone_augmented", "condition": cond, **m})


# ── Summary ──────────────────────────────────────────────────────────────────

def stage_summary(args):
    """Aggregate every stage CSV into mean +/- std across seeds."""
    out = {}
    specs = {
        "baseline.csv": ["arch", "cache", "weighted"],
        "prune.csv": ["arch", "schedule", "target_sparsity"],
        "distill.csv": ["mode", "pretrained"],
        "quantize.csv": ["arch", "base", "quant"],
        "latency.csv": ["arch", "base"],
        "external.csv": ["arch", "cohort"],
        "phone.csv": ["arch", "training", "condition"],
    }
    for fname, keys in specs.items():
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        keys = [k for k in keys if k in df.columns]
        metrics = [c for c in ("acc", "sens", "spec", "auc", "size_mb",
                               "lat_ms_median", "actual_sparsity")
                   if c in df.columns]
        agg = df.groupby(keys)[metrics].agg(["mean", "std", "count"]).round(4)
        agg.to_csv(os.path.join(RESULTS_DIR, f"summary_{fname}"))
        out[fname] = agg
        print(f"\n=== {fname} ===\n{agg}")
    with open(os.path.join(RESULTS_DIR, "environment.json"), "w") as fh:
        json.dump({
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "numpy": np.__version__, "pandas": pd.__version__,
        }, fh, indent=2)
    return out


STAGES = {"audit": stage_audit, "baseline": stage_baseline,
          "prune": stage_prune, "distill": stage_distill,
          "quantize": stage_quantize, "external": stage_external,
          "phone": stage_phone, "summary": stage_summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--arch", nargs="+", default=["compact"],
                    choices=["compact", "full"])
    ap.add_argument("--caches", nargs="+", default=["faithful"],
                    choices=["faithful", "simple"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--weighted", nargs="+", type=int, default=[0],
                    help="0 = plain CE, 1 = inverse-frequency class weights")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--ft-epochs", type=int, default=5)
    ap.add_argument("--distill-epochs", type=int, default=15)
    ap.add_argument("--iter-steps", type=int, default=8)
    ap.add_argument("--iter-amount", type=float, default=0.16)
    ap.add_argument("--iter-ft-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--pretrained-student", type=int, default=1)
    ap.add_argument("--phone-finetune", action="store_true")
    ap.add_argument("--external-path", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.weighted = [bool(w) for w in args.weighted]

    csv_name = f"{args.stage}.csv"
    if args.stage != "summary" and stage_done(csv_name) and not args.force:
        print(f"results/{csv_name} exists; skipping (use --force to rerun)")
        return

    print(f"stage={args.stage} device={DEVICE} seeds={args.seeds}")
    t0 = time.time()
    STAGES[args.stage](args)
    print(f"stage {args.stage} finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
