"""
Every experiment in the reproducibility study, as resumable stages.

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

import common
from common import (CachedTBDataset, append_result, count_params, evaluate,
                    loader_generator, model_dtype, phone_perturb, seed_worker,
                    set_seed, stage_done)
from models_repro import build_model, build_student

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = {s: os.path.join(REPO, "data_splits", f"{s}.csv")
          for s in ("train", "val", "test")}
CKPT_DIR = os.path.join(REPO, "checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- Data ---------------------------------------------------------------------

def make_loaders(cache, seed, batch_size=32, three_channel=False,
                 augment_train=False, train_perturb=None, dual_train=False):
    kw = dict(cache_name=cache, three_channel=three_channel)
    train = CachedTBDataset(SPLITS["train"], augment=augment_train,
                            perturb=train_perturb, perturb_seed=seed,
                            dual=dual_train, **kw)
    val = CachedTBDataset(SPLITS["val"], **kw)
    test = CachedTBDataset(SPLITS["test"], **kw)
    dl_kw = dict(num_workers=2, worker_init_fn=seed_worker,
                 generator=loader_generator(seed), pin_memory=True)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, **dl_kw),
        DataLoader(val, batch_size=batch_size, shuffle=False, **dl_kw),
        DataLoader(test, batch_size=batch_size, shuffle=False, **dl_kw),
    )


def class_weight_tensor():
    df = pd.read_csv(SPLITS["train"], header=None, names=["f", "y"])
    counts = df["y"].value_counts().sort_index().values
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


# -- Generic training loop ----------------------------------------------------

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
        for batch in train_loader:
            # A dual dataset yields (teacher_input, student_input, label); a
            # normal one yields (input, label) and both models share it.
            if len(batch) == 3:
                x_t, x_s, y = batch
                x_t = x_t.to(DEVICE, non_blocking=True)
            else:
                x_s, y = batch
                x_t = None
            x_s, y = x_s.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            if x_t is None:
                x_t = x_s
            opt.zero_grad(set_to_none=True)
            logits = model(x_s)
            loss = ce(logits, y)
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(x_t)
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
        # Learning curves: a model still improving at the last epoch is
        # undertrained, and that changes how its result should be read.
        append_result("history.csv", {
            "tag": tag, "epoch": ep + 1,
            "train_loss": total / max(len(train_loader), 1),
            "val_acc": m["acc"], "val_sens": m["sens"], "val_spec": m["spec"]})
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


@torch.no_grad()
def dump_probs(model, dataset, tag, device=DEVICE):
    """
    Persist per-sample TB probabilities.

    With these on disk, ROC curves, operating-point selection (paper section 5.8),
    paired bootstrap comparisons between models and DeLong tests are all
    plain post-processing - no retraining and no re-inference needed.
    """
    dl = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    model.eval()
    dtype = model_dtype(model)
    probs = []
    for x, _ in dl:
        logits = model(x.to(device=device, dtype=dtype))
        probs.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy().tolist())
    out_dir = os.path.join(common.RESULTS_DIR, "probs")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}.csv")
    pd.DataFrame({"filename": dataset.files, "label": dataset.labels,
                  "prob": probs}).to_csv(path, index=False)
    return path


# -- Stage: baselines ---------------------------------------------------------

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
                    # The epoch count belongs in the name. Without it a
                    # longer-budget arm overwrites the checkpoint of the
                    # default one, and every later stage silently loads the
                    # wrong model while the CSV still distinguishes them.
                    suffix = "" if args.epochs == 10 else f"_e{args.epochs}"
                    name = (f"{arch}_{cache}{'_w' if weighted else ''}"
                            f"{suffix}_s{seed}")
                    path = save_ckpt(model, name)
                    m = evaluate(model, te, DEVICE)
                    dump_probs(model, te.dataset, name)
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


# -- Stage: pruning -----------------------------------------------------------

def prunable(model):
    return [(m, "weight") for m in model.modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))]


def apply_prune(model, amount, structured=False):
    """
    structured=False: global L1 unstructured. Zeroes individual weights, so the
        .pth file size does not change - the point paper section 4.3 makes.
    structured=True: per-layer L1 filter pruning (whole output channels). The
        tensors keep their shape here too, but every zeroed filter is a channel
        that a compaction pass could physically remove, so the FLOP reduction
        is real in a way unstructured sparsity is not.
    """
    if not structured:
        prune.global_unstructured(prunable(model),
                                  pruning_method=prune.L1Unstructured,
                                  amount=amount)
        return
    for m, name in prunable(model):
        if isinstance(m, nn.Conv2d) and m.weight.shape[0] > 1 and m.groups == 1:
            prune.ln_structured(m, name, amount=amount, n=1, dim=0)


def channel_sparsity(model):
    """Fraction of output filters that are entirely zero."""
    tot = dead = 0
    for m, _ in prunable(model):
        if isinstance(m, nn.Conv2d):
            w = m.weight.detach()
            tot += w.shape[0]
            dead += int((w.flatten(1).abs().sum(1) == 0).sum())
    return 100.0 * dead / max(tot, 1)


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

            # One-shot pruning at each sparsity level, unstructured and
            # structured. Structured is the variant that can actually shrink
            # the model, which is the paper's stated deployment goal.
            for structured in (False, True):
                kind = "structured" if structured else "unstructured"
                for amount in (0.25, 0.50, 0.75):
                    set_seed(seed)
                    model, _ = load_baseline(arch, cache, seed)
                    apply_prune(model, amount, structured=structured)
                    pre = evaluate(model, te, DEVICE, n_boot=0)
                    model = fit(model, tr, va, args.ft_epochs, lr=args.lr,
                                tag=f"{kind[:6]}{int(amount * 100)}/{arch}/s{seed}")
                    strip_masks(model)
                    name = f"{arch}_{kind}{int(amount * 100)}_s{seed}"
                    path = save_ckpt(model, name)
                    m = evaluate(model, te, DEVICE)
                    dump_probs(model, te.dataset, name)
                    append_result("prune.csv", {
                        "arch": arch, "seed": seed, "schedule": "one-shot",
                        "kind": kind, "target_sparsity": amount,
                        "actual_sparsity": sparsity(model),
                        "channel_sparsity": channel_sparsity(model),
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
            dump_probs(model, te.dataset, f"{arch}_pruneiter_s{seed}")
            append_result("prune.csv", {
                "arch": arch, "seed": seed, "schedule": "iterative",
                "kind": "unstructured",
                "target_sparsity": 1 - (1 - args.iter_amount) ** args.iter_steps,
                "actual_sparsity": sparsity(model),
                "channel_sparsity": channel_sparsity(model),
                "size_mb": size_mb(path), **m})


# -- Stage: distillation (with a from-scratch control) ------------------------

def stage_distill(args):
    cache = args.caches[0]
    teacher_arch = args.arch[0]
    for seed in args.seeds:
        tr, va, te = make_loaders(cache, seed, three_channel=True,
                                  dual_train=True)
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
            dump_probs(student, te.dataset, f"student_{mode}_s{seed}")
            append_result("distill.csv", {
                "mode": mode, "seed": seed, "teacher": tname,
                "pretrained": args.pretrained_student,
                "params": count_params(student), "size_mb": size_mb(path),
                "train_s": round(time.time() - t0, 1), **m})
            print(f"  -> student/{mode}/s{seed}: acc {m['acc']:.2f} sens {m['sens']:.2f}")


# -- Stage: quantization, combined compression, ONNX latency ------------------

def onnx_latency(model, path, n=100, three_channel=False):
    import onnxruntime as ort
    ch = 3 if three_channel else 1
    dummy = torch.randn(1, ch, 224, 224)
    model = model.cpu().eval()

    # torch 2.6+ routes torch.onnx.export through the dynamo exporter by
    # default and the legacy path rejects some argument combinations, so try
    # the variants in order and report every failure rather than swallowing it.
    attempts = [
        dict(opset_version=17, dynamo=False),
        dict(opset_version=17),
        dict(opset_version=13, dynamo=False),
    ]
    errors = []
    for kw in attempts:
        try:
            torch.onnx.export(model, dummy, path, input_names=["input"],
                              output_names=["logits"], **kw)
            break
        except Exception as exc:                      # noqa: BLE001
            errors.append(f"{kw}: {type(exc).__name__}: {exc}")
    else:
        raise RuntimeError("all ONNX export attempts failed:\n  "
                           + "\n  ".join(errors))
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
    os.makedirs(os.path.join(REPO, "deploy_repro"), exist_ok=True)
    for arch in args.arch:
        _stage_quantize_arch(args, arch, cache)


def _stage_quantize_arch(args, arch, cache):
    for seed in args.seeds:
        _, _, te = make_loaders(cache, seed)

        variants = {"fp32": load_baseline(arch, cache, seed)[0]}
        p25 = os.path.join(CKPT_DIR, f"{arch}_unstructured25_s{seed}.pth")
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
                import traceback
                print(f"  ! ONNX/latency failed for {base_name}: {exc}")
                traceback.print_exc()
                append_result("latency_failures.csv", {
                    "arch": arch, "seed": seed, "base": base_name,
                    "error": str(exc)[:500]})
            base.to(DEVICE)


# -- Stage: external validation on Montgomery + Shenzhen ----------------------

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
    # Montgomery ships left/right lung masks whose filenames match the
    # radiographs; without this filter the cohort triples (414 vs 138).
    paths = [p for p in paths
             if os.path.basename(p)[:-4].endswith(("_0", "_1"))
             and "mask" not in p.lower()]
    if not paths:
        print(f"no MC/SZ-style images under {args.external_path}; skipping")
        return

    # Match the pipeline the models were trained under. With labels ruled out
    # as the cause of the sub-chance AUC, preprocessing is the open suspect,
    # so this has to be selectable rather than hardcoded.
    from build_cache import PROCESSORS
    process = PROCESSORS[args.caches[0]]
    print(f"external preprocessing: {args.caches[0]}")
    imgs, labels, cohort = [], [], []
    for p in paths:
        try:
            imgs.append(process(p))
        except Exception:                                    # noqa: BLE001
            imgs.append(cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (224, 224)))
        labels.append(int(os.path.basename(p)[:-4][-1]))
        cohort.append("montgomery" if "MCUCXR" in os.path.basename(p) else "shenzhen")
    x = torch.from_numpy(np.stack(imgs).astype(np.float32) / 255.0).unsqueeze(1)
    x = (x - 0.5) / 0.5
    y = np.array(labels)

    # Published class balances: Montgomery 80 normal / 58 TB, Shenzhen
    # 326 normal / 336 TB. If the parsed counts come out swapped, the _0/_1
    # convention is being read backwards and every AUC below 0.5 is explained.
    expected = {"montgomery": (80, 58), "shenzhen": (326, 336)}
    print(f"external set: {len(y)} images ({int(y.sum())} TB)")
    for name, (exp_neg, exp_pos) in expected.items():
        mask = np.array([c == name for c in cohort])
        if not mask.any():
            continue
        pos = int(y[mask].sum())
        neg = int(mask.sum() - pos)
        flag = "OK" if (neg, pos) == (exp_neg, exp_pos) else \
            ("LABELS LOOK INVERTED" if (neg, pos) == (exp_pos, exp_neg)
             else "UNEXPECTED")
        print(f"  {name}: {neg} normal / {pos} TB "
              f"(expected {exp_neg}/{exp_pos}) -> {flag}")
        append_result("external_labels.csv", {
            "cohort": name, "parsed_normal": neg, "parsed_tb": pos,
            "expected_normal": exp_neg, "expected_tb": exp_pos, "verdict": flag})

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
                dtype = model_dtype(model)
                for i in range(0, len(x), 64):
                    logits = model(x[i:i + 64].to(device=DEVICE, dtype=dtype))
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
                    "preproc": args.caches[0], "cohort": subset,
                    "n": int(mask.sum()), **m})


# -- Stage: cohort overlap ----------------------------------------------------

def _dhash(img, size=8):
    """64-bit difference hash; stable under rescaling and mild intensity shifts."""
    import cv2
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    h = 0
    for b in bits.flatten():
        h = (h << 1) | int(b)
    return h


def stage_overlap(args):
    """
    Are the Montgomery/Shenzhen images already inside the training cohort?

    The Rahman database aggregates several sources and the NLM sets are among
    them. If those images sit in the training split then the external stage is
    not external, and its numbers are optimistic rather than held out.
    """
    import glob

    import cv2
    if not args.external_path:
        print("--external-path not given; skipping")
        return

    members = []
    for sp in ("train", "val", "test"):
        df = pd.read_csv(SPLITS[sp], header=None, names=["f", "y"])
        members += [(os.path.basename(f), sp) for f in df["f"]]

    arr = np.load(os.path.join(common.CACHE_DIR, "faithful.npy"), mmap_mode="r")
    index = pd.read_csv(os.path.join(common.CACHE_DIR, "faithful_index.csv"))
    row_of = dict(zip(index["filename"], index["row"]))

    cohort = {}
    for name, sp in members:
        if name in row_of:
            cohort.setdefault(_dhash(np.array(arr[row_of[name]])),
                              []).append((name, sp))

    ext = [p for p in glob.glob(os.path.join(args.external_path, "**", "*.png"),
                                recursive=True)
           if os.path.basename(p)[:-4].endswith(("_0", "_1"))
           and "mask" not in p.lower()]

    from build_cache import process_faithful
    hits = 0
    distances = []
    for p in ext:
        try:
            img = process_faithful(p)
        except Exception:                                # noqa: BLE001
            img = cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (224, 224))
        h = _dhash(img)
        # Nearest neighbour, not first-hit: a chest radiograph downsampled to
        # 9x8 shares gross thoracic structure with every other chest
        # radiograph, so a loose radius matches almost anything. Record the
        # actual distance and let the threshold be justified from it.
        best_d, matches = 64, []
        for hh, v in cohort.items():
            d = bin(hh ^ h).count("1")
            if d < best_d:
                best_d, matches = d, v
                if d == 0:
                    break
        distances.append(best_d)
        if best_d <= args.hash_radius:
            hits += 1
            append_result("overlap.csv", {
                "external": os.path.basename(p),
                "cohort": "montgomery" if "MCUCXR" in p else "shenzhen",
                "matched_training_file": matches[0][0],
                "matched_split": matches[0][1],
                "hamming": best_d})
    print(f"overlap: {hits}/{len(ext)} external images within Hamming "
          f"{args.hash_radius} of a training image")
    hist = {d: distances.count(d) for d in sorted(set(distances))}
    print("nearest-neighbour distance histogram:", hist)
    print("  If the counts rise smoothly from small distances with no gap, the")
    print("  hash is measuring anatomy rather than identity and no threshold")
    print("  on it is trustworthy.")
    for d, n in hist.items():
        append_result("overlap_distances.csv", {"hamming": d, "count": n})
    if hits == 0:
        append_result("overlap.csv", {"external": "(none)", "cohort": "-",
                                      "matched_training_file": "-",
                                      "matched_split": "-", "hamming": -1})


# -- Stage: phone-capture robustness ------------------------------------------

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
                dump_probs(model, ds, f"{name}__{cond}")
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


# -- Stage: probability dumps for existing checkpoints ------------------------

def stage_dumpprobs(args):
    """
    Write results/probs/<name>.csv for checkpoints that predate probability
    dumping. Inference only, no training, so it costs seconds per model and
    unlocks the paired significance tests those checkpoints are missing.
    """
    if not os.path.isdir(CKPT_DIR):
        print("no checkpoints/ directory")
        return
    cache = args.caches[0]
    _, _, te = make_loaders(cache, 0)
    _, _, te3 = make_loaders(cache, 0, three_channel=True)

    for fname in sorted(os.listdir(CKPT_DIR)):
        if not fname.endswith(".pth"):
            continue
        name = fname[:-4]
        out = os.path.join(common.RESULTS_DIR, "probs", f"{name}.csv")
        if os.path.exists(out) and not args.force:
            continue
        if "fp16" in name or "int8" in name:
            continue                     # quantized state dicts, handled elsewhere
        is_student = name.startswith("student")
        arch = "full" if name.startswith("full") else "compact"
        model = build_student(pretrained=False) if is_student \
            else build_model(arch)
        try:
            model.load_state_dict(torch.load(os.path.join(CKPT_DIR, fname),
                                             map_location="cpu"))
        except Exception as exc:                          # noqa: BLE001
            print(f"  ! {name}: {type(exc).__name__}: {exc}")
            continue
        model.to(DEVICE)
        dump_probs(model, (te3 if is_student else te).dataset, name)
        print(f"  dumped {name}")


# -- Stage: FLOPs -------------------------------------------------------------

def stage_flops(args):
    """
    Dense and effective multiply-accumulate counts.

    Dense FLOPs are identical across every pruning condition, because a zeroed
    weight is still multiplied. The number that reflects a real compute saving
    is the effective count, which charges only for output channels that are
    not entirely zero. Input-channel propagation is not modelled, so the
    effective figure is conservative: the true saving is at least this large.
    """
    shapes = {}

    def hook(mod, inp, out):
        shapes[mod] = (tuple(inp[0].shape), tuple(out.shape))

    def macs(model, count_dead=True):
        handles = [m.register_forward_hook(hook)
                   for m in model.modules()
                   if isinstance(m, (nn.Conv2d, nn.Linear))]
        model.eval()
        with torch.no_grad():
            model(torch.randn(1, 1, 224, 224).to(next(model.parameters()).device))
        for h in handles:
            h.remove()

        dense = eff = 0
        for m, (ishape, oshape) in shapes.items():
            w = m.weight.detach()
            if isinstance(m, nn.Conv2d):
                per_out = (w.shape[1] * w.shape[2] * w.shape[3])
                spatial = oshape[2] * oshape[3]
                dense += per_out * w.shape[0] * spatial
                live = int((w.flatten(1).abs().sum(1) != 0).sum()) if count_dead \
                    else w.shape[0]
                eff += per_out * live * spatial
            else:
                dense += w.shape[0] * w.shape[1]
                live = int((w.abs().sum(1) != 0).sum()) if count_dead else w.shape[0]
                eff += live * w.shape[1]
        shapes.clear()
        return dense, eff

    for arch in args.arch:
        model = build_model(arch).to(DEVICE)
        dense, _ = macs(model, count_dead=False)
        append_result("flops.csv", {
            "arch": arch, "checkpoint": "(untrained dense)",
            "params": count_params(model),
            "macs_dense_M": round(dense / 1e6, 2),
            "macs_effective_M": round(dense / 1e6, 2),
            "saving_pct": 0.0})
        print(f"  {arch}: {dense / 1e6:.1f}M MACs dense, "
              f"{count_params(model) / 1e6:.2f}M params")

    if not os.path.isdir(CKPT_DIR):
        return
    for fname in sorted(os.listdir(CKPT_DIR)):
        if not fname.endswith(".pth") or "fp16" in fname or "int8" in fname \
                or fname.startswith("student"):
            continue
        name = fname[:-4]
        arch = "full" if name.startswith("full") else "compact"
        model = build_model(arch)
        try:
            model.load_state_dict(torch.load(os.path.join(CKPT_DIR, fname),
                                             map_location="cpu"))
        except Exception:                                  # noqa: BLE001
            continue
        model.to(DEVICE)
        dense, eff = macs(model)
        append_result("flops.csv", {
            "arch": arch, "checkpoint": name,
            "params": count_params(model),
            "macs_dense_M": round(dense / 1e6, 2),
            "macs_effective_M": round(eff / 1e6, 2),
            "saving_pct": round(100.0 * (1 - eff / max(dense, 1)), 2)})


# -- Summary ------------------------------------------------------------------

def stage_summary(args):
    """Aggregate every stage CSV into mean +/- std across seeds."""
    out = {}
    specs = {
        "baseline.csv": ["arch", "cache", "weighted"],
        "prune.csv": ["arch", "schedule", "kind", "target_sparsity"],
        "distill.csv": ["mode", "pretrained"],
        "quantize.csv": ["arch", "base", "quant"],
        "latency.csv": ["arch", "base"],
        "external.csv": ["arch", "cohort"],
        "phone.csv": ["arch", "training", "condition"],
    }
    for fname, keys in specs.items():
        path = os.path.join(common.RESULTS_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        keys = [k for k in keys if k in df.columns]
        metrics = [c for c in ("acc", "sens", "spec", "auc", "size_mb",
                               "lat_ms_median", "actual_sparsity",
                               "channel_sparsity")
                   if c in df.columns]
        agg = df.groupby(keys)[metrics].agg(["mean", "std", "count"]).round(4)
        agg.to_csv(os.path.join(common.RESULTS_DIR, f"summary_{fname}"))
        out[fname] = agg
        print(f"\n=== {fname} ===\n{agg}")
    with open(os.path.join(common.RESULTS_DIR, "environment.json"), "w") as fh:
        json.dump({
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "numpy": np.__version__, "pandas": pd.__version__,
        }, fh, indent=2)
    return out


STAGES = {"baseline": stage_baseline, "prune": stage_prune,
          "dumpprobs": stage_dumpprobs, "flops": stage_flops,
          "overlap": stage_overlap, "distill": stage_distill,
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
    ap.add_argument("--hash-radius", type=int, default=0,
                    help="overlap: max dHash Hamming distance "
                         "counted as a duplicate (0 = exact)")
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
