"""
Render every figure from results/*.csv into figures/.

Separate from analyze.py because this needs matplotlib while the analysis
deliberately does not.

  python src/make_figures.py
"""

import csv
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")
FIG = os.path.join(REPO, "figures")
COLORS = {"compact": "#1f77b4", "full": "#d62728",
          "distilled": "#1f77b4", "scratch": "#7f7f7f"}


def load(name):
    path = os.path.join(RES, name)
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def agg(rows, keys, metric):
    g = defaultdict(list)
    for r in rows:
        try:
            g[tuple(r.get(k, "") for k in keys)].append(float(r[metric]))
        except (KeyError, ValueError, TypeError):
            pass
    return {k: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)
            for k, v in g.items()}


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, REPO))


def fig_baseline():
    rows = load("baseline.csv")
    if not rows:
        return
    g = agg(rows, ["arch", "cache", "weighted"], "sens")
    keys = sorted(g)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = [f"{k[0]}\n{k[1]}{' + weighted' if k[2] == 'True' else ''}"
              for k in keys]
    ax.bar(range(len(keys)), [g[k][0] for k in keys],
           yerr=[g[k][1] for k in keys], capsize=4,
           color=[COLORS.get(k[0], "#888") for k in keys])
    ax.axhline(90, ls="--", lw=1, color="k")
    ax.text(len(keys) - 0.4, 90.4, "WHO triage target", fontsize=8, ha="right")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("sensitivity (%)")
    ax.set_ylim(80, 102)
    ax.set_title("Baseline sensitivity, mean +/- sd over 5 seeds")
    save(fig, "baseline_sensitivity.png")


def fig_pruning():
    rows = load("prune.csv")
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, arch in zip(axes, ("compact", "full")):
        sub = [r for r in rows if r["arch"] == arch]
        for kind, sched, style in (("unstructured", "one-shot", "-o"),
                                   ("structured", "one-shot", "-s"),
                                   ("unstructured", "iterative", "--^")):
            g = agg([r for r in sub if r.get("kind") == kind
                     and r["schedule"] == sched],
                    ["target_sparsity"], "sens")
            if not g:
                continue
            pts = sorted((float(k[0]) * 100, v) for k, v in g.items())
            ax.errorbar([p[0] for p in pts], [p[1][0] for p in pts],
                        yerr=[p[1][1] for p in pts], fmt=style, capsize=3,
                        label=f"{kind}, {sched}")
        ax.set_title(arch)
        ax.set_xlabel("target sparsity (%)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("sensitivity (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Gradual pruning removes the sensitivity cliff at 75%")
    save(fig, "pruning_sensitivity.png")


def fig_distill():
    rows = load("distill.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    modes = sorted({r["mode"] for r in rows})
    for i, metric in enumerate(("acc", "sens", "spec")):
        g = agg(rows, ["mode"], metric)
        xs = [i + (j - 0.5) * 0.35 for j in range(len(modes))]
        ax.bar(xs, [g[(m,)][0] for m in modes], width=0.32,
               yerr=[g[(m,)][1] for m in modes], capsize=3,
               color=[COLORS.get(m, "#888") for m in modes],
               label=None if i else modes)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS.get(m, "#888"))
               for m in modes]
    ax.legend(handles, modes, fontsize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["accuracy", "sensitivity", "specificity"])
    ax.set_ylabel("%")
    ax.set_ylim(80, 102)
    ax.set_title("Distillation vs from-scratch control")
    save(fig, "distillation_control.png")


def fig_quantize():
    rows = load("quantize.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    g_s = agg(rows, ["arch", "base", "quant"], "size_mb")
    g_y = agg(rows, ["arch", "base", "quant"], "sens")
    for k in sorted(g_s):
        ax.errorbar(g_s[k][0], g_y[k][0], yerr=g_y[k][1], fmt="o",
                    color=COLORS.get(k[0], "#888"), capsize=3)
        ax.annotate(f"{k[1]}/{k[2]}", (g_s[k][0], g_y[k][0]),
                    textcoords="offset points", xytext=(6, -3), fontsize=7)
    ax.set_xlabel("model size (MB)")
    ax.set_ylabel("sensitivity (%)")
    ax.grid(alpha=0.3)
    ax.set_title("Size against sensitivity after quantization")
    save(fig, "size_vs_sensitivity.png")


def fig_phone():
    rows = load("phone.csv")
    if not rows:
        return
    conds = ["clean", "only_brightness", "only_blur", "only_moire",
             "only_rotation", "only_glare", "phone_all"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, metric in zip(axes, ("acc", "sens")):
        for arch in ("compact", "full"):
            g = agg([r for r in rows if r["arch"] == arch
                     and r["training"] == "clean"], ["condition"], metric)
            xs = [c for c in conds if (c,) in g]
            ax.errorbar(range(len(xs)), [g[(c,)][0] for c in xs],
                        yerr=[g[(c,)][1] for c in xs], fmt="-o", capsize=3,
                        color=COLORS[arch], label=arch)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([c.replace("only_", "") for c in conds],
                           rotation=30, fontsize=8)
        ax.set_title({"acc": "accuracy", "sens": "sensitivity"}[metric])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("%")
    axes[0].legend(fontsize=8)
    fig.suptitle("Per-distortion attribution under simulated phone capture")
    save(fig, "phone_attribution.png")


def fig_roc():
    """ROC curves from the per-sample probability dumps."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for arch in ("compact", "full"):
        curves = []
        for seed in range(5):
            path = os.path.join(RES, "probs", f"{arch}_faithful_s{seed}.csv")
            if not os.path.exists(path):
                continue
            rows = list(csv.DictReader(open(path)))
            pts = sorted(((float(r["prob"]), int(r["label"])) for r in rows),
                         reverse=True)
            P = sum(y for _, y in pts) or 1
            N = len(pts) - P or 1
            tp = fp = 0
            xs, ys = [0.0], [0.0]
            for _, y in pts:
                tp += y
                fp += 1 - y
                xs.append(fp / N)
                ys.append(tp / P)
            curves.append((xs, ys))
        for i, (xs, ys) in enumerate(curves):
            ax.plot(xs, ys, color=COLORS[arch], alpha=0.45, lw=1,
                    label=arch if i == 0 else None)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC on the clean test split (5 seeds each)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "roc_curves.png")


def fig_external():
    rows = load("external.csv")
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    cohorts = ["montgomery", "shenzhen", "all"]
    for arch in ("compact", "full"):
        g = agg([r for r in rows if r["arch"] == arch], ["cohort"], "auc")
        xs = [c for c in cohorts if (c,) in g]
        ax.errorbar(range(len(xs)), [g[(c,)][0] for c in xs],
                    yerr=[g[(c,)][1] for c in xs], fmt="-o", capsize=3,
                    color=COLORS[arch], label=arch)
    ax.axhline(0.5, ls="--", color="k", lw=1)
    ax.text(0.02, 0.505, "chance", fontsize=8)
    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels(cohorts)
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("External cohorts: below chance, unresolved (see ANALYSIS.md)")
    save(fig, "external_auc.png")


if __name__ == "__main__":
    for f in (fig_baseline, fig_pruning, fig_distill, fig_quantize,
              fig_phone, fig_roc, fig_external):
        try:
            f()
        except Exception as exc:                      # noqa: BLE001
            print(f"  ! {f.__name__} failed: {type(exc).__name__}: {exc}")
