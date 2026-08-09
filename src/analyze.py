"""
Turn results/*.csv into results/ANALYSIS.md.

Standard library only, on purpose: the analysis of a finished run should not
need a GPU environment, or even numpy, to reproduce. Figures are a separate
step (src/make_figures.py) because those do need matplotlib.

  python src/analyze.py
"""

import csv
import json
import os
import random
import statistics as st
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")
OUT = os.path.join(RES, "ANALYSIS.md")
N_BOOT = 2000


# -- small metric helpers ------------------------------------------------------

def rates(labels, probs, thr=0.5):
    tp = tn = fp = fn = 0
    for y, p in zip(labels, probs):
        pred = 1 if p >= thr else 0
        if pred == 1 and y == 1: tp += 1
        elif pred == 0 and y == 0: tn += 1
        elif pred == 1 and y == 0: fp += 1
        else: fn += 1
    n = max(tp + tn + fp + fn, 1)
    return {"acc": 100.0 * (tp + tn) / n,
            "sens": 100.0 * tp / max(tp + fn, 1),
            "spec": 100.0 * tn / max(tn + fp, 1)}


def auc(labels, probs):
    """Mann-Whitney U formulation, with ties handled by average ranks."""
    pairs = sorted(zip(probs, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = sum(1 for _, y in pairs if y == 1)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def sens_at_spec(labels, probs, target_spec=95.0):
    """Highest sensitivity reachable while holding specificity at target."""
    best = 0.0
    for thr in sorted(set(probs)):
        r = rates(labels, probs, thr)
        if r["spec"] >= target_spec:
            best = max(best, r["sens"])
    return best


def agg(rows, keys, metrics):
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = tuple(r.get(x, "") for x in keys)
        for m in metrics:
            try:
                g[k][m].append(float(r[m]))
            except (KeyError, ValueError, TypeError):
                pass
    return g


def fmt(vals, dp=2):
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.{dp}f}"
    return f"{st.mean(vals):.{dp}f} +/- {st.stdev(vals):.{dp}f}"


def load(name):
    path = os.path.join(RES, name)
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def table(w, header, rows):
    w(f"| {' | '.join(header)} |")
    w(f"|{'|'.join(['---'] * len(header))}|")
    for r in rows:
        w(f"| {' | '.join(str(c) for c in r)} |")
    w("")


# -- per-sample probability files ---------------------------------------------

def read_probs(tag):
    path = os.path.join(RES, "probs", f"{tag}.csv")
    if not os.path.exists(path):
        return None
    labels, probs = [], []
    for row in csv.DictReader(open(path)):
        labels.append(int(row["label"]))
        probs.append(float(row["prob"]))
    return labels, probs


def paired_compare(tag_a, tag_b, seeds, label_a, label_b):
    """
    Per-seed paired difference on the shared test set, plus a bootstrap CI on
    the seed-averaged predictions. Reported as A minus B.
    """
    diffs = {"auc": [], "sens": [], "acc": []}
    stacks = []
    for s in seeds:
        a, b = read_probs(tag_a.format(s=s)), read_probs(tag_b.format(s=s))
        if not a or not b:
            continue
        (ya, pa), (yb, pb) = a, b
        if ya != yb:
            return None                       # different test order; unsafe
        diffs["auc"].append((auc(ya, pa) - auc(yb, pb)) * 100)
        ra, rb = rates(ya, pa), rates(yb, pb)
        diffs["sens"].append(ra["sens"] - rb["sens"])
        diffs["acc"].append(ra["acc"] - rb["acc"])
        stacks.append((ya, pa, pb))
    if not stacks:
        return None

    labels = stacks[0][0]
    mean_a = [st.mean(x) for x in zip(*[s[1] for s in stacks])]
    mean_b = [st.mean(x) for x in zip(*[s[2] for s in stacks])]
    rng = random.Random(0)
    n = len(labels)
    boot = []
    for _ in range(N_BOOT):
        idx = [rng.randrange(n) for _ in range(n)]
        ys = [labels[i] for i in idx]
        if len(set(ys)) < 2:
            continue
        boot.append((auc(ys, [mean_a[i] for i in idx])
                     - auc(ys, [mean_b[i] for i in idx])) * 100)
    boot.sort()
    lo = boot[int(0.025 * len(boot))]
    hi = boot[int(0.975 * len(boot))]
    wins = sum(1 for d in diffs["auc"] if d > 0)
    return {"label": f"{label_a} - {label_b}", "diffs": diffs,
            "ci": (lo, hi), "wins": wins, "n": len(diffs["auc"]),
            "significant": lo > 0 or hi < 0}


# -- report -------------------------------------------------------------------

def main():
    lines = []
    w = lines.append

    manifest = {}
    mpath = os.path.join(RES, "run_manifest.json")
    if os.path.exists(mpath):
        manifest = json.load(open(mpath))
    seeds = manifest.get("seeds", [0, 1, 2, 3, 4])

    w("# Analysis")
    w("")
    w("Generated by `python src/analyze.py` from `results/*.csv`. Every figure")
    w("in `figures/` is derived from the same files.")
    w("")
    w("## Provenance")
    w("")
    env = manifest.get("env", {})
    table(w, ["field", "value"], [
        ["commit", manifest.get("commit", "?")[:12]],
        ["seeds", manifest.get("seeds", "?")],
        ["architectures", manifest.get("archs", "?")],
        ["preprocessing", manifest.get("caches", "?")],
        ["torch", env.get("torch", "?")],
        ["GPU", env.get("gpu", "?")],
        ["splits", "3360 train / 420 val / 420 test, leak-checked"],
    ])

    # ---- baseline
    rows = load("baseline.csv")
    if rows:
        w("## 1. Baselines")
        w("")
        g = agg(rows, ["arch", "cache", "weighted"], ("acc", "sens", "spec", "auc"))
        table(w, ["arch", "preprocessing", "weighted", "accuracy", "sensitivity",
                  "specificity", "AUC"],
              [[k[0], k[1], k[2], fmt(v["acc"]), fmt(v["sens"]), fmt(v["spec"]),
                fmt(v["auc"], 4)] for k, v in sorted(g.items())])
        w("Sensitivity is the metric of record: the test split is 350 normal to")
        w("70 TB, so predicting 'normal' for everything already scores 83.3%")
        w("accuracy.")
        w("")

        cmp_pre = paired_compare("compact_faithful_s{s}", "compact_simple_s{s}",
                                 seeds, "faithful", "simple")
        cmp_w = paired_compare("compact_faithful_w_s{s}", "compact_faithful_s{s}",
                               seeds, "class-weighted", "plain CE")
        w("### Paired comparisons (same test set, A minus B)")
        w("")
        rowsc = []
        for c in (cmp_pre, cmp_w):
            if c:
                rowsc.append([c["label"], f"{fmt(c['diffs']['auc'], 3)} pp",
                              f"{fmt(c['diffs']['sens'])} pp",
                              f"[{c['ci'][0]:.2f}, {c['ci'][1]:.2f}]",
                              f"{c['wins']}/{c['n']}",
                              "yes" if c["significant"] else "no"])
        table(w, ["comparison", "d AUC", "d sensitivity", "95% CI on d AUC",
                  "seeds favouring A", "CI excludes 0"], rowsc)

    # ---- pruning
    rows = load("prune.csv")
    if rows:
        w("## 2. Pruning")
        w("")
        g = agg(rows, ["arch", "kind", "schedule", "target_sparsity"],
                ("sens", "acc", "auc", "actual_sparsity", "channel_sparsity",
                 "size_mb"))
        table(w, ["arch", "kind", "schedule", "target", "sensitivity",
                  "accuracy", "weight sparsity %", "channel sparsity %",
                  "size MB"],
              [[k[0], k[1], k[2], f"{float(k[3]):.2f}", fmt(v["sens"]),
                fmt(v["acc"]), fmt(v["actual_sparsity"], 1),
                fmt(v["channel_sparsity"], 1), fmt(v["size_mb"], 2)]
               for k, v in sorted(g.items())])
        w("The one-shot 75% row is the sensitivity cliff. The iterative row is")
        w("the same final sparsity reached gradually.")
        w("")

    # ---- distillation
    rows = load("distill.csv")
    if rows:
        w("## 3. Distillation against a from-scratch control")
        w("")
        g = agg(rows, ["mode"], ("acc", "sens", "spec", "auc", "size_mb"))
        table(w, ["mode", "accuracy", "sensitivity", "specificity", "AUC",
                  "size MB"],
              [[k[0], fmt(v["acc"]), fmt(v["sens"]), fmt(v["spec"]),
                fmt(v["auc"], 4), fmt(v["size_mb"], 2)]
               for k, v in sorted(g.items())])
        c = paired_compare("student_distilled_s{s}", "student_scratch_s{s}",
                           seeds, "distilled", "from scratch")
        if c:
            table(w, ["comparison", "d AUC", "d sensitivity", "95% CI on d AUC",
                      "seeds favouring A", "CI excludes 0"],
                  [[c["label"], f"{fmt(c['diffs']['auc'], 3)} pp",
                    f"{fmt(c['diffs']['sens'])} pp",
                    f"[{c['ci'][0]:.2f}, {c['ci'][1]:.2f}]",
                    f"{c['wins']}/{c['n']}",
                    "yes" if c["significant"] else "no"]])
            w("The control is what licenses any claim about distillation. "
              "Without it, a student trained on hard labels alone might reach "
              "the same place.")
            w("")

    # ---- quantization
    rows = load("quantize.csv")
    if rows:
        w("## 4. Quantization")
        w("")
        g = agg(rows, ["arch", "base", "quant"], ("acc", "sens", "auc", "size_mb"))
        table(w, ["arch", "base", "quantization", "accuracy", "sensitivity",
                  "AUC", "size MB"],
              [[k[0], k[1], k[2], fmt(v["acc"]), fmt(v["sens"]),
                fmt(v["auc"], 4), fmt(v["size_mb"], 3)]
               for k, v in sorted(g.items())])
    if not load("latency.csv"):
        w("`latency.csv` is absent: the ONNX export raised and the stage caught")
        w("it, so no latency was recorded. Any deployment-latency claim is")
        w("currently unsupported by measurement.")
        w("")

    # ---- phone robustness
    rows = load("phone.csv")
    if rows:
        w("## 5. Phone-capture robustness")
        w("")
        g = agg(rows, ["arch", "training", "condition"], ("acc", "sens", "spec"))
        table(w, ["arch", "training", "condition", "accuracy", "sensitivity",
                  "specificity"],
              [[k[0], k[1], k[2], fmt(v["acc"]), fmt(v["sens"]), fmt(v["spec"])]
               for k, v in sorted(g.items())])

        w("### Which distortion does the damage")
        w("")
        for arch in sorted({r["arch"] for r in rows}):
            base = [float(r["acc"]) for r in rows
                    if r["arch"] == arch and r["condition"] == "clean"
                    and r["training"] == "clean"]
            if not base:
                continue
            drops = []
            for cond in sorted({r["condition"] for r in rows}):
                if not cond.startswith("only_"):
                    continue
                v = [float(r["acc"]) for r in rows if r["arch"] == arch
                     and r["condition"] == cond and r["training"] == "clean"]
                if v:
                    drops.append((st.mean(base) - st.mean(v), cond[5:]))
            drops.sort(reverse=True)
            w(f"- **{arch}**: " + ", ".join(
                f"{name} ({d:.1f} pp)" for d, name in drops))
        w("")

    # ---- external
    rows = load("external.csv")
    if rows:
        w("## 6. External validation (Montgomery + Shenzhen)")
        w("")
        g = agg(rows, ["arch", "cohort"], ("acc", "sens", "spec", "auc", "n"))
        table(w, ["arch", "cohort", "n", "accuracy", "sensitivity",
                  "specificity", "AUC", "AUC if inverted"],
              [[k[0], k[1], int(v["n"][0]) if v["n"] else "-", fmt(v["acc"]),
                fmt(v["sens"]), fmt(v["spec"]), fmt(v["auc"], 4),
                fmt([1 - a for a in v["auc"]], 4)]
               for k, v in sorted(g.items())])
        worst = min((st.mean(v["auc"]), k) for k, v in g.items() if v["auc"])
        if worst[0] < 0.5:
            w(f"**Treat this stage as unresolved.** The lowest AUC is "
              f"{worst[0]:.3f} for {worst[1][0]}/{worst[1][1]}, which is below")
            w("chance. A systematically inverted ranking is more often a label")
            w("or preprocessing defect than a domain-shift result, and the")
            w("'AUC if inverted' column shows what the numbers would be under")
            w("that hypothesis. Diagnose before interpreting:")
            w("")
            w("1. Confirm the `_0` normal / `_1` TB filename convention against")
            w("   the cohort documentation.")
            w("2. Render a few processed Montgomery images and compare them")
            w("   with processed training images; the auto-crop assumes padding")
            w("   borders that these scans may not have.")
            w("3. Check whether Montgomery/Shenzhen images are already inside")
            w("   the Rahman training cohort, which would make this evaluation")
            w("   optimistic rather than external.")
            w("")

    # ---- operating points
    w("## 7. Operating points")
    w("")
    w("From `results/probs/`, so no retraining is involved. WHO guidance for")
    w("TB triage tools is 90% sensitivity at 70% specificity.")
    w("")
    op = []
    for arch in ("compact", "full"):
        s_at_95, s_at_90, aucs = [], [], []
        for s in seeds:
            pr = read_probs(f"{arch}_faithful_s{s}")
            if not pr:
                continue
            y, p = pr
            s_at_95.append(sens_at_spec(y, p, 95.0))
            s_at_90.append(sens_at_spec(y, p, 90.0))
            aucs.append(auc(y, p))
        if aucs:
            op.append([arch, fmt(aucs, 4), fmt(s_at_95), fmt(s_at_90)])
    table(w, ["arch", "AUC", "sensitivity at 95% specificity",
              "sensitivity at 90% specificity"], op)

    w("## 8. Open issues")
    w("")
    w("- External validation is below chance and unexplained (section 6).")
    w("- No latency measurement: the ONNX export failed (section 4).")
    w("- `full` (4.2M parameters) underperforms `compact` (0.27M). Ten epochs")
    w("  was tuned on the contaminated splits; the larger model is plausibly")
    w("  undertrained, so no capacity conclusion should be drawn yet.")
    w("- Possible overlap between the Rahman cohort and Montgomery/Shenzhen")
    w("  has not been tested.")
    w("- Structured pruning zeroes filters but no compaction pass removes")
    w("  them, so the FLOP saving is real and the file-size saving is not.")
    w("")

    open(OUT, "w").write("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
