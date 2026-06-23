"""
Generates publication-quality figures for all TB-Net variants.

All charts are generated from the reported metrics in tbnet_paper.md and the
known test-set class distribution (350 normal, 140 TB, 490 total) — so no
local data/ directory is required.

Usage:
    python visualize_results.py

Outputs (saved to figures/):
    operating_points.png    — sensitivity/specificity scatter for all models
    confusion_matrices.png  — reconstructed confusion matrices from metrics
    sparsity_tradeoff.png   — accuracy/sensitivity vs pruning sparsity
    size_vs_sensitivity.png — model size (MB) vs sensitivity scatter
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs('figures', exist_ok=True)

# ── Reported results from tbnet_paper.md ─────────────────────────────────────
# Test set: 350 normal (label=0), 140 TB (label=1), 490 total
N_NORMAL, N_TB = 350, 140

MODELS = [
    # name                size_mb  acc     sens    spec    auc
    ('TB-Net FP32',        1.07,   99.39,  97.86, 100.00, 0.9987),
    ('TB-Net FP16',        0.55,   99.39,  97.86, 100.00, 0.9986),
    ('TB-Net INT8',        0.82,   99.39,  97.86, 100.00, 0.9987),
    ('25% Pruned',         1.07,   99.59,  98.57, 100.00, 0.9996),
    ('50% Pruned',         1.07,   98.57,  97.14,  99.14, 0.9987),
    ('75% Pruned',         1.07,   93.47,  79.29,  99.14, 0.9903),
    ('MobileNetV3 (dist)', 5.91,   98.57,  96.43,  99.43, 0.9973),
    ('ONNX INT8',          0.30,   99.39,  97.86, 100.00, 0.9987),
]

COLORS = [
    '#2196F3', '#E91E63', '#FF9800', '#4CAF50',
    '#9C27B0', '#F44336', '#00BCD4', '#FF5722',
]

def reconstruct_cm(sens, spec):
    """Return [[TN, FP], [FN, TP]] from sensitivity and specificity."""
    tp = round(sens / 100 * N_TB)
    tn = round(spec / 100 * N_NORMAL)
    fn = N_TB - tp
    fp = N_NORMAL - tn
    return np.array([[tn, fp], [fn, tp]])

# ── 1. Sensitivity/Specificity Operating-Point Scatter ───────────────────────
print("Plotting sensitivity/specificity operating points...")
fig, ax = plt.subplots(figsize=(8, 6))

for (name, size, acc, sens, spec, auc), color in zip(MODELS, COLORS):
    # x-axis: 1 - specificity (FPR), y-axis: sensitivity (TPR)
    fpr = 100 - spec
    ax.scatter(fpr, sens, s=120, color=color, zorder=4)
    offset = (4, 4) if sens < 99 else (4, -10)
    ax.annotate(name, (fpr, sens), textcoords='offset points',
                xytext=offset, fontsize=8)

ax.axhline(90, color='red', linestyle=':', lw=1.3, label='WHO 90% sensitivity minimum')
ax.axvline(0,  color='grey', linestyle='-', lw=0.5, alpha=0.4)
ax.set_xlabel('False Positive Rate  (1 − Specificity, %)', fontsize=11)
ax.set_ylabel('Sensitivity (True Positive Rate, %)', fontsize=11)
ax.set_title('Sensitivity / Specificity Operating Points\n(Each dot = one model at default 0.5 threshold)',
             fontsize=12, fontweight='bold')
ax.set_xlim([-0.5, 5]); ax.set_ylim([75, 102])
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/operating_points.png', dpi=150)
plt.close()
print("  Saved figures/operating_points.png")

# ── 2. Confusion Matrices ─────────────────────────────────────────────────────
print("Plotting confusion matrices...")
n = len(MODELS)
cols = 4
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows))
axes_flat = axes.flatten()

for ax, (name, size, acc, sens, spec, auc), color in zip(axes_flat, MODELS, COLORS):
    cm = reconstruct_cm(sens, spec)
    ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0)
    ax.set_title(f"{name}\nAcc={acc:.1f}%  Sens={sens:.1f}%  Spec={spec:.1f}%",
                 fontsize=8.5, fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'TB'], fontsize=8)
    ax.set_yticklabels(['Normal', 'TB'], fontsize=8)
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('Actual', fontsize=8)
    thresh = cm.max() / 2
    labels_map = {(0,0): 'TN', (0,1): 'FP', (1,0): 'FN', (1,1): 'TP'}
    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            tag = labels_map[(i, j)]
            ax.text(j, i, f"{val}\n({tag})", ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=11, fontweight='bold')

for ax in axes_flat[n:]:
    ax.set_visible(False)

fig.suptitle('Confusion Matrices — All TB-Net Variants\n'
             '(Reconstructed from reported Sensitivity/Specificity on 490-image test set: 350 Normal, 140 TB)',
             fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figures/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved figures/confusion_matrices.png")

# ── 3. Pruning Sparsity Trade-off ─────────────────────────────────────────────
print("Plotting sparsity trade-off...")
sparsity_x  = [0,     25,    50,    75]
accuracy    = [99.39, 99.59, 98.57, 93.47]
sensitivity = [97.86, 98.57, 97.14, 79.29]
specificity = [100.0, 100.0, 99.14, 99.14]
auc_vals    = [0.9987, 0.9996, 0.9987, 0.9903]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: acc / sens / spec
ax1.plot(sparsity_x, accuracy,    'o-', color='#2196F3', label='Accuracy',    lw=2, ms=7)
ax1.plot(sparsity_x, sensitivity, 's-', color='#E91E63', label='Sensitivity', lw=2, ms=7)
ax1.plot(sparsity_x, specificity, '^-', color='#4CAF50', label='Specificity', lw=2, ms=7)
ax1.axhline(90, color='red', linestyle=':', lw=1.2, label='WHO 90% minimum')
ax1.fill_between(sparsity_x,
                 [min(s, 90) for s in sensitivity], 90,
                 alpha=0.12, color='red', label='Below WHO threshold')
for xi, sens_val in zip(sparsity_x, sensitivity):
    ax1.annotate(f'{sens_val:.1f}%', (xi, sens_val),
                 textcoords='offset points', xytext=(5, 6), fontsize=8.5, color='#E91E63')
ax1.set_xlabel('L1 Unstructured Pruning Sparsity (%)', fontsize=11)
ax1.set_ylabel('Metric (%)', fontsize=11)
ax1.set_title('Performance vs. Pruning Sparsity\n(post 5-epoch fine-tune)', fontsize=11, fontweight='bold')
ax1.set_xticks(sparsity_x); ax1.set_ylim([70, 102])
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

# Right: AUC
ax2.plot(sparsity_x, auc_vals, 'D-', color='#FF9800', lw=2, ms=8)
for xi, a in zip(sparsity_x, auc_vals):
    ax2.annotate(f'{a:.4f}', (xi, a),
                 textcoords='offset points', xytext=(5, 5), fontsize=9, color='#FF9800')
ax2.set_xlabel('L1 Unstructured Pruning Sparsity (%)', fontsize=11)
ax2.set_ylabel('AUC-ROC', fontsize=11)
ax2.set_title('AUC-ROC vs. Pruning Sparsity', fontsize=11, fontweight='bold')
ax2.set_xticks(sparsity_x); ax2.set_ylim([0.985, 1.001])
ax2.grid(alpha=0.3)

plt.suptitle('Pruning Sparsity Analysis', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/sparsity_tradeoff.png', dpi=150)
plt.close()
print("  Saved figures/sparsity_tradeoff.png")

# ── 4. Size vs Sensitivity Scatter ────────────────────────────────────────────
print("Plotting size vs sensitivity...")
fig, ax = plt.subplots(figsize=(9, 5.5))

for (name, size, acc, sens, spec, auc), color in zip(MODELS, COLORS):
    ax.scatter(size, sens, s=130, color=color, zorder=4)
    ax.annotate(f"{name}\n{size:.2f} MB", (size, sens),
                textcoords='offset points', xytext=(7, 3), fontsize=8)

# Highlight recommended combined model (estimated)
ax.scatter(0.55, 98.57, s=250, color='gold', edgecolors='black', lw=1.5,
           zorder=5, marker='*')
ax.annotate('★ Recommended\n  FP16 + 25% Pruned\n  (est. 0.55 MB, 98.57% sens)',
            (0.55, 98.57), textcoords='offset points',
            xytext=(10, -28), fontsize=8, color='#B8860B',
            arrowprops=dict(arrowstyle='->', color='#B8860B', lw=1))

ax.axhline(90, color='red', linestyle=':', lw=1.3, label='WHO 90% sensitivity minimum')
ax.axvline(50, color='grey', linestyle='--', lw=1, alpha=0.5, label='50 MB deployment target')
ax.set_xlabel('Model Size (MB)', fontsize=11)
ax.set_ylabel('Sensitivity (%)', fontsize=11)
ax.set_title('Model Size vs. Sensitivity Trade-off\n(All models far left of 50 MB target)',
             fontsize=12, fontweight='bold')
ax.set_xlim([-0.3, 7.5]); ax.set_ylim([75, 103])
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/size_vs_sensitivity.png', dpi=150)
plt.close()
print("  Saved figures/size_vs_sensitivity.png")

# ── 5. Summary Bar Chart ──────────────────────────────────────────────────────
print("Plotting summary comparison bar chart...")
names  = [m[0] for m in MODELS]
sens   = [m[3] for m in MODELS]
spec   = [m[4] for m in MODELS]
sizes  = [m[1] for m in MODELS]

x = np.arange(len(names))
width = 0.35

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

bars1 = ax1.bar(x - width/2, sens, width, label='Sensitivity', color='#E91E63', alpha=0.85)
bars2 = ax1.bar(x + width/2, spec, width, label='Specificity', color='#4CAF50', alpha=0.85)
ax1.axhline(90, color='red', linestyle=':', lw=1.2, label='WHO 90% threshold')
ax1.axhline(99.86, color='navy', linestyle='--', lw=1, alpha=0.7, label='Original TB-Net sensitivity (100%)')
ax1.set_ylabel('Score (%)', fontsize=11)
ax1.set_title('Sensitivity and Specificity — All Compressed Models', fontsize=12, fontweight='bold')
ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
ax1.set_ylim([70, 104])
ax1.legend(fontsize=9); ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars1, sens):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}', ha='center', va='bottom', fontsize=7.5, color='#C0003C')

size_colors = ['#4CAF50' if s < 1 else '#FF9800' if s < 3 else '#F44336' for s in sizes]
bars3 = ax2.bar(x, sizes, color=size_colors, alpha=0.85)
ax2.axhline(50, color='grey', linestyle='--', lw=1, label='50 MB deployment target')
ax2.set_ylabel('Model Size (MB)', fontsize=11)
ax2.set_title('Model File Size — All Compressed Models (log scale)', fontsize=12, fontweight='bold')
ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
ax2.set_yscale('log'); ax2.set_ylim([0.1, 200])
ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars3, sizes):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
             f'{val:.2f}', ha='center', va='bottom', fontsize=8)

green_patch = mpatches.Patch(color='#4CAF50', alpha=0.85, label='< 1 MB')
orange_patch = mpatches.Patch(color='#FF9800', alpha=0.85, label='1–3 MB')
ax2.legend(handles=[green_patch, orange_patch,
                    mpatches.Patch(color='none', label='')], fontsize=8)

plt.tight_layout()
plt.savefig('figures/summary_comparison.png', dpi=150)
plt.close()
print("  Saved figures/summary_comparison.png")

print("\nDone. All 5 figures saved to figures/")
