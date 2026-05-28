import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from model import LightweightTBNet
from dataset import TBDataset, get_transforms

FINE_TUNE_EPOCHS = 5
BATCH_SIZE = 16
LR = 0.0001
SPARSITY_LEVELS = [0.0, 0.25, 0.50, 0.75]

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

train_dataset = TBDataset('train_split.csv', 'data/', get_transforms(train=True))
test_dataset = TBDataset('test_split.csv', 'data/', get_transforms(train=False))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def apply_pruning(model, amount):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)


def remove_pruning(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')


def get_sparsity(model):
    total, zero = 0, 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_mask'):
                total += module.weight_mask.numel()
                zero += (module.weight_mask == 0).sum().item()
            else:
                total += module.weight.numel()
                zero += (module.weight == 0).sum().item()
    return zero / total


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, latencies = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            start = time.perf_counter()
            preds = model(imgs).argmax(dim=1)
            latencies.append((time.perf_counter() - start) * 1000 / imgs.size(0))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    matrix = confusion_matrix(all_labels, all_preds).astype(float)
    acc = matrix.diagonal().sum() / matrix.sum()
    sensitivity = matrix[1, 1] / matrix[1].sum()
    specificity = matrix[0, 0] / matrix[0].sum()
    return acc, sensitivity, specificity, np.mean(latencies)


def fine_tune(model, epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(epochs):
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'    Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}')


results = []

for sparsity in SPARSITY_LEVELS:
    label = f'{int(sparsity*100)}%'
    print(f'\n=== Sparsity: {label} ===')

    model = LightweightTBNet(num_classes=2).to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))

    if sparsity > 0:
        apply_pruning(model, sparsity)
        actual = get_sparsity(model)
        print(f'  Actual sparsity: {actual*100:.1f}%')
        print(f'  Fine-tuning for {FINE_TUNE_EPOCHS} epochs...')
        fine_tune(model, FINE_TUNE_EPOCHS)
        remove_pruning(model)
    else:
        actual = 0.0

    acc, sens, spec, lat = evaluate(model, test_loader)
    results.append({
        'target': sparsity,
        'actual': actual,
        'acc': acc,
        'sens': sens,
        'spec': spec,
        'lat': lat,
    })
    print(f'  Accuracy: {acc*100:.2f}% | Sensitivity: {sens*100:.2f}% | Specificity: {spec*100:.2f}% | Latency: {lat:.2f}ms')

    if sparsity > 0:
        torch.save(model.state_dict(), f'checkpoints/pruned_{int(sparsity*100)}.pth')

# --- Plot ---
sparsities = [r['actual'] * 100 for r in results]
accs = [r['acc'] * 100 for r in results]
senss = [r['sens'] * 100 for r in results]
specs = [r['spec'] * 100 for r in results]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sparsities, accs, 'o-', label='Accuracy')
ax.plot(sparsities, senss, 's--', label='Sensitivity')
ax.plot(sparsities, specs, '^:', label='Specificity')
ax.set_xlabel('Sparsity (%)')
ax.set_ylabel('Score (%)')
ax.set_title('Accuracy vs Sparsity (L1 Pruning + 5-epoch Fine-tune)')
ax.legend()
ax.grid(True)
ax.set_ylim(80, 101)
plt.tight_layout()
plt.savefig('pruning_curve.png', dpi=150)
print('\nSaved pruning_curve.png')

# --- Report ---
report = f"""
=== Phase 3 Pruning Report ===

Sparsity   Actual    Accuracy    Sensitivity  Specificity  Latency(ms)
--------   ------    --------    -----------  -----------  -----------
"""
for r in results:
    report += (f"{int(r['target']*100):>6}%   {r['actual']*100:>5.1f}%"
               f"    {r['acc']*100:>7.2f}%    {r['sens']*100:>10.2f}%"
               f"  {r['spec']*100:>10.2f}%  {r['lat']:>11.2f}\n")

report += f"""
Delta from baseline (0% sparsity):
  25% pruned | Acc: {(results[1]['acc']-results[0]['acc'])*100:+.2f}% | Sens: {(results[1]['sens']-results[0]['sens'])*100:+.2f}% | Spec: {(results[1]['spec']-results[0]['spec'])*100:+.2f}%
  50% pruned | Acc: {(results[2]['acc']-results[0]['acc'])*100:+.2f}% | Sens: {(results[2]['sens']-results[0]['sens'])*100:+.2f}% | Spec: {(results[2]['spec']-results[0]['spec'])*100:+.2f}%
  75% pruned | Acc: {(results[3]['acc']-results[0]['acc'])*100:+.2f}% | Sens: {(results[3]['sens']-results[0]['sens'])*100:+.2f}% | Spec: {(results[3]['spec']-results[0]['spec'])*100:+.2f}%
"""

print(report)
with open('phase3_pruning_report.txt', 'w') as f:
    f.write(report)
print('Saved to phase3_pruning_report.txt')
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, roc_auc_score
from tbnet_pytorch import TBNet
import copy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class TBDataset(Dataset):
    def __init__(self, csv_file, data_path, transform=None):
        self.df = pd.read_csv(csv_file, header=None, names=['filename','label'])
        self.data_path = data_path
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.data_path, os.path.basename(row['filename']))).convert('L')
        if self.transform: img = self.transform(img)
        return img, int(row['label'])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_ds    = TBDataset('train_split_new.csv', 'data/', transform)
test_ds     = TBDataset('test_split_new.csv',  'data/', transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0)

# ── Eval helper ───────────────────────────────────────────────────────────────
def evaluate(model, loader, label=""):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            out   = model(images)
            probs = torch.softmax(out, dim=1)[:, 1].cpu()
            preds = out.argmax(dim=1).cpu()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
            all_probs.extend(probs.numpy())
    cm   = confusion_matrix(all_labels, all_preds)
    acc  = 100 * np.trace(cm) / cm.sum()
    sens = 100 * cm[1,1] / cm[1].sum() if cm[1].sum() else 0
    spec = 100 * cm[0,0] / cm[0].sum() if cm[0].sum() else 0
    auc  = roc_auc_score(all_labels, all_probs)
    print(f"{label} → Acc: {acc:.2f}%  Sens: {sens:.2f}%  Spec: {spec:.2f}%  AUC: {auc:.4f}")
    return acc, sens, spec, auc

# ── Fine-tune helper ──────────────────────────────────────────────────────────
def finetune(model, loader, epochs=5):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (out.argmax(1) == labels).sum().item()
            total      += labels.size(0)
        print(f"  Finetune epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(loader):.4f}  Acc: {100*correct/total:.2f}%")

# ── Apply L1 unstructured pruning to all conv layers ─────────────────────────
def apply_pruning(model, amount):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model

def remove_pruning_masks(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            try:
                prune.remove(module, 'weight')
            except:
                pass
    return model

def model_size_mb(model):
    path = 'models/_tmp_pruned.pth'
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / 1024 / 1024
    os.remove(path)
    return size

# ── Load baseline ─────────────────────────────────────────────────────────────
print("\n── Baseline ──")
baseline_model = TBNet().to(DEVICE)
baseline_model.load_state_dict(torch.load('models/tbnet_best.pth',
    map_location=DEVICE, weights_only=False))
evaluate(baseline_model, test_loader, "Baseline")

# ── Pruning experiments ───────────────────────────────────────────────────────
results = []
for sparsity in [0.25, 0.50, 0.75]:
    print(f"\n── Pruning at {int(sparsity*100)}% sparsity ──")

    # Fresh copy of the model each time
    model = TBNet().to(DEVICE)
    model.load_state_dict(torch.load('models/tbnet_best.pth',
        map_location=DEVICE, weights_only=False))

    # Apply pruning
    apply_pruning(model, amount=sparsity)
    print(f"Before fine-tune:")
    evaluate(model, test_loader, f"  Pruned {int(sparsity*100)}%")

    # Fine-tune for 5 epochs
    print(f"Fine-tuning for 5 epochs...")
    finetune(model, train_loader, epochs=5)

    # Remove masks and evaluate
    remove_pruning_masks(model)
    print(f"After fine-tune:")
    acc, sens, spec, auc = evaluate(model, test_loader, f"  Pruned {int(sparsity*100)}% + finetune")

    size = model_size_mb(model)
    print(f"  Model size: {size:.2f} MB")

    # Save
    savepath = f'models/tbnet_pruned_{int(sparsity*100)}.pth'
    torch.save(model.state_dict(), savepath)
    results.append((sparsity, acc, sens, spec, auc, size))

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Pruning Summary ──")
print(f"{'Sparsity':<12} {'Acc':>8} {'Sens':>8} {'Spec':>8} {'AUC':>8} {'Size MB':>10}")
print("-" * 60)
for sparsity, acc, sens, spec, auc, size in results:
    print(f"{int(sparsity*100)}%{'':<10} {acc:>8.2f} {sens:>8.2f} {spec:>8.2f} {auc:>8.4f} {size:>10.2f}")
