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
