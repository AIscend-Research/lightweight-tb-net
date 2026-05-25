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