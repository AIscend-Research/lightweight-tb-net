import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from tbnet_pytorch import TBNet

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH   = 'data/'
EPOCHS      = 10
BATCH_SIZE  = 32
LR          = 0.0001
SAVE_PATH   = 'models/'
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── Dataset ──────────────────────────────────────────────────────────────────
class TBDataset(Dataset):
    def __init__(self, csv_file, data_path, transform=None):
        self.df = pd.read_csv(csv_file, header=None, names=['filename', 'label'])
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        filename = os.path.basename(row['filename'])
        label    = int(row['label'])
        img_path = os.path.join(self.data_path, filename)
        image    = Image.open(img_path).convert('L')   # grayscale
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ── Data loaders ─────────────────────────────────────────────────────────────
train_ds = TBDataset('train_split_new.csv', DATA_PATH, transform)
val_ds   = TBDataset('val_split_new.csv',   DATA_PATH, transform)
test_ds  = TBDataset('test_split_new.csv',  DATA_PATH, transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

# ── Eval helper ───────────────────────────────────────────────────────────────
def evaluate(model, loader, name=""):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)[:, 1]
            preds   = outputs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    cm       = confusion_matrix(all_labels, all_preds)
    acc      = 100 * np.trace(cm) / cm.sum()
    sens     = 100 * cm[1,1] / cm[1].sum() if cm[1].sum() else 0
    spec     = 100 * cm[0,0] / cm[0].sum() if cm[0].sum() else 0
    auc      = roc_auc_score(all_labels, all_probs)
    print(f"{name} → Acc: {acc:.2f}%  Sens: {sens:.2f}%  Spec: {spec:.2f}%  AUC: {auc:.4f}")
    return acc

# ── Model / loss / optimizer ──────────────────────────────────────────────────
# Class weights compensate for 2.5:1 normal-to-TB imbalance.
# Weight = total_samples / (num_classes * class_count): normal=4900/(2*3500)=0.7, TB=4900/(2*1400)=1.75
# Normalised to keep gradient scale consistent: [0.7, 1.75] / 0.7 = [1.0, 2.5]
CLASS_WEIGHTS = torch.tensor([1.0, 2.5]).to(DEVICE)
model     = TBNet().to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
os.makedirs(SAVE_PATH, exist_ok=True)

# ── Training loop ─────────────────────────────────────────────────────────────
best_acc = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    train_acc = 100 * correct / total
    print(f"\nEpoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.4f}  Train Acc: {train_acc:.2f}%")
    val_acc = evaluate(model, val_loader, "Val")

    scheduler.step()
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'tbnet_best.pth'))
        print(f"  ✓ Saved best model (val acc: {best_acc:.2f}%)")

# ── Final test evaluation ─────────────────────────────────────────────────────
print("\n── Final Test Results ──")
model.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'tbnet_best.pth')))
evaluate(model, test_loader, "Test")
print("Training complete!")