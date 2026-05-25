import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np

from model import LightweightTBNet
from dataset import TBDataset, get_transforms

DATA_DIR = 'data/'
EPOCHS = 10
BATCH_SIZE = 16
LR = 0.0001
SAVE_PATH = 'checkpoints/'

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

train_dataset = TBDataset('train_split.csv', DATA_DIR, get_transforms(train=True))
val_dataset = TBDataset('val_split.csv', DATA_DIR, get_transforms(train=False))
test_dataset = TBDataset('test_split.csv', DATA_DIR, get_transforms(train=False))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

model = LightweightTBNet(num_classes=2).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f'Model parameters: {total_params:,} ({total_params/1e6:.2f}M)')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

os.makedirs(SAVE_PATH, exist_ok=True)


def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    matrix = confusion_matrix(all_labels, all_preds).astype(float)
    acc = matrix.diagonal().sum() / matrix.sum()
    sensitivity = matrix[1, 1] / matrix[1].sum()
    specificity = matrix[0, 0] / matrix[0].sum()
    return acc, sensitivity, specificity


best_val_acc = 0
for epoch in range(EPOCHS):
    model.train()
    start = time.time()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    val_acc, val_sens, val_spec = evaluate(val_loader)
    elapsed = time.time() - start
    print(f'Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | '
          f'Val Acc: {val_acc*100:.2f}% | Sens: {val_sens*100:.2f}% | '
          f'Spec: {val_spec*100:.2f}% | Time: {elapsed:.1f}s')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'best_model.pth'))

print('\nLoading best model for test evaluation...')
model.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'best_model.pth'), map_location=device))
test_acc, test_sens, test_spec = evaluate(test_loader)
print(f'Test Accuracy:    {test_acc*100:.2f}%')
print(f'Test Sensitivity: {test_sens*100:.2f}%')
print(f'Test Specificity: {test_spec*100:.2f}%')
