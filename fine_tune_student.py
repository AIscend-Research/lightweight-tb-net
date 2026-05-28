import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import confusion_matrix

from model import LightweightTBNet
from dataset import TBDataset, get_transforms

EPOCHS = 15
BATCH_SIZE = 16
LR = 0.00005
TB_CLASS_WEIGHT = 3.0  # penalise missed TB cases 3x harder

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

train_dataset = TBDataset('train_split.csv', 'data/', get_transforms(train=True))
val_dataset = TBDataset('val_split.csv', 'data/', get_transforms(train=False))
test_dataset = TBDataset('test_split.csv', 'data/', get_transforms(train=False))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

student = mobilenet_v3_small(weights=None)
student.classifier[3] = nn.Linear(student.classifier[3].in_features, 2)
student.load_state_dict(torch.load('checkpoints/student_best.pth', map_location=device))
student = student.to(device)
print('Loaded student from checkpoints/student_best.pth')

teacher = LightweightTBNet(num_classes=2).to(device)
teacher.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False


def distillation_loss(student_logits, teacher_logits, labels, T=4, alpha=0.5):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T ** 2)
    weights = torch.tensor([1.0, TB_CLASS_WEIGHT], device=device)
    hard_loss = F.cross_entropy(student_logits, labels, weight=weights)
    return alpha * soft_loss + (1 - alpha) * hard_loss


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


optimizer = torch.optim.Adam(student.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_sens = 0
print(f'\nFine-tuning student with TB class weight={TB_CLASS_WEIGHT}, {EPOCHS} epochs...\n')

for epoch in range(EPOCHS):
    student.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            teacher_logits = teacher(imgs)
        student_logits = student(imgs)
        loss = distillation_loss(student_logits, teacher_logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    val_acc, val_sens, val_spec, _ = evaluate(student, val_loader)
    print(f'Epoch {epoch+1:>2}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | '
          f'Val Acc: {val_acc*100:.2f}% | Sens: {val_sens*100:.2f}% | Spec: {val_spec*100:.2f}%')

    if val_sens > best_val_sens:
        best_val_sens = val_sens
        torch.save(student.state_dict(), 'checkpoints/student_best.pth')
        print(f'  --> New best sensitivity: {val_sens*100:.2f}%')

student.load_state_dict(torch.load('checkpoints/student_best.pth', map_location=device))
test_acc, test_sens, test_spec, test_lat = evaluate(student, test_loader)

sens_gap = abs(1.0 - test_sens) * 100
within_2pct = 'YES' if sens_gap <= 2.0 else 'NO'

report = f"""
=== Student Fine-tune Report (sensitivity-focused) ===

Fine-tune setup:
  TB class weight:  {TB_CLASS_WEIGHT}x
  Epochs:           {EPOCHS}
  LR:               {LR}

Test Set Results:
  Accuracy:         {test_acc*100:.2f}%
  Sensitivity:      {test_sens*100:.2f}%   (teacher: 100.00%, gap: {sens_gap:.2f}%)
  Specificity:      {test_spec*100:.2f}%
  Latency (ms):     {test_lat:.2f}

Student matches teacher sensitivity within 2%: {within_2pct}
"""

print(report)
with open('phase3_distillation_report.txt', 'a') as f:
    f.write(report)
print('Appended to phase3_distillation_report.txt')
