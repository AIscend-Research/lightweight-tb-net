import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, roc_auc_score
from model import TBNet

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEMPERATURE = 4
ALPHA       = 0.7
EPOCHS      = 15
LR          = 0.0001

print(f"Using device: {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────────────────
transform_teacher = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

transform_student = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class TBDataset(Dataset):
    def __init__(self, csv_file, data_path, transform=None, grayscale=True):
        self.df        = pd.read_csv(csv_file, header=None, names=['filename','label'])
        self.data_path = data_path
        self.transform = transform
        self.grayscale = grayscale
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        mode = 'L' if self.grayscale else 'RGB'
        img  = Image.open(os.path.join(self.data_path, row['filename'])).convert(mode)
        if self.transform: img = self.transform(img)
        return img, int(row['label'])

train_teacher_loader = DataLoader(
    TBDataset('train_split_new.csv', 'data/', transform_teacher, grayscale=True),
    batch_size=32, shuffle=False, num_workers=0)
train_student_loader = DataLoader(
    TBDataset('train_split_new.csv', 'data/', transform_student, grayscale=False),
    batch_size=32, shuffle=False, num_workers=0)
test_teacher_loader  = DataLoader(
    TBDataset('test_split_new.csv', 'data/', transform_teacher, grayscale=True),
    batch_size=32, shuffle=False, num_workers=0)
test_student_loader  = DataLoader(
    TBDataset('test_split_new.csv', 'data/', transform_student, grayscale=False),
    batch_size=32, shuffle=False, num_workers=0)

# ── Eval helper ───────────────────────────────────────────────────────────────
def evaluate(model, loader, label=""):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            out    = model(images)
            probs  = torch.softmax(out, dim=1)[:, 1].cpu()
            preds  = out.argmax(dim=1).cpu()
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

def model_size_mb(model):
    path = 'models/_tmp.pth'
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / 1024 / 1024
    os.remove(path)
    return size

# ── Teacher ───────────────────────────────────────────────────────────────────
teacher = TBNet().to(DEVICE)
teacher.load_state_dict(torch.load('models/tbnet_best.pth',
    map_location=DEVICE, weights_only=False))
teacher.eval()
print(f"\nTeacher size: {model_size_mb(teacher):.2f} MB")
print("Teacher performance:")
evaluate(teacher, test_teacher_loader, "  Teacher")

# ── Student: MobileNetV3-Small ────────────────────────────────────────────────
student = models.mobilenet_v3_small(weights=None)
student.classifier[3] = nn.Linear(student.classifier[3].in_features, 2)
student = student.to(DEVICE)
print(f"\nStudent size: {model_size_mb(student):.2f} MB")

# ── Distillation loss ─────────────────────────────────────────────────────────
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T ** 2)
    hard_loss = F.cross_entropy(student_logits, labels)
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
train_losses, val_accs, val_senss = [], [], []

print(f'\nDistilling teacher → student (T={TEMPERATURE}, α={ALPHA}, {EPOCHS} epochs)...\n')

for epoch in range(EPOCHS):
    student.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            teacher_logits = teacher(imgs)
        student_logits = student(imgs)
        loss = distillation_loss(student_logits, teacher_logits, labels, TEMPERATURE, ALPHA)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    val_acc, val_sens, val_spec, _ = evaluate(student, val_loader)
    train_losses.append(total_loss / len(train_loader))
    val_accs.append(val_acc * 100)
    val_senss.append(val_sens * 100)

    print(f'Epoch {epoch+1:>2}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | '
          f'Val Acc: {val_acc*100:.2f}% | Sens: {val_sens*100:.2f}% | Spec: {val_spec*100:.2f}%')

    if val_sens > best_val_sens:
        best_val_sens = val_sens
        torch.save(student.state_dict(), 'checkpoints/student_best.pth')

# --- Final evaluation ---
student.load_state_dict(torch.load('checkpoints/student_best.pth', map_location=device))
teacher_acc, teacher_sens, teacher_spec, teacher_lat = evaluate(teacher, test_loader)
student_acc, student_sens, student_spec, student_lat = evaluate(student, test_loader)

# --- Plot training curves ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, label='Distillation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(val_accs, label='Val Accuracy')
ax2.plot(val_senss, label='Val Sensitivity')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Score (%)')
ax2.set_title('Validation Metrics')
ax2.legend()
ax2.grid(True)
ax2.set_ylim(80, 101)

plt.tight_layout()
plt.savefig('distillation_curve.png', dpi=150)
print('\nSaved distillation_curve.png')

sens_gap = abs(teacher_sens - student_sens) * 100
within_2pct = 'YES' if sens_gap <= 2.0 else 'NO'

report = f"""
=== Phase 3 Knowledge Distillation Report ===

Setup:
  Teacher:     LightweightTBNet (ResNet18 + AttentionCondenser)
  Student:     MobileNetV3-Small
  Temperature: {TEMPERATURE}
  Alpha:       {ALPHA} (distillation) / {1-ALPHA:.1f} (hard labels)
  Epochs:      {EPOCHS}

Model Comparison:
                    Teacher       Student
  Parameters:     {teacher_params:>10,}   {student_params:>10,}
  Size ratio:     baseline      {student_params/teacher_params*100:.1f}% of teacher

Test Set Results:
                    Teacher       Student     Delta
  Accuracy:        {teacher_acc*100:>6.2f}%       {student_acc*100:>6.2f}%     {(student_acc-teacher_acc)*100:+.2f}%
  Sensitivity:     {teacher_sens*100:>6.2f}%       {student_sens*100:>6.2f}%     {(student_sens-teacher_sens)*100:+.2f}%
  Specificity:     {teacher_spec*100:>6.2f}%       {student_spec*100:>6.2f}%     {(student_spec-teacher_spec)*100:+.2f}%
  Latency (ms):   {teacher_lat:>7.2f}       {student_lat:>7.2f}

Sensitivity Gap: {sens_gap:.2f}%
Student matches teacher sensitivity within 2%: {within_2pct}
"""

print(report)
with open('phase3_distillation_report.txt', 'w') as f:
    f.write(report)
print('Saved to phase3_distillation_report.txt')
print('Best student model saved to checkpoints/student_best.pth')
# ── Training ──────────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(student.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
best_acc  = 0

print(f"\nDistilling TB-Net → MobileNetV3-Small (T={TEMPERATURE}, α={ALPHA}, epochs={EPOCHS})\n")

for epoch in range(EPOCHS):
    student.train()
    total_loss, correct, total = 0, 0, 0

    for (t_images, labels), (s_images, _) in zip(train_teacher_loader, train_student_loader):
        t_images = t_images.to(DEVICE)
        s_images = s_images.to(DEVICE)
        labels   = labels.to(DEVICE)

        with torch.no_grad():
            teacher_logits = teacher(t_images)

        student_logits = student(s_images)
        loss = distillation_loss(student_logits, teacher_logits, labels, TEMPERATURE, ALPHA)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (student_logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    scheduler.step()
    train_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_student_loader):.4f}  Train Acc: {train_acc:.2f}%")

    if (epoch + 1) % 5 == 0:
        acc, sens, spec, auc = evaluate(student, test_student_loader, "  Student")
        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), 'models/student_mobilenet.pth')
            print(f"  ✓ Saved best student (acc: {best_acc:.2f}%)")

# ── Final comparison ──────────────────────────────────────────────────────────
print("\n── Final Comparison ──")
student.load_state_dict(torch.load('models/student_mobilenet.pth',
    map_location=DEVICE, weights_only=False))

print(f"Teacher ({model_size_mb(teacher):.2f} MB):")
evaluate(teacher, test_teacher_loader, "  Teacher")

print(f"Student ({model_size_mb(student):.2f} MB):")
evaluate(student, test_student_loader, "  Student")
