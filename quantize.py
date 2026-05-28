import time
import os
import torch
import torch.quantization
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from model import LightweightTBNet
from dataset import TBDataset, get_transforms


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, latencies = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            if device.type == 'cpu' and next(model.parameters()).dtype == torch.float16:
                imgs = imgs.half()
            start = time.perf_counter()
            preds = model(imgs).argmax(dim=1)
            latencies.append((time.perf_counter() - start) * 1000)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    matrix = confusion_matrix(all_labels, all_preds).astype(float)
    acc = matrix.diagonal().sum() / matrix.sum()
    sensitivity = matrix[1, 1] / matrix[1].sum()
    specificity = matrix[0, 0] / matrix[0].sum()
    return acc, sensitivity, specificity, np.mean(latencies), np.median(latencies)


def get_model_size_mb(model):
    path = '/tmp/_tmp_model.pth'
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / 1e6
    os.remove(path)
    return size


torch.backends.quantized.engine = 'qnnpack'
cpu = torch.device('cpu')

test_dataset = TBDataset('test_split.csv', 'data/', get_transforms(train=False))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# --- FP32 baseline on CPU ---
fp32_model = LightweightTBNet(num_classes=2).to(cpu)
fp32_model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=cpu))
fp32_model.eval()
fp32_size = get_model_size_mb(fp32_model)
fp32_acc, fp32_sens, fp32_spec, fp32_mean_lat, fp32_med_lat = evaluate(fp32_model, test_loader, cpu)

# --- INT8 Dynamic Quantization ---
int8_model = LightweightTBNet(num_classes=2).to(cpu)
int8_model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=cpu))
int8_model.eval()
int8_model = torch.quantization.quantize_dynamic(
    int8_model,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)
int8_size = get_model_size_mb(int8_model)
int8_acc, int8_sens, int8_spec, int8_mean_lat, int8_med_lat = evaluate(int8_model, test_loader, cpu)

# --- FP16 ---
fp16_model = LightweightTBNet(num_classes=2).to(cpu)
fp16_model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=cpu))
fp16_model.eval()
fp16_model = fp16_model.half()
fp16_size = get_model_size_mb(fp16_model)
fp16_acc, fp16_sens, fp16_spec, fp16_mean_lat, fp16_med_lat = evaluate(fp16_model, test_loader, cpu)

torch.save(int8_model.state_dict(), 'checkpoints/int8_model.pth')
torch.save(fp16_model.state_dict(), 'checkpoints/fp16_model.pth')

report = f"""
=== Phase 3 Quantization Report ===

                  FP32        INT8        FP16
Model Size (MB):  {fp32_size:>7.2f}     {int8_size:>7.2f}     {fp16_size:>7.2f}
Size Reduction:   baseline    {(1 - int8_size/fp32_size)*100:>6.1f}%     {(1 - fp16_size/fp32_size)*100:>6.1f}%

Latency Mean(ms): {fp32_mean_lat:>7.2f}     {int8_mean_lat:>7.2f}     {fp16_mean_lat:>7.2f}
Latency Med (ms): {fp32_med_lat:>7.2f}     {int8_med_lat:>7.2f}     {fp16_med_lat:>7.2f}
Speedup:          baseline    {fp32_mean_lat/int8_mean_lat:>6.2f}x     {fp32_mean_lat/fp16_mean_lat:>6.2f}x

Accuracy:         {fp32_acc*100:>6.2f}%     {int8_acc*100:>6.2f}%     {fp16_acc*100:>6.2f}%
Sensitivity:      {fp32_sens*100:>6.2f}%     {int8_sens*100:>6.2f}%     {fp16_sens*100:>6.2f}%
Specificity:      {fp32_spec*100:>6.2f}%     {int8_spec*100:>6.2f}%     {fp16_spec*100:>6.2f}%

Delta from FP32 Baseline:
  INT8 Accuracy:      {(int8_acc - fp32_acc)*100:+.2f}%
  INT8 Sensitivity:   {(int8_sens - fp32_sens)*100:+.2f}%
  INT8 Specificity:   {(int8_spec - fp32_spec)*100:+.2f}%
  FP16 Accuracy:      {(fp16_acc - fp32_acc)*100:+.2f}%
  FP16 Sensitivity:   {(fp16_sens - fp32_sens)*100:+.2f}%
  FP16 Specificity:   {(fp16_spec - fp32_spec)*100:+.2f}%

Smartphone Target: model < 50MB, inference < 2000ms
  INT8 meets size target: {'YES' if int8_size < 50 else 'NO'} ({int8_size:.2f} MB)
  FP16 meets size target: {'YES' if fp16_size < 50 else 'NO'} ({fp16_size:.2f} MB)
"""

print(report)
with open('phase3_quantization_report.txt', 'w') as f:
    f.write(report)
print('Saved to phase3_quantization_report.txt')
print('Saved INT8 model to checkpoints/int8_model.pth')
print('Saved FP16 model to checkpoints/fp16_model.pth')
import torch
import os
from tbnet_pytorch import TBNet
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Dataset ───────────────────────────────────────────────────────────────────
class TBDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, data_path, transform=None):
        self.df = pd.read_csv(csv_file, header=None, names=['filename','label'])
        self.data_path = data_path
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.data_path, row['filename'])).convert('L')
        if self.transform: img = self.transform(img)
        return img, int(row['label'])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_ds     = TBDataset('test_split_new.csv', 'data/', transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

# ── Eval helper ───────────────────────────────────────────────────────────────
def evaluate(model, loader, device, label=""):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
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

def model_size_mb(path):
    return os.path.getsize(path) / 1024 / 1024

# ── Load baseline ─────────────────────────────────────────────────────────────
model = TBNet()
model.load_state_dict(torch.load('models/tbnet_best.pth', map_location='cpu'))
model.eval()

torch.save(model.state_dict(), 'models/tbnet_fp32.pth')
print(f"FP32 model size: {model_size_mb('models/tbnet_fp32.pth'):.2f} MB")
baseline = evaluate(model, test_loader, torch.device('cpu'), "FP32 Baseline")

# ── FP16 Quantization ─────────────────────────────────────────────────────────
model_fp16 = TBNet().half()
model_fp16.load_state_dict({k: v.half() for k, v in
    torch.load('models/tbnet_best.pth', map_location='cpu').items()})
model_fp16.eval()
torch.save(model_fp16.state_dict(), 'models/tbnet_fp16.pth')
print(f"\nFP16 model size: {model_size_mb('models/tbnet_fp16.pth'):.2f} MB")

# FP16 eval on CPU needs float32 inputs cast to half
class FP16Wrapper:
    def __init__(self, m): self.m = m
    def eval(self): self.m.eval(); return self
    def __call__(self, x): return self.m(x.half())
    def to(self, d): self.m.to(d); return self

evaluate(FP16Wrapper(model_fp16), test_loader, torch.device('cpu'), "FP16")

# ── INT8 Dynamic Quantization ─────────────────────────────────────────────────
model_int8 = TBNet()
model_int8.load_state_dict(torch.load('models/tbnet_best.pth', map_location='cpu', weights_only=False))
model_int8.eval()

model_int8 = torch.quantization.quantize_dynamic(
    model_int8,
    {torch.nn.Linear, torch.nn.Conv2d},
    dtype=torch.qint8
)

torch.save(model_int8.state_dict(), 'models/tbnet_int8.pth')
print(f"\nINT8 model size: {model_size_mb('models/tbnet_int8.pth'):.2f} MB")
evaluate(model_int8, test_loader, torch.device('cpu'), "INT8")

print("\n── Summary ──")
print(f"FP32: {model_size_mb('models/tbnet_fp32.pth'):.2f} MB")
print(f"FP16: {model_size_mb('models/tbnet_fp16.pth'):.2f} MB")
print(f"INT8: {model_size_mb('models/tbnet_int8.pth'):.2f} MB")
