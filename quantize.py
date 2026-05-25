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