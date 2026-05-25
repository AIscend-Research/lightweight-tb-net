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
