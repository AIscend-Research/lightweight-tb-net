import os
import time
import tracemalloc
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from PIL import Image, ImageFilter
import onnxruntime as ort

from dataset import TBDataset, get_transforms

STUDENT_CKPT = 'checkpoints/student_best.pth'
ONNX_PATH = 'checkpoints/student.onnx'
INPUT_SHAPE = (1, 3, 224, 224)

device = torch.device('cpu')  # export on CPU for ONNX compatibility

# --- Load student ---
student = mobilenet_v3_small(weights=None)
student.classifier[3] = nn.Linear(student.classifier[3].in_features, 2)
student.load_state_dict(torch.load(STUDENT_CKPT, map_location=device))
student.eval()

student_params = sum(p.numel() for p in student.parameters())
pytorch_size_mb = os.path.getsize(STUDENT_CKPT) / 1e6
print(f'Student: {student_params:,} params, {pytorch_size_mb:.2f} MB (.pth)')

# --- Export to ONNX ---
dummy = torch.randn(*INPUT_SHAPE)
torch.onnx.export(
    student, dummy, ONNX_PATH,
    input_names=['image'],
    output_names=['logits'],
    dynamic_axes={'image': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version=18,
)
onnx_data_path = ONNX_PATH + '.data'
onnx_size_mb = (
    os.path.getsize(ONNX_PATH) +
    (os.path.getsize(onnx_data_path) if os.path.exists(onnx_data_path) else 0)
) / 1e6
print(f'Exported ONNX: {onnx_size_mb:.2f} MB total → {ONNX_PATH}')

# --- ONNX Runtime session (CPU = mobile proxy) ---
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 2  # simulate dual-core mobile CPU
session = ort.InferenceSession(ONNX_PATH, sess_options, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name


def onnx_predict(img_tensor):
    return session.run(None, {input_name: img_tensor.numpy()})[0]


# --- Standard test evaluation ---
test_dataset = TBDataset('test_split.csv', 'data/', get_transforms(train=False))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

all_preds, all_labels, latencies = [], [], []
tracemalloc.start()
for imgs, labels in test_loader:
    start = time.perf_counter()
    logits = onnx_predict(imgs)
    latencies.append((time.perf_counter() - start) * 1000)
    all_preds.append(np.argmax(logits))
    all_labels.append(labels.item())
_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()
peak_mem_mb = peak_mem / 1e6

matrix = confusion_matrix(all_labels, all_preds).astype(float)
acc = matrix.diagonal().sum() / matrix.sum()
sensitivity = matrix[1, 1] / matrix[1].sum()
specificity = matrix[0, 0] / matrix[0].sum()


# --- Simulated phone-capture degradation ---
phone_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=1.5))),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.531, 0.531, 0.531], [0.229, 0.229, 0.229]),
])

phone_dataset = TBDataset('test_split.csv', 'data/', phone_transform)
phone_loader = DataLoader(phone_dataset, batch_size=1, shuffle=False, num_workers=0)

phone_preds, phone_labels, phone_latencies = [], [], []
for imgs, labels in phone_loader:
    start = time.perf_counter()
    logits = onnx_predict(imgs)
    phone_latencies.append((time.perf_counter() - start) * 1000)
    phone_preds.append(np.argmax(logits))
    phone_labels.append(labels.item())

pm = confusion_matrix(phone_labels, phone_preds).astype(float)
phone_acc = pm.diagonal().sum() / pm.sum()
phone_sens = pm[1, 1] / pm[1].sum()
phone_spec = pm[0, 0] / pm[0].sum()

meets_size = onnx_size_mb < 50
meets_latency = np.mean(latencies) < 2000

report = f"""
=== Phase 3 Smartphone Deployment Report ===

Model: MobileNetV3-Small student (knowledge distillation)

Export:
  Format:          ONNX (opset 18)
  PyTorch .pth:    {pytorch_size_mb:.2f} MB
  ONNX size:       {onnx_size_mb:.2f} MB
  Parameters:      {student_params:,} (1.52M)

ONNX Runtime Inference (CPU, 2-thread — mobile proxy):
  Mean latency:    {np.mean(latencies):.2f} ms/image
  Median latency:  {np.median(latencies):.2f} ms/image
  Min latency:     {np.min(latencies):.2f} ms/image
  Max latency:     {np.max(latencies):.2f} ms/image
  Peak RAM:        {peak_mem_mb:.2f} MB

Standard Test Set (clean images):
  Accuracy:        {acc*100:.2f}%
  Sensitivity:     {sensitivity*100:.2f}%
  Specificity:     {specificity*100:.2f}%

Simulated Phone-Capture Test (blur + colour jitter):
  Accuracy:        {phone_acc*100:.2f}%
  Sensitivity:     {phone_sens*100:.2f}%
  Specificity:     {phone_spec*100:.2f}%
  Mean latency:    {np.mean(phone_latencies):.2f} ms/image

Smartphone Targets:
  Model < 50 MB:       {'PASS' if meets_size else 'FAIL'} ({onnx_size_mb:.2f} MB)
  Inference < 2000 ms: {'PASS' if meets_latency else 'FAIL'} ({np.mean(latencies):.2f} ms)

Notes:
  - Benchmarked on Apple Silicon CPU with 2 threads to simulate
    a dual-core mobile CPU. Actual Android inference via ONNX
    Runtime Mobile would vary by device.
  - ONNX export split into student.onnx (structure) +
    student.onnx.data (weights) by PyTorch dynamo exporter.
  - Phone-capture test shows model collapses to predicting TB for
    nearly all degraded images (100% sensitivity, ~0% specificity).
    Root cause: model was trained only on clean clinical images and
    is not robust to camera blur + colour shift. Mitigation: include
    degraded augmentations in training or fine-tune on real phone
    captures before deployment.
"""

print(report)
with open('phase3_deployment_report.txt', 'w') as f:
    f.write(report)
print('Saved to phase3_deployment_report.txt')
print(f'ONNX model ready for Android deployment: {ONNX_PATH}')
