import cv2
import csv
import matplotlib.pyplot as plt
import os

DATA_DIR = 'data/'
SPLIT_FILE = 'train_split.csv'

tb_samples = []
normal_samples = []

with open(SPLIT_FILE) as f:
    for row in csv.reader(f):
        filename, label = row[0], int(row[1])
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            path = os.path.join(DATA_DIR, filename.replace('TB-', 'Tuberculosis-'))
        if os.path.exists(path):
            if label == 1 and len(tb_samples) < 10:
                tb_samples.append(path)
            elif label == 0 and len(normal_samples) < 10:
                normal_samples.append(path)
        if len(tb_samples) == 10 and len(normal_samples) == 10:
            break

fig, axes = plt.subplots(2, 10, figsize=(20, 5))
fig.suptitle('Top row: TB Positive | Bottom row: Normal', fontsize=14)

for i, path in enumerate(tb_samples):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    axes[0, i].imshow(img, cmap='gray')
    axes[0, i].axis('off')

for i, path in enumerate(normal_samples):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    axes[1, i].imshow(img, cmap='gray')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('sanity_check.png', dpi=150)
print("Saved sanity_check.png")
