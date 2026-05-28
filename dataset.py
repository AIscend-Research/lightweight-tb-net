import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class TBDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        with open(csv_path) as f:
            for row in csv.reader(f):
                filename, label = row[0], int(row[1])
                path = os.path.join(data_dir, filename)
                if not os.path.exists(path):
                    path = os.path.join(data_dir, filename.replace('TB-', 'Tuberculosis-'))
                if os.path.exists(path):
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.531, 0.531, 0.531], [0.229, 0.229, 0.229]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.531, 0.531, 0.531], [0.229, 0.229, 0.229]),
    ])
