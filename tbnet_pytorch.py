import torch
import torch.nn as nn

class AttentionCondenser(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid = max(channels // 8, 8)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.attn(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class TBNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.layer1 = DepthwiseSeparable(32, 64)
        self.attn1 = AttentionCondenser(64)

        self.layer2 = DepthwiseSeparable(64, 128, stride=2)
        self.attn2 = AttentionCondenser(128)

        self.layer3 = DepthwiseSeparable(128, 256, stride=2)
        self.attn3 = AttentionCondenser(256)

        self.layer4 = DepthwiseSeparable(256, 512, stride=2)
        self.attn4 = AttentionCondenser(512)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.attn1(self.layer1(x))
        x = self.attn2(self.layer2(x))
        x = self.attn3(self.layer3(x))
        x = self.attn4(self.layer4(x))
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = TBNet()
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total/1e6:.2f}M")
    dummy = torch.randn(1, 1, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    print("Model looks good!")