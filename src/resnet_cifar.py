"""
ResNet for CIFAR-10 (He et al., 2015)
Architecture: ResNet-20 with BasicBlock
  - 3 groups × 3 blocks = 6×3 + 2 = 20 layers
  - Channels: 16 → 32 → 64
  - No max-pooling, uses stride-2 convolutions for downsampling
  - Global average pooling → FC(64, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for CIFAR ResNets."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection for dimension mismatch
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    """
    ResNet for CIFAR-10/100 (32×32 images).

    Args:
        num_blocks: list of 3 ints, number of blocks per group.
            ResNet-20: [3, 3, 3]
            ResNet-32: [5, 5, 5]
            ResNet-44: [7, 7, 7]
            ResNet-56: [9, 9, 9]
        num_classes: number of output classes.
    """

    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 16

        # Initial conv: 3 → 16, no pooling
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # 3 groups: 16 channels (32×32), 32 channels (16×16), 64 channels (8×8)
        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)

        self.fc = nn.Linear(64, num_classes)

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet20(num_classes=10):
    """ResNet-20 for CIFAR (0.27M parameters)."""
    return ResNetCIFAR([3, 3, 3], num_classes=num_classes)


def resnet32(num_classes=10):
    """ResNet-32 for CIFAR (0.46M parameters)."""
    return ResNetCIFAR([5, 5, 5], num_classes=num_classes)


def resnet56(num_classes=10):
    """ResNet-56 for CIFAR (0.85M parameters)."""
    return ResNetCIFAR([9, 9, 9], num_classes=num_classes)
