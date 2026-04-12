import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def load_cifar10(data_dir="./data", batch_size=128):
    transform = get_transform()

    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"CIFAR-10 test set: {len(dataset)} samples")
    return dataloader


def load_svhn(data_dir="./data", batch_size=128):
    transform = get_transform()

    dataset = torchvision.datasets.SVHN(
        root=data_dir,
        split="test",
        download=True,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"SVHN test set: {len(dataset)} samples")
    return dataloader


def load_gaussian_noise(num_samples=10000, image_shape=(3, 32, 32), batch_size=128):
    noise_data = torch.randn(num_samples, *image_shape)

    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    noise_data = (noise_data - mean) / std

    dummy_labels = torch.zeros(num_samples, dtype=torch.long)

    dataset = TensorDataset(noise_data, dummy_labels)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Gaussian Noise dataset: {num_samples} samples")
    return dataloader


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Data Loaders")
    print("=" * 50)

    cifar10_loader = load_cifar10()
    svhn_loader = load_svhn()
    noise_loader = load_gaussian_noise()

    for name, loader in [("CIFAR-10", cifar10_loader), ("SVHN", svhn_loader), ("Noise", noise_loader)]:
        images, labels = next(iter(loader))
        print(f"  {name}: batch shape = {images.shape}, labels shape = {labels.shape}")
