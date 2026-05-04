import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
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


def load_dtd(data_dir="./data", batch_size=128):
    """Load the Describable Textures Dataset (DTD) as an OOD benchmark.

    DTD images are high-resolution texture patches that trigger anomalous
    high activations in CIFAR-trained models, making it a challenging OOD
    dataset for demonstrating the value of ReAct.
    """
    dtd_transform = transforms.Compose([
        transforms.Resize(36),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    dataset = torchvision.datasets.DTD(
        root=data_dir,
        split="test",
        download=True,
        transform=dtd_transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"DTD test set: {len(dataset)} samples")
    return dataloader


def load_places365(data_dir="./data", batch_size=128, num_samples=2000):
    """Load a small subset of Places365 (small-256) as an OOD benchmark.

    Places365 contains natural-scene images that are semantically far from
    CIFAR-10 classes but trigger anomalously high penultimate-layer
    activations in CIFAR-trained ResNets — the exact failure mode that
    ReAct is designed to fix.

    Args:
        data_dir:    Root directory for the dataset.
        batch_size:  Batch size for the DataLoader.
        num_samples: Number of images to subsample (default: 2000).
    """
    places_transform = transforms.Compose([
        transforms.Resize(36),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_dataset = torchvision.datasets.Places365(
        root=data_dir,
        split="val",
        small=True,       # Use the 256×256 version (much lighter download)
        download=True,
        transform=places_transform,
    )

    # Subsample deterministically for reproducibility
    num_samples = min(num_samples, len(full_dataset))
    rng = np.random.RandomState(42)
    indices = rng.choice(len(full_dataset), size=num_samples, replace=False)
    dataset = Subset(full_dataset, indices.tolist())

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"Places365 subset: {len(dataset)} samples (from {len(full_dataset)} total)")
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
