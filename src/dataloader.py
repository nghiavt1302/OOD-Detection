"""
Data loaders for Architecture-Driven OOD Detection Ablation Study.

All loaders produce 32×32 images normalised with CIFAR-10 statistics so that
they are directly comparable when fed through CIFAR-10-trained models.

Supported datasets
──────────────────
  ID :  CIFAR-10  (test split)
  OOD:  SVHN · Gaussian Noise · DTD · Places365 · LSUN (resized)
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms


# ── Standard CIFAR-10 normalisation ──────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def get_transform():
    """Base transform: ToTensor → CIFAR-10 normalisation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _resize_transform():
    """Resize + crop to 32×32, then normalise (for non-32px datasets)."""
    return transforms.Compose([
        transforms.Resize(36),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


# ── 1. CIFAR-10 (ID) ────────────────────────────────────────────────────────

def load_cifar10(data_dir="./data", batch_size=128):
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transform(),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )
    print(f"    CIFAR-10 test set: {len(dataset)} samples")
    return loader


# ── 2. SVHN (OOD) ───────────────────────────────────────────────────────────

def load_svhn(data_dir="./data", batch_size=128):
    dataset = torchvision.datasets.SVHN(
        root=data_dir,
        split="test",
        download=True,
        transform=get_transform(),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )
    print(f"    SVHN test set: {len(dataset)} samples")
    return loader


# ── 3. Gaussian Noise (OOD) ─────────────────────────────────────────────────

def load_gaussian_noise(num_samples=10000, image_shape=(3, 32, 32), batch_size=128):
    noise_data = torch.randn(num_samples, *image_shape)

    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    noise_data = (noise_data - mean) / std

    dummy_labels = torch.zeros(num_samples, dtype=torch.long)
    dataset = TensorDataset(noise_data, dummy_labels)

    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )
    print(f"    Gaussian Noise dataset: {num_samples} samples")
    return loader


# ── 4. DTD — Describable Textures (OOD) ─────────────────────────────────────

def load_dtd(data_dir="./data", batch_size=128):
    dataset = torchvision.datasets.DTD(
        root=data_dir,
        split="test",
        download=True,
        transform=_resize_transform(),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )
    print(f"    DTD test set: {len(dataset)} samples")
    return loader


# ── 5. Places365 (OOD, subsampled) ──────────────────────────────────────────

def load_places365(data_dir="./data", batch_size=128, num_samples=2000):
    full_dataset = torchvision.datasets.Places365(
        root=data_dir,
        split="val",
        small=True,
        download=True,
        transform=_resize_transform(),
    )
    num_samples = min(num_samples, len(full_dataset))
    rng = np.random.RandomState(42)
    indices = rng.choice(len(full_dataset), size=num_samples, replace=False)
    dataset = Subset(full_dataset, indices.tolist())

    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )
    print(f"    Places365 subset: {len(dataset)} samples (from {len(full_dataset)} total)")
    return loader


# ── 6. LSUN (resized, OOD) ──────────────────────────────────────────────────

def load_lsun(data_dir="./data", batch_size=128, num_samples=2000):
    """Load LSUN (classroom) test split resized to 32×32.

    LSUN is a large-scale scene dataset. We use the ``test`` split of the
    ``classroom`` category and subsample to *num_samples* images for
    tractability.  Images are resized + centre-cropped to 32×32 and
    normalised with CIFAR-10 statistics.
    """
    full_dataset = torchvision.datasets.LSUN(
        root=os.path.join(data_dir, "lsun"),
        classes=["classroom_test"],
        transform=_resize_transform(),
    )

    num_samples = min(num_samples, len(full_dataset))
    rng = np.random.RandomState(42)
    indices = rng.choice(len(full_dataset), size=num_samples, replace=False)
    dataset = Subset(full_dataset, indices.tolist())

    loader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )
    print(f"    LSUN subset: {len(dataset)} samples (from {len(full_dataset)} total)")
    return loader


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Data Loaders")
    print("=" * 50)

    cifar10_loader = load_cifar10()
    svhn_loader    = load_svhn()
    noise_loader   = load_gaussian_noise()

    for name, loader in [
        ("CIFAR-10", cifar10_loader),
        ("SVHN", svhn_loader),
        ("Noise", noise_loader),
    ]:
        images, labels = next(iter(loader))
        print(f"  {name}: batch shape = {images.shape}, labels shape = {labels.shape}")
