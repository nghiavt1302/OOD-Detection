"""
dataloader.py - Tải & tiền xử lý dữ liệu (CIFAR-10, SVHN, Gaussian Noise)

Module này cung cấp các hàm để tải dữ liệu In-Distribution (CIFAR-10)
và Out-of-Distribution (SVHN, Gaussian Noise) cho bài toán OOD Detection.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms


# Cấu hình tiền xử lý chuẩn cho CIFAR-10
# Mean và Std được tính trước trên tập CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def get_transform():
    """Trả về phép biến đổi (transform) chuẩn cho dữ liệu ảnh.

    Áp dụng:
        - Chuyển đổi sang Tensor
        - Chuẩn hóa (Normalize) theo mean/std của CIFAR-10

    Returns:
        transforms.Compose: Pipeline biến đổi ảnh.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def load_cifar10(data_dir="./data", batch_size=128):
    """Tải tập dữ liệu CIFAR-10 (Test split) - In-Distribution.

    Args:
        data_dir (str): Đường dẫn thư mục lưu dữ liệu. Mặc định: "./data".
        batch_size (int): Kích thước batch. Mặc định: 128.

    Returns:
        DataLoader: DataLoader cho CIFAR-10 test set.
    """
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

    print(f"[INFO] Đã tải CIFAR-10 test set: {len(dataset)} mẫu")
    return dataloader


def load_svhn(data_dir="./data", batch_size=128):
    """Tải tập dữ liệu SVHN (Test split) - Out-of-Distribution #1.

    Args:
        data_dir (str): Đường dẫn thư mục lưu dữ liệu. Mặc định: "./data".
        batch_size (int): Kích thước batch. Mặc định: 128.

    Returns:
        DataLoader: DataLoader cho SVHN test set.
    """
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

    print(f"[INFO] Đã tải SVHN test set: {len(dataset)} mẫu")
    return dataloader


def load_gaussian_noise(num_samples=10000, image_shape=(3, 32, 32), batch_size=128):
    """Sinh tập dữ liệu Gaussian Noise - Out-of-Distribution #2.

    Sinh dữ liệu ngẫu nhiên hoàn toàn vô nghĩa (nhiễu Gauss) để kiểm thử
    khả năng phòng thủ của hệ thống OOD Detection trước dữ liệu vô nghĩa.

    Dữ liệu được chuẩn hóa theo cùng mean/std với CIFAR-10 để đảm bảo
    tính nhất quán trong pipeline.

    Args:
        num_samples (int): Số lượng mẫu cần sinh. Mặc định: 10000.
        image_shape (tuple): Kích thước ảnh (C, H, W). Mặc định: (3, 32, 32).
        batch_size (int): Kích thước batch. Mặc định: 128.

    Returns:
        DataLoader: DataLoader cho tập Gaussian Noise.
    """
    # Sinh dữ liệu ngẫu nhiên dạng uniform [0, 1], sau đó chuẩn hóa
    noise_data = torch.randn(num_samples, *image_shape)

    # Chuẩn hóa theo mean/std của CIFAR-10
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    noise_data = (noise_data - mean) / std

    # Tạo nhãn giả (không sử dụng trong OOD detection)
    dummy_labels = torch.zeros(num_samples, dtype=torch.long)

    dataset = TensorDataset(noise_data, dummy_labels)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"[INFO] Đã sinh Gaussian Noise dataset: {num_samples} mẫu")
    return dataloader


if __name__ == "__main__":
    # Test nhanh các data loaders
    print("=" * 50)
    print("Testing Data Loaders")
    print("=" * 50)

    cifar10_loader = load_cifar10()
    svhn_loader = load_svhn()
    noise_loader = load_gaussian_noise()

    # Kiểm tra shape
    for name, loader in [("CIFAR-10", cifar10_loader), ("SVHN", svhn_loader), ("Noise", noise_loader)]:
        images, labels = next(iter(loader))
        print(f"  {name}: batch shape = {images.shape}, labels shape = {labels.shape}")
