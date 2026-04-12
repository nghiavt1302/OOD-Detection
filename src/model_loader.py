"""
model_loader.py - Khởi tạo Pre-trained Model (ResNet)

Module này cung cấp hàm tải mô hình ResNet-18 đã huấn luyện sẵn trên CIFAR-10,
đặt ở chế độ evaluation và đóng băng gradient.

Lưu ý: ResNet-18 của torchvision được thiết kế cho ImageNet (1000 classes).
Ở đây ta điều chỉnh lớp fully-connected cuối cùng cho CIFAR-10 (10 classes)
và sử dụng trọng số huấn luyện sẵn hoặc fine-tune.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models


def _build_resnet18_cifar10():
    """Xây dựng kiến trúc ResNet-18 cho CIFAR-10.

    Thay đổi so với ResNet-18 gốc (ImageNet):
        - Lớp conv1 đầu tiên: kernel_size=3, stride=1, padding=1 (thay vì 7x7)
        - Bỏ MaxPool layer đầu tiên (ảnh CIFAR-10 chỉ 32x32)
        - Lớp FC cuối: output = 10 (thay vì 1000)

    Returns:
        nn.Module: Mô hình ResNet-18 đã điều chỉnh cho CIFAR-10.
    """
    model = models.resnet18(weights=None, num_classes=10)

    # Điều chỉnh conv1 cho ảnh 32x32
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Bỏ maxpool (ảnh CIFAR-10 quá nhỏ cho pooling sớm)
    model.maxpool = nn.Identity()

    return model


def load_pretrained_model(checkpoint_path=None, device=None):
    """Tải mô hình ResNet-18 pretrained cho CIFAR-10.

    Nếu có checkpoint_path, tải trọng số từ file.
    Nếu không, tải pretrained ResNet-18 từ torchvision (ImageNet) và điều chỉnh.

    Mô hình luôn được đặt ở chế độ eval() và đóng băng gradient.

    Args:
        checkpoint_path (str, optional): Đường dẫn file checkpoint (.pth).
            Nếu None, sử dụng pretrained ImageNet weights. Mặc định: None.
        device (torch.device, optional): Device để đặt model.
            Nếu None, tự động chọn GPU/CPU. Mặc định: None.

    Returns:
        tuple: (model, device)
            - model (nn.Module): Mô hình đã sẵn sàng inference.
            - device (torch.device): Device đang sử dụng.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Sử dụng device: {device}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        # Tải mô hình từ checkpoint đã train trên CIFAR-10
        print(f"[INFO] Tải checkpoint từ: {checkpoint_path}")
        model = _build_resnet18_cifar10()
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Hỗ trợ cả state_dict thuần và checkpoint chứa key 'model_state_dict'
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
    else:
        # Sử dụng pretrained ImageNet weights và điều chỉnh cho CIFAR-10
        print("[INFO] Không tìm thấy checkpoint. Sử dụng pretrained ImageNet weights.")
        print("[WARNING] Để có kết quả chính xác, nên train hoặc sử dụng checkpoint CIFAR-10.")
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 10)

    model = model.to(device)

    # Đặt chế độ evaluation
    model.eval()

    # Đóng băng tất cả gradient
    for param in model.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Đã tải ResNet-18 ({total_params:,} parameters) - eval mode, gradients frozen")

    return model, device


def train_cifar10_model(data_dir="./data", save_path="./data/resnet18_cifar10.pth",
                        epochs=50, lr=0.1, device=None):
    """Huấn luyện ResNet-18 trên CIFAR-10 nếu chưa có checkpoint.

    Sử dụng SGD + Cosine Annealing scheduler cho hiệu quả tốt.

    Args:
        data_dir (str): Thư mục dữ liệu. Mặc định: "./data".
        save_path (str): Đường dẫn lưu checkpoint. Mặc định: "./data/resnet18_cifar10.pth".
        epochs (int): Số epoch huấn luyện. Mặc định: 50.
        lr (float): Learning rate ban đầu. Mặc định: 0.1.
        device (torch.device, optional): Device. Nếu None, tự động chọn.

    Returns:
        nn.Module: Mô hình đã huấn luyện.
    """
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data augmentation cho training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    model = _build_resnet18_cifar10().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    print(f"[INFO] Bắt đầu huấn luyện ResNet-18 trên CIFAR-10 ({epochs} epochs)...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

        scheduler.step()

        # Đánh giá trên test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100. * test_correct / test_total
        print(f"  Epoch {epoch+1}/{epochs} - Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  [SAVE] Best model saved (Acc: {best_acc:.2f}%)")

    print(f"[INFO] Huấn luyện hoàn tất. Best Test Accuracy: {best_acc:.2f}%")
    return model


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Chạy: python -m src.model_loader train
        train_cifar10_model()
    else:
        model, device = load_pretrained_model()
        print(f"Model loaded on {device}")
