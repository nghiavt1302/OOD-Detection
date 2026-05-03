import os
import torch
import torch.nn as nn

from src.resnet_cifar import resnet20


def _build_resnet20_cifar10():
    return resnet20(num_classes=10)


def load_pretrained_model(checkpoint_path=None, device=None, model_name="resnet20"):
    """Load a pretrained CIFAR-10 model.

    Args:
        checkpoint_path: Optional path to a local checkpoint file.
        device: Target device (auto-detected if None).
        model_name: Model architecture name. Supports "resnet20" and "resnet56".

    Returns:
        Tuple of (model, device).
    """
    supported_models = ["resnet20", "resnet56"]
    if model_name not in supported_models:
        raise ValueError(f"Unsupported model_name='{model_name}'. Choose from {supported_models}")

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"  Device: {device}")

    # Nếu có truyền vào đường dẫn và file tồn tại trên máy -> Load model cục bộ
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading local checkpoint from: {checkpoint_path}")
        model = _build_resnet20_cifar10()
        state_dict = torch.load(checkpoint_path, map_location=device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)

    # NẾU KHÔNG CÓ FILE TRONG MÁY -> Tự động tải mô hình chuẩn từ PyTorch Hub
    else:
        hub_model_name = f"cifar10_{model_name}"
        print(f"  Downloading pretrained {hub_model_name} from PyTorch Hub...")
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", hub_model_name, pretrained=True)

    # Chuyển model vào thiết bị (CPU/GPU/MPS) và bật chế độ đánh giá
    model = model.to(device)
    model.eval()

    # Khóa gradient để tiết kiệm RAM và tăng tốc độ tính toán
    for param in model.parameters():
        param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded pretrained {model_name} ({total_params:,} params) — eval mode, gradients frozen")

    return model, device


def train_cifar10_model(data_dir="./data", save_path="./data/resnet20_cifar10.pth",
                        epochs=50, lr=0.1, device=None):
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = _build_resnet20_cifar10().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    print(f"Training ResNet-20 on CIFAR-10 ({epochs} epochs)...")

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
            print(f"  Best model saved (Acc: {best_acc:.2f}%)")

    print(f"Training completed. Best Test Accuracy: {best_acc:.2f}%")
    return model


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_cifar10_model()
    else:
        model, device = load_pretrained_model()
        print(f"Model loaded on {device}")