"""
ood_detector.py - Cốt lõi: Trích xuất Logits, tính Energy & Softmax Score

Module này chứa logic trung tâm của hệ thống OOD Detection:
    1. Trích xuất logits từ mô hình
    2. Tính Softmax Confidence Score (Baseline - MSP)
    3. Tính Energy Score (Proposed method)

Công thức Energy Score:
    E(x; f) = -T * log(sum(exp(f_i(x) / T)))

Sử dụng torch.logsumexp để đảm bảo an toàn toán học (tránh overflow/underflow).
"""

import torch
import numpy as np
from tqdm import tqdm


def extract_logits(model, dataloader, device):
    """Trích xuất logits (đầu ra thô) từ mô hình cho toàn bộ dataset.

    Args:
        model (nn.Module): Mô hình đã eval(), gradient frozen.
        dataloader (DataLoader): DataLoader chứa dữ liệu cần inference.
        device (torch.device): Device để chạy inference.

    Returns:
        np.ndarray: Mảng logits có shape (N, num_classes),
                     trong đó N là tổng số mẫu.
    """
    all_logits = []

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Extracting logits", leave=False):
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())

    return np.concatenate(all_logits, axis=0)


def compute_softmax_score(logits):
    """Tính Softmax Confidence Score (Maximum Softmax Probability - MSP).

    Đây là phương pháp baseline: lấy xác suất softmax cao nhất
    trong tất cả các lớp cho mỗi mẫu.

    Args:
        logits (np.ndarray): Mảng logits shape (N, num_classes).

    Returns:
        np.ndarray: Mảng softmax confidence scores shape (N,).
                     Giá trị cao → mô hình tự tin → khả năng ID cao.
    """
    # Sử dụng trick trừ max để tránh overflow
    logits_tensor = torch.from_numpy(logits).float()
    softmax_probs = torch.softmax(logits_tensor, dim=1)
    max_softmax, _ = torch.max(softmax_probs, dim=1)

    return max_softmax.numpy()


def compute_energy_score(logits, temperature=1.0):
    """Tính Energy Score theo công thức trong bài báo.

    Công thức: E(x; f) = -T * log(sum(exp(f_i(x) / T)))

    Sử dụng torch.logsumexp cho an toàn toán học.

    Lưu ý: Trả về ÂM Energy Score (Negative Energy) để thuận tiện
    so sánh với Softmax Score (cùng hướng: cao = ID, thấp = OOD).

    Args:
        logits (np.ndarray): Mảng logits shape (N, num_classes).
        temperature (float): Tham số nhiệt độ T. Mặc định: 1.0.

    Returns:
        np.ndarray: Mảng negative energy scores shape (N,).
                     Giá trị cao (ít âm) → khả năng ID cao.
                     Giá trị thấp (rất âm) → khả năng OOD cao.
    """
    logits_tensor = torch.from_numpy(logits).float()

    # E(x) = -T * logsumexp(f(x) / T)
    # Negative Energy = T * logsumexp(f(x) / T) (đổi dấu)
    neg_energy = temperature * torch.logsumexp(logits_tensor / temperature, dim=1)

    return neg_energy.numpy()


def compute_all_scores(model, dataloader, device, temperature=1.0):
    """Pipeline đầy đủ: Trích xuất logits → Tính Dual Scores.

    Tiện ích kết hợp extract_logits, compute_softmax_score,
    và compute_energy_score trong một lần gọi.

    Args:
        model (nn.Module): Mô hình inference.
        dataloader (DataLoader): DataLoader dữ liệu.
        device (torch.device): Device.
        temperature (float): Tham số T cho Energy Score. Mặc định: 1.0.

    Returns:
        dict: Dictionary chứa:
            - "logits": np.ndarray shape (N, num_classes)
            - "softmax_scores": np.ndarray shape (N,)
            - "energy_scores": np.ndarray shape (N,)  (negative energy)
    """
    logits = extract_logits(model, dataloader, device)
    softmax_scores = compute_softmax_score(logits)
    energy_scores = compute_energy_score(logits, temperature=temperature)

    return {
        "logits": logits,
        "softmax_scores": softmax_scores,
        "energy_scores": energy_scores,
    }


if __name__ == "__main__":
    # Demo nhanh với dữ liệu giả
    print("=" * 50)
    print("Demo: OOD Detector Scoring")
    print("=" * 50)

    # Giả lập logits
    np.random.seed(42)
    fake_id_logits = np.random.randn(100, 10) + 2.0  # ID: logits cao hơn
    fake_ood_logits = np.random.randn(100, 10)         # OOD: logits thấp hơn

    # Softmax scores
    id_softmax = compute_softmax_score(fake_id_logits)
    ood_softmax = compute_softmax_score(fake_ood_logits)
    print(f"\nSoftmax Score - ID  mean: {id_softmax.mean():.4f}")
    print(f"Softmax Score - OOD mean: {ood_softmax.mean():.4f}")

    # Energy scores
    id_energy = compute_energy_score(fake_id_logits, temperature=1.0)
    ood_energy = compute_energy_score(fake_ood_logits, temperature=1.0)
    print(f"\nEnergy Score  - ID  mean: {id_energy.mean():.4f}")
    print(f"Energy Score  - OOD mean: {ood_energy.mean():.4f}")
