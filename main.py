"""
main.py - File thực thi toàn bộ luồng xử lý tự động

Pipeline hoàn chỉnh:
    1. Tải dữ liệu (CIFAR-10, SVHN, Gaussian Noise)
    2. Tải Pre-trained Model (ResNet-18)
    3. Trích xuất Logits & tính Dual Scores (Softmax + Energy)
    4. Đánh giá Metrics (FPR95, AUROC, AUPR)
    5. Trực quan hóa (Histogram, Temperature Analysis)
    6. Lưu kết quả

Hướng dẫn chạy:
    python main.py                     # Chạy full pipeline (cần checkpoint)
    python main.py --train             # Train model trước rồi chạy pipeline
    python main.py --checkpoint PATH   # Chỉ định checkpoint cụ thể
"""

import os
import sys
import argparse
import numpy as np

from src.dataloader import load_cifar10, load_svhn, load_gaussian_noise
from src.model_loader import load_pretrained_model, train_cifar10_model
from src.ood_detector import extract_logits, compute_softmax_score, compute_energy_score
from src.evaluator import (
    compute_threshold,
    evaluate_ood,
    plot_score_histogram,
    save_results_to_file,
)
from src.temperature_exp import run_all_temperature_experiments


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Energy-Based OOD Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  python main.py                          # Chạy pipeline với checkpoint mặc định
  python main.py --train                  # Train model trước, sau đó chạy pipeline
  python main.py --train --epochs 100     # Train 100 epochs
  python main.py --checkpoint model.pth   # Dùng checkpoint cụ thể
  python main.py --batch-size 64          # Đổi batch size
        """,
    )
    parser.add_argument("--train", action="store_true",
                        help="Huấn luyện model trước khi chạy OOD detection")
    parser.add_argument("--checkpoint", type=str, default="./data/resnet18_cifar10.pth",
                        help="Đường dẫn checkpoint model (mặc định: ./data/resnet18_cifar10.pth)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Thư mục lưu dữ liệu (mặc định: ./data)")
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="Thư mục lưu kết quả (mặc định: ./results)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Kích thước batch (mặc định: 128)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Số epoch huấn luyện nếu --train (mặc định: 50)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Nhiệt độ T cho Energy Score (mặc định: 1.0)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Tạo thư mục kết quả
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    print("=" * 70)
    print("  🔬 Energy-Based Out-of-Distribution (OOD) Detection")
    print("  📚 Trustworthy AI - Vũ Trọng Nghĩa (20252021M)")
    print("=" * 70)

    # =========================================================================
    # Giai đoạn 1: Chuẩn bị Dữ liệu & Mô hình
    # =========================================================================
    print("\n" + "─" * 70)
    print("  📦 Giai đoạn 1: Chuẩn bị Dữ liệu & Mô hình")
    print("─" * 70)

    # 1.1 Tải dữ liệu
    print("\n[Step 1.1] Tải dữ liệu...")
    cifar10_loader = load_cifar10(data_dir=args.data_dir, batch_size=args.batch_size)
    svhn_loader = load_svhn(data_dir=args.data_dir, batch_size=args.batch_size)
    noise_loader = load_gaussian_noise(batch_size=args.batch_size)

    # 1.2 Tải/Train model
    print("\n[Step 1.2] Chuẩn bị mô hình...")
    if args.train:
        print("[INFO] Bắt đầu huấn luyện ResNet-18 trên CIFAR-10...")
        train_cifar10_model(
            data_dir=args.data_dir,
            save_path=args.checkpoint,
            epochs=args.epochs,
        )

    model, device = load_pretrained_model(checkpoint_path=args.checkpoint)

    # =========================================================================
    # Giai đoạn 2: Inference & Dual Scoring
    # =========================================================================
    print("\n" + "─" * 70)
    print("  🧠 Giai đoạn 2: Trích xuất Logits & Tính Dual Scores")
    print("─" * 70)

    # 2.1 Trích xuất logits
    print("\n[Step 2.1] Trích xuất logits từ mô hình...")
    print("  → CIFAR-10 (ID)...")
    cifar10_logits = extract_logits(model, cifar10_loader, device)
    print(f"    Shape: {cifar10_logits.shape}")

    print("  → SVHN (OOD)...")
    svhn_logits = extract_logits(model, svhn_loader, device)
    print(f"    Shape: {svhn_logits.shape}")

    print("  → Gaussian Noise (OOD)...")
    noise_logits = extract_logits(model, noise_loader, device)
    print(f"    Shape: {noise_logits.shape}")

    # 2.2 Tính Dual Scores
    print(f"\n[Step 2.2] Tính Dual Scores (T={args.temperature})...")

    T = args.temperature

    # Softmax Confidence Scores
    cifar10_softmax = compute_softmax_score(cifar10_logits)
    svhn_softmax = compute_softmax_score(svhn_logits)
    noise_softmax = compute_softmax_score(noise_logits)

    # Energy Scores (negative energy)
    cifar10_energy = compute_energy_score(cifar10_logits, temperature=T)
    svhn_energy = compute_energy_score(svhn_logits, temperature=T)
    noise_energy = compute_energy_score(noise_logits, temperature=T)

    print(f"  Softmax - CIFAR-10 mean: {cifar10_softmax.mean():.4f}, "
          f"SVHN mean: {svhn_softmax.mean():.4f}, "
          f"Noise mean: {noise_softmax.mean():.4f}")
    print(f"  Energy  - CIFAR-10 mean: {cifar10_energy.mean():.4f}, "
          f"SVHN mean: {svhn_energy.mean():.4f}, "
          f"Noise mean: {noise_energy.mean():.4f}")

    # =========================================================================
    # Giai đoạn 3: Đánh giá & Thử nghiệm Chuyên sâu
    # =========================================================================
    print("\n" + "─" * 70)
    print("  📊 Giai đoạn 3: Đánh giá Metrics & Phân tích Temperature")
    print("─" * 70)

    # 3.1 Thiết lập Threshold τ
    print("\n[Step 3.1] Thiết lập ngưỡng τ...")
    threshold_energy = compute_threshold(cifar10_energy, percentile=5)
    threshold_softmax = compute_threshold(cifar10_softmax, percentile=5)

    # 3.2 Đánh giá Baseline Metrics
    print("\n[Step 3.2] Đánh giá Metrics cho tất cả kịch bản...")
    all_results = []

    # Softmax: CIFAR-10 vs SVHN
    all_results.append(evaluate_ood(cifar10_softmax, svhn_softmax,
                                    method_name="Softmax", ood_name="SVHN"))
    # Softmax: CIFAR-10 vs Noise
    all_results.append(evaluate_ood(cifar10_softmax, noise_softmax,
                                    method_name="Softmax", ood_name="Noise"))
    # Energy: CIFAR-10 vs SVHN
    all_results.append(evaluate_ood(cifar10_energy, svhn_energy,
                                    method_name="Energy", ood_name="SVHN"))
    # Energy: CIFAR-10 vs Noise
    all_results.append(evaluate_ood(cifar10_energy, noise_energy,
                                    method_name="Energy", ood_name="Noise"))

    # 3.3 Temperature Scaling Analysis
    print("\n[Step 3.3] Phân tích Temperature Scaling...")
    ood_datasets = {
        "SVHN": svhn_logits,
        "Noise": noise_logits,
    }
    temp_results = run_all_temperature_experiments(
        id_logits=cifar10_logits,
        ood_datasets=ood_datasets,
        save_dir=args.results_dir,
    )

    # =========================================================================
    # Giai đoạn 4: Trực quan hóa & Báo cáo
    # =========================================================================
    print("\n" + "─" * 70)
    print("  📈 Giai đoạn 4: Trực quan hóa & Lưu kết quả")
    print("─" * 70)

    # 4.1 Nhóm 1: Histogram - Softmax
    print("\n[Step 4.1] Vẽ Histogram Softmax...")
    plot_score_histogram(
        cifar10_softmax, svhn_softmax,
        score_type="Softmax", ood_name="SVHN",
        save_path=f"{args.results_dir}/histogram_softmax_svhn.png",
        threshold=threshold_softmax,
    )
    plot_score_histogram(
        cifar10_softmax, noise_softmax,
        score_type="Softmax", ood_name="Gaussian Noise",
        save_path=f"{args.results_dir}/histogram_softmax_noise.png",
        threshold=threshold_softmax,
    )

    # 4.2 Nhóm 2: Histogram - Energy
    print("\n[Step 4.2] Vẽ Histogram Energy...")
    plot_score_histogram(
        cifar10_energy, svhn_energy,
        score_type="Energy", ood_name="SVHN",
        save_path=f"{args.results_dir}/histogram_energy_svhn.png",
        threshold=threshold_energy,
    )
    plot_score_histogram(
        cifar10_energy, noise_energy,
        score_type="Energy", ood_name="Gaussian Noise",
        save_path=f"{args.results_dir}/histogram_energy_noise.png",
        threshold=threshold_energy,
    )

    # 4.3 Lưu kết quả metrics
    print("\n[Step 4.3] Lưu kết quả...")
    save_results_to_file(all_results, save_path=f"{args.results_dir}/metrics.txt")

    # =========================================================================
    # Tổng kết
    # =========================================================================
    print("\n" + "=" * 70)
    print("  ✅ PIPELINE HOÀN TẤT!")
    print("=" * 70)
    print(f"\n  📁 Kết quả được lưu tại: {os.path.abspath(args.results_dir)}/")
    print(f"     ├── metrics.txt                        (Bảng metrics)")
    print(f"     ├── histogram_softmax_svhn.png         (Histogram Softmax vs SVHN)")
    print(f"     ├── histogram_softmax_noise.png        (Histogram Softmax vs Noise)")
    print(f"     ├── histogram_energy_svhn.png          (Histogram Energy vs SVHN)")
    print(f"     ├── histogram_energy_noise.png         (Histogram Energy vs Noise)")
    print(f"     ├── temperature_analysis_svhn.png      (Temperature vs FPR95 - SVHN)")
    print(f"     └── temperature_analysis_noise.png     (Temperature vs FPR95 - Noise)")
    print()


if __name__ == "__main__":
    main()
