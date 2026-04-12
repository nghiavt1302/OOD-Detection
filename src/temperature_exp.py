"""
temperature_exp.py - Thử nghiệm phân tích tham số Nhiệt độ (Temperature Scaling)

Module này chạy thử nghiệm với các mức nhiệt độ T ∈ {0.5, 1, 2, 10, 100}
để kiểm chứng ảnh hưởng của T lên hiệu năng OOD Detection (FPR@95).

Kỳ vọng lý thuyết: T càng lớn, FPR95 càng suy giảm,
chứng minh T=1 là lựa chọn tối ưu và đơn giản nhất.
"""

import numpy as np
from .ood_detector import compute_energy_score
from .evaluator import compute_fpr_at_tpr95, plot_temperature_analysis


# Các mức nhiệt độ cần thử nghiệm
TEMPERATURES = [0.5, 1.0, 2.0, 10.0, 100.0]


def run_temperature_experiment(id_logits, ood_logits, temperatures=None,
                               ood_name="SVHN", save_dir="./results"):
    """Chạy thử nghiệm Temperature Scaling Analysis.

    Lặp lại việc tính Energy Score với các mức T khác nhau,
    ghi nhận FPR@95 cho mỗi mức T.

    Args:
        id_logits (np.ndarray): Logits của dữ liệu ID (CIFAR-10).
        ood_logits (np.ndarray): Logits của dữ liệu OOD.
        temperatures (list, optional): Danh sách giá trị T.
            Nếu None, sử dụng TEMPERATURES mặc định.
        ood_name (str): Tên tập OOD. Mặc định: "SVHN".
        save_dir (str): Thư mục lưu kết quả. Mặc định: "./results".

    Returns:
        dict: Dictionary chứa:
            - "temperatures": list các giá trị T
            - "fpr95_values": list FPR@95 tương ứng
            - "ood_name": tên tập OOD
    """
    if temperatures is None:
        temperatures = TEMPERATURES

    fpr95_values = []

    print("\n" + "=" * 60)
    print(f"  Temperature Scaling Analysis: CIFAR-10 vs {ood_name}")
    print("=" * 60)
    print(f"  {'T':>8}  |  {'FPR@95':>10}")
    print(f"  {'-'*8}--+--{'-'*10}")

    for T in temperatures:
        # Tính Energy Score với nhiệt độ T
        id_energy = compute_energy_score(id_logits, temperature=T)
        ood_energy = compute_energy_score(ood_logits, temperature=T)

        # Tính FPR@95
        fpr95 = compute_fpr_at_tpr95(id_energy, ood_energy)
        fpr95_values.append(fpr95)

        print(f"  {T:>8.1f}  |  {fpr95:>10.4f}")

    print("=" * 60)

    # Vẽ biểu đồ
    save_path = f"{save_dir}/temperature_analysis_{ood_name.lower()}.png"
    plot_temperature_analysis(temperatures, fpr95_values, save_path=save_path)

    return {
        "temperatures": temperatures,
        "fpr95_values": fpr95_values,
        "ood_name": ood_name,
    }


def run_all_temperature_experiments(id_logits, ood_datasets, save_dir="./results"):
    """Chạy Temperature Analysis cho tất cả các tập OOD.

    Args:
        id_logits (np.ndarray): Logits của CIFAR-10.
        ood_datasets (dict): Dictionary {ood_name: ood_logits}.
        save_dir (str): Thư mục lưu kết quả.

    Returns:
        list[dict]: Kết quả cho từng tập OOD.
    """
    all_results = []

    for ood_name, ood_logits in ood_datasets.items():
        result = run_temperature_experiment(
            id_logits=id_logits,
            ood_logits=ood_logits,
            ood_name=ood_name,
            save_dir=save_dir,
        )
        all_results.append(result)

    return all_results


if __name__ == "__main__":
    # Demo với dữ liệu giả
    print("Demo: Temperature Experiment")
    np.random.seed(42)
    fake_id = np.random.randn(1000, 10) + 2.0
    fake_ood = np.random.randn(1000, 10)

    run_temperature_experiment(fake_id, fake_ood, ood_name="FakeOOD", save_dir="./results")
