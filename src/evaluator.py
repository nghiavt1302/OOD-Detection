"""
evaluator.py - Tính toán Metrics (FPR95, AUROC, AUPR) và Threshold

Module này cung cấp các hàm đánh giá hiệu năng OOD Detection:
    - FPR@95 (False Positive Rate at 95% True Positive Rate)
    - AUROC (Area Under ROC Curve)
    - AUPR (Area Under Precision-Recall Curve)
    - Threshold τ (dựa trên phân vị 5th percentile)

Kèm theo tính năng trực quan hóa (Histogram, Line Chart).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve


def compute_threshold(id_scores, percentile=5):
    """Thiết lập ngưỡng τ (Threshold) dựa trên phân vị của ID scores.

    Cắt ở phân vị thứ 5 (5th percentile) để giữ lại 95% dữ liệu ID
    (TPR = 95%).

    Args:
        id_scores (np.ndarray): Mảng scores của dữ liệu ID.
        percentile (int): Phân vị cắt. Mặc định: 5.

    Returns:
        float: Giá trị ngưỡng τ.
    """
    threshold = np.percentile(id_scores, percentile)
    print(f"[INFO] Threshold τ = {threshold:.4f} (percentile={percentile})")
    return threshold


def compute_fpr_at_tpr95(id_scores, ood_scores):
    """Tính FPR@95 (False Positive Rate khi TPR = 95%).

    Đây là metric quan trọng nhất trong OOD Detection:
    Tỷ lệ mẫu OOD bị phân loại nhầm là ID khi giữ lại 95% mẫu ID.

    Args:
        id_scores (np.ndarray): Scores của mẫu ID (cao = ID).
        ood_scores (np.ndarray): Scores của mẫu OOD (cao = ID).

    Returns:
        float: FPR@95 (0.0 = hoàn hảo, 1.0 = tệ nhất).
    """
    # labels: 1 = ID, 0 = OOD
    labels = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores, ood_scores])

    # Tính ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Tìm FPR tại TPR >= 95%
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx]

    return fpr95


def compute_auroc(id_scores, ood_scores):
    """Tính AUROC (Area Under ROC Curve).

    Đo khả năng phân biệt ID/OOD tổng thể.
    1.0 = phân biệt hoàn hảo, 0.5 = ngẫu nhiên.

    Args:
        id_scores (np.ndarray): Scores của mẫu ID.
        ood_scores (np.ndarray): Scores của mẫu OOD.

    Returns:
        float: AUROC score.
    """
    labels = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores, ood_scores])

    return roc_auc_score(labels, scores)


def compute_aupr(id_scores, ood_scores):
    """Tính AUPR (Area Under Precision-Recall Curve).

    Phù hợp khi tỷ lệ ID/OOD không cân bằng.

    Args:
        id_scores (np.ndarray): Scores của mẫu ID.
        ood_scores (np.ndarray): Scores của mẫu OOD.

    Returns:
        float: AUPR score.
    """
    labels = np.concatenate([
        np.ones(len(id_scores)),
        np.zeros(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores, ood_scores])

    precision, recall, _ = precision_recall_curve(labels, scores)
    return auc(recall, precision)


def evaluate_ood(id_scores, ood_scores, method_name="", ood_name=""):
    """Đánh giá toàn diện OOD Detection với tất cả metrics.

    Args:
        id_scores (np.ndarray): Scores của dữ liệu ID.
        ood_scores (np.ndarray): Scores của dữ liệu OOD.
        method_name (str): Tên phương pháp (Softmax/Energy).
        ood_name (str): Tên tập OOD (SVHN/Noise).

    Returns:
        dict: Dictionary chứa FPR95, AUROC, AUPR.
    """
    fpr95 = compute_fpr_at_tpr95(id_scores, ood_scores)
    auroc = compute_auroc(id_scores, ood_scores)
    aupr = compute_aupr(id_scores, ood_scores)

    results = {
        "method": method_name,
        "ood_dataset": ood_name,
        "FPR95": fpr95,
        "AUROC": auroc,
        "AUPR": aupr,
    }

    label = f"[{method_name}] CIFAR-10 vs {ood_name}" if method_name and ood_name else "Results"
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  FPR@95:  {fpr95:.4f}  (↓ thấp hơn = tốt hơn)")
    print(f"  AUROC:   {auroc:.4f}  (↑ cao hơn = tốt hơn)")
    print(f"  AUPR:    {aupr:.4f}  (↑ cao hơn = tốt hơn)")
    print(f"{'='*55}")

    return results


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_score_histogram(id_scores, ood_scores, score_type="Energy",
                         ood_name="OOD", save_path=None, threshold=None):
    """Vẽ Histogram so sánh phân phối scores giữa ID và OOD.

    Args:
        id_scores (np.ndarray): Scores của ID data.
        ood_scores (np.ndarray): Scores của OOD data.
        score_type (str): Loại score ("Softmax" hoặc "Energy").
        ood_name (str): Tên tập OOD.
        save_path (str, optional): Đường dẫn lưu biểu đồ.
        threshold (float, optional): Vẽ đường ngưỡng τ nếu cung cấp.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(id_scores, bins=100, alpha=0.6, color="#2196F3", label="CIFAR-10 (ID)",
            density=True, edgecolor="white", linewidth=0.5)
    ax.hist(ood_scores, bins=100, alpha=0.6, color="#FF5722", label=f"{ood_name} (OOD)",
            density=True, edgecolor="white", linewidth=0.5)

    if threshold is not None:
        ax.axvline(x=threshold, color="#4CAF50", linestyle="--", linewidth=2,
                   label=f"Threshold τ = {threshold:.2f}")

    ax.set_xlabel(f"{score_type} Score", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(f"Distribution of {score_type} Scores: CIFAR-10 vs {ood_name}",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] Biểu đồ đã lưu: {save_path}")

    plt.close()


def plot_temperature_analysis(temperatures, fpr95_values, save_path=None):
    """Vẽ Line Chart: FPR95 theo Temperature T (thang log).

    Args:
        temperatures (list): Danh sách giá trị T.
        fpr95_values (list): FPR95 tương ứng với từng T.
        save_path (str, optional): Đường dẫn lưu biểu đồ.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(temperatures, fpr95_values, marker="o", linewidth=2.5,
            color="#9C27B0", markersize=8, markerfacecolor="#E91E63",
            markeredgecolor="white", markeredgewidth=1.5)

    # Đánh dấu T=1 (giá trị mặc định/tối ưu)
    if 1.0 in temperatures:
        idx = temperatures.index(1.0)
        ax.annotate(f"T=1\nFPR95={fpr95_values[idx]:.4f}",
                    xy=(1.0, fpr95_values[idx]),
                    xytext=(2.0, fpr95_values[idx] + 0.02),
                    arrowprops=dict(arrowstyle="->", color="#333"),
                    fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))

    ax.set_xscale("log")
    ax.set_xlabel("Temperature T (log scale)", fontsize=13)
    ax.set_ylabel("FPR@95 (↓ lower is better)", fontsize=13)
    ax.set_title("Temperature Scaling Analysis: FPR@95 vs Temperature",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")

    # Thêm giá trị trên mỗi điểm
    for t, fpr in zip(temperatures, fpr95_values):
        if t != 1.0:  # T=1 đã có annotation riêng
            ax.annotate(f"{fpr:.4f}", xy=(t, fpr),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] Biểu đồ đã lưu: {save_path}")

    plt.close()


def save_results_to_file(all_results, save_path="./results/metrics.txt"):
    """Lưu toàn bộ kết quả metrics ra file text.

    Args:
        all_results (list[dict]): Danh sách kết quả từ evaluate_ood().
        save_path (str): Đường dẫn file lưu kết quả.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  Energy-Based OOD Detection - Evaluation Results\n")
        f.write("=" * 70 + "\n\n")

        for result in all_results:
            f.write(f"[{result['method']}] CIFAR-10 vs {result['ood_dataset']}\n")
            f.write(f"  FPR@95:  {result['FPR95']:.4f}\n")
            f.write(f"  AUROC:   {result['AUROC']:.4f}\n")
            f.write(f"  AUPR:    {result['AUPR']:.4f}\n")
            f.write("-" * 40 + "\n")

    print(f"[SAVE] Kết quả đã lưu: {save_path}")
