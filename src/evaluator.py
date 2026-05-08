"""
Evaluator module for OOD Detection.

Core metrics : FPR@95, AUROC, AUPR
Threshold    : Tau at TPR 95%
Plots        : Score histograms, Temperature ablation chart
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve


# ── Threshold ───────────────────────────────────────────────────────────────

def compute_threshold_tau(id_scores, tpr=0.95):
    """Calculate threshold Tau so that `tpr` fraction of ID scores are above it.

    For tpr=0.95: tau = 5th percentile of ID scores
    → 95% of ID samples score above tau.

    Args:
        id_scores: np.ndarray — ID scores (higher = more in-distribution).
        tpr:       float — target true-positive rate (default 0.95).

    Returns:
        float — threshold tau.
    """
    percentile = (1.0 - tpr) * 100.0
    return float(np.percentile(id_scores, percentile))


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_fpr_at_tpr95(id_scores, ood_scores):
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    return fpr[idx]


def compute_auroc(id_scores, ood_scores):
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    return roc_auc_score(labels, scores)


def compute_aupr(id_scores, ood_scores):
    labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    precision, recall, _ = precision_recall_curve(labels, scores)
    return auc(recall, precision)


def evaluate_ood(id_scores, ood_scores, method_name="", ood_name="", verbose=True):
    """Evaluate OOD detection and return a results dict."""
    fpr95 = compute_fpr_at_tpr95(id_scores, ood_scores)
    auroc = compute_auroc(id_scores, ood_scores)
    aupr  = compute_aupr(id_scores, ood_scores)

    results = {
        "method": method_name,
        "ood_dataset": ood_name,
        "FPR95": fpr95,
        "AUROC": auroc,
        "AUPR": aupr,
    }

    if verbose:
        label = f"[{method_name}] CIFAR-10 vs {ood_name}" if method_name and ood_name else "Results"
        print(f"\n{'='*55}")
        print(f"  {label}")
        print(f"{'='*55}")
        print(f"  FPR@95:  {fpr95:.4f}")
        print(f"  AUROC:   {auroc:.4f}")
        print(f"  AUPR:    {aupr:.4f}")
        print(f"{'='*55}")

    return results


# ── Plots ───────────────────────────────────────────────────────────────────

def plot_score_histogram(id_scores, ood_scores, score_type="Energy",
                         ood_name="OOD", save_path=None, threshold=None):
    """Plot ID vs OOD score distributions with optional Tau threshold line."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(id_scores, bins=100, alpha=0.6, color="#2196F3",
            label="CIFAR-10 (ID)", density=True, edgecolor="white", linewidth=0.5)
    ax.hist(ood_scores, bins=100, alpha=0.6, color="#FF5722",
            label=f"{ood_name} (OOD)", density=True, edgecolor="white", linewidth=0.5)

    if threshold is not None:
        ax.axvline(x=threshold, color="#4CAF50", linestyle="--", linewidth=2,
                   label=f"$\\tau$ = {threshold:.2f} (TPR95)")

    ax.set_xlabel(f"{score_type} Score", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(f"{score_type} Score Distribution: CIFAR-10 vs {ood_name}",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_temperature_analysis(temperatures, fpr95_values, save_path=None):
    """Plot T (log-scale X) vs Average FPR95 (Y). Highlights T=1 as optimal.

    Args:
        temperatures:  list[float] — T values tested.
        fpr95_values:  list[float] — corresponding avg FPR95 values.
        save_path:     optional path to save figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(temperatures, fpr95_values, marker="o", linewidth=2.5,
            color="#9C27B0", markersize=8, markerfacecolor="#E91E63",
            markeredgecolor="white", markeredgewidth=1.5)

    # Highlight T=1
    if 1.0 in temperatures:
        idx = temperatures.index(1.0)
        ax.annotate(f"T=1 (optimal)\nFPR95={fpr95_values[idx]:.4f}",
                    xy=(1.0, fpr95_values[idx]),
                    xytext=(2.5, fpr95_values[idx] + 0.03),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5),
                    fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="#FFEB3B", alpha=0.9))

    ax.set_xscale("log")
    ax.set_xlabel("Temperature T (log scale)", fontsize=13)
    ax.set_ylabel("Average FPR@95 (lower is better)", fontsize=13)
    ax.set_title("Temperature Ablation: Energy Score FPR@95 vs T (ResNet-20)",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")

    for t, fpr in zip(temperatures, fpr95_values):
        if t != 1.0:
            ax.annotate(f"T={t}\n{fpr:.4f}", xy=(t, fpr),
                        textcoords="offset points", xytext=(0, 14),
                        ha="center", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", edgecolor="#ccc", alpha=0.8))

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()