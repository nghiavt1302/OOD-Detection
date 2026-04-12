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
    parser = argparse.ArgumentParser(
        description="Energy-Based OOD Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train", action="store_true",
                        help="Train model before OOD detection")
    parser.add_argument("--checkpoint", type=str, default="./data/resnet18_cifar10.pth",
                        help="Checkpoint path (default: ./data/resnet18_cifar10.pth)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Data directory (default: ./data)")
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="Results directory (default: ./results)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs if --train (default: 50)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature T for Energy Score (default: 1.0)")

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    print("=" * 70)
    print("  🔬 Energy-Based Out-of-Distribution (OOD) Detection")
    print("=" * 70)

    print("\n" + "─" * 70)
    print("  Phase 1: Data & Model Preparation")
    print("─" * 70)

    print("\n[Step 1.1] Loading datasets...")
    cifar10_loader = load_cifar10(data_dir=args.data_dir, batch_size=args.batch_size)
    svhn_loader = load_svhn(data_dir=args.data_dir, batch_size=args.batch_size)
    noise_loader = load_gaussian_noise(batch_size=args.batch_size)

    print("\n[Step 1.2] Loading model...")
    if args.train:
        print("Training ResNet-18 on CIFAR-10...")
        train_cifar10_model(
            data_dir=args.data_dir,
            save_path=args.checkpoint,
            epochs=args.epochs,
        )

    model, device = load_pretrained_model(checkpoint_path=args.checkpoint)

    print("\n" + "─" * 70)
    print("  Phase 2: Extract Logits & Compute Dual Scores")
    print("─" * 70)

    print("\n[Step 2.1] Extracting logits...")
    print("  → CIFAR-10 (ID)...")
    cifar10_logits = extract_logits(model, cifar10_loader, device)
    print(f"    Shape: {cifar10_logits.shape}")

    print("  → SVHN (OOD)...")
    svhn_logits = extract_logits(model, svhn_loader, device)
    print(f"    Shape: {svhn_logits.shape}")

    print("  → Gaussian Noise (OOD)...")
    noise_logits = extract_logits(model, noise_loader, device)
    print(f"    Shape: {noise_logits.shape}")

    print(f"\n[Step 2.2] Computing Dual Scores (T={args.temperature})...")

    T = args.temperature

    cifar10_softmax = compute_softmax_score(cifar10_logits)
    svhn_softmax = compute_softmax_score(svhn_logits)
    noise_softmax = compute_softmax_score(noise_logits)

    cifar10_energy = compute_energy_score(cifar10_logits, temperature=T)
    svhn_energy = compute_energy_score(svhn_logits, temperature=T)
    noise_energy = compute_energy_score(noise_logits, temperature=T)

    print(f"  Softmax - CIFAR-10 mean: {cifar10_softmax.mean():.4f}, "
          f"SVHN mean: {svhn_softmax.mean():.4f}, "
          f"Noise mean: {noise_softmax.mean():.4f}")
    print(f"  Energy  - CIFAR-10 mean: {cifar10_energy.mean():.4f}, "
          f"SVHN mean: {svhn_energy.mean():.4f}, "
          f"Noise mean: {noise_energy.mean():.4f}")

    print("\n" + "─" * 70)
    print("  Phase 3: Evaluation Metrics & Temperature Analysis")
    print("─" * 70)

    print("\n[Step 3.1] Computing threshold τ...")
    threshold_energy = compute_threshold(cifar10_energy, percentile=5)
    threshold_softmax = compute_threshold(cifar10_softmax, percentile=5)

    print("\n[Step 3.2] Evaluating Metrics for all scenarios...")
    all_results = []

    all_results.append(evaluate_ood(cifar10_softmax, svhn_softmax,
                                    method_name="Softmax", ood_name="SVHN"))
    all_results.append(evaluate_ood(cifar10_softmax, noise_softmax,
                                    method_name="Softmax", ood_name="Noise"))
    all_results.append(evaluate_ood(cifar10_energy, svhn_energy,
                                    method_name="Energy", ood_name="SVHN"))
    all_results.append(evaluate_ood(cifar10_energy, noise_energy,
                                    method_name="Energy", ood_name="Noise"))

    print("\n[Step 3.3] Temperature Scaling Analysis...")
    ood_datasets = {
        "SVHN": svhn_logits,
        "Noise": noise_logits,
    }
    temp_results = run_all_temperature_experiments(
        id_logits=cifar10_logits,
        ood_datasets=ood_datasets,
        save_dir=args.results_dir,
    )

    print("\n" + "─" * 70)
    print("  Phase 4: Visualization & Saving Results")
    print("─" * 70)

    print("\n[Step 4.1] Plotting Softmax Histogram...")
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

    print("\n[Step 4.2] Plotting Energy Histogram...")
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

    print("\n[Step 4.3] Saving results...")
    save_results_to_file(all_results, save_path=f"{args.results_dir}/metrics.txt")

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETED")
    print("=" * 70)
    print(f"\n  Results saved at: {os.path.abspath(args.results_dir)}/")
    print(f"     ├── metrics.txt                        (Metrics table)")
    print(f"     ├── histogram_softmax_svhn.png         (Softmax Histogram vs SVHN)")
    print(f"     ├── histogram_softmax_noise.png        (Softmax Histogram vs Noise)")
    print(f"     ├── histogram_energy_svhn.png          (Histogram Energy vs SVHN)")
    print(f"     ├── histogram_energy_noise.png         (Histogram Energy vs Noise)")
    print(f"     ├── temperature_analysis_svhn.png      (Temperature vs FPR95 - SVHN)")
    print(f"     └── temperature_analysis_noise.png     (Temperature vs FPR95 - Noise)")
    print()

if __name__ == "__main__":
    main()
