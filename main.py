import os
import gc
import argparse

import torch
import numpy as np

from src.dataloader import (
    load_cifar10,
    load_svhn,
    load_gaussian_noise,
    load_dtd,
    load_places365,
    load_lsun,
)
from src.model_loader import load_pretrained_model
from src.ood_detector import extract_logits, compute_energy_score, compute_softmax_score
from src.evaluator import (
    compute_fpr_at_tpr95,
    compute_threshold,
    evaluate_ood,
    plot_score_histogram,
)


# ── Ablation configurations ─────────────────────────────────────────────────
CONFIGS = [
    {"name": "1_Base_ResNet20",    "model": "resnet20"},
    {"name": "2_Deep_ResNet56",    "model": "resnet56"},
    {"name": "3_Wide_VGG16", "model": "vgg16_bn"},
]

OOD_NAMES = ["SVHN", "Noise", "DTD", "Places365", "LSUN"]


# ── CLI arguments ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Architecture-Driven Ablation: Energy-Based OOD Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir",    type=str,   default="./data",
                        help="Data directory (default: ./data)")
    parser.add_argument("--results-dir", type=str,   default="./results",
                        help="Results directory (default: ./results)")
    parser.add_argument("--batch-size",  type=int,   default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature T for Energy Score (default: 1.0)")
    return parser.parse_args()


# ── Pretty-print helpers ────────────────────────────────────────────────────

def _header(text, char="=", width=76):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def _print_summary_table(all_results):
    """Print a nicely formatted comparison table to stdout."""
    _header("Architecture Ablation — Summary Table")

    hdr = (
        f"  {'Scenario':<24} {'Model':<14} {'Method':<10} "
        f"{'OOD Dataset':<12} {'FPR95↓':>8} {'AUROC↑':>8} {'AUPR↑':>8}"
    )
    print(hdr)
    print("  " + "─" * 86)

    for r in all_results:
        print(
            f"  {r['scenario']:<24} {r['model']:<14} {r['method']:<10} "
            f"{r['ood_dataset']:<12} {r['FPR95']:>8.4f} {r['AUROC']:>8.4f} {r['AUPR']:>8.4f}"
        )
    print()


def _save_summary_table(all_results, save_path):
    """Persist the summary table to a text file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 96 + "\n")
        f.write("  Architecture-Driven Ablation Study — Energy vs Softmax OOD Detection\n")
        f.write("=" * 96 + "\n\n")

        hdr = (
            f"  {'Scenario':<24} {'Model':<14} {'Method':<10} "
            f"{'OOD Dataset':<12} {'FPR95↓':>8} {'AUROC↑':>8} {'AUPR↑':>8}\n"
        )
        f.write(hdr)
        f.write("  " + "─" * 86 + "\n")

        for r in all_results:
            f.write(
                f"  {r['scenario']:<24} {r['model']:<14} {r['method']:<10} "
                f"{r['ood_dataset']:<12} {r['FPR95']:>8.4f} {r['AUROC']:>8.4f} {r['AUPR']:>8.4f}\n"
            )

        f.write("\n" + "=" * 96 + "\n")

    print(f"  📄 Summary saved → {save_path}")


# ── Cleanup helper ──────────────────────────────────────────────────────────

def _cleanup(model):
    """Delete the model and free GPU/MPS memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Main pipeline ───────────────────────────────────────────────────────────

def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    T = args.temperature

    _header("🔬 Architecture-Driven Ablation Study — Energy-Based OOD Detection")
    print(f"  Temperature T       = {T}")
    print(f"  Configurations      : {len(CONFIGS)}")
    print(f"  OOD datasets        : {OOD_NAMES}")

    # ==================================================================
    # Phase 1 — Load datasets (shared across all scenarios)
    # ==================================================================
    _header("Phase 1: Loading Datasets", char="─")

    print("\n  [1/5] CIFAR-10 (ID)...")
    cifar10_loader = load_cifar10(data_dir=args.data_dir, batch_size=args.batch_size)

    print("  [2/5] SVHN (OOD)...")
    svhn_loader = load_svhn(data_dir=args.data_dir, batch_size=args.batch_size)

    print("  [3/5] Gaussian Noise (OOD)...")
    noise_loader = load_gaussian_noise(batch_size=args.batch_size)

    print("  [4/5] DTD — Describable Textures (OOD)...")
    dtd_loader = load_dtd(data_dir=args.data_dir, batch_size=args.batch_size)

    print("  [5/5] Places365 (OOD, subsampled)...")
    places_loader = load_places365(data_dir=args.data_dir, batch_size=args.batch_size)

    # NOTE: Uncomment the block below and add "LSUN" to OOD_NAMES if you
    # have LSUN data downloaded.  LSUN requires a manual download of the
    # lmdb files — see https://www.yf.io/p/lsun for instructions.
    # print("  [6/6] LSUN (OOD, subsampled)...")
    # lsun_loader = load_lsun(data_dir=args.data_dir, batch_size=args.batch_size)

    ood_loaders = {
        "SVHN":      svhn_loader,
        "Noise":     noise_loader,
        "DTD":       dtd_loader,
        "Places365": places_loader,
        # "LSUN":    lsun_loader,   # uncomment when available
    }

    # Filter OOD_NAMES to only datasets we actually loaded
    active_ood = [n for n in OOD_NAMES if n in ood_loaders]

    # ==================================================================
    # Phase 2 — Run ablation loop
    # ==================================================================
    _header("Phase 2: Running Ablation Scenarios", char="─")

    all_results = []

    for idx, cfg in enumerate(CONFIGS, start=1):
        scenario   = cfg["name"]
        model_name = cfg["model"]

        print(f"\n{'━' * 76}")
        print(f"  [{idx}/{len(CONFIGS)}]  {scenario}   (model={model_name})")
        print(f"{'━' * 76}")

        # --- Load model --------------------------------------------------
        print(f"\n  Loading {model_name}...")
        model, device = load_pretrained_model(model_name=model_name)

        # --- Extract logits for CIFAR-10 (ID) ----------------------------
        print("\n  Extracting logits...")
        print("    → CIFAR-10 (ID)")
        cifar10_logits = extract_logits(model, cifar10_loader, device)
        print(f"      Shape: {cifar10_logits.shape}")

        # --- Extract logits for every OOD dataset -------------------------
        ood_logits = {}
        for ood_name in active_ood:
            print(f"    → {ood_name} (OOD)")
            ood_logits[ood_name] = extract_logits(model, ood_loaders[ood_name], device)
            print(f"      Shape: {ood_logits[ood_name].shape}")

        # --- Compute Scores (Energy and Softmax) --------------------------
        print(f"\n  Computing Scores (T={T})...")
        cifar10_energy = compute_energy_score(cifar10_logits, temperature=T)
        cifar10_msp = compute_softmax_score(cifar10_logits)
        print(f"    CIFAR-10   mean energy: {cifar10_energy.mean():.4f} | mean MSP: {cifar10_msp.mean():.4f}")

        ood_energy = {}
        ood_msp = {}
        for ood_name in active_ood:
            ood_energy[ood_name] = compute_energy_score(ood_logits[ood_name], temperature=T)
            ood_msp[ood_name] = compute_softmax_score(ood_logits[ood_name])
            print(f"    {ood_name:<10} mean energy: {ood_energy[ood_name].mean():.4f} | mean MSP: {ood_msp[ood_name].mean():.4f}")

        # --- Evaluate Metrics ---------------------------------------------
        print("\n  Evaluating metrics...")
        for ood_name in active_ood:
            # Softmax
            res_softmax = evaluate_ood(
                cifar10_msp, ood_msp[ood_name],
                method_name="Softmax",
                ood_name=ood_name,
                verbose=False
            )
            res_softmax["scenario"] = scenario
            res_softmax["model"]    = model_name
            all_results.append(res_softmax)

            # Energy
            res_energy = evaluate_ood(
                cifar10_energy, ood_energy[ood_name],
                method_name="Energy",
                ood_name=ood_name,
                verbose=False
            )
            res_energy["scenario"] = scenario
            res_energy["model"]    = model_name
            all_results.append(res_energy)

        # --- Histograms ---------------------------------------------------
        tau_energy = compute_threshold(cifar10_energy, percentile=5, verbose=False)
        tau_softmax = compute_threshold(cifar10_msp, percentile=5, verbose=False)

        print("\n  Plotting score histograms...")
        for ood_name in active_ood:
            # Softmax Histogram
            save_path_softmax = os.path.join(
                args.results_dir,
                f"{scenario}_Softmax_histogram_{ood_name}.png",
            )
            plot_score_histogram(
                cifar10_msp, ood_msp[ood_name],
                score_type="Softmax Confidence", ood_name=ood_name,
                save_path=save_path_softmax,
                threshold=tau_softmax,
            )

            # Energy Histogram
            save_path_energy = os.path.join(
                args.results_dir,
                f"{scenario}_Energy_histogram_{ood_name}.png",
            )
            plot_score_histogram(
                cifar10_energy, ood_energy[ood_name],
                score_type="Energy", ood_name=ood_name,
                save_path=save_path_energy,
                threshold=tau_energy,
            )

        # --- Cleanup ------------------------------------------------------
        _cleanup(model)
        print(f"\n  ✓ Scenario {idx} ({scenario}) complete.\n")

    # ==================================================================
    # Phase 3 — Save & display results
    # ==================================================================
    _header("Phase 3: Results", char="─")

    metrics_path = os.path.join(args.results_dir, "metrics_architecture_comparison.txt")
    _save_summary_table(all_results, metrics_path)
    _print_summary_table(all_results)

    _header("PIPELINE COMPLETED")
    print(f"\n  Results directory: {os.path.abspath(args.results_dir)}/")
    print(f"     ├── metrics_architecture_comparison.txt     (Summary table)")
    print(f"     └── <scenario>_histogram_<OOD>.png          (Energy score histograms)")
    print()


if __name__ == "__main__":
    main()
