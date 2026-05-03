"""
Ablation Study: Energy-Based OOD Detection
===========================================
Progressive storyline:
  1. ResNet-20 baseline — struggles but works on easy OOD (SVHN, Noise)
  2. ResNet-56 capacity — improves via larger model, but fails on DTD textures
  3. ResNet-56 + ReAct  — auto-clips anomalous activations → fixes DTD, SOTA
"""

import os
import argparse
import numpy as np

from src.dataloader import load_cifar10, load_svhn, load_gaussian_noise, load_dtd
from src.model_loader import load_pretrained_model
from src.ood_detector import extract_logits, compute_energy_score
from src.evaluator import (
    compute_fpr_at_tpr95,
    compute_threshold,
    evaluate_ood,
    plot_score_histogram,
    plot_activation_distribution,
    plot_temperature_analysis,
    save_results_to_file,
)
from src.react_utils import (
    apply_react_hook,
    remove_react_hook,
    calculate_react_threshold,
    extract_penultimate_features,
)


# ---------------------------------------------------------------------------
# Ablation study configurations
# ---------------------------------------------------------------------------
ABLATION_CONFIGS = [
    {"name": "1_Base_ResNet20",       "model_name": "resnet20", "use_react": False},
    {"name": "2_Capacity_ResNet56",   "model_name": "resnet56", "use_react": False},
    {"name": "3_SOTA_ResNet56_ReAct", "model_name": "resnet56", "use_react": True},
]

OOD_NAMES = ["SVHN", "Noise", "DTD"]
TEMPERATURES = [0.5, 1.0, 2.0, 10.0, 100.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation Study: Model Capacity × ReAct for Energy-Based OOD Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Data directory (default: ./data)")
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="Results directory (default: ./results)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature T for Energy Score (default: 1.0)")
    parser.add_argument("--react-percentile", type=int, default=90,
                        help="Percentile for auto-calculating ReAct threshold (default: 90)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _header(text, char="=", width=72):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def _print_summary_table(all_results):
    """Print a nicely formatted comparison table of all scenarios."""
    _header("Ablation Study — Summary Table")

    hdr = (
        f"  {'Scenario':<26} {'Model':<10} {'ReAct':<7} "
        f"{'OOD':<8} {'FPR95↓':>8} {'AUROC↑':>8} {'AUPR↑':>8}"
    )
    print(hdr)
    print("  " + "─" * 76)

    for r in all_results:
        react_str = "Yes" if r.get("react") else "No"
        print(
            f"  {r['scenario']:<26} {r['model']:<10} {react_str:<7} "
            f"{r['ood_dataset']:<8} {r['FPR95']:>8.4f} {r['AUROC']:>8.4f} {r['AUPR']:>8.4f}"
        )

    print()


def _save_ablation_results(all_results, save_path):
    """Save ablation results to a structured text file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 88 + "\n")
        f.write("  Ablation Study: Model Capacity × ReAct — Energy-Based OOD Detection\n")
        f.write("=" * 88 + "\n\n")

        hdr = (
            f"  {'Scenario':<26} {'Model':<10} {'ReAct':<7} "
            f"{'OOD':<8} {'FPR95':>8} {'AUROC':>8} {'AUPR':>8}\n"
        )
        sep = "  " + "─" * 78 + "\n"
        f.write(hdr)
        f.write(sep)

        for r in all_results:
            react_str = "Yes" if r.get("react") else "No"
            f.write(
                f"  {r['scenario']:<26} {r['model']:<10} {react_str:<7} "
                f"{r['ood_dataset']:<8} {r['FPR95']:>8.4f} {r['AUROC']:>8.4f} {r['AUPR']:>8.4f}\n"
            )

        f.write("\n" + "=" * 88 + "\n")

    print(f"  Results saved → {save_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    T = args.temperature

    _header("🔬 Ablation Study: Model Capacity × ReAct")
    print(f"  Temperature T       = {T}")
    print(f"  ReAct percentile    = {args.react_percentile}")
    print(f"  Configurations      : {len(ABLATION_CONFIGS)}")
    print(f"  OOD datasets        : {OOD_NAMES}")

    # ==================================================================
    # Phase 1 — Load datasets (shared across all configs)
    # ==================================================================
    _header("Phase 1: Loading Datasets", char="─")

    print("\n  [1.1] CIFAR-10 (ID)...")
    cifar10_loader = load_cifar10(data_dir=args.data_dir, batch_size=args.batch_size)

    print("  [1.2] SVHN (OOD)...")
    svhn_loader = load_svhn(data_dir=args.data_dir, batch_size=args.batch_size)

    print("  [1.3] Gaussian Noise (OOD)...")
    noise_loader = load_gaussian_noise(batch_size=args.batch_size)

    print("  [1.4] DTD — Describable Textures (OOD)...")
    dtd_loader = load_dtd(data_dir=args.data_dir, batch_size=args.batch_size)

    ood_loaders = {"SVHN": svhn_loader, "Noise": noise_loader, "DTD": dtd_loader}

    # ==================================================================
    # Phase 2 — Run ablation loop
    # ==================================================================
    _header("Phase 2: Running Ablation Scenarios", char="─")

    all_results = []

    for idx, cfg in enumerate(ABLATION_CONFIGS, start=1):
        scenario = cfg["name"]
        model_name = cfg["model_name"]
        use_react = cfg["use_react"]
        safe_name = scenario  # already filesystem-safe

        print(f"\n{'━' * 72}")
        print(f"  [INFO] Scenario {idx}/{len(ABLATION_CONFIGS)}: {scenario}")
        print(f"         Model={model_name}  |  ReAct={'ON' if use_react else 'OFF'}")
        print(f"{'━' * 72}")

        # --- Load model -------------------------------------------------
        print(f"\n  Loading {model_name}...")
        model, device = load_pretrained_model(model_name=model_name)

        # --- Extract penultimate features BEFORE ReAct (for vis) ---------
        #     We always extract ID features; for DTD we also extract OOD
        #     features so we can plot the activation distributions.
        print("\n  Extracting penultimate features (pre-ReAct)...")
        print("    → CIFAR-10 (ID)")
        id_features = extract_penultimate_features(model, cifar10_loader, device)
        print(f"      Shape: {id_features.shape}")

        print("    → DTD (OOD)")
        dtd_features = extract_penultimate_features(model, dtd_loader, device)
        print(f"      Shape: {dtd_features.shape}")

        # --- Calculate & apply ReAct if required -------------------------
        hook_handle = None
        threshold_c = None
        if use_react:
            threshold_c = calculate_react_threshold(
                model, cifar10_loader, device,
                percentile=args.react_percentile,
            )
            hook_handle = apply_react_hook(model, threshold_c=threshold_c)

        # --- Activation distribution plots (always, for each scenario) ---
        print("\n  Plotting activation distributions...")
        plot_activation_distribution(
            id_features, dtd_features,
            ood_name="DTD",
            threshold_c=threshold_c,
            save_path=f"{args.results_dir}/{safe_name}_activations_dtd.png",
        )

        # --- Extract logits (post-hook if ReAct is on) -------------------
        print("\n  Extracting logits...")
        print("    → CIFAR-10 (ID)")
        cifar10_logits = extract_logits(model, cifar10_loader, device)
        print(f"      Shape: {cifar10_logits.shape}")

        ood_logits = {}
        for ood_name, ood_loader in ood_loaders.items():
            print(f"    → {ood_name} (OOD)")
            ood_logits[ood_name] = extract_logits(model, ood_loader, device)
            print(f"      Shape: {ood_logits[ood_name].shape}")

        # --- Compute Energy Scores ---------------------------------------
        print(f"\n  Computing Energy Scores (T={T})...")
        cifar10_energy = compute_energy_score(cifar10_logits, temperature=T)
        print(f"    CIFAR-10 mean energy: {cifar10_energy.mean():.4f}")

        ood_energy = {}
        for ood_name in OOD_NAMES:
            ood_energy[ood_name] = compute_energy_score(ood_logits[ood_name], temperature=T)
            print(f"    {ood_name:<8} mean energy: {ood_energy[ood_name].mean():.4f}")

        # --- Evaluate Metrics --------------------------------------------
        print("\n  Evaluating metrics...")
        for ood_name in OOD_NAMES:
            result = evaluate_ood(
                cifar10_energy, ood_energy[ood_name],
                method_name=f"Energy ({scenario})",
                ood_name=ood_name,
            )
            result["scenario"] = scenario
            result["model"] = model_name
            result["react"] = use_react
            all_results.append(result)

        # --- Per-scenario histograms -------------------------------------
        tau = compute_threshold(cifar10_energy, percentile=5)

        print("\n  Plotting energy histograms...")
        for ood_name in OOD_NAMES:
            plot_score_histogram(
                cifar10_energy, ood_energy[ood_name],
                score_type="Energy", ood_name=ood_name,
                save_path=f"{args.results_dir}/{safe_name}_histogram_{ood_name.lower()}.png",
                threshold=tau,
            )

        # --- Per-scenario temperature analysis ---------------------------
        print("\n  Running temperature analysis...")
        for ood_name in OOD_NAMES:
            fpr95_vals = []
            for t_val in TEMPERATURES:
                id_e = compute_energy_score(cifar10_logits, temperature=t_val)
                ood_e = compute_energy_score(ood_logits[ood_name], temperature=t_val)
                fpr95_vals.append(compute_fpr_at_tpr95(id_e, ood_e))
            plot_temperature_analysis(
                TEMPERATURES, fpr95_vals,
                save_path=f"{args.results_dir}/{safe_name}_temperature_{ood_name.lower()}.png",
            )
            print(f"    Saved → {safe_name}_temperature_{ood_name.lower()}.png")

        # --- Cleanup -----------------------------------------------------
        if hook_handle is not None:
            remove_react_hook(hook_handle)

        del model, id_features, dtd_features
        print(f"\n  ✓ Scenario {idx} ({scenario}) complete.")

    # ==================================================================
    # Phase 3 — Save & display results
    # ==================================================================
    _header("Phase 3: Results", char="─")

    metrics_path = os.path.join(args.results_dir, "metrics.txt")
    _save_ablation_results(all_results, metrics_path)
    _print_summary_table(all_results)

    _header("PIPELINE COMPLETED")
    print(f"\n  Results directory: {os.path.abspath(args.results_dir)}/")
    print(f"     ├── metrics.txt                              (Comparison table)")
    print(f"     ├── <scenario>_histogram_<ood>.png           (Energy score histograms)")
    print(f"     ├── <scenario>_activations_dtd.png           (Activation distributions)")
    print(f"     └── <scenario>_temperature_<ood>.png         (Temperature analysis)")
    print()


if __name__ == "__main__":
    main()
