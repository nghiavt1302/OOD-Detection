import os
import sys
import argparse
import numpy as np

from src.dataloader import load_cifar10, load_svhn, load_gaussian_noise
from src.model_loader import load_pretrained_model
from src.ood_detector import extract_logits, compute_energy_score
from src.evaluator import (
    compute_fpr_at_tpr95,
    compute_threshold,
    evaluate_ood,
    plot_score_histogram,
    plot_temperature_analysis,
    save_results_to_file,
)
from src.ood_detector import compute_energy_score as _compute_energy  # alias for temp sweep
from src.react_utils import apply_react_hook, remove_react_hook


# ---------------------------------------------------------------------------
# Ablation study configurations
# ---------------------------------------------------------------------------
ABLATION_CONFIGS = [
    {"name": "Base",     "model_name": "resnet20", "react": False, "threshold_c": None},
    {"name": "Capacity", "model_name": "resnet56", "react": False, "threshold_c": None},
    {"name": "ReAct",    "model_name": "resnet20", "react": True,  "threshold_c": 2.5},
    {"name": "SOTA",     "model_name": "resnet56", "react": True,  "threshold_c": 2.5},
]

OOD_DATASETS = ["SVHN", "Noise"]
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
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_header(text, char="=", width=70):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def _print_summary_table(all_results):
    """Print a nicely formatted comparison table of all scenarios."""
    _print_header("Ablation Study — Summary Table")

    header = (
        f"  {'Scenario':<12} {'Model':<10} {'ReAct':<7} "
        f"{'OOD':<8} {'FPR95↓':>8} {'AUROC↑':>8} {'AUPR↑':>8}"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for r in all_results:
        react_str = "Yes" if r.get("react") else "No"
        print(
            f"  {r['scenario']:<12} {r['model']:<10} {react_str:<7} "
            f"{r['ood_dataset']:<8} {r['FPR95']:>8.4f} {r['AUROC']:>8.4f} {r['AUPR']:>8.4f}"
        )

    print()


def _save_ablation_results(all_results, save_path):
    """Save ablation results to a structured text file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  Ablation Study: Model Capacity × ReAct — Energy-Based OOD Detection\n")
        f.write("=" * 80 + "\n\n")

        header = (
            f"  {'Scenario':<12} {'Model':<10} {'ReAct':<7} "
            f"{'OOD':<8} {'FPR95':>8} {'AUROC':>8} {'AUPR':>8}\n"
        )
        sep = "  " + "─" * 72 + "\n"
        f.write(header)
        f.write(sep)

        for r in all_results:
            react_str = "Yes" if r.get("react") else "No"
            f.write(
                f"  {r['scenario']:<12} {r['model']:<10} {react_str:<7} "
                f"{r['ood_dataset']:<8} {r['FPR95']:>8.4f} {r['AUROC']:>8.4f} {r['AUPR']:>8.4f}\n"
            )

        f.write("\n" + "=" * 80 + "\n")

    print(f"  Results saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    T = args.temperature

    _print_header("🔬 Ablation Study: Model Capacity × ReAct")
    print(f"  Temperature T = {T}")
    print(f"  Configurations: {len(ABLATION_CONFIGS)}")
    print(f"  OOD datasets  : {OOD_DATASETS}")

    # ------------------------------------------------------------------
    # Phase 1 — Load datasets (shared across all configs)
    # ------------------------------------------------------------------
    _print_header("Phase 1: Loading Datasets", char="─")

    print("\n  [Step 1.1] CIFAR-10 (ID)...")
    cifar10_loader = load_cifar10(data_dir=args.data_dir, batch_size=args.batch_size)

    print("  [Step 1.2] SVHN (OOD)...")
    svhn_loader = load_svhn(data_dir=args.data_dir, batch_size=args.batch_size)

    print("  [Step 1.3] Gaussian Noise (OOD)...")
    noise_loader = load_gaussian_noise(batch_size=args.batch_size)

    ood_loaders = {"SVHN": svhn_loader, "Noise": noise_loader}

    # ------------------------------------------------------------------
    # Phase 2 — Run ablation loop
    # ------------------------------------------------------------------
    _print_header("Phase 2: Running Ablation Scenarios", char="─")

    all_results = []

    for idx, cfg in enumerate(ABLATION_CONFIGS, start=1):
        scenario = cfg["name"]
        model_name = cfg["model_name"]
        use_react = cfg["react"]
        threshold_c = cfg["threshold_c"]

        print(f"\n{'━' * 70}")
        print(f"  [INFO] Running Scenario {idx}: {scenario} "
              f"({model_name}{' + ReAct' if use_react else ''})")
        print(f"{'━' * 70}")

        # --- Load model ---
        print(f"\n  Loading {model_name}...")
        model, device = load_pretrained_model(model_name=model_name)

        # --- Apply ReAct hook if required ---
        hook_handle = None
        if use_react:
            hook_handle = apply_react_hook(model, threshold_c=threshold_c)

        # --- Extract logits ---
        print("\n  Extracting logits...")
        print("    → CIFAR-10 (ID)")
        cifar10_logits = extract_logits(model, cifar10_loader, device)
        print(f"      Shape: {cifar10_logits.shape}")

        ood_logits = {}
        for ood_name, ood_loader in ood_loaders.items():
            print(f"    → {ood_name} (OOD)")
            ood_logits[ood_name] = extract_logits(model, ood_loader, device)
            print(f"      Shape: {ood_logits[ood_name].shape}")

        # --- Compute Energy Scores ---
        print(f"\n  Computing Energy Scores (T={T})...")
        cifar10_energy = compute_energy_score(cifar10_logits, temperature=T)
        print(f"    CIFAR-10 energy mean: {cifar10_energy.mean():.4f}")

        ood_energy = {}
        for ood_name in OOD_DATASETS:
            ood_energy[ood_name] = compute_energy_score(ood_logits[ood_name], temperature=T)
            print(f"    {ood_name} energy mean: {ood_energy[ood_name].mean():.4f}")

        # --- Evaluate Metrics ---
        print("\n  Evaluating metrics...")
        for ood_name in OOD_DATASETS:
            result = evaluate_ood(
                cifar10_energy, ood_energy[ood_name],
                method_name=f"Energy ({scenario})",
                ood_name=ood_name,
            )
            # Attach extra metadata for the summary table
            result["scenario"] = scenario
            result["model"] = model_name
            result["react"] = use_react
            all_results.append(result)

        # --- Visualizations per scenario ---
        safe_name = cfg["name"].replace(" ", "_").replace("+", "plus")

        # Threshold for histograms (same ID threshold for both OOD sets)
        tau = compute_threshold(cifar10_energy, percentile=5)

        print("\n  Plotting histograms...")
        plot_score_histogram(
            cifar10_energy, ood_energy["SVHN"],
            score_type="Energy", ood_name="SVHN",
            save_path=f"{args.results_dir}/{safe_name}_histogram_svhn.png",
            threshold=tau,
        )
        plot_score_histogram(
            cifar10_energy, ood_energy["Noise"],
            score_type="Energy", ood_name="Noise",
            save_path=f"{args.results_dir}/{safe_name}_histogram_noise.png",
            threshold=tau,
        )

        # Temperature scaling analysis (sweep T while model+hook are still active)
        print("\n  Running temperature analysis...")
        for ood_name in OOD_DATASETS:
            fpr95_values = []
            for t_val in TEMPERATURES:
                id_e = _compute_energy(cifar10_logits, temperature=t_val)
                ood_e = _compute_energy(ood_logits[ood_name], temperature=t_val)
                fpr95_values.append(compute_fpr_at_tpr95(id_e, ood_e))
            plot_temperature_analysis(
                TEMPERATURES, fpr95_values,
                save_path=f"{args.results_dir}/{safe_name}_temperature_{ood_name.lower()}.png",
            )
            print(f"    Saved temperature chart → {safe_name}_temperature_{ood_name.lower()}.png")

        # --- Cleanup ---
        if hook_handle is not None:
            remove_react_hook(hook_handle)

        # Let Python GC reclaim model memory on next iteration
        del model
        print(f"\n  ✓ Scenario {idx} ({scenario}) complete.")

    # ------------------------------------------------------------------
    # Phase 3 — Save & display results
    # ------------------------------------------------------------------
    _print_header("Phase 3: Results", char="─")

    metrics_path = os.path.join(args.results_dir, "metrics.txt")
    _save_ablation_results(all_results, metrics_path)
    _print_summary_table(all_results)

    _print_header("PIPELINE COMPLETED")
    print(f"\n  Results saved at: {os.path.abspath(args.results_dir)}/")
    print(f"     └── metrics.txt  (Ablation Study comparison table)")
    print()


if __name__ == "__main__":
    main()
