"""
Softmax vs Energy — OOD Detection Final Pipeline
=================================================
  Phase 1 : Temperature (T) Ablation (ResNet-20, T in {0.1,1,10,100,1000})
  Phase 2 : Softmax vs Energy — ResNet-20, ResNet-56, VGG16-BN
  Phase 3 : Summary Table with Tau threshold

OOD datasets : SVHN, Gaussian Noise, DTD, Places365
Metrics      : FPR@95, AUROC, AUPR
"""

import os, gc, argparse
import torch
import numpy as np

from src.dataloader import (
    load_cifar10, load_svhn, load_gaussian_noise, load_dtd, load_places365,
)
from src.model_loader import load_pretrained_model
from src.ood_detector import extract_logits, compute_energy_score, compute_softmax_score
from src.evaluator import (
    compute_threshold_tau, evaluate_ood,
    plot_score_histogram, plot_temperature_analysis,
)

MODELS    = ["resnet20", "resnet56", "vgg16_bn"]
OOD_NAMES = ["SVHN", "Noise", "DTD", "Places365"]
T_VALUES  = [0.1, 1, 10, 100, 1000]
MODEL_LABELS = {"resnet20": "ResNet-20", "resnet56": "ResNet-56", "vgg16_bn": "VGG16-BN"}


def parse_args():
    p = argparse.ArgumentParser(description="Softmax vs Energy OOD Detection")
    p.add_argument("--data-dir",    type=str, default="./data")
    p.add_argument("--results-dir", type=str, default="./results")
    p.add_argument("--batch-size",  type=int, default=128)
    return p.parse_args()


def _header(text, char="=", width=78):
    print(f"\n{char * width}\n  {text}\n{char * width}")


def _print_summary_table(all_results):
    _header("FINAL RESULTS -- Softmax vs Energy (T=1)")
    hdr = (f"  {'Model':<11} {'Method':<10} {'OOD Dataset':<12} "
           f"{'FPR95':>8} {'AUROC':>8} {'AUPR':>8}")
    print(hdr)
    print("  " + "-" * 62)
    prev = None
    for r in all_results:
        if prev and r["model"] != prev:
            print("  " + "." * 62)
        prev = r["model"]
        print(f"  {MODEL_LABELS[r['model']]:<11} {r['method']:<10} "
              f"{r['ood_dataset']:<12} {r['FPR95']:>8.4f} "
              f"{r['AUROC']:>8.4f} {r['AUPR']:>8.4f}")
    print("  " + "-" * 62 + "\n")


def _save_summary_table(all_results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("  Softmax vs Energy -- Final Results (T=1)\n")
        f.write("=" * 80 + "\n\n")
        hdr = (f"  {'Model':<11} {'Method':<10} {'OOD Dataset':<12} "
               f"{'FPR95':>8} {'AUROC':>8} {'AUPR':>8}\n")
        f.write(hdr)
        f.write("  " + "-" * 62 + "\n")
        prev = None
        for r in all_results:
            if prev and r["model"] != prev:
                f.write("  " + "." * 62 + "\n")
            prev = r["model"]
            f.write(f"  {MODEL_LABELS[r['model']]:<11} {r['method']:<10} "
                    f"{r['ood_dataset']:<12} {r['FPR95']:>8.4f} "
                    f"{r['AUROC']:>8.4f} {r['AUPR']:>8.4f}\n")
        f.write("  " + "-" * 62 + "\n")
        f.write("\n" + "=" * 80 + "\n")
    print(f"  Saved: {path}")


def _print_energy_vs_softmax(all_results):
    _header("Energy vs Softmax -- Average FPR95 Comparison")
    for m in MODELS:
        s_fprs = [r["FPR95"] for r in all_results if r["model"]==m and r["method"]=="Softmax"]
        e_fprs = [r["FPR95"] for r in all_results if r["model"]==m and r["method"]=="Energy"]
        if s_fprs and e_fprs:
            avg_s, avg_e = np.mean(s_fprs), np.mean(e_fprs)
            d = avg_s - avg_e
            pct = (d / avg_s * 100) if avg_s > 0 else 0
            tag = "Energy wins" if d > 0 else "Softmax wins"
            print(f"  {MODEL_LABELS[m]:<11}  Softmax={avg_s:.4f}  Energy={avg_e:.4f}  "
                  f"Delta={d:+.4f} ({pct:+.1f}%)  {tag}")
    print()


def _cleanup(model):
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    _header("Softmax vs Energy -- OOD Detection Final Pipeline")
    print(f"  Models       : {[MODEL_LABELS[m] for m in MODELS]}")
    print(f"  OOD datasets : {OOD_NAMES}")
    print(f"  T ablation   : {T_VALUES}")

    # ── Phase 0: Load datasets ──────────────────────────────────────
    _header("Phase 0: Loading Datasets", char="-")
    cifar10_loader = load_cifar10(data_dir=args.data_dir, batch_size=args.batch_size)
    svhn_loader    = load_svhn(data_dir=args.data_dir, batch_size=args.batch_size)
    noise_loader   = load_gaussian_noise(batch_size=args.batch_size)
    dtd_loader     = load_dtd(data_dir=args.data_dir, batch_size=args.batch_size)
    places_loader  = load_places365(data_dir=args.data_dir, batch_size=args.batch_size)
    ood_loaders = {"SVHN": svhn_loader, "Noise": noise_loader,
                   "DTD": dtd_loader, "Places365": places_loader}
    active_ood = [n for n in OOD_NAMES if n in ood_loaders]
    print("  All datasets loaded.\n")

    # ── Phase 1: Temperature Ablation (ResNet-20) ───────────────────
    _header("Phase 1: Temperature Ablation (ResNet-20)", char="-")
    print(f"  T values : {T_VALUES}")
    print(f"  Goal     : Prove T=1 is optimal (Liu et al.)\n")

    model_t, device = load_pretrained_model(model_name="resnet20")
    print("  Extracting logits...")
    c10_logits_t = extract_logits(model_t, cifar10_loader, device)
    ood_logits_t = {n: extract_logits(model_t, ood_loaders[n], device) for n in active_ood}

    t_results = []
    for T_val in T_VALUES:
        print(f"\n  -- T = {T_val} --")
        id_e = compute_energy_score(c10_logits_t, temperature=T_val)
        fprs = []
        for oname in active_ood:
            ood_e = compute_energy_score(ood_logits_t[oname], temperature=T_val)
            r = evaluate_ood(id_e, ood_e, method_name="Energy", ood_name=oname, verbose=False)
            fprs.append(r["FPR95"])
            print(f"    {oname:<12} FPR95 = {r['FPR95']:.4f}")
        avg = np.mean(fprs)
        t_results.append((T_val, avg))
        print(f"    {'Average':<12} FPR95 = {avg:.4f}")

    ts   = [r[0] for r in t_results]
    fprs = [r[1] for r in t_results]
    plot_temperature_analysis(ts, fprs,
        save_path=os.path.join(args.results_dir, "T_Ablation_Analysis.png"))
    best_i = int(np.argmin(fprs))
    print(f"\n  Best T = {ts[best_i]}  (avg FPR95 = {fprs[best_i]:.4f})")
    _cleanup(model_t)

    # ── Phase 2: Softmax vs Energy (T=1) ────────────────────────────
    _header("Phase 2: Softmax vs Energy Evaluation (T=1.0)", char="-")
    T = 1.0
    all_results = []

    for idx, mname in enumerate(MODELS, 1):
        print(f"\n  [{idx}/{len(MODELS)}] {MODEL_LABELS[mname]}")
        print(f"  {'=' * 40}")
        model, device = load_pretrained_model(model_name=mname)

        c10_logits = extract_logits(model, cifar10_loader, device)
        ood_logits = {n: extract_logits(model, ood_loaders[n], device) for n in active_ood}

        c10_energy = compute_energy_score(c10_logits, temperature=T)
        c10_msp    = compute_softmax_score(c10_logits)

        tau_e = compute_threshold_tau(c10_energy, tpr=0.95)
        tau_s = compute_threshold_tau(c10_msp, tpr=0.95)
        print(f"  Tau_energy={tau_e:.4f}  Tau_softmax={tau_s:.4f}")

        for oname in active_ood:
            ood_e = compute_energy_score(ood_logits[oname], temperature=T)
            ood_s = compute_softmax_score(ood_logits[oname])
            for method, id_sc, ood_sc, tau in [
                ("Softmax", c10_msp, ood_s, tau_s),
                ("Energy",  c10_energy, ood_e, tau_e),
            ]:
                res = evaluate_ood(id_sc, ood_sc, method_name=method,
                                   ood_name=oname, verbose=False)
                res.update(model=mname, tau=tau)
                all_results.append(res)
                print(f"    [{method:<7}] vs {oname:<10}  "
                      f"FPR95={res['FPR95']:.4f}  AUROC={res['AUROC']:.4f}")

        # Histograms
        for oname in active_ood:
            plot_score_histogram(
                c10_msp, compute_softmax_score(ood_logits[oname]),
                score_type="Softmax", ood_name=oname, threshold=tau_s,
                save_path=os.path.join(args.results_dir,
                    f"{MODEL_LABELS[mname]}_Softmax_histogram_{oname}.png"))
            plot_score_histogram(
                c10_energy, compute_energy_score(ood_logits[oname], T),
                score_type="Energy", ood_name=oname, threshold=tau_e,
                save_path=os.path.join(args.results_dir,
                    f"{MODEL_LABELS[mname]}_Energy_histogram_{oname}.png"))

        print(f"  Done: {MODEL_LABELS[mname]}")
        _cleanup(model)

    # ── Phase 3: Results ────────────────────────────────────────────
    _header("Phase 3: Results", char="-")
    mpath = os.path.join(args.results_dir, "final_results.txt")
    _save_summary_table(all_results, mpath)
    _print_summary_table(all_results)
    _print_energy_vs_softmax(all_results)

    _header("PIPELINE COMPLETED")
    print(f"\n  Results: {os.path.abspath(args.results_dir)}/")
    print(f"    T_Ablation_Analysis.png")
    print(f"    final_results.txt")
    for m in MODELS:
        print(f"    {MODEL_LABELS[m]}_Softmax_histogram_*.png")
        print(f"    {MODEL_LABELS[m]}_Energy_histogram_*.png")
    print()


if __name__ == "__main__":
    main()
