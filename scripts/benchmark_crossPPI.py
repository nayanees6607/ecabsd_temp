"""
ECABSD Benchmark — CrossPPI Benchmark Comparison.

Runs ECABSD predictions on standard PPI benchmark structures and
compares performance against baseline methods.

Usage:
    python scripts/benchmark_crossPPI.py --checkpoint checkpoints/best_model.pt
"""

import os
import sys
import csv
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.ecabsd_model import ECABSDModel
from models.graph_construction import build_residue_graph
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


# Published baseline results (approximate values from literature)
BASELINE_RESULTS = {
    "SPPIDER": {"precision": 0.45, "recall": 0.52, "f1": 0.48, "mcc": 0.25},
    "ProMate": {"precision": 0.42, "recall": 0.48, "f1": 0.45, "mcc": 0.22},
    "PSIVER": {"precision": 0.50, "recall": 0.45, "f1": 0.47, "mcc": 0.24},
    "PAIRpred": {"precision": 0.55, "recall": 0.50, "f1": 0.52, "mcc": 0.30},
    "DELPHI": {"precision": 0.58, "recall": 0.53, "f1": 0.55, "mcc": 0.33},
}


def run_benchmark(
    benchmark_dir: str = "data/raw/pdbs",
    checkpoint_path: str = "checkpoints/best_model.pt",
    output_path: str = "results/benchmark.csv",
    threshold: float = 0.5,
):
    """
    Run ECABSD on benchmark structures and compare with baselines.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ECABSDModel().to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Benchmark] Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"[Benchmark] WARNING: No checkpoint found. Using random weights.")

    model.eval()

    # Find benchmark PDBs
    import glob
    pdb_files = sorted(glob.glob(os.path.join(benchmark_dir, "*.pdb")))

    if not pdb_files:
        print(f"[Benchmark] No PDB files found in: {benchmark_dir}")
        print(f"[Benchmark] Run 'python scripts/download_pdbbind.py' first.")
        return

    print(f"[Benchmark] Running on {len(pdb_files)} structures...\n")

    all_labels = []
    all_preds = []
    per_structure = []

    for pdb_path in pdb_files:
        pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
        try:
            data_a = build_residue_graph(pdb_path, "A").to(device)
            probs, labels, _ = model.predict(data_a, threshold=threshold)

            # For benchmark, we need ground truth labels
            # Check if processed labels exist
            processed_path = os.path.join("data/processed", f"{pdb_name}_A.pt")
            if os.path.exists(processed_path):
                gt_data = torch.load(processed_path, weights_only=False)
                if hasattr(gt_data, "y") and gt_data.y is not None:
                    gt_labels = gt_data.y.numpy()
                    pred_labels = labels.cpu().numpy()

                    # Ensure same length
                    min_len = min(len(gt_labels), len(pred_labels))
                    gt_labels = gt_labels[:min_len]
                    pred_labels = pred_labels[:min_len]

                    all_labels.extend(gt_labels.tolist())
                    all_preds.extend(pred_labels.tolist())

                    p = precision_score(gt_labels, pred_labels, zero_division=0)
                    r = recall_score(gt_labels, pred_labels, zero_division=0)
                    f = f1_score(gt_labels, pred_labels, zero_division=0)
                    m = matthews_corrcoef(gt_labels, pred_labels)

                    per_structure.append({
                        "pdb_id": pdb_name,
                        "precision": f"{p:.4f}",
                        "recall": f"{r:.4f}",
                        "f1": f"{f:.4f}",
                        "mcc": f"{m:.4f}",
                        "num_residues": min_len,
                        "num_binding": int(gt_labels.sum()),
                    })

        except Exception as e:
            print(f"  [SKIP] {pdb_name}: {e}")

    # Compute overall ECABSD metrics
    if all_labels:
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        ecabsd_metrics = {
            "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
            "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
            "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
            "mcc": float(matthews_corrcoef(all_labels, all_preds)),
        }
    else:
        ecabsd_metrics = {"precision": 0, "recall": 0, "f1": 0, "mcc": 0}
        print("[Benchmark] WARNING: No ground truth labels found for comparison.")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  Cross-PPI Benchmark Comparison")
    print(f"{'='*70}")
    print(f"  {'Method':<15s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'MCC':>10s}")
    print(f"  {'─'*55}")

    for method, scores in BASELINE_RESULTS.items():
        print(
            f"  {method:<15s} {scores['precision']:>10.4f} {scores['recall']:>10.4f} "
            f"{scores['f1']:>10.4f} {scores['mcc']:>10.4f}"
        )

    print(f"  {'─'*55}")
    print(
        f"  {'ECABSD (ours)':<15s} {ecabsd_metrics['precision']:>10.4f} "
        f"{ecabsd_metrics['recall']:>10.4f} {ecabsd_metrics['f1']:>10.4f} "
        f"{ecabsd_metrics['mcc']:>10.4f}"
    )
    print(f"{'='*70}")

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save comparison CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "precision", "recall", "f1", "mcc"])
        writer.writeheader()
        for method, scores in BASELINE_RESULTS.items():
            writer.writerow({"method": method, **scores})
        writer.writerow({"method": "ECABSD (ours)", **ecabsd_metrics})

    print(f"\n  Benchmark saved to: {output_path}")

    # Save per-structure results
    if per_structure:
        per_struct_path = output_path.replace(".csv", "_per_structure.csv")
        with open(per_struct_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_structure[0].keys())
            writer.writeheader()
            writer.writerows(per_structure)
        print(f"  Per-structure results: {per_struct_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECABSD CrossPPI Benchmark")
    parser.add_argument("--benchmark-dir", default="data/raw/pdbs")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--output", default="results/benchmark.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    run_benchmark(
        benchmark_dir=args.benchmark_dir,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        threshold=args.threshold,
    )
