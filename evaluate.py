"""
ECABSD Evaluation Pipeline.

Evaluates a trained model on the test set and produces:
- Accuracy, Precision, Recall, F1, MCC, AUC-ROC, AUC-PR
- Confusion matrix plot
- Per-structure breakdown
"""

import os
import json
import yaml
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from models.ecabsd_model import ECABSDModel


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def plot_confusion_matrix(cm, output_path):
    """Save confusion matrix as an image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Non-binding", "Binding"],
            yticklabels=["Non-binding", "Binding"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("ECABSD — Confusion Matrix")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"  Confusion matrix saved to: {output_path}")
    except ImportError:
        print("  [WARN] matplotlib/seaborn not available; skipping confusion matrix plot.")


def run_evaluation(config_path: str = "config.yaml", checkpoint_path: str = "checkpoints/best_model.pt"):
    """Run full evaluation on test set."""
    cfg = load_config(config_path)
    mcfg = cfg["model"]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ECABSD] Evaluating on device: {device}")

    # Load model — must match training architecture exactly
    model = ECABSDModel(
        input_dim=mcfg["input_dim"],
        hidden_dim=mcfg["hidden_dim"],
        num_heads=mcfg["num_heads"],
        dropout=0.0,  # No dropout during evaluation
        edge_dim=mcfg["edge_feature_dim"],
        num_cross_attn_layers=mcfg.get("num_cross_attn_layers", 1),
        num_gcn_layers=mcfg.get("num_gcn_layers", 4),
    ).to(device)

    # Load checkpoint and recover saved threshold
    saved_threshold = cfg["prediction"].get("threshold", 0.5)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        saved_threshold = checkpoint.get("best_threshold", saved_threshold)
        print(f"[ECABSD] Loaded checkpoint from: {checkpoint_path}")
        print(f"[ECABSD] Using saved threshold: {saved_threshold:.4f}")
    else:
        print(f"[ECABSD] WARNING: No checkpoint found at {checkpoint_path}")
        print(f"[ECABSD] Running with random weights for demonstration.")

    model.eval()

    # Load test data
    processed_dir = cfg["data"]["processed_dir"]
    splits_csv = cfg["data"]["splits_csv"]

    all_probs = []
    all_labels = []

    if os.path.exists(processed_dir) and os.path.exists(splits_csv):
        from data.dataset import BindingSiteDataset, collate_fn
        from torch.utils.data import DataLoader

        test_dataset = BindingSiteDataset(processed_dir, splits_csv, split="test")
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )

        with torch.no_grad():
            for batch in test_loader:
                data_a = batch["data_a"].to(device)
                data_b = batch["data_b"].to(device)   # always a Batch (collate_fn guarantees)
                labels = batch["labels"]

                logits, _ = model(data_a, data_b)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
    else:
        print(f"[ECABSD] No processed test data found. Using sample PDB for demonstration.")
        from models.graph_construction import build_residue_graph

        sample_pdb = "1AY7.pdb"
        if os.path.exists(sample_pdb):
            data_a = build_residue_graph(sample_pdb, "A")
            data_a = data_a.to(device)

            with torch.no_grad():
                logits, attn = model(data_a)
                probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

            # Create dummy labels for demonstration
            dummy_labels = np.zeros(len(probs))
            dummy_labels[:10] = 1.0
            all_probs = probs.tolist()
            all_labels = dummy_labels.tolist()
        else:
            print("[ECABSD] ERROR: No data available for evaluation.")
            return

    # Compute metrics
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    best_threshold = saved_threshold  # comes from checkpoint["best_threshold"]
    print(f"  [Threshold] Using val-optimised threshold: {best_threshold:.4f}")

    all_preds = (all_probs >= best_threshold).astype(int)

    metrics = {
        "accuracy":              float(accuracy_score(all_labels, all_preds)),
        "precision":             float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall":                float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1":                    float(f1_score(all_labels, all_preds, zero_division=0)),
        "mcc":                   float(matthews_corrcoef(all_labels, all_preds)),
        "threshold":             float(best_threshold),
        "num_samples":           len(all_labels),
        "num_positive":          int(all_labels.sum()),
        "num_predicted_positive": int(all_preds.sum()),
    }

    # AUC metrics (need both classes present)
    if len(np.unique(all_labels)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(all_labels, all_probs))
        metrics["auc_pr"] = float(average_precision_score(all_labels, all_probs))
    else:
        metrics["auc_roc"] = None
        metrics["auc_pr"] = None

    # Print results
    print(f"\n{'='*50}")
    print("  ECABSD Evaluation Results")
    print(f"{'='*50}")
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:>25s}: {val:.4f}")
        else:
            print(f"  {key:>25s}: {val}")
    print(f"{'='*50}\n")

    # Save metrics
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to: {metrics_path}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, cm_path)

    return metrics


if __name__ == "__main__":
    run_evaluation()
