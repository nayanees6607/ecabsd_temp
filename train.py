"""
ECABSD Training Pipeline.

Handles:
- Config loading
- Dataset construction
- Model initialization
- Training loop with Focal / BCE loss, auto class weighting, early stopping
- Checkpoint saving
- Metric logging
"""

import os
import json
import time
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

from models.ecabsd_model import ECABSDModel
from data.dataset import BindingSiteDataset, collate_fn


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary Focal Loss.

    Suppresses the gradient contribution from easy (well-classified) negatives
    so training focuses on hard, ambiguous residues — exactly the scenario in
    imbalanced binding site detection.

    Parameters
    ----------
    alpha : float
        Balancing factor for the positive class (0.75 → up-weight positives).
    gamma : float
        Focusing exponent.  Higher values down-weight easy samples more
        aggressively.  gamma=2 is the standard starting point.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce        = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t        = torch.exp(-bce)                            # model confidence
        alpha_t    = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_w    = alpha_t * (1 - p_t) ** self.gamma
        return (focal_w * bce).mean()


def build_criterion(tcfg: dict, pos_weight: float, device: torch.device) -> nn.Module:
    """Instantiate loss function from config."""
    loss_name = tcfg.get("loss", "bce").lower()
    if loss_name == "focal":
        print(
            f"[ECABSD] Using Focal Loss  "
            f"(alpha={tcfg['focal_alpha']}, gamma={tcfg['focal_gamma']})"
        )
        return FocalLoss(alpha=tcfg["focal_alpha"], gamma=tcfg["focal_gamma"])
    else:
        print(f"[ECABSD] Using BCEWithLogitsLoss  (pos_weight={pos_weight:.2f})")
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight(dataset) -> float:
    """
    Compute negative/positive ratio from the training dataset.
    Used as pos_weight for BCEWithLogitsLoss.
    """
    total_pos = 0
    total_neg = 0
    for sample in dataset:
        labels = sample["labels"]
        total_pos += int(labels.sum().item())
        total_neg += int((labels == 0).sum().item())
    if total_pos == 0:
        return 7.0  # safe fallback
    return total_neg / total_pos


def compute_metrics(all_labels, all_preds) -> dict:
    """Compute per-epoch classification metrics."""
    return {
        "accuracy":  float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall":    float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1":        float(f1_score(all_labels, all_preds, zero_division=0)),
        "mcc":       float(matthews_corrcoef(all_labels, all_preds)),
    }


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, gradient_clip):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    all_labels, all_preds = [], []

    for sample in loader:
        data_a  = sample["data_a"].to(device)
        data_b  = sample["data_b"].to(device) if sample["data_b"] is not None else None
        labels  = sample["labels"].to(device)

        optimizer.zero_grad()
        logits, _ = model(data_a, data_b)
        logits = logits.squeeze(-1)

        loss = criterion(logits, labels.float())
        loss.backward()

        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        probs       = torch.sigmoid(logits)
        binary_preds = (probs >= 0.5).long().cpu().numpy()
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(binary_preds.tolist())

    avg_loss        = total_loss / max(len(all_labels), 1)
    metrics         = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation and sweep threshold for best F1."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs  = []

    for sample in loader:
        data_a  = sample["data_a"].to(device)
        data_b  = sample["data_b"].to(device) if sample["data_b"] is not None else None
        labels  = sample["labels"].to(device)

        logits, _ = model(data_a, data_b)
        logits = logits.squeeze(-1)

        loss   = criterion(logits, labels.float())
        total_loss += loss.item() * labels.size(0)

        probs        = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss        = total_loss / max(len(all_labels), 1)

    import numpy as np
    from sklearn.metrics import precision_recall_curve
    all_probs_np  = np.array(all_probs)
    all_labels_np = np.array(all_labels)

    if len(np.unique(all_labels_np)) > 1:
        precisions, recalls, thresh_vals = precision_recall_curve(all_labels_np, all_probs_np)
        f1_vals = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = int(np.argmax(f1_vals[:-1]))
        best_threshold = float(thresh_vals[best_idx])
    else:
        best_threshold = 0.5

    all_preds = (all_probs_np >= best_threshold).astype(int).tolist()
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    metrics["best_threshold"] = best_threshold
    return metrics


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run_training(config_path: str = "config.yaml", resume_from: str = None):
    """Main training function."""
    cfg  = load_config(config_path)
    tcfg = cfg["training"]
    mcfg = cfg["model"]
    pcfg = cfg["paths"]

    set_seed(tcfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ECABSD] Training on device: {device}")

    os.makedirs(pcfg["checkpoints_dir"], exist_ok=True)
    os.makedirs(pcfg["logs_dir"], exist_ok=True)

    # Build model
    model = ECABSDModel(
        input_dim=mcfg["input_dim"],
        hidden_dim=mcfg["hidden_dim"],
        num_heads=mcfg["num_heads"],
        dropout=mcfg["dropout"],
        edge_dim=mcfg["edge_feature_dim"],
    ).to(device)

    print(f"[ECABSD] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )

    # Dataset & loaders
    processed_dir = cfg["data"]["processed_dir"]
    splits_csv    = cfg["data"]["splits_csv"]

    if os.path.exists(processed_dir) and os.path.exists(splits_csv):
        train_dataset = BindingSiteDataset(processed_dir, splits_csv, split="train")
        val_dataset   = BindingSiteDataset(processed_dir, splits_csv, split="val")

        train_loader = DataLoader(
            train_dataset, batch_size=tcfg["batch_size"], shuffle=True,
            num_workers=tcfg["num_workers"], collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=tcfg["batch_size"], shuffle=False,
            num_workers=tcfg["num_workers"], collate_fn=collate_fn,
        )

        # Auto pos_weight from data
        pos_weight_val = tcfg.get("pos_weight", "auto")
        if pos_weight_val == "auto" or pos_weight_val is None:
            print("[ECABSD] Computing pos_weight from training data...")
            pos_weight_val = compute_pos_weight(train_dataset)
            print(f"[ECABSD] pos_weight = {pos_weight_val:.2f}")
    else:
        print(f"[ECABSD] WARNING: Processed data not found at '{processed_dir}'.")
        print(f"[ECABSD] Run 'python scripts/prepare_db5.py' first.")
        return

    # Loss function
    criterion = build_criterion(tcfg, pos_weight_val, device)

    # LR scheduler
    if tcfg["lr_scheduler"] == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min",
            patience=tcfg["lr_patience"], factor=tcfg["lr_factor"],
        )
    elif tcfg["lr_scheduler"] == "step":
        scheduler = StepLR(optimizer, step_size=tcfg["lr_patience"], gamma=tcfg["lr_factor"])
    elif tcfg["lr_scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=tcfg["epochs"])
    else:
        scheduler = None

    # Resume from checkpoint
    start_epoch  = 0
    best_val_loss = float("inf")
    if resume_from and os.path.exists(resume_from):
        checkpoint   = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch  = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"[ECABSD] Resumed from epoch {start_epoch}")

    # Training loop
    patience_counter = 0
    history          = []
    best_threshold   = 0.5

    print(f"\n{'='*60}")
    print(f"  ECABSD Training — {tcfg['epochs']} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, tcfg["epochs"]):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, tcfg["gradient_clip"]
        )
        val_metrics = validate(model, val_loader, criterion, device)
        best_threshold = val_metrics["best_threshold"]

        elapsed = time.time() - t0

        # LR scheduler step
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:03d}/{tcfg['epochs']} | "
            f"Train Loss: {train_metrics['loss']:.4f} F1: {train_metrics['f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} F1: {val_metrics['f1']:.4f} | "
            f"LR: {lr:.6f} | {elapsed:.1f}s"
        )

        epoch_record = {
            "epoch":     epoch + 1,
            "train":     train_metrics,
            "val":       val_metrics,
            "lr":        lr,
            "time":      elapsed,
            "threshold": best_threshold,
        }
        history.append(epoch_record)

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss    = val_metrics["loss"]
            patience_counter = 0
            ckpt_path        = os.path.join(pcfg["checkpoints_dir"], "best_model.pt")
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss":        best_val_loss,
                    "best_threshold":       best_threshold,
                    "config":               cfg,
                },
                ckpt_path,
            )
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f}, best_threshold={best_threshold:.4f})")
        else:
            patience_counter += 1

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(pcfg["checkpoints_dir"], f"epoch_{epoch+1}.pt")
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss":        best_val_loss,
                    "best_threshold":       best_threshold,
                    "config":               cfg,
                },
                ckpt_path,
            )

        # Early stopping
        if patience_counter >= tcfg["early_stopping_patience"]:
            print(f"\n[ECABSD] Early stopping at epoch {epoch+1}")
            break

    # Save training history
    history_path = os.path.join(pcfg["logs_dir"], "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Write best threshold back to config
    cfg_out = load_config(config_path)
    cfg_out["prediction"]["threshold"] = round(best_threshold, 4)
    with open(config_path, "w") as f:
        yaml.dump(cfg_out, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"  Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"  Best threshold:    {best_threshold:.4f}")
    print(f"  History saved to:  {history_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_training()
