"""
ECABSD Training Pipeline — v2 (Excellence Edition).

Key upgrades over v1:
- Combined Focal + Soft-Dice loss (directly optimises F1)
- Cosine-warmup LR scheduler (LinearLR warmup → CosineAnnealingLR)
- Early stopping on val F1 (not val loss)
- Full AUC-ROC + AUC-PR logging every epoch
- PR-curve-based threshold sweep on val set after every epoch
- AdamW optimizer with decoupled weight decay
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LinearLR, SequentialLR,
)
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef,
    roc_auc_score, average_precision_score,
    precision_recall_curve,
)

from models.ecabsd_model import ECABSDModel
from data.dataset import BindingSiteDataset, collate_fn


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary Focal Loss — suppresses easy negatives."""

    def __init__(self, alpha: float = 0.90, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t     = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_w = alpha_t * (1 - p_t) ** self.gamma
        return (focal_w * bce).mean()


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss — directly optimises the F1 / Dice coefficient.

    Unlike cross-entropy, Dice loss treats prediction as a soft mask
    and measures overlap, which directly corresponds to the F1 metric.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs        = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice         = (2.0 * intersection + self.smooth) / \
                       (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Focal + Soft-Dice combined loss.

    Focal handles class imbalance; Dice directly optimises F1.
    dice_weight = 0.4 → 60% focal + 40% dice works well in practice.
    """

    def __init__(self, focal_alpha=0.90, focal_gamma=2.0, dice_weight=0.4):
        super().__init__()
        self.focal       = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice        = SoftDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (1 - self.dice_weight) * self.focal(logits, targets) + \
                    self.dice_weight  * self.dice(logits, targets)


def build_criterion(tcfg: dict, pos_weight: float, device: torch.device) -> nn.Module:
    """Instantiate loss from config."""
    loss_name = tcfg.get("loss", "combined").lower()
    if loss_name == "combined":
        dw = tcfg.get("dice_weight", 0.4)
        print(f"[ECABSD] Using CombinedLoss  "
              f"(focal_alpha={tcfg['focal_alpha']}, gamma={tcfg['focal_gamma']}, "
              f"dice_weight={dw})")
        return CombinedLoss(
            focal_alpha=tcfg["focal_alpha"],
            focal_gamma=tcfg["focal_gamma"],
            dice_weight=dw,
        )
    elif loss_name == "focal":
        print(f"[ECABSD] Using FocalLoss  "
              f"(alpha={tcfg['focal_alpha']}, gamma={tcfg['focal_gamma']})")
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
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight(dataset) -> float:
    total_pos = total_neg = 0
    for sample in dataset:
        labels = sample["labels"]
        total_pos += int(labels.sum().item())
        total_neg += int((labels == 0).sum().item())
    return (total_neg / total_pos) if total_pos > 0 else 7.0


def compute_metrics(all_labels, all_preds, all_probs=None) -> dict:
    m = {
        "accuracy":  float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall":    float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1":        float(f1_score(all_labels, all_preds, zero_division=0)),
        "mcc":       float(matthews_corrcoef(all_labels, all_preds)),
    }
    if all_probs is not None:
        labels_np = np.array(all_labels)
        probs_np  = np.array(all_probs)
        if len(np.unique(labels_np)) > 1:
            m["auc_roc"] = float(roc_auc_score(labels_np, probs_np))
            m["auc_pr"]  = float(average_precision_score(labels_np, probs_np))
        else:
            m["auc_roc"] = 0.0
            m["auc_pr"]  = 0.0
    return m


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, gradient_clip):
    model.train()
    total_loss = 0.0
    all_labels, all_preds = [], []

    for sample in loader:
        data_a  = sample["data_a"].to(device)
        data_b  = sample["data_b"].to(device) if sample["data_b"] is not None else None
        labels  = sample["labels"].to(device)

        optimizer.zero_grad()
        logits, _ = model(data_a, data_b)
        logits    = logits.squeeze(-1)

        loss = criterion(logits, labels.float())
        loss.backward()

        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss  += loss.item() * labels.size(0)
        probs        = torch.sigmoid(logits)
        binary_preds = (probs >= 0.5).long().cpu().numpy()
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(binary_preds.tolist())

    avg_loss        = total_loss / max(len(all_labels), 1)
    metrics         = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate and sweep PR-curve for best threshold."""
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for sample in loader:
        data_a  = sample["data_a"].to(device)
        data_b  = sample["data_b"].to(device) if sample["data_b"] is not None else None
        labels  = sample["labels"].to(device)

        logits, _ = model(data_a, data_b)
        logits    = logits.squeeze(-1)

        loss = criterion(logits, labels.float())
        total_loss += loss.item() * labels.size(0)

        probs = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss      = total_loss / max(len(all_labels), 1)
    all_probs_np  = np.array(all_probs)
    all_labels_np = np.array(all_labels)

    # PR-curve threshold sweep (on val set — legally)
    if len(np.unique(all_labels_np)) > 1:
        precisions, recalls, thresh_vals = precision_recall_curve(all_labels_np, all_probs_np)
        f1_vals    = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx   = int(np.argmax(f1_vals[:-1]))
        best_thr   = float(thresh_vals[best_idx])
    else:
        best_thr = 0.5

    all_preds = (all_probs_np >= best_thr).astype(int).tolist()
    metrics   = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"]           = avg_loss
    metrics["best_threshold"] = best_thr
    return metrics


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run_training(config_path: str = "config.yaml", resume_from: str = None):
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
        num_cross_attn_layers=mcfg.get("num_cross_attn_layers", 2),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[ECABSD] Model parameters: {total_params:,}")

    # AdamW optimizer (better generalisation than Adam)
    optimizer = AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
        betas=(0.9, 0.999),
    )

    # Dataset & loaders
    processed_dir = cfg["data"]["processed_dir"]
    splits_csv    = cfg["data"]["splits_csv"]

    if not (os.path.exists(processed_dir) and os.path.exists(splits_csv)):
        print(f"[ECABSD] ERROR: Processed data not found. Run prepare_db5.py first.")
        return

    train_dataset = BindingSiteDataset(processed_dir, splits_csv, split="train")
    val_dataset   = BindingSiteDataset(processed_dir, splits_csv, split="val")
    print(f"[Dataset] Loaded {len(train_dataset)} samples for split 'train'")
    print(f"[Dataset] Loaded {len(val_dataset)} samples for split 'val'")

    train_loader = DataLoader(
        train_dataset, batch_size=tcfg["batch_size"], shuffle=True,
        num_workers=tcfg["num_workers"], collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=tcfg["batch_size"], shuffle=False,
        num_workers=tcfg["num_workers"], collate_fn=collate_fn,
    )

    # Pos-weight
    print("[ECABSD] Computing pos_weight from training data...")
    pos_weight_val = compute_pos_weight(train_dataset)
    print(f"[ECABSD] pos_weight = {pos_weight_val:.2f}")

    # Loss
    criterion = build_criterion(tcfg, pos_weight_val, device)

    # Cosine-warmup LR scheduler
    warmup_epochs  = tcfg.get("warmup_epochs", 15)
    cosine_epochs  = max(tcfg["epochs"] - warmup_epochs, 1)
    warmup_sched   = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched   = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6)
    scheduler      = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

    print(f"[ECABSD] LR: warmup {warmup_epochs} epochs → cosine {cosine_epochs} epochs")

    # Resume
    start_epoch   = 0
    best_val_loss = float("inf")
    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[ECABSD] Resumed from epoch {start_epoch}")

    # Loop
    patience_counter = 0
    best_val_f1      = -1.0
    best_threshold   = 0.5
    history          = []

    print(f"\n{'='*60}")
    print(f"  ECABSD v2 Training — {tcfg['epochs']} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, tcfg["epochs"]):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, tcfg["gradient_clip"]
        )
        val_metrics   = validate(model, val_loader, criterion, device)
        best_threshold = val_metrics["best_threshold"]

        scheduler.step()
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:03d}/{tcfg['epochs']} | "
            f"Train Loss: {train_metrics['loss']:.4f} F1: {train_metrics['f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} F1: {val_metrics['f1']:.4f} "
            f"AUC-ROC: {val_metrics.get('auc_roc', 0):.4f} "
            f"AUC-PR: {val_metrics.get('auc_pr', 0):.4f} | "
            f"LR: {lr:.6f} | {elapsed:.1f}s"
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val":   val_metrics,
            "lr":    lr,
            "time":  elapsed,
            "threshold": best_threshold,
        }
        history.append(epoch_record)

        # Save best model on val F1
        current_f1 = val_metrics["f1"]
        if current_f1 > best_val_f1:
            best_val_f1      = current_f1
            best_val_loss    = val_metrics["loss"]
            patience_counter = 0
            ckpt_path        = os.path.join(pcfg["checkpoints_dir"], "best_model.pt")
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss":        best_val_loss,
                    "best_val_f1":          best_val_f1,
                    "best_threshold":       best_threshold,
                    "config":               cfg,
                },
                ckpt_path,
            )
            print(f"  -> New best! val_F1={best_val_f1:.4f} | "
                  f"AUC-ROC={val_metrics.get('auc_roc',0):.4f} | "
                  f"threshold={best_threshold:.4f}")
        else:
            patience_counter += 1

        # Periodic checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            ckpt_path = os.path.join(pcfg["checkpoints_dir"], f"epoch_{epoch+1}.pt")
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_f1":      best_val_f1,
                    "best_threshold":   best_threshold,
                    "config":           cfg,
                },
                ckpt_path,
            )

        # Early stopping on F1
        if patience_counter >= tcfg["early_stopping_patience"]:
            print(f"\n[ECABSD] Early stopping at epoch {epoch+1} (best val F1={best_val_f1:.4f})")
            break

    # Save history
    history_path = os.path.join(pcfg["logs_dir"], "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Write best threshold to config
    cfg_out = load_config(config_path)
    cfg_out["prediction"]["threshold"] = round(best_threshold, 4)
    with open(config_path, "w") as f:
        yaml.dump(cfg_out, f, default_flow_style=False, sort_keys=False)

    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Best val F1:    {best_val_f1:.4f}")
    print(f"  Best val loss:  {best_val_loss:.4f}")
    print(f"  Best threshold: {best_threshold:.4f}")
    print(f"  History:        {history_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_training()
