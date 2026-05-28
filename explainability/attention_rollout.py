"""
ECABSD Attention Rollout — Explainability via cross-attention weights.

Extracts and visualizes per-residue importance scores from the
CrossAttention layer using attention rollout.

For ECABSD the cross-attention is a stack of CrossEncoder layers. We capture
the attention weights from the first layer's CrossFusion sub-module using a
forward hook — this is the clearest spatial signal before deep stacking.

Usage:
    from explainability.attention_rollout import AttentionRollout
    rollout = AttentionRollout(model)
    scores = rollout.compute(data_a, data_b)
    rollout.plot_heatmap(scores, residue_ids)
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ecabsd_model import ECABSDModel


class AttentionRollout:
    """
    Computes per-residue importance scores from cross-attention weights.

    Hooks into the first CrossFusion layer of the CrossAttention stack to
    capture raw attention weights (chain A attending to chain B).
    When multiple heads are present, they are fused by `head_fusion`.

    Parameters
    ----------
    model : ECABSDModel
        Trained ECABSD model.
    head_fusion : str
        How to fuse multi-head attention: 'mean', 'max', or 'min'.
    """

    def __init__(self, model: ECABSDModel, head_fusion: str = "mean"):
        self.model        = model
        self.head_fusion  = head_fusion
        self._attn_a      = None  # will hold (1, H, N_a, N_b) after hook fires

        # Hook into the CrossFusion module inside the first CrossEncoder layer.
        # CrossFusion.forward() returns ((ctx_a, ctx_b), (attn_a, attn_b)).
        # We capture attn_a (chain A attending to chain B) via the output hook.
        first_fusion = model.cross_attention.layers[0].fusion
        self._hook = first_fusion.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        """Capture chain-A attention weights from CrossFusion output."""
        # CrossFusion.forward returns: ((ctx_a, ctx_b), (attn_a, attn_b))
        _, (attn_a, _) = output
        self._attn_a = attn_a.detach().cpu()   # (1, H, N_a, N_b)

    def remove_hook(self):
        """Remove the forward hook to avoid memory leaks."""
        self._hook.remove()

    def compute(self, data_a, data_b=None):
        """
        Compute per-residue importance scores.

        Parameters
        ----------
        data_a : torch_geometric.data.Data
            Graph for chain A.
        data_b : torch_geometric.data.Data, optional
            Graph for chain B. If None, chain A attends to itself.

        Returns
        -------
        scores : np.ndarray
            Per-residue importance scores for chain A, shape (N_a,),
            normalized to [0, 1].
        attn_matrix : np.ndarray
            Head-fused attention matrix, shape (N_a, N_b).
        """
        self.model.eval()
        self._attn_a = None
        with torch.no_grad():
            self.model(data_a, data_b)  # forward pass — hook captures attention

        if self._attn_a is None:
            raise RuntimeError(
                "AttentionRollout hook did not fire. "
                "Ensure model has at least one CrossEncoder layer."
            )

        # self._attn_a shape: (1, H, N_a, N_b)
        attn = self._attn_a.squeeze(0)   # (H, N_a, N_b)

        # Fuse heads
        if self.head_fusion == "mean":
            attn_matrix = attn.mean(dim=0).numpy()    # (N_a, N_b)
        elif self.head_fusion == "max":
            attn_matrix = attn.max(dim=0).values.numpy()
        elif self.head_fusion == "min":
            attn_matrix = attn.min(dim=0).values.numpy()
        else:
            raise ValueError(f"Unknown head_fusion: '{self.head_fusion}'. Use 'mean', 'max', or 'min'.")

        # Per-residue score: how much each chain-A residue attends to chain B overall
        scores = attn_matrix.sum(axis=1)   # (N_a,)

        # Normalize to [0, 1]
        s_min, s_max = scores.min(), scores.max()
        if s_max > s_min:
            scores = (scores - s_min) / (s_max - s_min)

        return scores, attn_matrix

    def plot_heatmap(self, scores, residue_labels=None, output_path=None, title="Attention Rollout"):
        """
        Plot per-residue attention importance as a bar chart.

        Parameters
        ----------
        scores : np.ndarray
            Per-residue importance scores, shape (N,).
        residue_labels : list, optional
            Residue labels for the x-axis (e.g., ['ALA1', 'GLY2', ...]).
        output_path : str, optional
            Path to save the figure. If None, calls plt.show().
        title : str
            Plot title.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(max(12, len(scores) // 5), 3))

            bar_colors = plt.cm.RdYlGn_r(scores)
            ax.bar(range(len(scores)), scores, color=bar_colors, width=1.0)

            ax.set_xlabel("Residue Index", fontsize=12)
            ax.set_ylabel("Attention Score (normalized)", fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.set_xlim(-0.5, len(scores) - 0.5)
            ax.set_ylim(0, 1.05)

            if residue_labels and len(residue_labels) <= 50:
                ax.set_xticks(range(len(residue_labels)))
                ax.set_xticklabels(residue_labels, rotation=90, fontsize=8)

            # Colorbar
            sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, 1))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label="Importance")

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                print(f"  Attention rollout saved to: {output_path}")
            else:
                plt.show()
            plt.close()

        except ImportError:
            print("[WARN] matplotlib not available. Cannot plot attention rollout.")

    def plot_matrix(self, attn_matrix, output_path=None, title="Cross-Attention Matrix"):
        """
        Plot the full attention weight matrix as a heatmap.

        Parameters
        ----------
        attn_matrix : np.ndarray
            Attention matrix, shape (N_a, N_b).
        output_path : str, optional
            Path to save the figure. If None, calls plt.show().
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(attn_matrix, cmap="Blues", aspect="auto")
            ax.set_xlabel("Chain B Residue Index", fontsize=12)
            ax.set_ylabel("Chain A Residue Index", fontsize=12)
            ax.set_title(title, fontsize=14)
            plt.colorbar(im, ax=ax, label="Attention Weight")
            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                print(f"  Attention matrix saved to: {output_path}")
            else:
                plt.show()
            plt.close()

        except ImportError:
            print("[WARN] matplotlib not available. Cannot plot attention matrix.")


def explain_prediction(
    model: ECABSDModel,
    data_a,
    data_b=None,
    residues_a=None,
    output_dir: str = "results",
):
    """
    Convenience function: run attention rollout and save plots.

    Parameters
    ----------
    model : ECABSDModel
        Trained ECABSD model.
    data_a : torch_geometric.data.Data
        Chain A graph.
    data_b : torch_geometric.data.Data, optional
        Chain B graph.
    residues_a : list, optional
        List of Bio.PDB residue objects for axis labels.
    output_dir : str
        Directory to save plots.

    Returns
    -------
    scores : np.ndarray, shape (N_a,)
    attn_matrix : np.ndarray, shape (N_a, N_b)
    """
    os.makedirs(output_dir, exist_ok=True)

    rollout = AttentionRollout(model)
    scores, attn_matrix = rollout.compute(data_a, data_b)

    # Build residue labels for axis ticks
    labels = None
    if residues_a:
        labels = [f"{r.get_resname()}{r.get_id()[1]}" for r in residues_a]

    rollout.plot_heatmap(
        scores,
        residue_labels=labels,
        output_path=os.path.join(output_dir, "attention_rollout.png"),
    )
    rollout.plot_matrix(
        attn_matrix,
        output_path=os.path.join(output_dir, "attention_matrix.png"),
    )
    rollout.remove_hook()

    return scores, attn_matrix
