"""
ECABSD Grad-CAM — Gradient-weighted Class Activation Mapping for GNNs.

Implements Grad-CAM adapted for graph neural networks to produce
per-residue saliency maps explaining binding site predictions.

Reference: Selvaraju et al. (2017) Grad-CAM, adapted for GNNs.

Usage:
    from explainability.gradcam import GradCAM
    gradcam = GradCAM(model)
    saliency = gradcam.compute(data_a)
    gradcam.plot(saliency, residue_ids)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ecabsd_model import ECABSDModel


class GradCAM:
    """
    Grad-CAM for ECABSD GCN layers.

    Hooks into the last GCN layer to capture:
    - Forward activations (feature maps)
    - Backward gradients (from binding probability w.r.t. activations)

    The per-residue saliency score is:
        saliency_i = ReLU( sum_k( alpha_k * A_ik ) )
    where alpha_k = (1/N) * sum_i( dY/dA_ik ) (global average pooling of grads)

    Parameters
    ----------
    model : ECABSDModel
        Trained ECABSD model.
    target_layer : str
        Which GCN layer to target. One of 'conv1', 'conv2', 'conv3', 'conv4'.
    """

    def __init__(self, model: ECABSDModel, target_layer: str = "conv4"):
        self.model = model
        self.target_layer = target_layer

        self._activations_list = []
        self._gradients_list = []

        # Select target layer
        gcn = self.model.gcn_encoder
        layer = getattr(gcn, target_layer, None)
        if layer is None:
            raise ValueError(
                f"Layer '{target_layer}' not found in GCNEncoder. "
                f"Available: conv1, conv2, conv3, conv4"
            )

        # Register hooks
        self._fwd_hook = layer.register_forward_hook(self._fwd_hook_fn)
        self._bwd_hook = layer.register_full_backward_hook(self._bwd_hook_fn)

    def _fwd_hook_fn(self, module, input, output):
        """Capture forward activations."""
        self._activations_list.append(output.detach())

    def _bwd_hook_fn(self, module, grad_input, grad_output):
        """Capture backward gradients."""
        self._gradients_list.append(grad_output[0].detach())

    def remove_hooks(self):
        """Remove all hooks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def compute(self, data_a, data_b=None, target_residue_idx=None):
        """
        Compute per-residue Grad-CAM saliency scores.

        Parameters
        ----------
        data_a : torch_geometric.data.Data
            Chain A graph.
        data_b : torch_geometric.data.Data, optional
            Chain B graph.
        target_residue_idx : int, optional
            Specific residue index to compute saliency for.
            If None, uses the mean binding probability across all residues.

        Returns
        -------
        saliency : np.ndarray
            Per-residue saliency scores, shape (N_a,), normalized to [0, 1].
        """
        self.model.eval()
        
        # Reset lists for this run
        self._activations_list = []
        self._gradients_list = []

        # Requires gradient computation
        data_a.x.requires_grad_(True)

        # Forward pass
        pred, _ = self.model(data_a, data_b)
        pred = pred.squeeze(-1)  # (N_a,)

        # Backprop target
        if target_residue_idx is not None:
            score = pred[target_residue_idx]
        else:
            # Mean binding probability (global explanation)
            score = pred.mean()

        self.model.zero_grad()
        score.backward()

        # Activations: the first forward call corresponds to chain A
        activations = self._activations_list[0].cpu().numpy()
        # Gradients: the last backward call corresponds to chain A
        gradients = self._gradients_list[-1].cpu().numpy()

        # Global average pooling of gradients → importance weights per channel
        alpha = gradients.mean(axis=0)  # (hidden_dim,)

        # Weighted combination of activations
        saliency = np.dot(activations, alpha)  # (N,)

        # Apply ReLU (we only care about features that increase the score)
        saliency = np.maximum(saliency, 0)

        # Normalize to [0, 1]
        if saliency.max() > saliency.min():
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

        return saliency

    def plot(
        self,
        saliency,
        residue_labels=None,
        output_path=None,
        title="Grad-CAM Saliency",
    ):
        """
        Plot per-residue saliency scores.

        Parameters
        ----------
        saliency : np.ndarray
            Per-residue saliency scores.
        residue_labels : list, optional
            Labels for x-axis.
        output_path : str, optional
            Path to save the plot.
        title : str
            Plot title.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(max(14, len(saliency) // 5), 6))

            # Bar chart
            ax1 = axes[0]
            colors = plt.cm.hot_r(saliency)
            ax1.bar(range(len(saliency)), saliency, color=colors, width=1.0)
            ax1.set_xlabel("Residue Index", fontsize=11)
            ax1.set_ylabel("Grad-CAM Score", fontsize=11)
            ax1.set_title(title, fontsize=13)
            ax1.set_xlim(-0.5, len(saliency) - 0.5)
            ax1.set_ylim(0, 1.05)

            if residue_labels and len(residue_labels) <= 50:
                ax1.set_xticks(range(len(residue_labels)))
                ax1.set_xticklabels(residue_labels, rotation=90, fontsize=7)

            # 1D heatmap strip
            ax2 = axes[1]
            ax2.imshow(
                saliency.reshape(1, -1),
                cmap="hot_r",
                aspect="auto",
                vmin=0,
                vmax=1,
            )
            ax2.set_xlabel("Residue Index", fontsize=11)
            ax2.set_yticks([])
            ax2.set_title("Saliency Heatmap Strip", fontsize=11)

            # Colorbar
            sm = plt.cm.ScalarMappable(cmap="hot_r", norm=plt.Normalize(0, 1))
            sm.set_array([])
            plt.colorbar(sm, ax=axes, label="Saliency Score", fraction=0.02)

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                print(f"  Grad-CAM saved to: {output_path}")
            else:
                plt.show()
            plt.close()

        except ImportError:
            print("[WARN] matplotlib not available. Cannot plot Grad-CAM.")


def explain_with_gradcam(
    model: ECABSDModel,
    data_a,
    data_b=None,
    residues_a=None,
    output_dir: str = "results",
    target_layer: str = "conv4",
):
    """
    Convenience function: run Grad-CAM and save plots.

    Returns
    -------
    saliency : np.ndarray
        Per-residue saliency scores.
    """
    os.makedirs(output_dir, exist_ok=True)

    gradcam = GradCAM(model, target_layer=target_layer)
    saliency = gradcam.compute(data_a, data_b)

    labels = None
    if residues_a:
        labels = [f"{r.get_resname()}{r.get_id()[1]}" for r in residues_a]

    gradcam.plot(
        saliency,
        residue_labels=labels,
        output_path=os.path.join(output_dir, "gradcam_saliency.png"),
        title=f"Grad-CAM ({target_layer}) — Binding Site Saliency",
    )
    gradcam.remove_hooks()

    return saliency
