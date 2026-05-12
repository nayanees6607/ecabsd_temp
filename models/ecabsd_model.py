"""
ECABSDModel v2 — Full upgraded pipeline.

Architecture:
    Protein A  →  [GATv2(6L) → SE3-Gated-FFN]  →  h_a ─┐
                                                          ├→ CrossAttention-FFN → GlobalPool
    Protein B  →  [GATv2(6L) → SE3-Gated-FFN]  →  h_b ─┘
                                                   ↓
                                          concat(h_a, global_ctx)
                                                   ↓
                                          BindingSiteClassifier → logits

Improvements over v1:
  - 6-layer GATv2 encoder (up from 4)
  - Gated-FFN SE3 module (up from 2-linear stub)
  - CrossAttention with pre-norm + post-FFN block
  - Global context pooling: a summary vector of the whole complex
    is broadcast back to every residue before the final classification
  - 5-layer deep classifier head with skip connections
  - 33-dim node features (up from 23)
  - 5-dim edge features with edge-type encoding (up from 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn_model      import GCNEncoder
from .se3_model      import SE3Transformer
from .cross_attention import CrossAttention
from .classifier     import BindingSiteClassifier


class ECABSDModel(nn.Module):
    """
    Full ECABSD v2 pipeline.

    Parameters
    ----------
    input_dim  : node feature dim  (33 after upgrade)
    hidden_dim : representation dim (512 recommended)
    num_heads  : cross-attention heads
    dropout    : dropout probability
    edge_dim   : edge feature dim  (5 after upgrade)
    """

    def __init__(
        self,
        input_dim:  int   = 33,
        hidden_dim: int   = 512,
        num_heads:  int   = 8,
        dropout:    float = 0.2,
        edge_dim:   int   = 5,
        num_cross_attn_layers: int = 2,
    ):
        super().__init__()

        self.gcn_encoder = GCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.se3_refine = SE3Transformer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.cross_attention = CrossAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_cross_attn_layers,
        )

        # Global context: mean-pool fused repr + project back to hidden_dim
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Norm before classifier
        self.norm_fuse = nn.LayerNorm(hidden_dim)

        # Classifier head (takes hidden_dim = fused + global)
        self.classifier = BindingSiteClassifier(
            input_dim=hidden_dim,
            dropout=dropout,
        )

    def encode_chain(self, x, edge_index, edge_attr) -> torch.Tensor:
        """GATv2 + SE3 gated-FFN refinement."""
        h = self.gcn_encoder(x, edge_index, edge_attr)
        h = self.se3_refine(h)
        return h

    def forward(self, data_a, data_b=None):
        """
        Parameters
        ----------
        data_a : Data — chain A (query chain for binding prediction)
        data_b : Data or None — chain B (interaction partner).
                 If None, chain A attends to itself.

        Returns
        -------
        logits      : (N_a, 1) — raw logits, apply sigmoid at inference
        attn_weights: (N_a, N_b) — cross-attention weights
        """
        # Encode both chains
        h_a = self.encode_chain(data_a.x, data_a.edge_index, data_a.edge_attr)  # (N_a, D)
        h_b = self.encode_chain(data_b.x, data_b.edge_index, data_b.edge_attr) \
              if data_b is not None else h_a                                      # (N_b, D)

        # Cross-attention  (batch dim required for nn.MultiheadAttention)
        h_a_seq = h_a.unsqueeze(0)   # (1, N_a, D)
        h_b_seq = h_b.unsqueeze(0)   # (1, N_b, D)

        cross_out, attn_weights = self.cross_attention(h_a_seq, h_b_seq)
        cross_out = cross_out.squeeze(0)     # (N_a, D)
        attn_weights = attn_weights.squeeze(0)  # (N_a, N_b)

        # Global context: mean-pool over all residues of fused representation
        # broadcast as a constant background signal to every residue
        global_ctx = self.global_proj(cross_out.mean(dim=0, keepdim=True))  # (1, D)
        h_fused = self.norm_fuse(cross_out + global_ctx)                     # (N_a, D)

        # Per-residue classification
        logits = self.classifier(h_fused)   # (N_a, 1)

        return logits, attn_weights

    def predict(self, data_a, data_b=None, threshold: float = 0.5):
        """Convenience inference method."""
        self.eval()
        with torch.no_grad():
            logits, attn = self.forward(data_a, data_b)
            probs  = torch.sigmoid(logits)
            labels = (probs.squeeze(-1) >= threshold).long()
        return probs, labels, attn
