"""
ECABSDModel — End-to-end Equivariant Cross-Attention Binding Site Detection model.

Architecture:
    Protein A  →  [GATv2 Encoder → SE3 Refinement]  →  repr_A  ─┐
                                                                   ├─→ CrossAttention → Classifier → per-residue logit
    Protein B  →  [GATv2 Encoder → SE3 Refinement]  →  repr_B  ─┘

Notes
-----
- The classifier outputs RAW LOGITS (no sigmoid).  Apply torch.sigmoid() at
  inference time and use BCEWithLogitsLoss during training.
- Edge features (distance + 3D unit vector) are now consumed by GATv2Conv.
"""

import torch
import torch.nn as nn

from .gcn_model import GCNEncoder
from .se3_model import SE3Transformer
from .cross_attention import CrossAttention
from .classifier import BindingSiteClassifier


class ECABSDModel(nn.Module):
    """
    Full ECABSD pipeline.

    Parameters
    ----------
    input_dim : int
        Node feature dimension (default 26: 20 AA + 3 SS + 3 physicochemical).
    hidden_dim : int
        Hidden representation dimension.
    num_heads : int
        Number of attention heads in cross-attention.
    dropout : float
        Dropout probability (applied inside GATv2 and classifier).
    edge_dim : int
        Edge feature dimension (default 4: distance + 3D unit vector).
    """

    def __init__(
        self,
        input_dim: int = 26,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.3,
        edge_dim: int = 4,
    ):
        super(ECABSDModel, self).__init__()

        # Shared encoder for both chains (weight sharing)
        self.gcn_encoder = GCNEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.se3_refine = SE3Transformer(input_dim=hidden_dim, hidden_dim=hidden_dim)

        # Cross-attention: chain A attends to chain B
        self.cross_attention = CrossAttention(embed_dim=hidden_dim, num_heads=num_heads)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.norm_a    = nn.LayerNorm(hidden_dim)
        self.norm_b    = nn.LayerNorm(hidden_dim)
        self.norm_cross = nn.LayerNorm(hidden_dim)

        # Per-residue binding site classifier (outputs raw logits)
        self.classifier = BindingSiteClassifier(input_dim=hidden_dim, dropout=dropout)

    def encode_chain(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a single protein chain through GATv2 + SE3 refinement."""
        h = self.gcn_encoder(x, edge_index, edge_attr)
        h = self.se3_refine(h)
        return h

    def forward(self, data_a, data_b=None):
        """
        Forward pass.

        Parameters
        ----------
        data_a : torch_geometric.data.Data
            Graph for protein chain A (the target chain for binding prediction).
        data_b : torch_geometric.data.Data or None
            Graph for protein chain B (interaction partner).
            If None, self-attention on chain A is used.

        Returns
        -------
        logits : torch.Tensor
            Per-residue binding site logits for chain A, shape (N_a, 1).
            Apply sigmoid to get probabilities.
        attn_weights : torch.Tensor
            Cross-attention weight matrix, shape (N_a, N_b).
        """
        # Encode chain A (pass edge_attr to GATv2)
        h_a = self.encode_chain(data_a.x, data_a.edge_index, data_a.edge_attr)
        h_a = self.norm_a(h_a)

        # Encode chain B (or use chain A for self-attention)
        if data_b is not None:
            h_b = self.encode_chain(data_b.x, data_b.edge_index, data_b.edge_attr)
            h_b = self.norm_b(h_b)
        else:
            h_b = h_a

        # Cross-attention: chain A attends to chain B
        # Add batch dimension for nn.MultiheadAttention: (1, N, D)
        h_a_seq = h_a.unsqueeze(0)
        h_b_seq = h_b.unsqueeze(0)

        cross_out, attn_weights = self.cross_attention(h_a_seq, h_b_seq)
        cross_out = cross_out.squeeze(0)  # (N_a, D)

        # Residual connection + norm
        h_fused = self.norm_cross(h_a + self.dropout(cross_out))

        # Per-residue classification → raw logits
        logits = self.classifier(h_fused)  # (N_a, 1)

        # Squeeze attention weights
        attn_weights = attn_weights.squeeze(0)  # (N_a, N_b)

        return logits, attn_weights

    def predict(self, data_a, data_b=None, threshold: float = 0.5):
        """
        Convenience inference method: returns binary predictions + probabilities.

        Returns
        -------
        probs  : torch.Tensor  — (N_a, 1) probabilities after sigmoid
        labels : torch.Tensor  — (N_a,) binary labels
        attn   : torch.Tensor  — attention weights
        """
        self.eval()
        with torch.no_grad():
            logits, attn = self.forward(data_a, data_b)
            probs  = torch.sigmoid(logits)
            labels = (probs.squeeze(-1) >= threshold).long()
        return probs, labels, attn
