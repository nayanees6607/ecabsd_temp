"""
SE(3)-inspired refinement block.

Replaces the original 2-linear stub with a proper 4-layer FFN with
gating, LayerNorm, GELU, residual connection, and dropout.
This acts as a learnable refinement step after the GATv2 message-passing,
giving the model a chance to re-weight residue representations before
cross-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SE3Transformer(nn.Module):
    """
    Gated FFN refinement module (SE3-inspired).

    Architecture (applied after GATv2 encoder):
        LayerNorm → Linear → GELU → Dropout
        → Linear (gate, sigmoid) ⊙ Linear (value)
        → residual → LayerNorm

    Parameters
    ----------
    input_dim  : feature dimensionality in (= hidden_dim from GCNEncoder)
    hidden_dim : inner expansion dim (default 4× input for Transformer-style FFN)
    dropout    : dropout probability
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()

        inner = hidden_dim * 2  # expansion factor

        self.norm1 = nn.LayerNorm(input_dim)

        # Value branch
        self.fc_v1 = nn.Linear(input_dim, inner)
        self.fc_v2 = nn.Linear(inner, input_dim)

        # Gate branch (GLU-style gating)
        self.fc_g  = nn.Linear(input_dim, input_dim)

        self.drop  = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm
        h = self.norm1(x)

        # Value path
        v = F.gelu(self.fc_v1(h))
        v = self.drop(self.fc_v2(v))

        # Gate
        g = torch.sigmoid(self.fc_g(h))

        # Gated residual
        out = x + g * v

        # Post-norm
        return self.norm2(out)