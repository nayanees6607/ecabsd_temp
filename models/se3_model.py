"""
SE(3)-inspired refinement block (Gated FFN).

Replaces the original 2-linear stub with a proper gated FFN with
LayerNorm, GELU, GLU-style gating, residual connection, and dropout.
This acts as a learnable refinement step applied after the GATv2
message-passing and before cross-attention, giving the model a chance
to re-weight residue representations.

Architecture:
    pre-norm (LayerNorm)
    → value path  : Linear → GELU → Dropout → Linear
    → gate        : Linear → Sigmoid
    → gated residual: x + gate ⊙ value
    → post-norm (LayerNorm)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SE3Transformer(nn.Module):
    """
    Gated FFN refinement module (SE3-inspired).

    Parameters
    ----------
    input_dim  : feature dimensionality in/out (= hidden_dim from GCNEncoder)
    hidden_dim : inner expansion dim for the value path (default = input_dim,
                 effectively 2× because inner = hidden_dim * 2 inside)
    dropout    : dropout probability
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()

        inner = hidden_dim * 2   # Transformer-FFN style 2× expansion

        self.norm1 = nn.LayerNorm(input_dim)

        # Value branch: expand → GELU → drop → project back
        self.fc_v1 = nn.Linear(input_dim, inner)
        self.fc_v2 = nn.Linear(inner, input_dim)

        # GLU-style gate: sigmoid-activated scalar gate per feature
        self.fc_g  = nn.Linear(input_dim, input_dim)

        self.drop  = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, input_dim) — node representations from GCNEncoder

        Returns
        -------
        out : (N, input_dim) — refined representations (same shape as input)
        """
        # Pre-norm for stable gradient flow
        h = self.norm1(x)

        # Value path
        v = F.gelu(self.fc_v1(h))
        v = self.drop(self.fc_v2(v))

        # Gate: sigmoid squashes to (0, 1), controlling how much value to add
        g = torch.sigmoid(self.fc_g(h))

        # Gated residual connection
        out = x + g * v

        # Post-norm before passing to cross-attention
        return self.norm2(out)