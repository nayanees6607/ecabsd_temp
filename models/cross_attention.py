"""
Stacked Cross-Attention module.

Each layer applies:
  1. Pre-LayerNorm cross-attention (chain A queries chain B)
  2. Residual connection + dropout
  3. Pre-LayerNorm feed-forward network (2× expansion)
  4. Residual connection + dropout

Multiple layers are stacked so the model can iteratively refine
the partner-aware residue representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer with pre-norm + FFN."""

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_q  = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn    = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.drop    = nn.Dropout(dropout)

        self.norm_ff = nn.LayerNorm(embed_dim)
        self.ff      = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x_a, x_b):
        # Pre-norm cross attention
        q   = self.norm_q(x_a)
        kv  = self.norm_kv(x_b)
        out, attn_w = self.attn(q, kv, kv)
        x_a = x_a + self.drop(out)

        # FFN
        x_a = x_a + self.ff(self.norm_ff(x_a))
        return x_a, attn_w


class CrossAttention(nn.Module):
    """
    Stacked cross-attention: `num_layers` layers of CrossAttentionLayer.

    Returns the output of the final layer and the attention weights
    of the first layer (for interpretability / visualisation).
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8,
                 dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor):
        """
        x_a : (1, N_a, D)
        x_b : (1, N_b, D)
        Returns enriched x_a and first-layer attention weights.
        """
        first_attn = None
        for i, layer in enumerate(self.layers):
            x_a, attn_w = layer(x_a, x_b)
            if i == 0:
                first_attn = attn_w
        return x_a, first_attn