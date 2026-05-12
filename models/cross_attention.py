"""
Cross-Attention module with:
  - Pre-norm (more stable than post-norm)
  - Dropout on attention output
  - Residual connection
  - Feed-forward refinement after attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    Bidirectional cross-attention: chain A attends to chain B.

    Improvements over the original stub:
      - Pre-LayerNorm for training stability
      - Dropout inside attention
      - Feed-forward refinement block after attention
      - Full residual connection

    Parameters
    ----------
    embed_dim : token (residue) embedding dimension
    num_heads : number of attention heads
    dropout   : dropout probability
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.norm_a  = nn.LayerNorm(embed_dim)
        self.norm_b  = nn.LayerNorm(embed_dim)
        self.attn    = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop    = nn.Dropout(dropout)

        # Post-attention FFN
        self.norm_ff = nn.LayerNorm(embed_dim)
        self.ff      = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor):
        """
        Parameters
        ----------
        x_a : (1, N_a, D) — chain A residues (query)
        x_b : (1, N_b, D) — chain B residues (key/value)

        Returns
        -------
        out         : (1, N_a, D) — enriched chain A representations
        attn_weights: (1, N_a, N_b) — attention weight matrix
        """
        # Pre-norm cross attention
        q = self.norm_a(x_a)
        k = self.norm_b(x_b)
        v = self.norm_b(x_b)

        attn_out, attn_weights = self.attn(q, k, v)
        x_a = x_a + self.drop(attn_out)   # residual

        # Post-attention FFN + residual
        x_a = x_a + self.ff(self.norm_ff(x_a))

        return x_a, attn_weights