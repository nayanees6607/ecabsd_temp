"""
BindingSiteClassifier — deep per-residue classification head.

Architecture:
    Linear(D→D) → LN → GELU → Dropout
    → Linear(D→D/2) → LN → GELU → Dropout
    → Linear(D/2→D/4) → LN → GELU → Dropout
    → Linear(D/4→64) → GELU
    → Linear(64→1)   — raw logit (no sigmoid)

Uses a skip connection from the input to the first hidden layer for
gradient flow, and an increasing compression ratio for stable training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BindingSiteClassifier(nn.Module):
    """
    Deep per-residue binding site classifier.

    Parameters
    ----------
    input_dim : feature dimension from fused representation (hidden_dim)
    dropout   : dropout probability
    """

    def __init__(self, input_dim: int = 512, dropout: float = 0.3):
        super().__init__()

        d1 = input_dim
        d2 = input_dim // 2
        d3 = input_dim // 4

        self.fc1  = nn.Linear(d1, d1)
        self.ln1  = nn.LayerNorm(d1)

        self.fc2  = nn.Linear(d1, d2)
        self.ln2  = nn.LayerNorm(d2)

        self.fc3  = nn.Linear(d2, d3)
        self.ln3  = nn.LayerNorm(d3)

        self.fc4  = nn.Linear(d3, 64)
        self.fc5  = nn.Linear(64, 1)

        self.drop = nn.Dropout(dropout)

        # Skip from input to layer-1 output (for gradient flow)
        self.skip = nn.Linear(d1, d1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1 with skip
        h1 = F.gelu(self.ln1(self.fc1(x))) + self.skip(x)
        h1 = self.drop(h1)

        # Layer 2
        h2 = F.gelu(self.ln2(self.fc2(h1)))
        h2 = self.drop(h2)

        # Layer 3
        h3 = F.gelu(self.ln3(self.fc3(h2)))
        h3 = self.drop(h3)

        # Layer 4-5
        h4 = F.gelu(self.fc4(h3))
        out = self.fc5(h4)   # raw logit — no sigmoid

        return out