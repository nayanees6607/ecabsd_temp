"""
BindingSiteClassifier — deep per-residue classification head.

Architecture:
    Linear(D→D) → LayerNorm → GELU → Dropout    [+ skip from input]
    → Linear(D→D/2) → LayerNorm → GELU → Dropout
    → Linear(D/2→D/4) → LayerNorm → GELU → Dropout
    → Linear(D/4→64) → GELU
    → Linear(64→1)   — raw logit (no sigmoid; apply after with BCEWithLogits or sigmoid)

Uses a skip connection from input to the first hidden layer to improve
gradient flow during early training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BindingSiteClassifier(nn.Module):
    """
    Deep per-residue binding site classifier.

    Parameters
    ----------
    input_dim : feature dimension from fused cross-attention representation (hidden_dim)
    dropout   : dropout probability
    """

    def __init__(self, input_dim: int = 128, dropout: float = 0.3):
        super().__init__()

        d1 = input_dim
        d2 = input_dim // 2
        d3 = input_dim // 4

        self.fc1 = nn.Linear(d1, d1)
        self.ln1 = nn.LayerNorm(d1)

        self.fc2 = nn.Linear(d1, d2)
        self.ln2 = nn.LayerNorm(d2)

        self.fc3 = nn.Linear(d2, d3)
        self.ln3 = nn.LayerNorm(d3)

        self.fc4 = nn.Linear(d3, 64)
        self.fc5 = nn.Linear(64, 1)

        self.drop = nn.Dropout(dropout)

        # Skip connection from input to layer-1 output for gradient flow
        self.skip = nn.Linear(d1, d1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, input_dim) — fused per-residue representations

        Returns
        -------
        logits : (N, 1) — raw binding logits (no sigmoid applied)
        """
        # Layer 1 with skip connection
        h1 = F.gelu(self.ln1(self.fc1(x))) + self.skip(x)
        h1 = self.drop(h1)

        # Layer 2
        h2 = F.gelu(self.ln2(self.fc2(h1)))
        h2 = self.drop(h2)

        # Layer 3
        h3 = F.gelu(self.ln3(self.fc3(h2)))
        h3 = self.drop(h3)

        # Layers 4 and 5 — compress to scalar logit
        h4  = F.gelu(self.fc4(h3))
        out = self.fc5(h4)

        return out