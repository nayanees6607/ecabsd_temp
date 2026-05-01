import torch
import torch.nn as nn


class BindingSiteClassifier(nn.Module):
    """
    Deep per-residue binding site classifier.

    Outputs raw logits (no sigmoid) — pair with BCEWithLogitsLoss during
    training and apply torch.sigmoid() at inference time.

    Architecture: Linear → LN → GELU → Dropout → Linear → LN → GELU → Dropout
                  → Linear → GELU → Linear (logit)
    """

    def __init__(self, input_dim: int = 128, dropout: float = 0.3):
        super(BindingSiteClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            # No sigmoid — use BCEWithLogitsLoss during training
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # raw logits, shape (N, 1)