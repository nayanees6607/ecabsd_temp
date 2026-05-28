"""
GCN Encoder — multi-layer GATv2Conv stack with edge attributes.

Uses the GATv2Conv operator (Brody et al., 2022) which fixes the expressiveness
limitations of standard GATv1. The anisotropic attention weights are computed
from concatenated [source, target] features, making attention both dynamic and
input-dependent.

Architecture per layer:
  - Layer 0..N-2 : GATv2Conv(concat=True) → hidden_dim via (head_dim × num_heads)
                   → GELU → LayerNorm → Dropout
                   → residual added from layer 2 onwards
  - Layer N-1    : GATv2Conv(concat=False, heads=1) → hidden_dim
                   → residual added (no norm/activation on final output)

Parameters
----------
input_dim  : node feature dim (23 for structural, 983 for ESM-2-augmented)
hidden_dim : representation dim (must be divisible by num_heads)
edge_dim   : edge feature dim (4 — normalised distance + 3D unit vector)
num_heads  : attention heads per intermediate layer
dropout    : dropout rate
num_layers : total number of GATv2Conv layers (default 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GCNEncoder(torch.nn.Module):
    """
    Multi-layer GATv2Conv encoder with GELU activations,
    LayerNorm, and residual connections from layer 2 onwards.
    Supports 4-dimensional spatial edge attributes throughout.
    """

    def __init__(
        self,
        input_dim: int  = 23,
        hidden_dim: int = 128,
        edge_dim: int   = 4,
        num_heads: int  = 4,
        dropout: float  = 0.3,
        num_layers: int = 4,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        head_dim   = hidden_dim // num_heads
        self.drop  = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            # Last layer: single head, no concat → output is hidden_dim
            if i == num_layers - 1:
                self.convs.append(
                    GATv2Conv(in_dim, hidden_dim, heads=1,
                              edge_dim=edge_dim, dropout=dropout, concat=False)
                )
            else:
                # Intermediate layers: multi-head with concat, output is head_dim × num_heads = hidden_dim
                self.convs.append(
                    GATv2Conv(in_dim, head_dim, heads=num_heads,
                              edge_dim=edge_dim, dropout=dropout, concat=True)
                )
                self.norms.append(nn.LayerNorm(hidden_dim))

        self.num_layers = num_layers

        # Expose conv layers as named properties for GradCAM / hook compatibility
        # (conv1..conv4 for the default 4-layer case; always uses self.convs[i])

    @property
    def conv1(self): return self.convs[0]

    @property
    def conv2(self): return self.convs[1]

    @property
    def conv3(self): return self.convs[2]

    @property
    def conv4(self): return self.convs[3]

    def forward(self, x, edge_index, edge_attr=None):
        """
        Parameters
        ----------
        x          : (N, input_dim)  — node features
        edge_index : (2, E)          — edge connectivity
        edge_attr  : (E, edge_dim)   — spatial edge features (optional but recommended)
        """
        h = x
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index, edge_attr)
            if i < self.num_layers - 1:
                h_new = F.gelu(self.norms[i](h_new))
                if i > 0:   # residual from layer 2 onwards (shapes match from layer 1+)
                    h_new = h_new + h
                h = self.drop(h_new)
            else:
                # Final layer: plain residual (no norm/activation to preserve representation scale)
                h = h_new + h
        return h