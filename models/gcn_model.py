"""
GCN Encoder — configurable GATv2Conv stack with residual connections.

Number of layers is controlled by num_layers parameter (default 4).
hidden_dim must be divisible by num_heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GCNEncoder(torch.nn.Module):
    """
    Multi-layer GATv2Conv encoder with GELU activations,
    LayerNorm, and residual connections from layer 2 onwards.

    Parameters
    ----------
    input_dim  : node feature dim (33)
    hidden_dim : representation dim (must be divisible by num_heads)
    edge_dim   : edge feature dim  (5)
    num_heads  : attention heads per layer
    dropout    : dropout rate
    num_layers : total GATv2 layers (default 4)
    """

    def __init__(
        self,
        input_dim: int = 33,
        hidden_dim: int = 128,
        edge_dim: int = 5,
        num_heads: int = 4,
        dropout: float = 0.5,
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
            in_dim  = input_dim  if i == 0 else hidden_dim
            # Last layer: single head, no concat → outputs hidden_dim
            if i == num_layers - 1:
                self.convs.append(
                    GATv2Conv(in_dim, hidden_dim, heads=1,
                              edge_dim=edge_dim, dropout=dropout, concat=False)
                )
            else:
                self.convs.append(
                    GATv2Conv(in_dim, head_dim, heads=num_heads,
                              edge_dim=edge_dim, dropout=dropout, concat=True)
                )
                self.norms.append(nn.LayerNorm(hidden_dim))

        self.num_layers = num_layers

    def forward(self, x, edge_index, edge_attr):
        h = x
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index, edge_attr)
            if i < self.num_layers - 1:
                h_new = F.gelu(self.norms[i](h_new))
                if i > 0:         # residual from layer 2 onwards
                    h_new = h_new + h
                h = self.drop(h_new)
            else:
                # Final layer: residual only (no norm/activation on output)
                h = h_new + h
        return h