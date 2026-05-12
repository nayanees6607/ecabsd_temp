"""
GCN Encoder — 6-layer GATv2Conv stack with residual connections,
dropout, and batch normalisation.

Accepts the enriched 5-dim edge features from graph_construction.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GCNEncoder(torch.nn.Module):
    """
    6-layer GATv2Conv encoder with:
      - GELU activations
      - LayerNorm after each layer
      - Residual connections from layer 2 onwards
      - Edge features consumed by every layer

    Parameters
    ----------
    input_dim  : node feature dim (33 after upgrade)
    hidden_dim : representation dim (must be divisible by num_heads)
    edge_dim   : edge feature dim  (5 after upgrade)
    num_heads  : attention heads per intermediate layer
    dropout    : dropout rate
    """

    def __init__(
        self,
        input_dim: int = 33,
        hidden_dim: int = 512,
        edge_dim: int = 5,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()

        head_dim = hidden_dim // num_heads

        # Input projection: input_dim → hidden_dim
        self.conv1 = GATv2Conv(
            input_dim, head_dim, heads=num_heads,
            edge_dim=edge_dim, dropout=dropout, concat=True,
        )

        # Layers 2-5: hidden_dim → hidden_dim (with residuals)
        self.conv2 = GATv2Conv(
            hidden_dim, head_dim, heads=num_heads,
            edge_dim=edge_dim, dropout=dropout, concat=True,
        )
        self.conv3 = GATv2Conv(
            hidden_dim, head_dim, heads=num_heads,
            edge_dim=edge_dim, dropout=dropout, concat=True,
        )
        self.conv4 = GATv2Conv(
            hidden_dim, head_dim, heads=num_heads,
            edge_dim=edge_dim, dropout=dropout, concat=True,
        )
        self.conv5 = GATv2Conv(
            hidden_dim, head_dim, heads=num_heads,
            edge_dim=edge_dim, dropout=dropout, concat=True,
        )

        # Final layer: single head, no concat
        self.conv6 = GATv2Conv(
            hidden_dim, hidden_dim, heads=1,
            edge_dim=edge_dim, dropout=dropout, concat=False,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)
        self.norm5 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        h1 = F.gelu(self.norm1(self.conv1(x, edge_index, edge_attr)))

        h2 = F.gelu(self.norm2(self.conv2(h1, edge_index, edge_attr))) + h1

        h3 = F.gelu(self.norm3(self.conv3(h2, edge_index, edge_attr))) + h2

        h4 = F.gelu(self.norm4(self.conv4(h3, edge_index, edge_attr))) + h3

        h5 = F.gelu(self.norm5(self.conv5(h4, edge_index, edge_attr))) + h4

        h6 = self.conv6(h5, edge_index, edge_attr) + h5

        return self.dropout(h6)