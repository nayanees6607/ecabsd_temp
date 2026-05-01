import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GCNEncoder(torch.nn.Module):
    """
    Graph encoder using GATv2Conv layers that explicitly use edge features.

    Replaces the original GCNConv stack (which silently ignored edge_attr).
    Uses multi-head attention with 8 heads per layer and residual connections
    to prevent over-smoothing across 4 message-passing steps.

    Parameters
    ----------
    input_dim : int
        Node feature dimension.
    hidden_dim : int
        Hidden representation dimension (must be divisible by num_heads).
    edge_dim : int
        Edge feature dimension (distance + 3D unit vector = 4).
    num_heads : int
        Number of attention heads for intermediate layers.
    dropout : float
        Dropout rate inside attention layers.
    """

    def __init__(
        self,
        input_dim: int = 26,
        hidden_dim: int = 256,
        edge_dim: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super(GCNEncoder, self).__init__()

        head_dim = hidden_dim // num_heads  # features per head

        # Layer 1: input_dim → hidden_dim
        self.conv1 = GATv2Conv(
            input_dim, head_dim, heads=num_heads, edge_dim=edge_dim,
            dropout=dropout, concat=True
        )
        # Layers 2-3: hidden_dim → hidden_dim (with residual)
        self.conv2 = GATv2Conv(
            hidden_dim, head_dim, heads=num_heads, edge_dim=edge_dim,
            dropout=dropout, concat=True
        )
        self.conv3 = GATv2Conv(
            hidden_dim, head_dim, heads=num_heads, edge_dim=edge_dim,
            dropout=dropout, concat=True
        )
        # Layer 4: hidden_dim → hidden_dim (single head, no concat)
        self.conv4 = GATv2Conv(
            hidden_dim, hidden_dim, heads=1, edge_dim=edge_dim,
            dropout=dropout, concat=False
        )

        # LayerNorm after each intermediate layer
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        # Layer 1
        h1 = F.gelu(self.norm1(self.conv1(x, edge_index, edge_attr)))

        # Layer 2 + residual
        h2 = F.gelu(self.norm2(self.conv2(h1, edge_index, edge_attr))) + h1

        # Layer 3 + residual
        h3 = F.gelu(self.norm3(self.conv3(h2, edge_index, edge_attr))) + h2

        # Layer 4 + residual
        h4 = self.conv4(h3, edge_index, edge_attr) + h3

        return h4