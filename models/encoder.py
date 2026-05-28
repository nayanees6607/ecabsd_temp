"""
Standalone Encoder — GCNEncoder + SE3Transformer pipeline.

This module provides a self-contained encoder that can be used independently
of the full ECABSDModel for feature extraction experiments, transfer learning,
or integration into custom pipelines.

Architecture:
  GCNEncoder (GATv2Conv, 4 layers) → SE3Transformer (Gated FFN)

Input : PyG Data with .x (N, 23) and .edge_index / .edge_attr (E, 4)
Output: Node representations of shape (N, 128)
"""

import torch
from .gcn_model import GCNEncoder
from .se3_model import SE3Transformer


class Encoder(torch.nn.Module):
    """
    Standalone GCN + SE3 encoder.

    Encodes a single protein graph (x, edge_index, edge_attr) into
    fixed-size 128-dimensional node representations.
    """

    def __init__(self):
        super().__init__()
        # GCNEncoder: 23-dim structural node features, 4-dim edge features, 128-dim output
        self.gcn = GCNEncoder(input_dim=23, hidden_dim=128, edge_dim=4)
        # SE3Transformer: 128-dim input → 128-dim output
        self.se3 = SE3Transformer(input_dim=128, hidden_dim=128)

    def forward(self, data):
        """
        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph with .x, .edge_index, and optionally .edge_attr.

        Returns
        -------
        x : torch.Tensor of shape (N, 128)
        """
        edge_attr = getattr(data, "edge_attr", None)
        x = self.gcn(data.x, data.edge_index, edge_attr)
        x = self.se3(x)
        return x