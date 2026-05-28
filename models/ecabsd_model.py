"""
ECABSD model architecture matching the committed `checkpoints/best_model.pt`.

The repository includes 23-dimensional processed residue graphs and a checkpoint
trained with this compact GCN + cross-attention model. Keep this file aligned
with that checkpoint so `predict.py` and `evaluate.py` work out of the box.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import unbatch


class GCNEncoder(nn.Module):
    """Four-layer GCN encoder used by the committed checkpoint."""

    def __init__(self, input_dim: int = 23, hidden_dim: int = 128, dropout: float = 0.3, **_):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, edge_index))
        h = self.dropout(h)
        h = F.relu(self.conv3(h, edge_index))
        h = self.dropout(h)
        return F.relu(self.conv4(h, edge_index))


class SE3Refinement(nn.Module):
    """Lightweight refinement block stored in the checkpoint."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h):
        return self.linear2(F.relu(self.linear1(h)))


class CrossAttention(nn.Module):
    """Single target-to-partner multi-head attention layer."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, query, key_val):
        return self.attention(query, key_val, key_val)


class BindingSiteClassifier(nn.Module):
    """Two-layer per-residue classifier used by `best_model.pt`."""

    def __init__(self, input_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        return self.linear2(self.dropout(F.relu(self.linear1(h))))


class ECABSDModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 23,
        hidden_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.3,
        edge_dim: int = 4,
        num_gcn_layers: int = 4,
        num_cross_attn_layers: int = 1,
    ):
        super().__init__()
        self.gcn_encoder = GCNEncoder(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.se3_refine = SE3Refinement(hidden_dim=hidden_dim)
        self.cross_attention = CrossAttention(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.norm_a = nn.LayerNorm(hidden_dim)
        self.norm_b = nn.LayerNorm(hidden_dim)
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.classifier = BindingSiteClassifier(hidden_dim, dropout=dropout)

    def encode_chain(self, data):
        h = self.gcn_encoder(data.x, data.edge_index, getattr(data, "edge_attr", None))
        return self.se3_refine(h)

    def forward(self, data_a, data_b=None):
        h_a = self.norm_a(self.encode_chain(data_a))
        h_b = self.norm_b(self.encode_chain(data_b)) if data_b is not None else h_a

        batch_a = (
            data_a.batch if hasattr(data_a, "batch") and data_a.batch is not None
            else torch.zeros(data_a.num_nodes, dtype=torch.long, device=data_a.x.device)
        )
        batch_b = (
            data_b.batch if data_b is not None and hasattr(data_b, "batch") and data_b.batch is not None
            else torch.zeros(h_b.size(0), dtype=torch.long, device=h_b.device)
        )

        h_a_list = unbatch(h_a, batch_a)
        h_b_list = unbatch(h_b, batch_b)
        fused = []
        attn_list = []

        for h_a_single, h_b_single in zip(h_a_list, h_b_list):
            cross_out, attn = self.cross_attention(
                h_a_single.unsqueeze(0),
                h_b_single.unsqueeze(0),
            )
            fused.append(self.norm_cross(cross_out.squeeze(0)))
            attn_list.append(attn.squeeze(0))

        logits = self.classifier(torch.cat(fused, dim=0))
        return logits, attn_list

    def predict(self, data_a, data_b=None, threshold: float = 0.5):
        self.eval()
        with torch.no_grad():
            logits, attn = self.forward(data_a, data_b)
            probs = torch.sigmoid(logits)
            labels = (probs.squeeze(-1) >= threshold).long()
        return probs, labels, attn
