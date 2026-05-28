import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, TransformerConv
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


class GATv2Encoder(nn.Module):
    """Multi-layer GATv2 encoder supporting edge attributes and residual connections."""

    def __init__(
        self,
        input_dim: int = 23,
        hidden_dim: int = 128,
        edge_dim: int = 4,
        num_heads: int = 4,
        dropout: float = 0.3,
        num_layers: int = 4,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        head_dim = hidden_dim // num_heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            if i == num_layers - 1:
                self.convs.append(
                    GATv2Conv(in_dim, hidden_dim, heads=1, edge_dim=edge_dim, dropout=dropout, concat=False)
                )
            else:
                self.convs.append(
                    GATv2Conv(in_dim, head_dim, heads=num_heads, edge_dim=edge_dim, dropout=dropout, concat=True)
                )
                self.norms.append(nn.LayerNorm(hidden_dim))

    @property
    def conv1(self): return self.convs[0]
    @property
    def conv2(self): return self.convs[1]
    @property
    def conv3(self): return self.convs[2]
    @property
    def conv4(self): return self.convs[3]

    def forward(self, x, edge_index, edge_attr=None):
        h = x
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index, edge_attr)
            if i < self.num_layers - 1:
                h_new = F.gelu(self.norms[i](h_new))
                if i > 0:
                    h_new = h_new + h
                h = self.dropout(h_new)
            else:
                if h.size(-1) == h_new.size(-1):
                    h = h_new + h
                else:
                    h = h_new
        return h


class TransformerEncoder(nn.Module):
    """Multi-layer TransformerConv encoder supporting edge attributes and residual connections."""

    def __init__(
        self,
        input_dim: int = 23,
        hidden_dim: int = 128,
        edge_dim: int = 4,
        num_heads: int = 4,
        dropout: float = 0.3,
        num_layers: int = 4,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        head_dim = hidden_dim // num_heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            if i == num_layers - 1:
                self.convs.append(
                    TransformerConv(in_dim, hidden_dim, heads=1, edge_dim=edge_dim, dropout=dropout, concat=False)
                )
            else:
                self.convs.append(
                    TransformerConv(in_dim, head_dim, heads=num_heads, edge_dim=edge_dim, dropout=dropout, concat=True)
                )
                self.norms.append(nn.LayerNorm(hidden_dim))

    @property
    def conv1(self): return self.convs[0]
    @property
    def conv2(self): return self.convs[1]
    @property
    def conv3(self): return self.convs[2]
    @property
    def conv4(self): return self.convs[3]

    def forward(self, x, edge_index, edge_attr=None):
        h = x
        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index, edge_attr)
            if i < self.num_layers - 1:
                h_new = F.gelu(self.norms[i](h_new))
                if i > 0:
                    h_new = h_new + h
                h = self.dropout(h_new)
            else:
                if h.size(-1) == h_new.size(-1):
                    h = h_new + h
                else:
                    h = h_new
        return h


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
    """Two-layer per-residue classifier with optional SASA prediction head."""

    def __init__(self, input_dim: int = 128, dropout: float = 0.3, predict_sasa: bool = False):
        super().__init__()
        self.predict_sasa = predict_sasa
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 1)
        if predict_sasa:
            self.linear_sasa = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h1 = self.dropout(F.relu(self.linear1(h)))
        logits = self.linear2(h1)
        if self.predict_sasa:
            sasa_preds = torch.sigmoid(self.linear_sasa(h1))
            return logits, sasa_preds
        return logits


class ECABSDModel(nn.Module):
    def __init__(
        self,
        input_dim: int = None,
        hidden_dim: int = None,
        num_heads: int = None,
        dropout: float = None,
        edge_dim: int = None,
        num_gcn_layers: int = None,
        num_cross_attn_layers: int = None,
        gnn_type: str = None,
        gnn_heads: int = None,
        predict_sasa: bool = None,
    ):
        super().__init__()
        # Fallback to config.yaml if any hyperparameter is unspecified
        import yaml
        import os
        cfg = {}
        if os.path.exists("config.yaml"):
            try:
                with open("config.yaml", "r") as f:
                    cfg = yaml.safe_load(f)
            except Exception:
                pass
        mcfg = cfg.get("model", {})

        use_esm = mcfg.get("use_esm", False)
        input_dim_default = 983 if use_esm else 23
        input_dim = input_dim if input_dim is not None else mcfg.get("input_dim", input_dim_default)
        if use_esm and input_dim == 23:
            input_dim = 983
        hidden_dim = hidden_dim if hidden_dim is not None else mcfg.get("hidden_dim", 128)
        num_heads = num_heads if num_heads is not None else mcfg.get("num_heads", 8)
        dropout = dropout if dropout is not None else mcfg.get("dropout", 0.3)
        edge_dim = edge_dim if edge_dim is not None else mcfg.get("edge_feature_dim", 4)
        num_gcn_layers = num_gcn_layers if num_gcn_layers is not None else mcfg.get("num_gcn_layers", 4)
        num_cross_attn_layers = num_cross_attn_layers if num_cross_attn_layers is not None else mcfg.get("num_cross_attn_layers", 1)
        gnn_type = gnn_type if gnn_type is not None else mcfg.get("gnn_type", "gcn")
        gnn_heads = gnn_heads if gnn_heads is not None else mcfg.get("gnn_heads", 4)
        predict_sasa = predict_sasa if predict_sasa is not None else mcfg.get("predict_sasa", False)

        self.predict_sasa = predict_sasa
        self.gnn_type = gnn_type.lower()
        if self.gnn_type == "gat":
            self.gcn_encoder = GATv2Encoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                num_heads=gnn_heads,
                dropout=dropout,
                num_layers=num_gcn_layers,
            )
        elif self.gnn_type == "transformer":
            self.gcn_encoder = TransformerEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                num_heads=gnn_heads,
                dropout=dropout,
                num_layers=num_gcn_layers,
            )
        else:
            self.gcn_encoder = GCNEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

        self.se3_refine = SE3Refinement(hidden_dim=hidden_dim)
        
        self.cross_attentions = nn.ModuleList([
            CrossAttention(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_cross_attn_layers)
        ])
        
        self.norm_a = nn.LayerNorm(hidden_dim)
        self.norm_b = nn.LayerNorm(hidden_dim)
        
        self.norm_crosses = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_cross_attn_layers)
        ])
        
        self.classifier = BindingSiteClassifier(hidden_dim, dropout=dropout, predict_sasa=predict_sasa)

    @property
    def cross_attention(self):
        return self.cross_attentions[0]

    @property
    def norm_cross(self):
        return self.norm_crosses[0]

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
            q = h_a_single.unsqueeze(0)
            kv = h_b_single.unsqueeze(0)
            last_attn = None
            for layer_idx, (cross_attn, norm_cross) in enumerate(zip(self.cross_attentions, self.norm_crosses)):
                cross_out, attn = cross_attn(q, kv)
                if len(self.cross_attentions) == 1:
                    q = norm_cross(cross_out)
                else:
                    q = norm_cross(cross_out) + q
                last_attn = attn
            fused.append(q.squeeze(0))
            attn_list.append(last_attn.squeeze(0))

        fused_tensor = torch.cat(fused, dim=0)
        if self.predict_sasa:
            logits, sasa_preds = self.classifier(fused_tensor)
            return logits, sasa_preds, attn_list
        else:
            logits = self.classifier(fused_tensor)
            return logits, attn_list

    def predict(self, data_a, data_b=None, threshold: float = 0.5):
        self.eval()
        with torch.no_grad():
            if self.predict_sasa:
                logits, sasa_preds, attn = self.forward(data_a, data_b)
            else:
                logits, attn = self.forward(data_a, data_b)
            probs = torch.sigmoid(logits)
            labels = (probs.squeeze(-1) >= threshold).long()
        return probs, labels, attn
