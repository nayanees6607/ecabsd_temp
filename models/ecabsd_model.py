"""
ECABSD Full Model — assembles all sub-modules into one PyTorch Module.

Architecture (per chain):
  1. GCNEncoder (gcn_model.py)     — GATv2Conv stack with edge features,
                                      residual connections, and LayerNorm
  2. SE3Transformer (se3_model.py) — Gated FFN refinement with GLU-style
                                      gating and pre/post LayerNorm
  3. CrossAttention (cross_attention.py) — Stacked bidirectional CrossPPI
                                      cross-attention between both chains
  4. BindingSiteClassifier (classifier.py) — Deep 5-layer per-residue head
                                      with skip connections

Optional:
  predict_sasa : bool  — jointly predicts relative solvent accessibility as
                          a regularization signal (multi-task learning)
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import unbatch

# Import the single canonical implementation of each sub-module
from .gcn_model import GCNEncoder
from .se3_model import SE3Transformer
from .cross_attention import CrossAttention
from .classifier import BindingSiteClassifier


class SASAHead(nn.Module):
    """
    Lightweight per-residue head to predict relative solvent accessibility
    (SASA proxy). Used as a multi-task regularizer alongside the main
    binding-site classification head.

    Parameters
    ----------
    hidden_dim : int
        Input feature dimension (= hidden_dim of the encoder).
    dropout : float
        Dropout probability.
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.fc1  = nn.Linear(hidden_dim, 64)
        self.fc2  = nn.Linear(64, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Output is in (0, 1) via sigmoid — represents normalized exposure
        h1 = self.drop(F.relu(self.fc1(h)))
        return torch.sigmoid(self.fc2(h1))  # (N, 1)


class ECABSDModel(nn.Module):
    """
    ECABSD — Equivariant Cross-Attention Binding Site Detection.

    All hyperparameters fall back to config.yaml if not passed explicitly,
    so the class can be instantiated with zero arguments in Colab.

    Parameters
    ----------
    input_dim          : node feature dimension (23 structural / 983 ESM-2)
    hidden_dim         : hidden representation size (must be divisible by num_heads)
    num_heads          : cross-attention heads
    dropout            : dropout probability
    edge_dim           : edge feature dimension (4)
    num_gcn_layers     : number of GATv2Conv layers in the GCN encoder
    num_cross_attn_layers : number of stacked CrossEncoder layers
    gnn_heads          : attention heads inside each GATv2Conv layer
    predict_sasa       : whether to also predict relative solvent accessibility
    """

    def __init__(
        self,
        input_dim: int            = None,
        hidden_dim: int           = None,
        num_heads: int            = None,
        dropout: float            = None,
        edge_dim: int             = None,
        num_gcn_layers: int       = None,
        num_cross_attn_layers: int = None,
        gnn_heads: int            = None,
        predict_sasa: bool        = None,
    ):
        super().__init__()

        # ── Hyperparameter resolution (config.yaml → sensible defaults) ──
        cfg = {}
        if os.path.exists("config.yaml"):
            try:
                with open("config.yaml", "r") as f:
                    cfg = yaml.safe_load(f)
            except Exception:
                pass
        mcfg = cfg.get("model", {})

        # If ESM is enabled the graph builder produces 983-dim nodes; otherwise 23
        use_esm = mcfg.get("use_esm", False)
        _default_input = 983 if use_esm else 23

        input_dim             = input_dim             if input_dim             is not None else mcfg.get("input_dim", _default_input)
        hidden_dim            = hidden_dim            if hidden_dim            is not None else mcfg.get("hidden_dim", 128)
        num_heads             = num_heads             if num_heads             is not None else mcfg.get("num_heads", 8)
        dropout               = dropout               if dropout               is not None else mcfg.get("dropout", 0.3)
        edge_dim              = edge_dim              if edge_dim              is not None else mcfg.get("edge_feature_dim", 4)
        num_gcn_layers        = num_gcn_layers        if num_gcn_layers        is not None else mcfg.get("num_gcn_layers", 4)
        num_cross_attn_layers = num_cross_attn_layers if num_cross_attn_layers is not None else mcfg.get("num_cross_attn_layers", 1)
        gnn_heads             = gnn_heads             if gnn_heads             is not None else mcfg.get("gnn_heads", 4)
        predict_sasa          = predict_sasa          if predict_sasa          is not None else mcfg.get("predict_sasa", False)

        # Auto-correct input_dim when ESM is on but config.yaml still says 23
        if use_esm and input_dim == 23:
            input_dim = 983

        self.predict_sasa = predict_sasa

        # ── Sub-modules ──────────────────────────────────────────────────

        # GATv2Conv-based GNN encoder — uses 4D spatial edge attributes
        self.gcn_encoder = GCNEncoder(
            input_dim  = input_dim,
            hidden_dim = hidden_dim,
            edge_dim   = edge_dim,
            num_heads  = gnn_heads,
            dropout    = dropout,
            num_layers = num_gcn_layers,
        )

        # SE(3)-inspired gated FFN applied after the GNN encoder
        self.se3_refine = SE3Transformer(
            input_dim  = hidden_dim,
            hidden_dim = hidden_dim,
            dropout    = dropout,
        )

        # Per-chain LayerNorms applied before cross-attention
        self.norm_a = nn.LayerNorm(hidden_dim)
        self.norm_b = nn.LayerNorm(hidden_dim)

        # Stacked bidirectional CrossPPI cross-attention
        self.cross_attention = CrossAttention(
            embed_dim  = hidden_dim,
            num_heads  = num_heads,
            dropout    = dropout,
            num_layers = num_cross_attn_layers,
        )

        # Deep per-residue classification head
        self.classifier = BindingSiteClassifier(
            input_dim = hidden_dim,
            dropout   = dropout,
        )

        # Optional SASA prediction head for multi-task regularization
        if predict_sasa:
            self.sasa_head = SASAHead(hidden_dim=hidden_dim, dropout=dropout)

    def encode_chain(self, data) -> torch.Tensor:
        """
        Run GNN + SE3 refinement for a single chain graph.

        Parameters
        ----------
        data : torch_geometric.data.Data
            PyG graph with .x (node features) and .edge_index / .edge_attr.

        Returns
        -------
        h : torch.Tensor of shape (N, hidden_dim)
        """
        edge_attr = getattr(data, "edge_attr", None)
        h = self.gcn_encoder(data.x, data.edge_index, edge_attr)
        return self.se3_refine(h)

    def forward(self, data_a, data_b=None):
        """
        Full forward pass.

        Parameters
        ----------
        data_a : PyG Data or Batch — chain A (target, will be classified)
        data_b : PyG Data or Batch, optional — chain B (partner, context)

        Returns
        -------
        logits     : torch.Tensor (total_nodes, 1) — raw binding logits
        sasa_preds : torch.Tensor (total_nodes, 1) — SASA predictions (only if predict_sasa=True)
        attn_list  : list[torch.Tensor]            — per-sample attention maps
        """
        # Encode both chains
        h_a = self.norm_a(self.encode_chain(data_a))
        h_b = self.norm_b(self.encode_chain(data_b)) if data_b is not None else h_a

        # Batch indices (handle both batched and single-graph inputs)
        batch_a = (
            data_a.batch
            if hasattr(data_a, "batch") and data_a.batch is not None
            else torch.zeros(data_a.num_nodes, dtype=torch.long, device=data_a.x.device)
        )
        batch_b = (
            data_b.batch
            if data_b is not None and hasattr(data_b, "batch") and data_b.batch is not None
            else torch.zeros(h_b.size(0), dtype=torch.long, device=h_b.device)
        )

        # Cross-attention is applied per sample (variable sequence length)
        h_a_list   = unbatch(h_a, batch_a)
        h_b_list   = unbatch(h_b, batch_b)
        fused      = []
        attn_list  = []

        for h_a_single, h_b_single in zip(h_a_list, h_b_list):
            # CrossAttention expects (1, L, D) batched tensors
            q  = h_a_single.unsqueeze(0)   # (1, N_a, D)
            kv = h_b_single.unsqueeze(0)   # (1, N_b, D)
            q, attn = self.cross_attention(q, kv)
            fused.append(q.squeeze(0))
            attn_list.append(attn.squeeze(0))   # (H, N_a, N_b) → squeeze batch dim

        fused_tensor = torch.cat(fused, dim=0)  # (total_N_a, hidden_dim)

        logits = self.classifier(fused_tensor)   # (total_N_a, 1)

        if self.predict_sasa:
            sasa_preds = self.sasa_head(fused_tensor)   # (total_N_a, 1)
            return logits, sasa_preds, attn_list

        return logits, attn_list

    def predict(self, data_a, data_b=None, threshold: float = 0.5):
        """
        Convenience inference method.

        Returns
        -------
        probs  : torch.Tensor (N, 1) — sigmoid probabilities
        labels : torch.Tensor (N,)   — binary predictions at `threshold`
        attn   : list[torch.Tensor]  — per-sample attention maps
        """
        self.eval()
        with torch.no_grad():
            if self.predict_sasa:
                logits, _, attn = self.forward(data_a, data_b)
            else:
                logits, attn = self.forward(data_a, data_b)
            probs  = torch.sigmoid(logits)
            labels = (probs.squeeze(-1) >= threshold).long()
        return probs, labels, attn
