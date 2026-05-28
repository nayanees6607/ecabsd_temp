"""
CrossAttention module — adapted from CrossPPI (drugparadigm/CrossPPI).

Key design: BIDIRECTIONAL CrossFusion.
Chain A queries chain B AND chain B simultaneously queries chain A.
Both chains enrich each other before classification, exactly as in CrossPPI.

Architecture per CrossEncoder layer:
  CrossFusion   : receptor queries ligand → context_receptor
                  ligand queries receptor → context_ligand
  SelfOutput    : dense + dropout + LayerNorm + residual (separate norms per chain)
  Intermediate  : dense + GELU (for both chains)
  CrossPPIOutput: dense + dropout + LayerNorm + residual (separate norms per chain)

Multiple CrossEncoder layers are stacked (controlled by num_layers, default 4).
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Custom LayerNorm with learned gamma/beta parameters."""

    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta  = nn.Parameter(torch.zeros(hidden_size))
        self.eps   = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var  = (x - mean).pow(2).mean(-1, keepdim=True)
        x    = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x + self.beta


class CrossFusion(nn.Module):
    """
    Bidirectional cross-attention.
    Receptor queries Ligand; Ligand queries Receptor simultaneously.
    Directly adapted from CrossPPI CrossFusion class.
    """

    def __init__(self, hidden_size, num_heads, attn_dropout):
        super().__init__()
        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        self.num_heads     = num_heads
        self.head_size     = hidden_size // num_heads
        self.all_head_size = hidden_size

        # Separate Q/K/V projections for each chain to avoid weight sharing
        self.q_a = nn.Linear(hidden_size, hidden_size)
        self.k_a = nn.Linear(hidden_size, hidden_size)
        self.v_a = nn.Linear(hidden_size, hidden_size)

        self.q_b = nn.Linear(hidden_size, hidden_size)
        self.k_b = nn.Linear(hidden_size, hidden_size)
        self.v_b = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(attn_dropout)

    def _split_heads(self, x):
        """(B, L, D) → (B, H, L, d_h)"""
        B, L, _ = x.shape
        x = x.view(B, L, self.num_heads, self.head_size)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """(B, H, L, d_h) → (B, L, D)"""
        x = x.permute(0, 2, 1, 3).contiguous()
        B, L, _, _ = x.shape
        return x.view(B, L, self.all_head_size)

    def forward(self, h_a, h_b):
        """
        h_a : (1, N_a, D) — chain A (receptor)
        h_b : (1, N_b, D) — chain B (ligand)
        Returns enriched (ctx_a, ctx_b) and (attn_a, attn_b) weight tensors.
        """
        scale = math.sqrt(self.head_size)

        # Chain A queries Chain B
        q_a    = self._split_heads(self.q_a(h_a))
        k_b    = self._split_heads(self.k_b(h_b))
        v_b    = self._split_heads(self.v_b(h_b))
        attn_a = torch.softmax(torch.matmul(q_a, k_b.transpose(-1, -2)) / scale, dim=-1)
        attn_a = self.dropout(attn_a)
        ctx_a  = self._merge_heads(torch.matmul(attn_a, v_b))   # (1, N_a, D)

        # Chain B queries Chain A
        q_b    = self._split_heads(self.q_b(h_b))
        k_a    = self._split_heads(self.k_a(h_a))
        v_a    = self._split_heads(self.v_a(h_a))
        attn_b = torch.softmax(torch.matmul(q_b, k_a.transpose(-1, -2)) / scale, dim=-1)
        attn_b = self.dropout(attn_b)
        ctx_b  = self._merge_heads(torch.matmul(attn_b, v_a))   # (1, N_b, D)

        return (ctx_a, ctx_b), (attn_a, attn_b)


class SelfOutput(nn.Module):
    """
    Dense projection + dropout + LayerNorm + residual (both chains).
    Uses SEPARATE LayerNorm instances for chain A and chain B to allow
    independent learned scale/shift parameters per chain.
    """

    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.dense_a = nn.Linear(hidden_size, hidden_size)
        self.dense_b = nn.Linear(hidden_size, hidden_size)
        # Separate norms — chain A and B have different distribution characteristics
        self.norm_a  = LayerNorm(hidden_size)
        self.norm_b  = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ctx, residual):
        ctx_a = self.norm_a(self.dropout(self.dense_a(ctx[0])) + residual[0])
        ctx_b = self.norm_b(self.dropout(self.dense_b(ctx[1])) + residual[1])
        return ctx_a, ctx_b


class Intermediate(nn.Module):
    """
    Position-wise FFN after attention (GELU activation, both chains).
    Expands to 2× hidden size then projects back (similar to Transformer FFN).
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.dense_a = nn.Linear(hidden_size, hidden_size * 2)
        self.dense_b = nn.Linear(hidden_size, hidden_size * 2)
        self.proj_a  = nn.Linear(hidden_size * 2, hidden_size)
        self.proj_b  = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, h_a, h_b):
        h_a = self.proj_a(F.gelu(self.dense_a(h_a)))
        h_b = self.proj_b(F.gelu(self.dense_b(h_b)))
        return h_a, h_b


class CrossPPIOutput(nn.Module):
    """
    Final projection + dropout + LayerNorm + residual (both chains).
    Uses SEPARATE LayerNorm instances for chain A and chain B.
    """

    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.dense_a = nn.Linear(hidden_size, hidden_size)
        self.dense_b = nn.Linear(hidden_size, hidden_size)
        # Separate norms — chain A and B have different distribution characteristics
        self.norm_a  = LayerNorm(hidden_size)
        self.norm_b  = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_a, h_b, residual_a, residual_b):
        h_a = self.norm_a(self.dropout(self.dense_a(h_a)) + residual_a)
        h_b = self.norm_b(self.dropout(self.dense_b(h_b)) + residual_b)
        return h_a, h_b


class CrossEncoder(nn.Module):
    """
    One full CrossPPI encoder layer:
      CrossFusion → SelfOutput → Intermediate → CrossPPIOutput
    """

    def __init__(self, hidden_size, num_heads, attn_dropout, hidden_dropout):
        super().__init__()
        self.fusion = CrossFusion(hidden_size, num_heads, attn_dropout)
        self.out1   = SelfOutput(hidden_size, hidden_dropout)
        self.inter  = Intermediate(hidden_size)
        self.out2   = CrossPPIOutput(hidden_size, hidden_dropout)

    def forward(self, h_a, h_b):
        (ctx_a, ctx_b), attn = self.fusion(h_a, h_b)
        h_a2, h_b2 = self.out1((ctx_a, ctx_b), (h_a, h_b))
        h_a3, h_b3 = self.inter(h_a2, h_b2)
        h_a4, h_b4 = self.out2(h_a3, h_b3, h_a2, h_b2)
        return h_a4, h_b4, attn


class CrossAttention(nn.Module):
    """
    Stacked CrossPPI-style bidirectional cross-attention.

    Parameters
    ----------
    embed_dim  : token embedding dimension (= hidden_dim from the GCN encoder)
    num_heads  : number of attention heads
    dropout    : dropout probability
    num_layers : number of stacked CrossEncoder layers (default 4, same as CrossPPI)
    """

    def __init__(self, embed_dim=128, num_heads=8, dropout=0.1, num_layers=4):
        super().__init__()
        layer = CrossEncoder(embed_dim, num_heads, dropout, dropout)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor):
        """
        x_a : (1, N_a, D)
        x_b : (1, N_b, D)

        Returns:
          x_a    : (1, N_a, D)      — enriched chain A representations
          attn_a : (1, H, N_a, N_b) — chain A's attention over chain B from first layer
        """
        first_attn = None
        for i, layer in enumerate(self.layers):
            x_a, x_b, (attn_a, _) = layer(x_a, x_b)
            if i == 0:
                # Store first-layer attention (cleaner than last-layer for visualisation)
                first_attn = attn_a
        return x_a, first_attn