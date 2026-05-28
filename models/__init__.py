"""
ECABSD Models Package.

Exports the canonical implementation of each sub-module.
All modules are defined in their own file and imported here for convenience:

  ECABSDModel          — full model (ecabsd_model.py)
  GCNEncoder           — GATv2Conv encoder (gcn_model.py)
  SE3Transformer       — Gated FFN refinement (se3_model.py)
  CrossAttention       — Bidirectional CrossPPI cross-attention (cross_attention.py)
  BindingSiteClassifier — Deep per-residue classification head (classifier.py)
  Encoder              — Standalone GCN+SE3 encoder for experimentation (encoder.py)
  build_residue_graph  — Graph construction from PDB (graph_construction.py)
"""

from .ecabsd_model import ECABSDModel
from .gcn_model import GCNEncoder
from .se3_model import SE3Transformer
from .cross_attention import CrossAttention
from .classifier import BindingSiteClassifier
from .encoder import Encoder
from .graph_construction import build_residue_graph

__all__ = [
    "ECABSDModel",
    "GCNEncoder",
    "SE3Transformer",
    "CrossAttention",
    "BindingSiteClassifier",
    "Encoder",
    "build_residue_graph",
]
