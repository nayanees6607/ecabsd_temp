"""
Test — Standalone Encoder pipeline (GCNEncoder + SE3Transformer).

Verifies that the Encoder can process a real PDB graph and produce
representations of the correct shape.
"""

import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.encoder import Encoder
from models.graph_construction import build_residue_graph

PDB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "1AY7.pdb"))


def test_encoder_output_shape():
    """Encoder output must be (N, 128) for any valid chain."""
    data = build_residue_graph(PDB_PATH, "A")
    model = Encoder()
    model.eval()
    with torch.no_grad():
        output = model(data)
    assert output.shape == (data.num_nodes, 128), \
        f"Expected ({data.num_nodes}, 128), got {output.shape}"


def test_encoder_output_is_float():
    """Encoder output must be a float32 tensor."""
    data = build_residue_graph(PDB_PATH, "A")
    model = Encoder()
    model.eval()
    with torch.no_grad():
        output = model(data)
    assert output.dtype == torch.float32
