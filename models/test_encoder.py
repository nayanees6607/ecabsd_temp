import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.encoder import Encoder
from models.graph_construction import build_residue_graph

# Load graph
pdb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "1AY7.pdb"))
data = build_residue_graph(pdb_path, "A")

# Initialize model
model = Encoder()

# Forward pass
output = model(data)

print("Output shape:", output.shape)