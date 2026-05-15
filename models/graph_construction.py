"""
Graph construction for ECABSD.

Node features (33-dim):
    0-19  : amino acid one-hot (20)
    20-22 : secondary structure one-hot — helix/sheet/coil (3)
    23    : Kyte-Doolittle hydrophobicity (normalised)
    24    : formal charge at pH 7
    25    : relative sequence position 0→1
    26    : relative solvent accessibility proxy (neighbour count, normalised)
    27    : B-factor (normalised by chain mean)
    28    : sin(2π·i/L) — sinusoidal positional encoding
    29    : cos(2π·i/L) — sinusoidal positional encoding
    30    : is N-terminal region (first 10% of chain)
    31    : is C-terminal region (last 10% of chain)
    32    : side-chain size proxy (normalised MW)

Edge features (5-dim):
    0   : Euclidean Cα–Cα distance (Å, normalised by cutoff)
    1-3 : unit direction vector
    4   : edge type (0 = spatial, 1 = sequential i±1, 2 = sequential i±2)
"""

import math
import numpy as np
import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa, three_to_one
import pydssp

# Local imports
try:
    from models.embedding import get_esm_embedding
except ImportError:
    from .embedding import get_esm_embedding

# ── Amino acid lookup ────────────────────────────────────────────────────────
STANDARD_AA = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL'
]
AA_TO_IDX = {aa: i for i, aa in enumerate(STANDARD_AA)}

SS_MAPPING = {
    'H': [1, 0, 0], 'G': [1, 0, 0], 'I': [1, 0, 0],
    'E': [0, 1, 0], 'B': [0, 1, 0],
    '-': [0, 0, 1], 'S': [0, 0, 1], 'T': [0, 0, 1],
}

# Kyte-Doolittle hydrophobicity (normalised to [-1, 1])
_HYDRO_RAW = {
    'ALA': 1.8,  'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8,  'LYS': -3.9, 'MET': 1.9,  'PHE': 2.8,  'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2,
}
HYDROPHOBICITY = {aa: v / 4.5 for aa, v in _HYDRO_RAW.items()}

# Formal charge at pH 7
CHARGE = {'ARG': 1.0, 'LYS': 1.0, 'ASP': -1.0, 'GLU': -1.0}

# Side-chain molecular weight proxy (normalised by TRP=186)
_MW = {
    'ALA': 15,  'ARG': 100, 'ASN': 58,  'ASP': 59,  'CYS': 47,
    'GLN': 72,  'GLU': 73,  'GLY': 1,   'HIS': 82,  'ILE': 57,
    'LEU': 57,  'LYS': 72,  'MET': 75,  'PHE': 91,  'PRO': 42,
    'SER': 31,  'THR': 45,  'TRP': 130, 'TYR': 107, 'VAL': 43,
}
SC_MW = {aa: v / 130.0 for aa, v in _MW.items()}

GRAPH_CUTOFF = 10.0  # Å — increased from 8 for better long-range context


def get_residues(chain):
    residues, skipped = [], 0
    for r in chain:
        if is_aa(r, standard=True):
            residues.append(r)
        else:
            skipped += 1
    return residues, skipped


def get_backbone_coords(residues):
    """Extract N, CA, C, O coords for pydssp — shape (L, 4, 3)."""
    coords = []
    for r in residues:
        try:
            n  = r['N'].get_vector().get_array()
            ca = r['CA'].get_vector().get_array()
            c  = r['C'].get_vector().get_array()
            o  = r['O'].get_vector().get_array()
            coords.append([n, ca, c, o])
        except KeyError:
            coords.append([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    return np.array(coords, dtype=np.float32)


def compute_rsa_proxy(ca_coords, cutoff=8.0):
    """
    Proxy for relative solvent accessibility:
    fewer neighbours within `cutoff` Å ↔ more exposed.
    Returns normalised exposure score in [0, 1].
    """
    n = len(ca_coords)
    counts = np.zeros(n)
    for i in range(n):
        diff = ca_coords - ca_coords[i]
        dists = np.linalg.norm(diff, axis=1)
        counts[i] = np.sum(dists < cutoff) - 1  # exclude self
    # More buried → higher count; invert and normalise
    max_c = counts.max() if counts.max() > 0 else 1
    return 1.0 - (counts / max_c)  # 1 = most exposed


def get_node_features(residues, ss_labels, ca_coords) -> torch.Tensor:
    """Build 33-dim per-residue node feature vectors."""
    n = len(residues)
    rsa = compute_rsa_proxy(ca_coords)

    # B-factor: normalise within chain
    bfactors = []
    for r in residues:
        try:
            bfactors.append(r['CA'].get_bfactor())
        except KeyError:
            bfactors.append(0.0)
    bfactors = np.array(bfactors, dtype=np.float32)
    bf_mean = bfactors.mean() if bfactors.mean() != 0 else 1.0
    bf_norm = np.clip(bfactors / (bf_mean + 1e-8), 0, 3) / 3.0  # [0, 1]

    features = []
    for i, r in enumerate(residues):
        resname = r.get_resname()

        # 20-dim amino acid one-hot
        one_hot = [0.0] * 20
        if resname in AA_TO_IDX:
            one_hot[AA_TO_IDX[resname]] = 1.0

        # 3-dim secondary structure
        ss = SS_MAPPING.get(str(ss_labels[i]), [0, 0, 1])

        # Scalar features
        hydro    = HYDROPHOBICITY.get(resname, 0.0)
        charge   = float(CHARGE.get(resname, 0.0))
        rel_pos  = i / max(n - 1, 1)
        rsa_val  = float(rsa[i])
        bf_val   = float(bf_norm[i])
        sin_pos  = math.sin(2 * math.pi * i / max(n, 1))
        cos_pos  = math.cos(2 * math.pi * i / max(n, 1))
        n_term   = 1.0 if i < max(n * 0.10, 1) else 0.0
        c_term   = 1.0 if i >= n - max(n * 0.10, 1) else 0.0
        sc_mw    = SC_MW.get(resname, 0.0)

        features.append(
            one_hot + ss +
            [hydro, charge, rel_pos, rsa_val, bf_val,
             sin_pos, cos_pos, n_term, c_term, sc_mw]
        )

    return torch.tensor(features, dtype=torch.float)


def get_edges(residues, cutoff: float = GRAPH_CUTOFF):
    """
    Build enriched edge set:
      - Spatial edges: Cα–Cα distance ≤ cutoff
      - Sequential edges: |i - j| ≤ 2 (always included)

    Edge features (5-dim):
      0   : normalised distance
      1-3 : unit direction vector
      4   : edge type (0=spatial, 1=seq±1, 2=seq±2)
    """
    ca_coords = []
    for r in residues:
        try:
            ca_coords.append(r['CA'].get_vector().get_array())
        except KeyError:
            ca_coords.append(np.array([0.0, 0.0, 0.0]))
    ca_coords = np.array(ca_coords)

    edge_src, edge_dst, edge_features = [], [], []
    n = len(residues)

    # Build set of sequential edges first so we can type them
    seq_edges = set()
    for i in range(n):
        for delta in [-2, -1, 1, 2]:
            j = i + delta
            if 0 <= j < n:
                seq_edges.add((i, j))

    # Spatial edges
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = ca_coords[j] - ca_coords[i]
            dist = float(np.linalg.norm(diff))
            if dist > cutoff and (i, j) not in seq_edges:
                continue  # skip far, non-sequential pairs
            # include if within cutoff OR is sequential
            if dist == 0:
                dist = 1e-8
            unit_vec = diff / dist
            norm_dist = min(dist / cutoff, 1.0)

            if (i, j) in seq_edges:
                edge_type = 1.0 if abs(i - j) == 1 else 2.0
            else:
                edge_type = 0.0

            edge_src.append(i)
            edge_dst.append(j)
            edge_features.append([norm_dist] + unit_vec.tolist() + [edge_type])

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr  = torch.tensor(edge_features, dtype=torch.float)
    return edge_index, edge_attr, ca_coords


def build_residue_graph(pdb_path: str, chain_id: str) -> Data:
    parser    = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model     = structure[0]
    chain     = model[chain_id]

    residues, skipped = get_residues(chain)

    if len(residues) < 30:
        raise ValueError(f"Chain {chain_id}: {len(residues)} residues — below minimum 30")
    if len(residues) > 800:
        raise ValueError(f"Chain {chain_id}: {len(residues)} residues — above maximum 800")

    # Secondary structure via pydssp
    backbone     = get_backbone_coords(residues)
    coord_tensor = torch.tensor(backbone).unsqueeze(0)
    ss_labels    = pydssp.assign(coord_tensor)[0]

    # Build the 1-letter amino acid sequence
    seq_str = ""
    for r in residues:
        try:
            seq_str += three_to_one(r.get_resname())
        except KeyError:
            seq_str += "X"  # Unknown

    # Get 480-dim ESM-2 embedding
    esm_emb = get_esm_embedding(seq_str, chain_id=chain_id)  # (L, 480)

    # Get 33-dim physical features
    x_phys = get_node_features(residues, ss_labels, ca_coords)  # (L, 33)

    # Concatenate: 33 + 480 = 513 dimensions
    x = torch.cat([x_phys, esm_emb], dim=1)

    data               = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_residues  = len(residues)
    data.protein_len   = len(residues)
    data.chain_id      = chain_id
    return data


if __name__ == "__main__":
    graph = build_residue_graph("1AY7.pdb", "A")
    print("Node features:", graph.x.shape)        # expect (N, 513)
    print("Edge index:   ", graph.edge_index.shape)
    print("Edge features:", graph.edge_attr.shape) # expect (E, 5)