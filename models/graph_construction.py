"""
Graph construction for ECABSD.

Node features (23-dim):
    0-19  : amino acid one-hot (20)
    20-22 : secondary structure one-hot — helix/sheet/coil (3)

Edge features (4-dim):
    0   : Euclidean Cα–Cα distance (Å, normalised by cutoff)
    1-3 : unit direction vector
"""

import numpy as np
import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
import pydssp

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

GRAPH_CUTOFF = 8.0


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
    """Build 23-dim per-residue node feature vectors."""
    features = []
    for i, r in enumerate(residues):
        resname = r.get_resname()

        # 20-dim amino acid one-hot
        one_hot = [0.0] * 20
        if resname in AA_TO_IDX:
            one_hot[AA_TO_IDX[resname]] = 1.0

        # 3-dim secondary structure
        ss = SS_MAPPING.get(str(ss_labels[i]), [0, 0, 1])

        features.append(one_hot + ss)

    return torch.tensor(features, dtype=torch.float)


def get_edges(residues, cutoff: float = GRAPH_CUTOFF):
    """
    Build spatial Cα-Cα edges within cutoff.

    Edge features (4-dim):
      0   : normalised distance
      1-3 : unit direction vector
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

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = ca_coords[j] - ca_coords[i]
            dist = float(np.linalg.norm(diff))
            if dist > cutoff:
                continue
            if dist == 0:
                dist = 1e-8
            unit_vec = diff / dist
            norm_dist = min(dist / cutoff, 1.0)

            edge_src.append(i)
            edge_dst.append(j)
            edge_features.append([norm_dist] + unit_vec.tolist())

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

    edge_index, edge_attr, ca_coords = get_edges(residues, cutoff=GRAPH_CUTOFF)

    x = get_node_features(residues, ss_labels, ca_coords)

    data               = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_residues  = len(residues)
    data.protein_len   = len(residues)
    data.chain_id      = chain_id
    return data


if __name__ == "__main__":
    graph = build_residue_graph("1AY7.pdb", "A")
    print("Node features:", graph.x.shape)        # expect (N, 23)
    print("Edge index:   ", graph.edge_index.shape)
    print("Edge features:", graph.edge_attr.shape) # expect (E, 4)
