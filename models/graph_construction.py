import numpy as np
import torch
from torch_geometric.data import Data
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
import pydssp

STANDARD_AA = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL'
]
AA_TO_IDX = {aa: i for i, aa in enumerate(STANDARD_AA)}

SS_MAPPING = {
    'H': [1, 0, 0], 'G': [1, 0, 0], 'I': [1, 0, 0],   # helix
    'E': [0, 1, 0], 'B': [0, 1, 0],                      # sheet
    '-': [0, 0, 1], 'S': [0, 0, 1], 'T': [0, 0, 1],     # coil
}

# Kyte-Doolittle hydrophobicity scale (normalised to [-1, 1] range)
_HYDRO_RAW = {
    'ALA': 1.8,  'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8,  'LYS': -3.9, 'MET': 1.9,  'PHE': 2.8,  'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2,
}
_HYDRO_MAX = 4.5
HYDROPHOBICITY = {aa: v / _HYDRO_MAX for aa, v in _HYDRO_RAW.items()}

# Formal charge at physiological pH
CHARGE = {
    'ARG':  1.0, 'LYS':  1.0,
    'ASP': -1.0, 'GLU': -1.0,
}


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
    return np.array(coords, dtype=np.float32)  # (L, 4, 3)


def get_node_features(residues, ss_labels) -> torch.Tensor:
    """
    Build per-residue node feature vectors (26-dim).

    Dimensions:
        0-19  : 20-dim one-hot amino acid type
        20-22 : 3-dim secondary structure one-hot (helix/sheet/coil)
        23    : Kyte-Doolittle hydrophobicity (normalised, range ≈ -1..1)
        24    : Formal charge (+1 / 0 / -1)
        25    : Relative sequence position (0 = N-term, 1 = C-term)
    """
    n = len(residues)
    features = []
    for i, r in enumerate(residues):
        resname = r.get_resname()

        # 20-dim amino acid one-hot
        one_hot = [0.0] * 20
        if resname in AA_TO_IDX:
            one_hot[AA_TO_IDX[resname]] = 1.0

        # 3-dim secondary structure
        ss = SS_MAPPING.get(str(ss_labels[i]), [0, 0, 1])

        # Physicochemical features
        hydro   = [HYDROPHOBICITY.get(resname, 0.0)]
        charge  = [float(CHARGE.get(resname, 0.0))]
        rel_pos = [i / max(n - 1, 1)]                 # 0..1

        features.append(one_hot + ss + hydro + charge + rel_pos)  # 26 total

    return torch.tensor(features, dtype=torch.float)


def get_edges(residues, cutoff: float = 8.0):
    """
    Build Cα-distance graph with edge features.

    Edge features (4-dim):
        0   : Euclidean distance (Å)
        1-3 : Unit direction vector (normalised)
    """
    ca_coords = []
    for r in residues:
        try:
            ca_coords.append(r['CA'].get_vector().get_array())
        except KeyError:
            ca_coords.append(np.array([0.0, 0.0, 0.0]))

    ca_coords = np.array(ca_coords)
    edge_src, edge_dst, edge_features = [], [], []

    for i in range(len(residues)):
        for j in range(len(residues)):
            if i == j:
                continue
            diff = ca_coords[j] - ca_coords[i]
            dist = np.linalg.norm(diff)
            if dist <= cutoff:
                edge_src.append(i)
                edge_dst.append(j)
                unit_vec = diff / (dist + 1e-8)
                edge_features.append([dist] + unit_vec.tolist())

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr  = torch.tensor(edge_features, dtype=torch.float)
    return edge_index, edge_attr


def build_residue_graph(pdb_path: str, chain_id: str) -> Data:
    parser    = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model     = structure[0]
    chain     = model[chain_id]

    residues, skipped = get_residues(chain)
    print(f"Chain {chain_id}: {len(residues)} residues, {skipped} skipped")

    if len(residues) < 50:
        raise ValueError(f"Chain {chain_id}: {len(residues)} residues — below minimum 50")
    if len(residues) > 512:
        raise ValueError(f"Chain {chain_id}: {len(residues)} residues — above maximum 512")

    # Secondary structure via pydssp
    backbone     = get_backbone_coords(residues)               # (L, 4, 3)
    coord_tensor = torch.tensor(backbone).unsqueeze(0)         # (1, L, 4, 3)
    ss_labels    = pydssp.assign(coord_tensor)[0]              # (L,)

    x                      = get_node_features(residues, ss_labels)
    edge_index, edge_attr  = get_edges(residues, cutoff=8.0)

    data               = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_residues  = len(residues)
    data.protein_len   = len(residues)
    data.chain_id      = chain_id
    return data


if __name__ == "__main__":
    graph = build_residue_graph("1AY7.pdb", "A")
    print("Node features:", graph.x.shape)         # expect (N, 26)
    print("Edge index:   ", graph.edge_index.shape)
    print("Edge features:", graph.edge_attr.shape)  # expect (E, 4)