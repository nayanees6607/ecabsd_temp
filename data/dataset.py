"""
ECABSD Dataset — PyTorch Geometric Dataset for binding site detection.

Loads preprocessed .pt graph files and returns paired protein graphs
with per-residue binding labels. Uses PyG Batch for efficient batching.
"""

import os
import csv
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class BindingSiteDataset(Dataset):
    """
    Dataset for protein-protein binding site detection.

    Each sample returns:
        - data_a : PyG Data for chain A (target chain, with .y labels)
        - data_b : PyG Data for chain B (partner chain)
        - labels : per-residue binding labels for chain A (float tensor)
        - pdb_id : PDB identifier string

    splits_csv columns: pdb_id, chain_a, chain_b, split
    """

    def __init__(self, processed_dir: str, splits_csv: str, split: str = "train"):
        self.processed_dir = processed_dir
        self.split         = split
        self.samples       = []

        if not os.path.exists(splits_csv):
            raise FileNotFoundError(f"Splits CSV not found: {splits_csv}")

        with open(splits_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    chain_b = row.get("chain_b", "").strip()
                    path_a  = os.path.join(processed_dir, f"{row['pdb_id']}_{row['chain_a']}.pt")
                    path_b  = os.path.join(processed_dir, f"{row['pdb_id']}_{chain_b}.pt") \
                              if chain_b else None
                    # Only include samples where chain A file exists
                    if os.path.exists(path_a):
                        self.samples.append({
                            "pdb_id":  row["pdb_id"],
                            "chain_a": row["chain_a"],
                            "chain_b": chain_b,
                            "path_a":  path_a,
                            "path_b":  path_b if (path_b and os.path.exists(path_b)) else None,
                        })

        print(f"[Dataset] Loaded {len(self.samples)} samples for split '{split}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s      = self.samples[idx]
        data_a = torch.load(s["path_a"], weights_only=False)
        data_b = torch.load(s["path_b"], weights_only=False) if s["path_b"] else None

        # Labels must come from data_a.y set during prepare_db5.py
        if hasattr(data_a, "y") and data_a.y is not None:
            labels = data_a.y.float()
        else:
            # Fallback: all-negative (will hurt training — means labels weren't saved)
            n_nodes = data_a.x.shape[0]
            labels  = torch.zeros(n_nodes, dtype=torch.float)

        return {
            "data_a":  data_a,
            "data_b":  data_b,
            "labels":  labels,
            "pdb_id":  s["pdb_id"],
            "partner_missing": data_b is None,
        }


def collate_fn(batch):
    """
    Collate a list of samples into a single batched sample.

    Uses PyG Batch.from_data_list to correctly batch variable-size graphs.
    Labels are concatenated to match the flattened node order in the batch.

    Note: data_b can be None if the partner chain was not available.
    We fall back to self-attention for those rows so inference can still run,
    and return partner_missing flags so training can avoid treating fallback
    partners as true chain-swap augmentation.
    """
    data_a_list  = [s["data_a"]  for s in batch]
    labels_list  = [s["labels"]  for s in batch]

    # For data_b: replace None with the corresponding data_a (self-attention fallback)
    data_b_list  = [s["data_b"] if s["data_b"] is not None else s["data_a"] for s in batch]

    batch_a  = Batch.from_data_list(data_a_list)
    batch_b  = Batch.from_data_list(data_b_list)
    labels   = torch.cat(labels_list, dim=0)   # (total_nodes_in_batch,)

    return {
        "data_a":  batch_a,
        "data_b":  batch_b,
        "labels":  labels,
        "pdb_id":  [s["pdb_id"] for s in batch],
        "partner_missing": [s["partner_missing"] for s in batch],
    }
