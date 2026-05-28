"""
Prepare ECABSD Dataset (Optimized with Multi-processing and Resume).

Processes raw PDB files into PyTorch Geometric graph objects with
per-residue binding site labels.

Usage:
    python scripts/prepare_dataset.py --pdb-dir data/raw/pdbs --output-dir data/processed --threads 8
"""

import os
import sys
import csv
import random
import argparse
import numpy as np
import torch
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.graph_construction import build_residue_graph
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa


def compute_binding_labels(pdb_path, chain_id, partner_chain_id=None, distance_cutoff=4.5):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]
    chain = model[chain_id]

    residues = [r for r in chain if is_aa(r, standard=True)]
    partner_atoms = []
    if partner_chain_id and partner_chain_id in [c.id for c in model]:
        partner_chain = model[partner_chain_id]
        for r in partner_chain:
            for atom in r:
                partner_atoms.append(atom)
    else:
        for c in model:
            for r in c:
                if not is_aa(r, standard=True) and r.get_id()[0] not in (" ", "W"):
                    for atom in r:
                        partner_atoms.append(atom)

    if not partner_atoms:
        return [0] * len(residues)

    ns = NeighborSearch(partner_atoms)
    labels = []
    for residue in residues:
        is_binding = False
        for atom in residue:
            nearby = ns.search(atom.get_vector().get_array(), distance_cutoff, level="A")
            if nearby:
                is_binding = True
                break
        labels.append(1 if is_binding else 0)
    return labels


def process_single_pdb(pdb_path, output_dir, distance_cutoff):
    """Worker function to process all chains in a single PDB file."""
    pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
    results = []
    errors = []
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        model = structure[0]
        chains = [c.id for c in model]

        for chain_id in chains:
            save_path = os.path.join(output_dir, f"{pdb_name}_{chain_id}.pt")
            
            # Check if already processed (Resume feature)
            if os.path.exists(save_path):
                try:
                    graph = torch.load(save_path, map_location="cpu", weights_only=False)
                    results.append({
                        "pdb_id": pdb_name,
                        "chain_a": chain_id,
                        "chain_b": "", # Simplified for resume, will be fixed in post-processing if needed
                        "num_residues": graph.num_residues if hasattr(graph, 'num_residues') else len(graph.x),
                        "num_binding": int(graph.y.sum()),
                    })
                    continue
                except:
                    pass # If load fails, re-process

            try:
                graph = build_residue_graph(pdb_path, chain_id)
                partner_chains = [c for c in chains if c != chain_id]
                partner_id = partner_chains[0] if partner_chains else None
                labels = compute_binding_labels(pdb_path, chain_id, partner_id, distance_cutoff)
                graph.y = torch.tensor(labels, dtype=torch.float)
                torch.save(graph, save_path)

                results.append({
                    "pdb_id": pdb_name,
                    "chain_a": chain_id,
                    "chain_b": partner_id or "",
                    "num_residues": len(labels),
                    "num_binding": sum(labels),
                })
            except Exception as e:
                errors.append(f"{pdb_name}_{chain_id}: {str(e)}")
    except Exception as e:
        errors.append(f"{pdb_name}: {str(e)}")
    
    return results, errors


def prepare_dataset(pdb_dir, output_dir, distance_cutoff, train_ratio, val_ratio, seed, threads):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    pdb_files = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")) + glob.glob(os.path.join(pdb_dir, "*.PDB")))
    if not pdb_files:
        print(f"[ERROR] No PDB files found in: {pdb_dir}")
        return

    print(f"[ECABSD] Processing {len(pdb_files)} PDB files using {threads} processes...")

    successful = []
    all_errors = []

    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(process_single_pdb, f, output_dir, distance_cutoff): f for f in pdb_files}
        for future in tqdm(as_completed(futures), total=len(pdb_files), desc="Processing PDBs"):
            results, errors = future.result()
            successful.extend(results)
            all_errors.extend(errors)

    # Create train/val/test splits
    random.shuffle(successful)
    n = len(successful)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    for i, s in enumerate(successful):
        if i < n_train:
            s["split"] = "train"
        elif i < n_train + n_val:
            s["split"] = "val"
        else:
            s["split"] = "test"

    # Write splits CSV
    splits_path = os.path.join(os.path.dirname(output_dir), "splits.csv")
    fieldnames = ["pdb_id", "chain_a", "chain_b", "split", "num_residues", "num_binding"]
    with open(splits_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(successful)

    print(f"\n{'='*60}\n  Dataset Preparation Complete\n{'='*60}")
    print(f"  Chains processed:     {len(successful)}")
    print(f"  Errors:               {len(all_errors)}")
    print(f"  Train / Val / Test:   {n_train} / {n_val} / {n - n_train - n_val}")
    print(f"  Splits CSV:           {splits_path}\n{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ECABSD Dataset (Optimized)")
    parser.add_argument("--pdb-dir", default="data/raw/pdbs", help="PDB files directory")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--cutoff", type=float, default=4.5, help="Binding distance cutoff (Å)")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8, help="Number of parallel processes")
    args = parser.parse_args()

    prepare_dataset(args.pdb_dir, args.output_dir, args.cutoff, args.train_ratio, args.val_ratio, args.seed, args.threads)
