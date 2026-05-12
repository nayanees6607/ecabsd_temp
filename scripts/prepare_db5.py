import os
import sys
import csv
import glob
import random
import argparse
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.graph_construction import build_residue_graph
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa

def compute_binding_labels(pdb_path, target_chain_id, partner_chain_id, distance_cutoff=4.5):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]
    
    target_chain = model[target_chain_id]
    partner_chain = model[partner_chain_id]

    residues = [r for r in target_chain if is_aa(r, standard=True)]
    partner_atoms = [atom for r in partner_chain for atom in r]

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
    """Process both Chain A and Chain B for DB5 target PDBs."""
    pdb_name = os.path.basename(pdb_path).split('_')[0]
    results = []
    errors = []
    
    # DB5 standard: A is receptor, B is ligand
    chains_to_process = [('A', 'B'), ('B', 'A')]
    
    for target_id, partner_id in chains_to_process:
        save_path = os.path.join(output_dir, f"{pdb_name}_{target_id}.pt")
        
        if os.path.exists(save_path):
            try:
                graph = torch.load(save_path, map_location="cpu")
                results.append({
                    "pdb_id": pdb_name,
                    "chain_a": target_id,
                    "chain_b": partner_id,
                    "num_residues": len(graph.y),
                    "num_binding": int(graph.y.sum()),
                })
                continue
            except:
                pass

        try:
            graph = build_residue_graph(pdb_path, target_id)
            labels = compute_binding_labels(pdb_path, target_id, partner_id, distance_cutoff)
            graph.y = torch.tensor(labels, dtype=torch.float)
            torch.save(graph, save_path)

            results.append({
                "pdb_id": pdb_name,
                "chain_a": target_id,
                "chain_b": partner_id,
                "num_residues": len(labels),
                "num_binding": sum(labels),
            })
        except Exception as e:
            errors.append(f"{pdb_name}_{target_id}: {str(e)}")
            
    return results, errors

def prepare_db5(db5_dir, output_dir, distance_cutoff, train_ratio, val_ratio, seed, threads):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    # Find all target.pdb files in DB5 HADDOCK-ready
    pdb_files = sorted(glob.glob(os.path.join(db5_dir, "*", "*_target.pdb")))
    if not pdb_files:
        print(f"[ERROR] No DB5 target files found in: {db5_dir}")
        return

    print(f"[ECABSD] Processing {len(pdb_files)} DB5 structures using {threads} processes...")

    successful = []
    all_errors = []

    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(process_single_pdb, f, output_dir, distance_cutoff): f for f in pdb_files}
        for future in tqdm(as_completed(futures), total=len(pdb_files), desc="Processing DB5"):
            results, errors = future.result()
            successful.extend(results)
            all_errors.extend(errors)

    # Group by PDB ID so A and B of the same complex stay in the same split
    complexes = list(set([s["pdb_id"] for s in successful]))
    random.shuffle(complexes)
    
    n = len(complexes)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_complexes = set(complexes[:n_train])
    val_complexes = set(complexes[n_train:n_train + n_val])
    test_complexes = set(complexes[n_train + n_val:])

    for s in successful:
        if s["pdb_id"] in train_complexes:
            s["split"] = "train"
        elif s["pdb_id"] in val_complexes:
            s["split"] = "val"
        else:
            s["split"] = "test"

    # Write splits CSV
    splits_path = os.path.join(os.path.dirname(output_dir), "db5_splits.csv")
    fieldnames = ["pdb_id", "chain_a", "chain_b", "split", "num_residues", "num_binding"]
    with open(splits_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(successful)

    print(f"\n{'='*60}\n  DB5 Dataset Preparation Complete\n{'='*60}")
    print(f"  Complexes processed:  {len(complexes)}")
    print(f"  Total Chains:         {len(successful)}")
    print(f"  Errors:               {len(all_errors)}")
    print(f"  Train/Val/Test split: {len(train_complexes)} / {len(val_complexes)} / {len(test_complexes)} complexes")
    print(f"  Splits CSV:           {splits_path}\n{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db5-dir", default="data/BM5-clean/HADDOCK-ready", help="DB5 HADDOCK-ready directory")
    parser.add_argument("--output-dir", default="data/db5_processed", help="Output directory")
    parser.add_argument("--cutoff", type=float, default=4.5)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    prepare_db5(args.db5_dir, args.output_dir, args.cutoff, args.train_ratio, args.val_ratio, args.seed, args.threads)
