"""
ECABSD Batch Prediction.

Runs binding site prediction on all PDB files in a directory.
Aggregates results into a summary CSV.
"""

import os
import json
import csv
import yaml
import glob

from tqdm import tqdm
from predict import run_prediction


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_batch_prediction(
    input_dir: str,
    checkpoint_path: str = "checkpoints/best_model.pt",
    chain_a: str = "A",
    chain_b: str = None,
    threshold: float = 0.5,
    output_dir: str = "results/batch",
    config_path: str = "config.yaml",
):
    """
    Run predictions on all PDB files in a directory.

    Parameters
    ----------
    input_dir : str
        Directory containing PDB files.
    checkpoint_path : str
        Path to model checkpoint.
    chain_a : str
        Default chain ID for target protein.
    chain_b : str, optional
        Default chain ID for partner protein.
    threshold : float
        Probability threshold.
    output_dir : str
        Directory to save individual and summary results.
    config_path : str
        Path to config file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all PDB files
    pdb_files = sorted(
        glob.glob(os.path.join(input_dir, "*.pdb"))
        + glob.glob(os.path.join(input_dir, "*.PDB"))
    )

    if not pdb_files:
        print(f"[ECABSD] No PDB files found in: {input_dir}")
        return

    print(f"[ECABSD] Found {len(pdb_files)} PDB files in: {input_dir}")
    print(f"[ECABSD] Output directory: {output_dir}\n")

    # Pre-load model and device
    import torch
    from models.ecabsd_model import ECABSDModel

    cfg = load_config(config_path)
    mcfg = cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ECABSDModel(
        input_dim=mcfg["input_dim"],
        hidden_dim=mcfg["hidden_dim"],
        num_heads=mcfg["num_heads"],
        # Use training dropout; model.eval() inside predict() disables it
        dropout=mcfg.get("dropout", 0.3),
        edge_dim=mcfg["edge_feature_dim"],
    ).to(device)

    cfg_threshold = cfg["prediction"].get("threshold", 0.5)
    ckpt_threshold = cfg_threshold
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        ckpt_threshold = checkpoint.get("best_threshold", cfg_threshold)
        if ckpt_threshold is None:
            ckpt_threshold = cfg_threshold
        print(f"[ECABSD Batch] Loaded model from: {checkpoint_path}")
        print(f"[ECABSD Batch] Checkpoint threshold: {ckpt_threshold:.4f}")
    else:
        print(f"[ECABSD Batch] WARNING: No checkpoint at {checkpoint_path}. Using random weights.")
        ckpt_threshold = cfg_threshold

    if threshold is None:
        threshold = ckpt_threshold

    # Process each PDB
    summary_rows = []
    errors = []

    for pdb_path in tqdm(pdb_files, desc="Predicting", unit="pdb"):
        pdb_name = os.path.splitext(os.path.basename(pdb_path))[0]
        output_path = os.path.join(output_dir, f"predictions_{pdb_name}.json")

        try:
            results = run_prediction(
                pdb_path=pdb_path,
                chain_a=chain_a,
                chain_b=chain_b,
                checkpoint_path=checkpoint_path,
                threshold=threshold,
                output_path=output_path,
                config_path=config_path,
                model=model,
                device=device,
            )

            # Collect summary row
            binding_residue_ids = [
                f"{r['resname']}{r['resid']}"
                for r in results["residues"]
                if r["is_binding"]
            ]
            avg_prob = sum(r["probability"] for r in results["residues"]) / max(
                len(results["residues"]), 1
            )

            summary_rows.append(
                {
                    "pdb_file": results["pdb_file"],
                    "chain_a": chain_a,
                    "chain_b": chain_b or "",
                    "total_residues": results["total_residues"],
                    "binding_residues": results["binding_residues_count"],
                    "avg_probability": f"{avg_prob:.4f}",
                    "binding_residue_ids": ";".join(binding_residue_ids),
                }
            )
        except Exception as e:
            error_msg = f"{pdb_name}: {str(e)}"
            errors.append(error_msg)
            tqdm.write(f"  ERROR: {error_msg}")

    # Write summary CSV
    summary_path = os.path.join(output_dir, "batch_summary.csv")
    if summary_rows:
        fieldnames = summary_rows[0].keys()
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Batch Prediction Summary")
    print(f"{'='*60}")
    print(f"  Total PDB files:   {len(pdb_files)}")
    print(f"  Successful:        {len(summary_rows)}")
    print(f"  Errors:            {len(errors)}")
    print(f"  Summary CSV:       {summary_path}")
    print(f"{'='*60}")

    if errors:
        print(f"\n  Errors:")
        for err in errors:
            print(f"    - {err}")

    return summary_rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ECABSD Batch Prediction")
    parser.add_argument("--input-dir", required=True, help="Directory of PDB files")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--chain-a", default="A")
    parser.add_argument("--chain-b", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="results/batch")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    run_batch_prediction(
        input_dir=args.input_dir,
        checkpoint_path=args.checkpoint,
        chain_a=args.chain_a,
        chain_b=args.chain_b,
        threshold=args.threshold,
        output_dir=args.output_dir,
        config_path=args.config,
    )
