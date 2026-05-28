import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse

from models.ecabsd_model import ECABSDModel
from models.graph_construction import build_residue_graph

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for ECABSD")
    parser.add_argument("--pdb", required=True, help="Path to PDB file")
    parser.add_argument("--chain-a", required=True, help="Target chain ID")
    parser.add_argument("--chain-b", default=None, help="Partner chain ID")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    # Dynamic output directory
    pdb_name = os.path.splitext(os.path.basename(args.pdb))[0]
    out_dir = f"results/{pdb_name}"
    os.makedirs(out_dir, exist_ok=True)

    out_json = f"{out_dir}/gradcam.json"
    out_png  = f"{out_dir}/gradcam.png"

    cfg = load_config(args.config)
    mcfg = cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ECABSDModel(
        input_dim=mcfg["input_dim"],
        hidden_dim=mcfg["hidden_dim"],
        num_heads=mcfg["num_heads"],
        dropout=0.0,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_a = build_residue_graph(args.pdb, args.chain_a).to(device)
    data_b = None
    if args.chain_b:
        data_b = build_residue_graph(args.pdb, args.chain_b).to(device)

    # Enable gradient on residue node features
    data_a.x = data_a.x.float()
    data_a.x.requires_grad_(True)

    pred, _ = model(data_a, data_b)
    pred = pred.squeeze(-1)
    score = pred.sum()

    model.zero_grad()
    score.backward()

    # Residue importance = average absolute gradient per residue
    grads = data_a.x.grad.detach().cpu().numpy()
    saliency = np.abs(grads).mean(axis=1)

    # Normalize 0 to 1
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    residues = []
    for i, s in enumerate(saliency):
        residues.append({
            "index": int(i),
            "gradcam_score": float(s)
        })

    with open(out_json, "w") as f:
        json.dump({
            "pdb_file": args.pdb,
            "chain": args.chain_a,
            "method": "gradcam_saliency",
            "residues": residues
        }, f, indent=2)

    plt.figure(figsize=(14, 3))
    plt.imshow(saliency.reshape(1, -1), aspect="auto", cmap="viridis")
    plt.colorbar(label="Grad-CAM importance")
    plt.title(f"Grad-CAM Residue Importance - {pdb_name} Chain {args.chain_a}")
    plt.xlabel("Residue Index")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    # plt.show()

    print("Saved:", out_json)
    print("Saved:", out_png)

if __name__ == "__main__":
    main()