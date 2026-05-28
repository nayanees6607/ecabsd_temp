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

def generate_heatmap(json_path, out_png, title):
    with open(json_path, "r") as f:
        data = json.load(f)
    residues = data["residues"]
    probs = np.array([r["probability"] for r in residues])
    heatmap = probs.reshape(1, -1)
    
    plt.figure(figsize=(14, 2))
    plt.imshow(heatmap, aspect="auto", cmap="viridis")
    plt.colorbar(label="Binding Probability")
    plt.title(title)
    plt.xlabel("Residue Index")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Heatmap saved to: {out_png}")

def generate_gradcam(pdb_path, chain_a, chain_b, checkpoint_path, config_path, out_json, out_png, title):
    cfg = load_config(config_path)
    mcfg = cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ECABSDModel(
        input_dim=mcfg["input_dim"],
        hidden_dim=mcfg["hidden_dim"],
        num_heads=mcfg["num_heads"],
        dropout=0.0,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_a = build_residue_graph(pdb_path, chain_a).to(device)
    data_b = None
    if chain_b:
        data_b = build_residue_graph(pdb_path, chain_b).to(device)

    data_a.x = data_a.x.float()
    data_a.x.requires_grad_(True)

    pred, _ = model(data_a, data_b)
    pred = pred.squeeze(-1)
    score = pred.sum()

    model.zero_grad()
    score.backward()

    grads = data_a.x.grad.detach().cpu().numpy()
    saliency = np.abs(grads).mean(axis=1)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    residues = []
    for i, s in enumerate(saliency):
        residues.append({"index": int(i), "gradcam_score": float(s)})

    with open(out_json, "w") as f:
        json.dump({
            "pdb_file": pdb_path,
            "chain": chain_a,
            "method": "gradcam_saliency",
            "residues": residues
        }, f, indent=2)

    plt.figure(figsize=(14, 2))
    plt.imshow(saliency.reshape(1, -1), aspect="auto", cmap="plasma")
    plt.colorbar(label="Grad-CAM Importance")
    plt.title(title)
    plt.xlabel("Residue Index")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Grad-CAM JSON saved to: {out_json}")
    print(f"Grad-CAM PNG saved to: {out_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--chain-a", required=True)
    parser.add_argument("--chain-b", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    sample_id = os.path.splitext(os.path.basename(args.pdb))[0]
    
    # Paths
    pred_json = f"results/predictions_{sample_id}_{args.chain_a}.json"
    heatmap_png = f"results/heatmap_{sample_id}_{args.chain_a}.png"
    gradcam_json = f"results/gradcam_{sample_id}_{args.chain_a}.json"
    gradcam_png = f"results/gradcam_{sample_id}_{args.chain_a}.png"

    # Heatmap
    if os.path.exists(pred_json):
        generate_heatmap(pred_json, heatmap_png, f"Binding Probability Heatmap - {sample_id} Chain {args.chain_a}")
    else:
        print(f"Warning: Prediction JSON not found at {pred_json}. Run predict.py first.")

    # Grad-CAM
    generate_gradcam(args.pdb, args.chain_a, args.chain_b, args.checkpoint, args.config, gradcam_json, gradcam_png, f"Grad-CAM Residue Importance - {sample_id} Chain {args.chain_a}")
