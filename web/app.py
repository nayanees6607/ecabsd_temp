"""
ECABSD Web API — FastAPI application for serving predictions.

Endpoints:
    GET  /          → Serves the frontend HTML
    GET  /health    → Health check
    POST /predict   → Upload PDB, get per-residue binding predictions
    POST /explain   → Upload PDB, get attention rollout scores
"""

import os
import sys
import json
import shutil
import tempfile
import yaml
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ecabsd_model import ECABSDModel
from models.graph_construction import build_residue_graph, get_residues
from Bio.PDB import PDBParser

# Global model instance
_model = None
_device = None
_config = None


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model(config_path: str = "config.yaml"):
    """Load model (singleton)."""
    global _model, _device, _config
    if _model is None:
        _config = load_config(config_path)
        mcfg = _config["model"]
        wcfg = _config.get("web", {})
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _model = ECABSDModel(
            input_dim=mcfg["input_dim"],
            hidden_dim=mcfg["hidden_dim"],
            num_heads=mcfg["num_heads"],
            dropout=0.0,
        ).to(_device)

        checkpoint_path = wcfg.get("checkpoint", "checkpoints/best_model.pt")
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=_device, weights_only=False)
            _model.load_state_dict(ckpt["model_state_dict"])
            print(f"[Web] Model loaded from: {checkpoint_path}")
        else:
            print(f"[Web] WARNING: No checkpoint at '{checkpoint_path}'. Using random weights.")

        _model.eval()
    return _model, _device, _config


def create_app(config_path: str = "config.yaml") -> FastAPI:
    """Create and configure the FastAPI application."""
    get_model(config_path)  # Pre-load model

    app = FastAPI(
        title="ECABSD — Binding Site Detection",
        description="Equivariant Cross-Attention for Protein-Protein Binding Site Detection",
        version="1.0.0",
    )

    # CORS
    wcfg = load_config(config_path).get("web", {})
    allow_origins = wcfg.get("allow_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    templates_dir = os.path.join(os.path.dirname(__file__), "templates")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the frontend."""
        html_path = os.path.join(templates_dir, "index.html")
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse("<h1>ECABSD Web Interface</h1><p>Frontend not found.</p>")

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        model, device, cfg = get_model()
        return {
            "status": "ok",
            "model": "ECABSDModel",
            "device": str(device),
            "version": "1.0.0",
        }

    async def validate_pdb_file(pdb_file: UploadFile):
        if not pdb_file.filename.lower().endswith(".pdb"):
            raise HTTPException(status_code=400, detail="Invalid file type. Only .pdb files are accepted.")
        MAX_SIZE = 50 * 1024 * 1024  # 50MB
        await pdb_file.seek(0, 2)
        file_size = await pdb_file.tell()
        await pdb_file.seek(0)
        if file_size > MAX_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 50MB.")

    @app.post("/predict")
    async def predict(
        pdb_file: UploadFile = File(...),
        chain_a: str = Form("A"),
        chain_b: Optional[str] = Form(None),
        threshold: float = Form(0.5),
    ):
        """
        Predict binding sites from an uploaded PDB file.

        Returns per-residue binding probabilities.
        """
        await validate_pdb_file(pdb_file)
        model, device, cfg = get_model()

        # Save uploaded PDB to temp file
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            shutil.copyfileobj(pdb_file.file, tmp)
            tmp_path = tmp.name

        try:
            # Build graphs
            try:
                data_a = build_residue_graph(tmp_path, chain_a).to(device)
            except (ValueError, KeyError) as e:
                raise HTTPException(status_code=400, detail=f"Chain {chain_a}: {str(e)}")

            data_b = None
            if chain_b and chain_b.strip():
                try:
                    data_b = build_residue_graph(tmp_path, chain_b).to(device)
                except Exception:
                    data_b = None

            # Predict
            probs, labels, attn = model.predict(data_a, data_b, threshold=threshold)
            probs_np = probs.squeeze(-1).cpu().tolist()
            labels_np = labels.cpu().tolist()

            # Get residue info
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", tmp_path)
            chain_obj = structure[0][chain_a]
            residues, _ = get_residues(chain_obj)

            results = []
            for i, r in enumerate(residues):
                results.append({
                    "index": i,
                    "resname": r.get_resname(),
                    "resid": r.get_id()[1],
                    "chain": chain_a,
                    "probability": round(probs_np[i], 4),
                    "is_binding": bool(labels_np[i]),
                })

            binding_count = sum(1 for r in results if r["is_binding"])

            return JSONResponse({
                "status": "success",
                "pdb_file": pdb_file.filename,
                "chain_a": chain_a,
                "chain_b": chain_b,
                "threshold": threshold,
                "total_residues": len(results),
                "binding_residues_count": binding_count,
                "residues": results,
            })

        finally:
            os.unlink(tmp_path)

    @app.post("/explain")
    async def explain(
        pdb_file: UploadFile = File(...),
        chain_a: str = Form("A"),
        chain_b: Optional[str] = Form(None),
    ):
        """
        Get attention rollout explanation for a prediction.
        """
        await validate_pdb_file(pdb_file)
        model, device, cfg = get_model()

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            shutil.copyfileobj(pdb_file.file, tmp)
            tmp_path = tmp.name

        try:
            from explainability.attention_rollout import AttentionRollout

            data_a = build_residue_graph(tmp_path, chain_a).to(device)
            data_b = None
            if chain_b and chain_b.strip():
                try:
                    data_b = build_residue_graph(tmp_path, chain_b).to(device)
                except Exception:
                    pass

            rollout = AttentionRollout(model)
            scores, attn_matrix = rollout.compute(data_a, data_b)
            rollout.remove_hook()

            return JSONResponse({
                "status": "success",
                "attention_scores": scores.tolist(),
                "attention_matrix_shape": list(attn_matrix.shape),
            })
        finally:
            os.unlink(tmp_path)

    return app


# App instance for uvicorn
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
