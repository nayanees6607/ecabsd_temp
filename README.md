# ECABSD — Explainable Cross Attention Model for Binding Site Discovery

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange?logo=pytorch)
![PyG](https://img.shields.io/badge/PyG-2.7-red)
![License](https://img.shields.io/badge/License-MIT-green)

**Deep learning model for per-residue protein–protein binding site discovery using graph neural networks and explainable cross-attention.**

</div>

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Usage](#cli-usage)
- [Web Interface](#web-interface)
- [Training](#training)
- [Evaluation](#evaluation)
- [Explainability](#explainability)
- [Docking Integration](#docking-integration)
- [Exports](#exports)
- [Project Structure](#project-structure)

---

## Overview

ECABSD predicts which residues in a target protein chain form the binding interface with a partner protein chain. The included processed graphs and checkpoint use:

1. **Graph Construction** — each protein chain becomes a residue graph with 8 Å distance cutoff edges
2. **GCN Encoder** — 4-layer graph convolution stack (23 → 128 features)
3. **SE(3)-inspired Refinement** — lightweight learned feature refinement block
4. **Target-to-partner Cross-Attention** — 8-head attention from target chain A to partner chain B
5. **Per-residue Classifier** — 2-layer MLP with sigmoid for binding probability

---

## Architecture

```
Protein A (target)  ─→ [Graph Construction] ─→ [GCN × 4] ─→ [SE3-inspired Refine] ─┐
                                                                                    ├─→ A→B CrossAttention (8 heads) ─→ Classifier ─→ P(binding) per target residue
Protein B (partner) ─→ [Graph Construction] ─→ [GCN × 4] ─→ [SE3-inspired Refine] ─┘
```

**Node features (23-dim):** 20-dim amino acid one-hot + 3-dim secondary structure (helix/sheet/coil)  
**Edge features (4-dim):** distance + 3D unit direction vector  
**Graph cutoff:** 8.0 Å (Cα–Cα distance)

---

## Installation

```bash
# Clone repository
git clone https://github.com/nayanees6607/ecabsd_temp.git
cd ecabsd

# Create environment
conda create -n ecabsd python=3.10 -y
conda activate ecabsd

# Install PyTorch (CPU)
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric==2.7.0
pip install torch-scatter torch-sparse torch-cluster --find-links https://data.pyg.org/whl/torch-2.1.0+cpu.html

# Install remaining dependencies
pip install biopython pydssp fastapi uvicorn typer pyyaml scikit-learn tqdm matplotlib seaborn
```

---

## Quick Start

### 1. Predict binding sites on 1AY7.pdb

```bash
python predict.py --pdb 1AY7.pdb --chain-a A --chain-b B
```

### 2. Run tests

```bash
pytest tests/
```

### 3. Launch web interface

```bash
cd web && python app.py
# → Open http://localhost:8000
```

---

## CLI Usage

```
python main.py --help

Commands:
  train          Train the ECABSD model
  evaluate       Evaluate on test set
  predict        Predict binding sites for a single PDB
  batch-predict  Batch predict for a directory of PDBs
  export         Export results to CSV / JSON / PyMOL
  web            Launch the web interface
```

### Examples

```bash
# Train (needs processed data)
python main.py train --config config.yaml

# Single prediction
python main.py predict --pdb 1AY7.pdb --chain-a A --chain-b B --threshold 0.5

# Batch prediction
python main.py batch-predict --input-dir data/raw/pdbs --output-dir results/batch

# Export to PyMOL script
python main.py export --results results/predictions_1AY7_A.json --format pymol
```

---

## Web Interface

```bash
# From project root
python web/app.py
```

Opens at **http://localhost:8000**. Features:
- Drag-and-drop PDB upload
- Chain selection + probability threshold slider
- Interactive probability chart (Chart.js)
- Per-residue results table with filter
- One-click export: CSV, JSON, PyMOL script

---

## Training

### Step 1: Download PDB structures

```bash
python scripts/download_pdbbind.py --benchmark
```

### Step 2: Prepare dataset

```bash
python scripts/prepare_dataset.py \
    --pdb-dir data/raw/pdbs \
    --output-dir data/processed \
    --cutoff 4.5
```

### Step 3: Train

```bash
python train.py
# or
python main.py train
```

Training config is in `config.yaml`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 128 | Model hidden dimension |
| `num_heads` | 8 | Cross-attention heads |
| `graph_cutoff` | 8.0 Å | Edge distance cutoff |
| `epochs` | 200 | Max training epochs |
| `learning_rate` | 0.0003 | Adam LR |
| `chain_swap_prob` | 0.5 | Probability of valid chain-swap augmentation |
| `early_stopping_patience` | 60 | Epochs to wait before stopping |

The included checkpoint is `checkpoints/best_model.pt`. New training runs save the best checkpoint to the same path and logs to `logs/training_history.json`.

---

## Evaluation

```bash
python main.py evaluate --checkpoint checkpoints/best_model.pt
```

Outputs:
- `results/metrics.json` — Accuracy, Precision, Recall, F1, MCC, AUC-ROC, AUC-PR
- `results/confusion_matrix.png` — Confusion matrix plot

Current verified evaluation after exact PDB-ID de-leakage:

| Metric | Score |
|---|---:|
| Accuracy | `0.8318` |
| Precision | `0.1403` |
| Recall | `0.0132` |
| F1 Score | `0.0241` |
| MCC | `-0.0058` |
| ROC-AUC | `0.4736` |
| PR-AUC | `0.1545` |
| Threshold | `0.5000` |

Before using benchmark numbers in a publication, run the leakage checks:

```bash
python check_leakage.py --mmseqs
```

Tier 1 checks exact PDB-ID overlap. Tier 2 requires MMseqs2 and checks sequence-similarity leakage at 30% identity.

### Benchmark vs. Baselines

```bash
python scripts/benchmark_crossPPI.py --checkpoint checkpoints/best_model.pt
```

---

## Explainability

```python
from models.ecabsd_model import ECABSDModel
from models.graph_construction import build_residue_graph
from explainability.attention_rollout import explain_prediction
from explainability.gradcam import explain_with_gradcam

model = ECABSDModel()
data_a = build_residue_graph("1AY7.pdb", "A")

# Attention rollout
scores, attn_matrix = explain_prediction(model, data_a, output_dir="results/")

# Grad-CAM
saliency = explain_with_gradcam(model, data_a, output_dir="results/")
```

---

## Docking Integration

Requires AutoDock Vina: `conda install -c conda-forge autodock-vina`

```python
from predict import run_prediction
from docking.docking_input import binding_residues_to_box, write_vina_config
from docking.vina_runner import VinaRunner

# Get predictions
results = run_prediction("1AY7.pdb", "A", "B")
binding_residues = [r for r in results["residues"] if r["is_binding"]]

# Compute docking box
center, box_size = binding_residues_to_box(binding_residues, "1AY7.pdb", "A")

# Run docking
runner = VinaRunner(exhaustiveness=8)
result = runner.dock("receptor.pdbqt", "ligand.pdbqt", center, box_size)
```

---

## Exports

```bash
# CSV
python main.py export --results results/predictions_1AY7_A.json --format csv

# JSON (with metadata + confidence bands)
python main.py export --results results/predictions_1AY7_A.json --format json

# PyMOL script (probability-gradient coloring)
python main.py export --results results/predictions_1AY7_A.json --format pymol
```

---

## Project Structure

```
ecabsd/
├── 1AY7.pdb                    # Sample PDB structure
├── config.yaml                 # Central configuration
├── main.py                     # Entry point
├── cli.py                      # Typer CLI
├── train.py                    # Training pipeline
├── evaluate.py                 # Evaluation pipeline
├── predict.py                  # Single-structure prediction
├── batch_predict.py            # Batch prediction
├── check_leakage.py            # Split leakage checks
│
├── models/
│   ├── __init__.py
│   ├── ecabsd_model.py         # End-to-end model
│   ├── encoder.py              # Experimental encoder components
│   ├── gcn_model.py            # Experimental GATv2 encoder
│   ├── se3_model.py            # Experimental SE(3)-inspired block
│   ├── cross_attention.py      # Experimental bidirectional cross-attention
│   ├── classifier.py           # Experimental deep classifier
│   └── graph_construction.py  # PDB → residue graph
│
├── data/
│   ├── __init__.py
│   ├── dataset.py              # PyG Dataset
│   ├── raw/                    # Raw PDB files
│   └── processed/              # Preprocessed .pt graphs
│
├── scripts/
│   ├── prepare_dataset.py      # PDB → labeled graphs
│   ├── download_pdbbind.py     # Download PDB structures
│   └── benchmark_crossPPI.py  # Benchmark comparison
│
├── explainability/
│   ├── __init__.py
│   ├── attention_rollout.py    # Attention-based explainability
│   └── gradcam.py              # Grad-CAM for GNNs
│
├── docking/
│   ├── __init__.py
│   ├── vina_runner.py          # AutoDock Vina wrapper
│   ├── docking_input.py        # Box definition + PDBQT prep
│   └── rmsd.py                 # Docking pose RMSD
│
├── exports/
│   ├── __init__.py
│   ├── csv_export.py           # CSV export
│   ├── json_export.py          # JSON export with metadata
│   └── pymol_export.py         # PyMOL .pml script
│
├── web/
│   ├── app.py                  # FastAPI backend
│   ├── templates/index.html    # Web UI
│   └── static/
│       ├── style.css           # Dark-mode CSS
│       └── app.js              # Frontend JavaScript
│
├── tests/
│   └── test_graph_construction.py
│
├── checkpoints/                # Saved model weights
├── logs/                       # Training logs
├── results/                    # Prediction outputs
└── requirements.txt
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
