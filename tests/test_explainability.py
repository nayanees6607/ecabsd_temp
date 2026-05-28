import os
import sys
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ecabsd_model import ECABSDModel
from models.graph_construction import build_residue_graph
from explainability.attention_rollout import AttentionRollout
from explainability.gradcam import GradCAM

PDB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '1AY7.pdb'))

def test_explainability_rollout_and_gradcam():
    # 1. Load model and graphs
    device = torch.device("cpu")
    model = ECABSDModel(
        input_dim=23,
        hidden_dim=128,
        num_heads=8,
        dropout=0.0,
        edge_dim=4
    ).to(device)
    
    data_a = build_residue_graph(PDB_PATH, 'A').to(device)
    data_b = build_residue_graph(PDB_PATH, 'B').to(device)
    
    # 2. Test AttentionRollout with single chain
    rollout = AttentionRollout(model)
    scores_single, attn_matrix_single = rollout.compute(data_a)
    assert isinstance(scores_single, np.ndarray)
    assert len(scores_single) == data_a.num_residues
    assert isinstance(attn_matrix_single, np.ndarray)
    assert attn_matrix_single.shape == (data_a.num_residues, data_a.num_residues)
    rollout.remove_hook()
    
    # 3. Test AttentionRollout with partner chain
    rollout_pair = AttentionRollout(model)
    scores_pair, attn_matrix_pair = rollout_pair.compute(data_a, data_b)
    assert isinstance(scores_pair, np.ndarray)
    assert len(scores_pair) == data_a.num_residues
    assert isinstance(attn_matrix_pair, np.ndarray)
    assert attn_matrix_pair.shape == (data_a.num_residues, data_b.num_residues)
    rollout_pair.remove_hook()
    
    # 4. Test GradCAM with single chain
    gradcam = GradCAM(model, target_layer="conv4")
    saliency_single = gradcam.compute(data_a)
    assert isinstance(saliency_single, np.ndarray)
    assert len(saliency_single) == data_a.num_residues
    gradcam.remove_hooks()
    
    # 5. Test GradCAM with partner chain (no shape mismatch crash)
    gradcam_pair = GradCAM(model, target_layer="conv4")
    saliency_pair = gradcam_pair.compute(data_a, data_b)
    assert isinstance(saliency_pair, np.ndarray)
    assert len(saliency_pair) == data_a.num_residues
    gradcam_pair.remove_hooks()


def test_gnn_encoder_types():
    device = torch.device("cpu")
    data_a = build_residue_graph(PDB_PATH, 'A').to(device)
    data_b = build_residue_graph(PDB_PATH, 'B').to(device)

    for gnn_type in ["gat", "transformer"]:
        for predict_sasa in [True, False]:
            model = ECABSDModel(
                input_dim=23,
                hidden_dim=128,
                num_heads=8,
                dropout=0.0,
                edge_dim=4,
                gnn_type=gnn_type,
                gnn_heads=4,
                num_cross_attn_layers=2,
                predict_sasa=predict_sasa
            ).to(device)

            if predict_sasa:
                logits, sasa_preds, attn = model(data_a, data_b)
                assert logits.shape == (data_a.num_nodes, 1)
                assert sasa_preds.shape == (data_a.num_nodes, 1)
                assert len(attn) == 1
            else:
                probs, labels, attn = model.predict(data_a, data_b)
                assert probs.shape == (data_a.num_nodes, 1)
                assert labels.shape == (data_a.num_nodes,)
                assert len(attn) == 1
                assert attn[0].shape == (data_a.num_nodes, data_b.num_nodes)

