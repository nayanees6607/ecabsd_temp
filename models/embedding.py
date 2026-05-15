"""
ESM-2 Embedding Extraction Module.

Uses the ESM-2 35M parameter model to extract rich evolutionary and 
structural features (480-dim) for each residue in a protein sequence.
These embeddings are crucial for detecting binding interfaces.
"""

import torch

# Global cache for the ESM model to avoid reloading
_ESM_MODEL = None
_ALPHABET = None
_BATCH_CONVERTER = None
_DEVICE = None

def load_esm_model():
    """Lazily load the ESM-2 model (35M parameters, 480-dim)."""
    global _ESM_MODEL, _ALPHABET, _BATCH_CONVERTER, _DEVICE
    if _ESM_MODEL is None:
        import esm
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ESM] Loading ESM-2 35M model onto {_DEVICE}...")
        _ESM_MODEL, _ALPHABET = esm.pretrained.esm2_t12_35M_UR50D()
        _BATCH_CONVERTER = _ALPHABET.get_batch_converter()
        _ESM_MODEL = _ESM_MODEL.to(_DEVICE)
        _ESM_MODEL.eval()
    return _ESM_MODEL, _BATCH_CONVERTER, _DEVICE

@torch.no_grad()
def get_esm_embedding(sequence: str, chain_id: str = "A") -> torch.Tensor:
    """
    Get multi-layer (6 and 12) ESM-2 embeddings for a single sequence.
    Returns:
        Tensor of shape (L, 960) containing the fused embeddings.
    """
    model, converter, device = load_esm_model()
    
    # Prepare data for ESM
    data = [(chain_id, sequence)]
    _, _, batch_tokens = converter(data)
    batch_tokens = batch_tokens.to(device)

    # Forward pass (extract layer 6 and 12 representations)
    # Layer 6 captures local geometry/secondary structure
    # Layer 12 captures global evolutionary context
    results = model(batch_tokens, repr_layers=[6, 12], return_contacts=False)
    
    # Combine layers: (1, L+2, 480) -> (1, L+2, 960)
    rep6  = results["representations"][6]
    rep12 = results["representations"][12]
    fused = torch.cat([rep6, rep12], dim=-1)

    # Extract only the actual sequence residues (skip CLS/EOS)
    sequence_len = len(sequence)
    embedding = fused[0, 1 : sequence_len + 1]
    
    return embedding.cpu()

if __name__ == "__main__":
    # Test
    seq = "MKVTIKVTEG"
    emb = get_esm_embedding(seq)
    print(f"Sequence: {seq}")
    print(f"Embedding shape: {emb.shape} (Expected: 10, 480)")
