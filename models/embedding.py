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
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ESM] Loading ESM-2 35M model onto {_DEVICE}...")
        # esm2_t12_35M_UR50D is fast and provides strong 480-dim embeddings
        _ESM_MODEL, _ALPHABET = torch.hub.load("facebookresearch/esm:main", "esm2_t12_35M_UR50D")
        _BATCH_CONVERTER = _ALPHABET.get_batch_converter()
        _ESM_MODEL = _ESM_MODEL.to(_DEVICE)
        _ESM_MODEL.eval()
    return _ESM_MODEL, _BATCH_CONVERTER, _DEVICE

@torch.no_grad()
def get_esm_embedding(sequence: str, chain_id: str = "A") -> torch.Tensor:
    """
    Get per-residue ESM-2 embeddings for a single sequence.
    Returns:
        Tensor of shape (L, 480) containing the embedding for each residue.
    """
    model, converter, device = load_esm_model()
    
    # Prepare data for ESM
    data = [(chain_id, sequence)]
    _, _, batch_tokens = converter(data)
    batch_tokens = batch_tokens.to(device)

    # Forward pass (extract layer 12 representations)
    results = model(batch_tokens, repr_layers=[12], return_contacts=False)
    token_representations = results["representations"][12]

    # ESM adds <cls> and <eos> tokens at start and end. 
    # Extract only the actual sequence residues: [1 : L + 1]
    sequence_len = len(sequence)
    embedding = token_representations[0, 1 : sequence_len + 1]
    
    # Move to CPU to free up GPU memory during dataset preprocessing
    return embedding.cpu()

if __name__ == "__main__":
    # Test
    seq = "MKVTIKVTEG"
    emb = get_esm_embedding(seq)
    print(f"Sequence: {seq}")
    print(f"Embedding shape: {emb.shape} (Expected: 10, 480)")
