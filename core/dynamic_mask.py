import torch
from .spiral_attention import build_spiral_attention_mask

SILENCE_TOKEN_ID = 0

def build_glyph_conditioned_mask(tokens: torch.Tensor, base_mask: torch.Tensor) -> torch.Tensor:
    B, L = tokens.shape
    
    # We are creating a 3D mask, but the error indicates a 2D mask is sometimes expected.
    # The MultiheadAttention layer can accept a 2D mask which is then broadcast across the batch.
    # Let's create a single, combined mask from the batch. A simple approach is to
    # take the union of all silence positions across the batch.
    
    # Create a single 2D mask of shape (L, L)
    final_mask = base_mask.clone()

    # Find all positions that have a silence token anywhere in the batch
    is_silent = tokens == SILENCE_TOKEN_ID
    batch_wide_silent_indices = is_silent.any(dim=0).nonzero(as_tuple=True)[0]
    
    if batch_wide_silent_indices.numel() > 0:
        final_mask[batch_wide_silent_indices, :] = False
        final_mask[:, batch_wide_silent_indices] = False
    
    return final_mask 