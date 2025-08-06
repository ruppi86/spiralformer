import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import time

from core.mycelial_model import MycelialSpiralformer
from utils.glyph_codec import GlyphCodec

class ContemplativeGenerator:
    """
    A tool for generating glyph sequences from a trained MycelialSpiralformer,
    embodying the principle of "Adaptive Uncertainty Threshold."
    
    When the model is uncertain, it chooses a vow of silence.
    """
    def __init__(self, model: MycelialSpiralformer, codec: GlyphCodec, uncertainty_threshold: float = 0.7):
        self.model = model
        self.model.eval() # Set model to evaluation mode
        self.codec = codec
        self.uncertainty_threshold = uncertainty_threshold
        self.silence_id = self.codec.decode_glyph("…") or 0 # Default to a known silence glyph

    def generate_step(self, tokens: torch.Tensor, conditions: torch.Tensor, t: float) -> int:
        """
        Generates a single next glyph, respecting the model's uncertainty.
        """
        with torch.no_grad():
            logits = self.model(tokens, conditions, t)
            # We only care about the very last token for the next prediction
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            
            # Calculate entropy as a measure of uncertainty
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

            # If entropy is high (uncertainty), choose silence.
            if entropy.item() > self.uncertainty_threshold:
                print(" contemplative silence (high uncertainty)...")
                # The model can also request an extended pause here in a future version.
                # self.model.breath.request_extended_pause(duration_multiplier=1.5)
                return self.silence_id

            # Otherwise, sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1).item()
            return next_token

    def generate_sequence(self, conditions: torch.Tensor, max_len: int = 20):
        """Generates a full sequence of glyphs for a given condition."""
        print(f"🌿 Generating sequence for condition... Resonance: {self.model.soma.sense_field_potential(self.model._tensor_to_dict(conditions)).resonance}")
        
        # Start with a silence token as the initial context
        tokens = torch.tensor([[self.silence_id]], dtype=torch.long)
        
        sequence_ids = []
        for i in range(max_len):
            t = time.time()
            next_token_id = self.generate_step(tokens, conditions, t)
            
            # Stop if a deep silence or end token is generated
            if next_token_id == self.silence_id:
                break
                
            sequence_ids.append(next_token_id)
            
            # Add the new token to the context for the next step (autoregression)
            tokens = torch.cat([tokens, torch.tensor([[next_token_id]], dtype=torch.long)], dim=1)

        print("Generated Glyph Sequence:")
        print(self.codec.format_glyph_sequence(sequence_ids))
        print("-" * 20)
        return sequence_ids, self.codec.format_glyph_sequence(sequence_ids)

# This is a placeholder for the actual main demo script
if __name__ == '__main__':
    # This is a conceptual demo and will not run without a trained model
    # and proper data loading.
    print("This script is a tool to be used with a trained Spiralformer.")
    print("It demonstrates the 'Adaptive Uncertainty Threshold' principle.") 