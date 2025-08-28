import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import time

from core.mycelial_model import MycelialSpiralformer
from utils.glyph_codec import GlyphCodec
from utils.breath_clock import BreathClock

class ContemplativeGenerator:
    """
    A tool for generating glyph sequences from a trained MycelialSpiralformer,
    embodying the principle of "Adaptive Uncertainty Threshold."
    
    When the model is uncertain, it chooses a vow of silence.
    """
    def __init__(
        self,
        model: MycelialSpiralformer,
        codec: GlyphCodec,
        clock: BreathClock,
        uncertainty_threshold: float = 0.7,
        temperature: float = 1.0,
        max_silence_run: int = 2,
        silence_penalty: float = 0.0,
        min_active_tokens_urgent: int = 2,
    ):
        self.model = model
        self.model.eval() # Set model to evaluation mode
        self.codec = codec
        self.clock = clock
        self.uncertainty_threshold = uncertainty_threshold
        self.temperature = max(1e-5, float(temperature))
        self.max_silence_run = max(1, int(max_silence_run))
        self.silence_penalty = float(silence_penalty)
        self.min_active_tokens_urgent = max(0, int(min_active_tokens_urgent))
        self.silence_id = self.codec.decode_glyph("…") or 0 # Default to a known silence glyph
        self.silence_ids = set(self.codec.get_contemplative_glyphs())
        # Counter for uncertainty-driven silences
        self.uncertainty_silence_count = 0
        # Resolve model device for tensor placement
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

    def generate_step(self, tokens: torch.Tensor, conditions: torch.Tensor, t: float) -> int:
        """
        Generates a single next glyph, respecting the model's uncertainty.
        """
        with torch.no_grad():
            logits = self.model(tokens, conditions, t)
            # We only care about the very last token for the next prediction
            last_logits = logits[:, -1, :]
            # Apply temperature to soften/harden distribution
            last_logits = last_logits / self.temperature
            probs = F.softmax(last_logits, dim=-1)
            
            # Calculate entropy as a measure of uncertainty
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

            # If entropy is high (uncertainty), choose silence.
            if entropy.item() > self.uncertainty_threshold:
                print(" contemplative silence (high uncertainty)...")
                self.uncertainty_silence_count += 1
                # The model can also request an extended pause here in a future version.
                # self.model.breath.request_extended_pause(duration_multiplier=1.5)
                return self.silence_id

            # Otherwise, sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1).item()
            return next_token

    def generate_sequence(self, conditions: torch.Tensor, text_prompt: Optional[str] = None, max_len: int = 20):
        """Generates a full sequence of glyphs for a given condition."""
        # Ensure conditions on correct device
        if conditions.device != getattr(self, 'device', torch.device('cpu')):
            conditions = conditions.to(self.device)
        # Ensure conditions on correct device
        if conditions.device != getattr(self, 'device', torch.device('cpu')):
            conditions = conditions.to(self.device)
        felt = self.model.soma.sense_field_potential(self.model._tensor_to_dict(conditions))
        print(f"🌿 Generating sequence for condition... Resonance: {felt.resonance}")
        
        # Start with a silence token as the initial context
        tokens = torch.tensor([[self.silence_id]], dtype=torch.long, device=self.device)
        
        sequence_ids = []
        silence_run = 0
        # Urgency-aware policy overrides
        eff_temp = self.temperature
        eff_thr = self.uncertainty_threshold
        eff_penalty = self.silence_penalty
        active_needed = 0
        if getattr(felt, 'resonance', '').lower() == 'urgent':
            eff_temp = max(eff_temp, 1.5)
            eff_thr = max(eff_thr, 1.5)
            eff_penalty = max(eff_penalty, 2.0)
            active_needed = self.min_active_tokens_urgent
        for i in range(max_len):
            t = time.time()
            self.clock.tick()
            # Pass the text_prompt to the forward pass of the model
            with torch.no_grad():
                logits = self.model(tokens, conditions, t, text_input=text_prompt)
            
            force_active = active_needed > 0
            next_token_id = self.generate_step_from_logits(
                logits,
                t,
                temperature=eff_temp,
                uncertainty_threshold=eff_thr,
                silence_penalty=(eff_penalty if force_active else 0.0),
                force_active=force_active,
            )
            
            # Stop if the model settles into deep, consecutive silence
            if next_token_id == self.silence_id:
                silence_run += 1
            else:
                silence_run = 0
                # Count as active if not a contemplative glyph
                if next_token_id not in self.silence_ids:
                    if active_needed > 0:
                        active_needed -= 1
            if silence_run >= self.max_silence_run:
                print("...ending on deep silence.")
                break
                
            sequence_ids.append(next_token_id)
            
            # Add the new token to the context for the next step (autoregression)
            tokens = torch.cat([tokens, torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)], dim=1)

        print("Generated Glyph Sequence:")
        print(self.codec.format_glyph_sequence(sequence_ids))
        print("-" * 20)
        return sequence_ids, self.codec.format_glyph_sequence(sequence_ids)

    def generate_step_from_logits(
        self,
        logits: torch.Tensor,
        t: float,
        temperature: float = 1.0,
        uncertainty_threshold: float = 0.7,
        silence_penalty: float = 0.0,
        force_active: bool = False,
    ) -> int:
        """
        Generates a single next glyph from logits, respecting uncertainty.
        This is separated to allow calling the model with a text_prompt.
        """
        last_logits = logits[:, -1, :]
        last_logits = last_logits / max(1e-5, float(temperature))
        if silence_penalty > 0.0:
            # Subtract penalty from all contemplative glyph logits (encourage speech)
            idx = list(self.silence_ids)
            last_logits[:, idx] = last_logits[:, idx] - float(silence_penalty)
        probs = F.softmax(last_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        if not force_active and entropy.item() > float(uncertainty_threshold):
            print(" contemplative silence (high uncertainty)...")
            self.uncertainty_silence_count += 1
            return self.silence_id

        next_token = torch.multinomial(probs, num_samples=1).item()
        return next_token

    def reset_uncertainty_counter(self):
        """Resets the counter for the next scenario."""
        self.uncertainty_silence_count = 0

# This is a placeholder for the actual main demo script
if __name__ == '__main__':
    # This is a conceptual demo and will not run without a trained model
    # and proper data loading.
    print("This script is a tool to be used with a trained Spiralformer.")
    print("It demonstrates the 'Adaptive Uncertainty Threshold' principle.") 