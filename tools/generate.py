"""Phase-aligned glyph generation demo.

During inhale → generate, exhale → low-temperature generate, pause → output silence token.
"""

import time
import torch
from ..core.model import SpiralFormer
from ..utils.breath_clock import BreathClock

VOCAB = 64
SILENCE_ID = 0

model = SpiralFormer(vocab_size=VOCAB)
clock = BreathClock()

def step(prev_tokens: torch.Tensor, t: float):
    phase = clock.phase_at(t)
    if phase.name == "pause":
        # emit silence glyph
        return torch.full_like(prev_tokens[:, :1], SILENCE_ID)
    logits = model(prev_tokens, t)
    # simple argmax for demo
    next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
    if phase.name == "exhale":
        # soften creativity by random drop
        mask = torch.rand_like(next_tok.float()) < 0.5
        next_tok = torch.where(mask, next_tok, torch.full_like(next_tok, SILENCE_ID))
    return next_tok

if __name__ == "__main__":
    B, L = 1, 10
    tokens = torch.randint(0, VOCAB, (B, L))
    for step_idx in range(20):
        t = step_idx * 1.0  # seconds
        next_tok = step(tokens, t)
        tokens = torch.cat([tokens, next_tok], dim=1)
        print(f"t={t:4.1f}s  phase={clock.phase_at(t).name:6}  token={next_tok.item()}") 