"""Spore-level LoRA fine-tuning demo with breath-synchronised rank.

Loads a pre-trained SpiralFormer (random for now) and trains LoRA adapters whose
rank varies with BreathClock phase.
"""
import time
import torch
from ...core.model import SpiralFormer
from ...utils.breath_clock import BreathClock
from ...utils.lora import attach_lora

VOCAB = 64
clock = BreathClock()
model = SpiralFormer(vocab_size=VOCAB)
# attach LoRA adapters to all linear layers for demo
lora_map = attach_lora(model, r=4)
optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
SILENCE = 0


def sample(batch=32, seq_len=256):
    noise = torch.randint(1, VOCAB, (batch, seq_len))
    mask = torch.rand(batch, seq_len) < 0.1
    return torch.where(mask, torch.full_like(noise, SILENCE), noise)

for step in range(100):
    tokens = sample()
    t_now = time.time() % clock._cycle
    # schedule LoRA rank according to phase
    phase = clock.phase_at(t_now)
    rank_map = {"inhale": 8, "hold": 4, "exhale": 2, "pause": 0}
    for _, lora in lora_map.values():
        lora.set_rank(rank_map[phase.name])
    out = model(tokens, t_now)
    loss = out.mean()  # dummy loss
    optim.zero_grad(); loss.backward(); optim.step()
    if step % 20 == 0:
        print(f"step {step} phase={phase.name} rank={rank_map[phase.name]} loss={loss.item():.4f}") 