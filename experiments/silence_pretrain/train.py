"""Silence-first pretraining demo.

Synthetic dataset: random glyphs with prescribed silence_ratio.
Objective: regress span length until next silence token.
"""

import yaml
import time
import torch
import torch.nn as nn
from pathlib import Path
from ...core.model import SpiralFormer
from ...utils.breath_clock import BreathClock
from ...utils.rhythmic_loss import RhythmicLossWrapper

CFG_PATH = Path(__file__).with_suffix('.yaml')

with open(CFG_PATH, 'r') as fp:
    cfg = yaml.safe_load(fp)

clock = BreathClock()
model = SpiralFormer(seq_len=cfg['seq_len'])
criterion = RhythmicLossWrapper(nn.MSELoss(), clock)
optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

SILENCE = cfg['silence_id']

def make_batch(batch, seq_len):
    # generate sequences respecting silence_ratio
    noise = torch.randint(1, 63, (batch, seq_len))
    silence_mask = torch.rand(batch, seq_len) < cfg['silence_ratio']
    tokens = torch.where(silence_mask, torch.full_like(noise, SILENCE), noise)
    # compute span-to-next-silence target
    target = torch.zeros_like(tokens)
    for i in range(seq_len-2, -1, -1):
        target[:, i] = torch.where(tokens[:, i] == SILENCE, 0, target[:, i+1] + 1)
    return tokens, target.float()

for epoch in range(cfg['epochs']):
    tokens, target = make_batch(cfg['batch'], cfg['seq_len'])
    t_now = time.time() % clock._cycle
    pred = model(tokens, t_now).squeeze(-1)  # assume last dim 1 for regression stub
    loss = criterion(pred, target, t=t_now)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f"epoch {epoch}: loss={loss.item():.4f}  phase={clock.phase_at(t_now).name}") 