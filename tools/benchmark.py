"""Benchmark SpiralFormer vs vanilla Transformer."""

import time
import torch
import torch.nn as nn
from ..core.model import SpiralFormer

SEQ_LEN = 256
BATCH = 8
VOCAB = 64
DEVICE = "cpu"

def synthetic_batch(batch=BATCH):
    return torch.randint(0, VOCAB, (batch, SEQ_LEN))

def bench(func, name):
    steps = 10
    tokens = synthetic_batch()
    start = time.time()
    with torch.no_grad():
        for s in range(steps):
            _ = func(tokens, s * 0.5)
    dur = (time.time() - start) / steps * 1000
    print(f"{name:18} {dur:6.2f} ms/step")

def main():
    spiral = SpiralFormer(seq_len=SEQ_LEN, vocab_size=VOCAB)

    def vanilla(tokens, t):
        embed = nn.Embedding(VOCAB, 128)
        model = nn.Transformer(d_model=128, nhead=4, num_encoder_layers=4, batch_first=True)
        x = embed(tokens)
        return model(x)

    bench(spiral, "SpiralFormer")
    bench(vanilla, "VanillaTransformer")

if __name__ == "__main__":
    main() 