"""Demo: Trigger a breath-synchronized, single-step LoRA update on resonance.

Usage:
  python experiments/online_learning/demo.py --config piko_mycelial_cpu
"""
import argparse
import time
from pathlib import Path
import os
import sys

import torch
import yaml

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.mycelial_model import MycelialSpiralformer
from utils.glyph_codec import GlyphCodec
from utils.breath_clock import BreathClock
from experiments.online_learning.online_learner import OnlineLearner, LearningTrigger, LearningEvent


def make_sample(codec: GlyphCodec, seq_len: int = 32, device: torch.device = torch.device("cpu")):
    # Start with silence and a few active glyphs to create a target
    silence_id = codec.decode_glyph("…") or 0
    context = torch.full((1, seq_len - 1), silence_id, dtype=torch.long, device=device)
    # Insert a couple of active glyphs near the end so next token is learnable
    if seq_len >= 6:
        context[0, -3] = max(1, silence_id - 1)
        context[0, -2] = max(2, silence_id - 2)
    target = torch.roll(context, shifts=-1, dims=1)
    # Dummy conditions: induce moderate urgency to increase chance of resonance
    conditions = torch.tensor([[0.8, 0.3, 0.7, 0.2, 0.4]], dtype=torch.float32, device=device)
    return context, target, conditions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_file", type=str, default="spiralformer_parameters.yml")
    parser.add_argument("--config", type=str, default="piko_mycelial_cpu")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    params = yaml.safe_load(open(args.param_file, "r"))
    cfg = params["models"][args.config]
    shared = params["shared"]
    lora_cfg = cfg.get("lora", shared.get("lora", {}))

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu"))

    codec = GlyphCodec()
    model = MycelialSpiralformer(
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        seq_len=cfg["seq_len"],
        num_layers=cfg["num_layers"],
        vocab_size=len(codec.symbol_to_id),
        condition_dim=cfg["condition_dim"],
        lora_config=lora_cfg,
    ).to(device)

    clock = BreathClock(
        inhale=shared["contemplative"]["breath_clock"]["inhale"],
        hold=shared["contemplative"]["breath_clock"]["hold"],
        exhale=shared["contemplative"]["breath_clock"]["exhale"],
        pause=shared["contemplative"]["breath_clock"]["pause"],
    )

    # Make a resonant event
    context, target, conditions = make_sample(codec, seq_len=cfg["seq_len"], device=device)

    # Get logits to evaluate uncertainty
    t0 = time.time()
    with torch.no_grad():
        logits = model(context, conditions, t0)

    trigger = LearningTrigger(entropy_threshold=1.0, memory_query_threshold=1)
    should = trigger.should_learn(model, logits)
    print(f"Resonance check → entropy/memory trigger = {should}")

    # Force time to inhale boundary for the single step, then run learner
    # For demo simplicity we just reuse current time; BreathClock is stateless to t.
    learner = OnlineLearner(model, clock, learning_rate=5e-4, max_grad_norm=1.0, device=device)
    event = LearningEvent(
        text_input=None,
        conditions=conditions,
        context_tokens=context,
        target_tokens=target,
        t=t0,
    )

    phase = clock.phase_at(event.t)
    print(f"Breath phase at t0: {phase.name}")
    if phase.name != "inhale":
        # Nudge time forward until inhale
        # Compute rough phase advance; we simply increment time in small steps
        t_search = event.t
        for _ in range(100):
            t_search += 0.25
            if clock.phase_at(t_search).name == "inhale":
                event.t = t_search
                break
        print(f"Adjusted t to inhale at phase: {clock.phase_at(event.t).name}")

    stats = {"updated": False}
    if should:
        stats = learner.single_step(event)
    print(f"Learning step stats: {stats}")

    # Sanity forward after update
    with torch.no_grad():
        logits2 = model(context, conditions, time.time())
    print(f"Logits delta (L2 norm) after step: {float(torch.norm(logits2 - logits)):.6f}")


if __name__ == "__main__":
    main()


