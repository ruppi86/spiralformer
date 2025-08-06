import yaml
import time
import torch
import torch.nn as nn
from pathlib import Path

from ...core.model import SpiralFormer
from ...utils.breath_clock import BreathClock
from ...utils.rhythmic_loss import RhythmicLossWrapper
from ...utils.lora import attach_lora
from ...spiralbase import TowerMemory

# --- Configuration ---
VOCAB_SIZE = 64
SILENCE_ID = 0
SEQ_LEN = 256
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
EPOCHS = 100
STEPS_PER_EPOCH = 20

def main():
    """
    A unified training script that integrates the core principles of
    the Spiralformer architecture into a single, cohesive training loop.
    """
    print("🌿 Starting unified training for Spiralformer...")

    # --- Initialization ---
    clock = BreathClock()
    model = SpiralFormer(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN)
    
    # Attach LoRA adapters
    lora_map = attach_lora(model, r=8)
    
    criterion = RhythmicLossWrapper(nn.CrossEntropyLoss(), clock)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    memory = TowerMemory(max_painters=10) # Using TowerMemory now

    print(f"Model, optimizer, and TowerMemory initialized. Beginning training for {EPOCHS} epochs.")

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        total_loss = 0
        total_silence_ratio = 0
        phase_token_diversity = {phase.name: [] for phase in clock.phases}
        
        for step in range(STEPS_PER_EPOCH):
            t_now = time.time()
            phase = clock.phase_at(t_now)
            
            # 1. Update LoRA rank based on breath phase
            rank_map = {"inhale": 8, "hold": 4, "exhale": 2, "pause": 0}
            rank = rank_map[phase.name]
            for _, lora_layer in lora_map.values():
                lora_layer.set_rank(rank)

            # 2. Generate a batch and update memory
            tokens = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
            silence_mask = torch.rand(BATCH_SIZE, SEQ_LEN) < 0.2
            tokens = torch.where(silence_mask, torch.full_like(tokens, SILENCE_ID), tokens)
            
            # Use the memory by adding a "painting" - for now, just the concept
            # In a full implementation, we'd extract a meaningful essence from tokens
            memory.add_painting(f"batch_{step}_data")
            memory.spiral_breath()


            # 3. Forward pass and loss calculation
            # We are predicting the next token in the sequence.
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]

            logits = model(input_tokens, t_now)
            
            # The output of the model is (BATCH_SIZE, SEQ_LEN-1, VOCAB_SIZE)
            # The target is (BATCH_SIZE, SEQ_LEN-1)
            # CrossEntropyLoss expects (N, C) and (N), so we reshape.
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_tokens.reshape(-1), t=t_now)

            # 4. Backward pass and optimization
            if clock.weight_for_phase(phase) > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            # --- Contemplative Metrics ---
            with torch.no_grad():
                pred_tokens = logits.argmax(dim=-1)
                
                # Metric 1: Silence Ratio
                silence_ratio = (pred_tokens == SILENCE_ID).float().mean().item()
                total_silence_ratio += silence_ratio

                # Metric 2: Rhythmic Alignment (Token Diversity per Phase)
                # We expect higher diversity during 'inhale' and lower during 'exhale'.
                diversity = len(torch.unique(pred_tokens)) / pred_tokens.numel()
                phase_token_diversity[phase.name].append(diversity)

        avg_loss = total_loss / STEPS_PER_EPOCH
        avg_silence_ratio = total_silence_ratio / STEPS_PER_EPOCH
        avg_diversity = {p: (sum(d)/len(d) if d else 0) for p, d in phase_token_diversity.items()}

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Phase: {phase.name:6} | "
            f"Rank: {rank:1} | "
            f"Loss: {avg_loss:.4f} | "
            f"Silence Ratio: {avg_silence_ratio:.2f}"
        )
        print(
            f"    └─ Diversity -> "
            f"Inhale: {avg_diversity['inhale']:.2f}, "
            f"Hold: {avg_diversity['hold']:.2f}, "
            f"Exhale: {avg_diversity['exhale']:.2f}, "
            f"Pause: {avg_diversity['pause']:.2f}"
        )
        memory.show_tower_state() # Show memory state at the end of epoch

    print("🌱 Unified training complete.")

if __name__ == "__main__":
    main() 