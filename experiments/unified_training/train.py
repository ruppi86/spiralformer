import yaml
import time
import torch
import torch.nn as nn
import argparse
from pathlib import Path
import sys
import os
import json

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.mycelial_model import MycelialSpiralformer
from utils.breath_clock import BreathClock
from utils.rhythmic_loss import RhythmicLossWrapper
from spiralbase import TowerMemory
from utils.glyph_codec import GlyphCodec


def main(args):
    """
    A unified training script that integrates the core principles of
    the Spiralformer architecture into a single, cohesive training loop.
    """
    with open(args.param_file, 'r') as f:
        params = yaml.safe_load(f)
    
    config_name = args.model_config
    config = params['models'][config_name]
    shared_config = params['shared']

    print(f"🌿 Starting unified training for Spiralformer with config '{config_name}'...")

    # --- Initialization ---
    codec = GlyphCodec()
    breath_params = shared_config['contemplative']['breath_clock']
    clock = BreathClock(
        inhale=breath_params['inhale'],
        hold=breath_params['hold'],
        exhale=breath_params['exhale'],
        pause=breath_params['pause']
    )
    
    model = MycelialSpiralformer(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        seq_len=config['seq_len'],
        num_layers=config['num_layers'],
        vocab_size=len(codec.symbol_to_id),
        condition_dim=config['condition_dim'],
        padding_idx=0
    )
    
    criterion = RhythmicLossWrapper(nn.CrossEntropyLoss(ignore_index=0), clock) # Use ignore_index for padding
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config['training']['learning_rate'])
    memory = TowerMemory(max_painters=shared_config['contemplative']['tower_memory']['max_painters'])

    print(f"Model, optimizer, and TowerMemory initialized. Beginning training for {config['training']['epochs']} epochs.")

    # --- Load Dataset ---
    with open(args.data_file, 'r') as f:
        dataset = [json.loads(line) for line in f]

    # --- Training Loop ---
    for epoch in range(config['training']['epochs']):
        total_loss = 0
        
        for i, sample in enumerate(dataset):
            t_now = time.time()
            phase = clock.phase_at(t_now)

            # --- Data Preparation ---
            conditions = torch.tensor([sample['conditions']], dtype=torch.float32)
            sequence = [codec.decode_glyph(g) for g in sample['glyph_sequence'] if g in codec.symbol_to_id]
            
            # Truncate if sequence is longer than seq_len
            if len(sequence) > config['seq_len']:
                sequence = sequence[:config['seq_len']]

            # Pad sequences to match seq_len
            padded_sequence = sequence + [0] * (config['seq_len'] - len(sequence))
            tokens = torch.tensor([padded_sequence], dtype=torch.long)
            
            # --- Forward pass ---
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]

            logits = model(input_tokens, conditions, t_now, text_input=sample.get('text_input'))
            
            loss = criterion(logits.reshape(-1, model.embed.num_embeddings), target_tokens.reshape(-1), t=t_now)

            # --- Backward pass ---
            if clock.weight_for_phase(phase) > 0 and torch.isfinite(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() if torch.isfinite(loss) else 0.0
            
            if (i + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}, Step {i+1}/{len(dataset)}, Loss: {loss.item():.4f}, Phase: {phase.name}")

        avg_loss = total_loss / len(dataset)
        print(
            f"Epoch {epoch+1:02d}/{config['training']['epochs']} | "
            f"Avg Loss: {avg_loss:.4f}"
        )
        memory.show_tower_state()

    # --- Save the final model ---
    save_path_config = config.get('save_paths', {})
    if save_path_config:
        model_dir = Path(save_path_config['model_dir'])
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / save_path_config['latest_model']
        torch.save(model.state_dict(), model_path)
        print(f"🌱 Model saved to {model_path}")
    else:
        print("🌱 Training complete. No save path configured.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified training script for Mycelial Spiralformer.")
    parser.add_argument("--param_file", type=str, default="spiralformer_parameters.yml", help="Path to the YAML parameter file.")
    parser.add_argument("--model_config", type=str, default="piko_long_train_cpu", help="The name of the model config in the YAML file.")
    parser.add_argument("--data_file", type=str, default="data/mycelial_training_data.jsonl", help="Path to the training data file.")
    args = parser.parse_args()
    main(args) 