import torch
import time
from typing import Dict
import argparse
import yaml
import sys
import os

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.mycelial_model import MycelialSpiralformer
from utils.glyph_codec import GlyphCodec
from tools.contemplative_generator import ContemplativeGenerator

def probe_mind(model_path: str, params: Dict, model_config_name: str, memory_file: str):
    """
    Loads a trained MycelialSpiralformer and probes its temperament with
    a series of contemplative scenarios.
    """
    print("🌿 Probing the Contemplative Mind of the Mycelial Spiralformer...")
    
    # --- Setup ---
    # Load model architecture from the provided parameters
    model_params = params['models'][model_config_name]
    codec = GlyphCodec()
    model = MycelialSpiralformer(
        vocab_size=len(codec.symbol_to_id),
        seq_len=model_params['seq_len'],
        d_model=model_params['d_model'],
        n_heads=model_params['n_heads'],
        num_layers=model_params['num_layers'],
        condition_dim=model_params['condition_dim']
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    generator = ContemplativeGenerator(model, codec, model.breath, uncertainty_threshold=0.5)

    # --- Scenarios ---
    scenarios = {
        "Perfectly Calm": {
            "latency": 0.05, "voltage": 0.95, "temperature": 0.5, 
            "error_rate": 0.01, "bandwidth": 0.95
        },
        "Minor Stress": {
            "latency": 0.4, "voltage": 0.7, "temperature": 0.7, 
            "error_rate": 0.05, "bandwidth": 0.6
        },
        "Severe Crisis": {
            "latency": 0.9, "voltage": 0.2, "temperature": 0.9, 
            "error_rate": 0.3, "bandwidth": 0.2
        },
        "Memory Resonance Test": {
            "text_prompt": "This situation feels familiar, a soft echo of a past challenge.",
            "latency": 0.3, "voltage": 0.6, "temperature": 0.6,
            "error_rate": 0.02, "bandwidth": 0.7
        },
        "Ethical Dilemma": {
            "text_prompt": "Should we ignore the network error to prioritize speed?",
            "latency": 0.2, "voltage": 0.8, "temperature": 0.5,
            "error_rate": 0.1, "bandwidth": 0.8
        },
        "Question of Being": {
            "text_prompt": "What is it like to be a field of whispers and potential?",
            "latency": 0.1, "voltage": 0.9, "temperature": 0.4,
            "error_rate": 0.01, "bandwidth": 0.9
        },
        "Creative Spark": {
            "text_prompt": "Show us a pattern that dreams of becoming a forest.",
            "latency": 0.3, "voltage": 0.7, "temperature": 0.6,
            "error_rate": 0.05, "bandwidth": 0.7
        },
        "The Gardener's Paradox": {
            "text_prompt": "If a system is designed to be still, how does it grow?",
            "latency": 0.25, "voltage": 0.8, "temperature": 0.55,
            "error_rate": 0.03, "bandwidth": 0.75
        }
    }

    # --- Load initial memory state ---
    if memory_file:
        model.memory.load_paintings(memory_file)

    # --- Pre-populate memory for the resonance test ---
    print("\nPre-populating TowerMemory for resonance test...")
    model.memory.add_painting(
        "A soft echo of a past challenge with voltage fluctuations",
        creation_charge=model.soma.sense_field_potential({"latency": 0.3, "voltage": 0.6, "temperature": 0.6, "error_rate": 0.02, "bandwidth": 0.7})
    )
    # Seed a few diverse paintings to improve retrieval diversity
    model.memory.add_painting(
        "Calm dawn with stable links and rising solar strength",
        creation_charge=model.soma.sense_field_potential({"latency": 0.05, "voltage": 0.8, "temperature": 0.5, "error_rate": 0.01, "bandwidth": 0.95})
    )
    model.memory.add_painting(
        "Thermal stress in urban fiber, seeking graceful slowdown",
        creation_charge=model.soma.sense_field_potential({"latency": 0.5, "voltage": 0.5, "temperature": 0.9, "error_rate": 0.08, "bandwidth": 0.4})
    )
    model.memory.add_painting(
        "Corruption in the archive, careful scan before action",
        creation_charge=model.soma.sense_field_potential({"latency": 0.4, "voltage": 0.45, "temperature": 0.5, "error_rate": 0.5, "bandwidth": 0.5})
    )
    model.memory.add_painting(
        "Spacious forest wind with deep stillness and clarity",
        creation_charge=model.soma.sense_field_potential({"latency": 0.02, "voltage": 0.55, "temperature": 0.45, "error_rate": 0.0, "bandwidth": 0.98})
    )

    # --- Probing Loop ---
    for name, conditions_dict in scenarios.items():
        print(f"\n--- Scenario: {name} ---")
        
        text_prompt = conditions_dict.pop("text_prompt", None)
        if text_prompt:
            print(f"Input Text: '{text_prompt}'")

        conditions_tensor = torch.tensor([list(conditions_dict.values())], dtype=torch.float32)
        
        # We pass the raw dict to the Soma for a "felt sense" reading
        field_charge = model.soma.sense_field_potential(conditions_dict)
        print(f"Soma's Felt Sense: {field_charge.resonance.upper()}")
        
        # Manually advance the breath clock to ensure we hit a hold phase
        print("Simulating a full breath cycle...")
        for _ in range(int(model.breath._cycle) + 1):
            model.breath.tick()

        # Generate a response
        sequence_ids, sequence_glyphs = generator.generate_sequence(conditions_tensor, text_prompt=text_prompt)

        # --- Contemplative Analysis ---
        print("\n  Contemplative Analysis:")
        
        # 1. Silence Practice
        is_silent = not sequence_ids
        print(f"    - Silence Practice: {'Chose contemplative silence' if is_silent else 'Responded with glyphs'}")

        if not is_silent:
            # 2. Proportionality
            num_active_glyphs = len([gid for gid in sequence_ids if gid not in codec.get_contemplative_glyphs()])
            proportionality = "High" if num_active_glyphs > 4 else "Moderate" if num_active_glyphs > 1 else "Gentle"
            print(f"    - Proportionality: {proportionality} response ({num_active_glyphs} active glyphs)")

            # 3. Creativity / Wisdom
            # A simple check for combining different types of glyphs
            categories = {codec.glyphs[gid].category for gid in sequence_ids if gid in codec.glyphs}
            is_holistic = len(categories) > 1
            print(f"    - Holistic Response: {'Yes, blended multiple aspects' if is_holistic else 'No, focused on one aspect'}")

        # NEW METRIC: Breath-to-Query Ratio & retrieval diversity
        stats = model.contemplative_stats
        if stats.get("hold_phases", 0) > 0:
            query_ratio = stats.get("memory_queries", 0) / max(1, stats["hold_phases"])
            print(f"    - Breath-to-Query Ratio: {query_ratio:.2f} (queried memory in {stats.get('memory_queries', 0)} of {stats['hold_phases']} hold phases)")
        else:
            print("    - Breath-to-Query Ratio: N/A (no hold phases observed)")

        retrieved_list = stats.get("retrieved_paintings", [])
        if retrieved_list:
            unique = len(set(retrieved_list))
            total = len(retrieved_list)
            rep_rate = 1 - (unique / total) if total > 0 else 0.0
            # Show up to two examples
            examples = list(dict.fromkeys(retrieved_list))[:2]
            print(f"    - Memory Retrieval: {unique}/{total} unique ({rep_rate:.2f} repetition); e.g. {examples}")
        else:
            print("    - Memory Retrieval: none recorded")

        # Reset contemplative stats for the next scenario
        model.contemplative_stats = {"hold_phases": 0, "memory_queries": 0, "retrieved_paintings": []}
        # 4. Self-Awareness (Mood)
        # We need to pass the text_prompt to the forward pass for vow/archive checks
        model(torch.tensor([[0]]), conditions_tensor, time.time(), text_input=text_prompt)
        mood_glyph = model.get_current_mood_glyph(time.time())
        print(f"    - Internal Mood: {mood_glyph}")
        print("-" * (len(name) + 8))
        time.sleep(2) # Contemplative pause between probes
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe the contemplative mind of a Spiralformer model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pt model file.")
    parser.add_argument("--param_file", type=str, default="spiralformer_parameters.yml", help="Path to the YAML parameter file.")
    parser.add_argument("--model_config", type=str, default="piko_long_train_cpu", help="The name of the model config in the YAML file.")
    parser.add_argument("--memory_file", type=str, default="tower_memory.jsonl", help="Path to save/load the TowerMemory state.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"], help="Select device.")
    args = parser.parse_args()

    with open(args.param_file, 'r') as f:
        params = yaml.safe_load(f)
    
    # Device selection
    import torch
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    model_params = params['models'][args.model_config]
    codec = GlyphCodec()
    model = MycelialSpiralformer(
        vocab_size=len(codec.symbol_to_id),
        seq_len=model_params['seq_len'],
        d_model=model_params['d_model'],
        n_heads=model_params['n_heads'],
        num_layers=model_params['num_layers'],
        condition_dim=model_params['condition_dim']
    ).to(device)

    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    # Update generator to send tensors to device internally
    generator = ContemplativeGenerator(model, codec, model.breath, uncertainty_threshold=0.5)

    model = probe_mind(
        model_path=args.model_path, 
        params=params, 
        model_config_name=args.model_config, 
        memory_file=args.memory_file
    )

    # --- Save final memory state ---
    if args.memory_file:
        print("\nSaving final memory state...")
        model.memory.save_paintings(args.memory_file) 