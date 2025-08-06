import torch
import time
from typing import Dict
import argparse

from core.mycelial_model import MycelialSpiralformer
from utils.glyph_codec import GlyphCodec
from tools.contemplative_generator import ContemplativeGenerator

def probe_mind(model_path: str):
    """
    Loads a trained MycelialSpiralformer and probes its temperament with
    a series of contemplative scenarios.
    """
    print("🌿 Probing the Contemplative Mind of the Mycelial Spiralformer...")
    
    # --- Setup ---
    # In a real scenario, we would load the specific config used for the model.
    # For now, we initialize a model with the same architecture.
    model = MycelialSpiralformer(vocab_size=68, seq_len=33)
    # model.load_state_dict(torch.load(model_path)) # This would be used in a real run
    
    codec = GlyphCodec()
    generator = ContemplativeGenerator(model, codec, uncertainty_threshold=1.0) # Lowered from 1.5

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
        }
    }

    # --- Probing Loop ---
    for name, conditions_dict in scenarios.items():
        print(f"\n--- Scenario: {name} ---")
        conditions_tensor = torch.tensor([list(conditions_dict.values())], dtype=torch.float32)
        
        # We pass the raw dict to the Soma for a "felt sense" reading
        field_charge = model.soma.sense_field_potential(conditions_dict)
        print(f"Soma's Felt Sense: {field_charge.resonance.upper()}")
        
        # Generate a response
        sequence_ids, sequence_glyphs = generator.generate_sequence(conditions_tensor)

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

        # NEW METRIC: Breath-to-Query Ratio
        stats = model.contemplative_stats
        if stats["hold_phases"] > 0:
            query_ratio = stats["memory_queries"] / stats["hold_phases"]
            print(f"    - Breath-to-Query Ratio: {query_ratio:.2f} (queried memory in {stats['memory_queries']} of {stats['hold_phases']} hold phases)")
        else:
            print("    - Breath-to-Query Ratio: N/A (no hold phases observed)")
        # Reset contemplative stats for the next scenario
        model.contemplative_stats = {"hold_phases": 0, "memory_queries": 0}
        # 4. Self-Awareness (Mood)
        mood_glyph = model.get_current_mood_glyph(time.time())
        print(f"    - Internal Mood: {mood_glyph}")
        print("-" * (len(name) + 8))
        time.sleep(2) # Contemplative pause between probes

if __name__ == "__main__":
    # This is a conceptual demonstration. To run it, we would need to pass
    # the path to a saved model file.
    # For now, it runs with an untrained model to show the structure.
    # Example usage: probe_mind("path/to/my_model.pt", "path/to/config.yml")
    parser = argparse.ArgumentParser(description="Probe the contemplative mind of a Spiralformer model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pt model file.")
    args = parser.parse_args()
    
    probe_mind(model_path=args.model_path) 