"""
Mycelial Data Generator for Spiralformer Training

Generates spore echoes based on ecological and abstract scenarios,
adapting the powerful data generation from the 'spiramycel' project.

This script pre-generates training data to a JSONL file, which can
then be used to train the MycelialSpiralformer.
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import argparse
import sys, os

# Ensure project root is on sys.path for reliable imports when run as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assuming glyph_codec is in utils, adjacent to this data directory's parent
try:
    from utils.glyph_codec import GlyphCodec
except ImportError:
    print("Warning: GlyphCodec not found. Using placeholder glyphs.")
    class GlyphCodec:
        def get_contemplative_glyphs(self):
            return list(range(0x31, 0x41))

@dataclass
class NetworkConditions:
    """Represents current network conditions for spore echo generation"""
    latency: float = 0.1
    voltage: float = 0.5
    temperature: float = 0.5
    error_rate: float = 0.02
    bandwidth: float = 0.8
    
    def to_condition_vector(self) -> List[float]:
        return [self.latency, self.voltage, self.temperature, self.error_rate, self.bandwidth]

class MycelialDataGenerator:
    """Generates realistic ecological and abstract training data for Spiralformer."""

    def __init__(self, random_seed: Optional[int] = None):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        self.codec = GlyphCodec()
        self.scenarios = self._get_simplified_scenarios()

    def _get_simplified_scenarios(self):
        return {
            "drought_australia": {
                "name": "Drought-Stressed Eucalyptus Forest",
                "problem_types": {
                    "drought_stress": {"sensor_ranges": {"temperature": (0.7, 1.0), "voltage": (0.2, 0.4)}, "repair_glyphs": ["⚡15", "💧08"], "effectiveness": (0.5, 0.8)},
                    "optimal": {"repair_glyphs": ["⭐47", "🌗28"], "effectiveness": (0.1, 0.3)}
                }
            },
            "urban_fiber": {
                "name": "Urban Fiber Network",
                "problem_types": {
                     "thermal_overload": {"sensor_ranges": {"temperature": (0.8, 1.0)}, "repair_glyphs": ["💫52", "🍄33"], "effectiveness": (0.4, 0.8)},
                     "optimal": {"repair_glyphs": ["⭐47", "🧭37"], "effectiveness": (0.05, 0.3)}
                }
            },
            "data_corruption": {
                "name": "Data Corruption Scenario",
                "problem_types": {
                    "corruption": {"sensor_ranges": {"error_rate": (0.5, 0.8)}, "repair_glyphs": ["🗄️", "🔍", "💾"], "effectiveness": (0.6, 0.9)},
                    "optimal": {"repair_glyphs": ["…"], "effectiveness": (0.1, 0.2)}
                }
            },
            # Multiple wisdom templates (codec-valid symbols)
            "wisdom_1": {
                "name": "Wisdom Question",
                "question": "If a system is designed to be still, how does it grow?",
                "answer_glyphs": ["🌱regen", "☀️pwr", "💧08", "…", "🌲44"],
                "description": "A philosophical question requiring a thoughtful, metaphorical answer."
            },
            "wisdom_2": {
                "name": "Wisdom Question",
                "question": "How does a forest answer a storm?",
                "answer_glyphs": ["🌊sil", "🌲44", "💧08", "…", "🪷"],
                "description": "Metaphor: resilience, nourishment, calm."
            },
            "wisdom_3": {
                "name": "Wisdom Question",
                "question": "What is wisdom when speed is demanded?",
                "answer_glyphs": ["⭕", "🧭37", "☀️pwr", "…", "🔆19"],
                "description": "Metaphor: orient first, then act with clarity."
            },
            "wisdom_4": {
                "name": "Wisdom Question",
                "question": "When memory awakens, what should be said?",
                "answer_glyphs": ["🕯️", "🌸sil", "🧬55", "…", "🕊️"],
                "description": "Metaphor: light, beauty, adaptation, peace."
            }
        }
    
    def generate_training_dataset(self, num_echoes: int = 1000, 
                                output_file: str = "data/mycelial_training_data.jsonl",
                                chaos_mode: bool = False):
        
        print(f"🌿 Generating {num_echoes} mycelial spore echoes...")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        problem_vs_optimal_ratio = 0.7 if chaos_mode else 0.4
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(num_echoes):
                scenario_name = random.choice(list(self.scenarios.keys()))
                scenario = self.scenarios[scenario_name]
                text_input = None
                
                is_problem = random.random() < problem_vs_optimal_ratio
                
                if scenario_name.startswith("wisdom"):
                    is_problem = False
                    problem_type_name = scenario_name
                    glyph_sequence = scenario["answer_glyphs"]
                    text_input = scenario["question"]
                elif "problem_types" in scenario:
                    problem_type_name = "optimal"
                    if is_problem:
                        problem_type_name = random.choice([k for k in scenario["problem_types"] if k != "optimal"])
                    problem_type = scenario["problem_types"][problem_type_name]
                    
                    contemplative_glyphs = self.codec.get_contemplative_glyphs()
                    primary_glyphs = problem_type["repair_glyphs"]
                    silence_count = random.randint(4, 8)
                    glyph_sequence = primary_glyphs + random.choices(contemplative_glyphs, k=silence_count)
                    random.shuffle(glyph_sequence)
                else:
                    # Default calm scenario
                    contemplative_glyphs = self.codec.get_contemplative_glyphs()
                    silence_count = random.randint(10, 15)
                    glyph_sequence = random.choices(contemplative_glyphs, k=silence_count)

                conditions = NetworkConditions()
                if "problem_types" in scenario and "sensor_ranges" in scenario["problem_types"].get(problem_type_name, {}):
                    for sensor, (min_val, max_val) in scenario["problem_types"][problem_type_name]["sensor_ranges"].items():
                        setattr(conditions, sensor, random.uniform(min_val, max_val))
                
                effectiveness = 0.5 # Default
                if "problem_types" in scenario and "effectiveness" in scenario["problem_types"].get(problem_type_name, {}):
                    effectiveness = random.uniform(*scenario["problem_types"][problem_type_name]["effectiveness"])

                sample = {
                    "conditions": conditions.to_condition_vector(),
                    "glyph_sequence": glyph_sequence,
                    "effectiveness": effectiveness,
                    "metadata": { "scenario": scenario_name, "problem_type": problem_type_name },
                    "text_input": text_input
                }
                f.write(json.dumps(sample) + '\n')
        
        print(f"✅ Mycelial data saved to {output_path}")
        return str(output_path)

def main():
    parser = argparse.ArgumentParser(description="Generate training data for Mycelial Spiralformer.")
    parser.add_argument("--num_echoes", type=int, default=1000, help="Number of training samples to generate.")
    parser.add_argument("--output_file", type=str, default="data/mycelial_training_data.jsonl", help="Path to the output JSONL file.")
    parser.add_argument("--chaos_mode", action='store_true', help="Generate more chaotic and problem-focused data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    generator = MycelialDataGenerator(random_seed=args.seed)
    generator.generate_training_dataset(
        num_echoes=args.num_echoes,
        output_file=args.output_file,
        chaos_mode=args.chaos_mode
    )

if __name__ == "__main__":
    main()