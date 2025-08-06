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
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

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
        # (In a full version, we would load the JSON scenarios here)
        # For this adaptation, we'll embed simplified versions.
        self.scenarios = self._get_simplified_scenarios()

    def _get_simplified_scenarios(self):
        # A simplified, embedded version of the external JSON files
        return {
            "drought_australia": {
                "name": "Drought-Stressed Eucalyptus Forest",
                "problem_types": {
                    "drought_stress": {"sensor_ranges": {"temperature": (0.7, 1.0), "voltage": (0.2, 0.4)}, "repair_glyphs": [0x17, 0x04], "effectiveness": (0.5, 0.8)},
                    "optimal": {"repair_glyphs": [0x31, 0x32], "effectiveness": (0.1, 0.3)}
                }
            },
            "urban_fiber": {
                "name": "Urban Fiber Network",
                "problem_types": {
                     "thermal_overload": {"sensor_ranges": {"temperature": (0.8, 1.0)}, "repair_glyphs": [0x16, 0x03], "effectiveness": (0.4, 0.8)},
                     "optimal": {"repair_glyphs": [0x31, 0x37], "effectiveness": (0.05, 0.3)}
                }
            }
        }
    
    def generate_training_dataset(self, num_echoes: int = 1000, 
                                output_file: str = "data/mycelial_training_data.jsonl",
                                chaos_mode: bool = False) -> str:
        
        print(f"🌿 Generating {num_echoes} mycelial spore echoes...")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        problem_vs_optimal_ratio = 0.7 if chaos_mode else 0.4
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(num_echoes):
                scenario_name = random.choice(list(self.scenarios.keys()))
                scenario = self.scenarios[scenario_name]
                
                is_problem = random.random() < problem_vs_optimal_ratio
                problem_type_name = "optimal"
                if is_problem:
                    problem_type_name = random.choice([k for k in scenario["problem_types"] if k != "optimal"])

                problem_type = scenario["problem_types"][problem_type_name]
                
                # Generate sensor readings
                conditions = NetworkConditions()
                if "sensor_ranges" in problem_type:
                    for sensor, (min_val, max_val) in problem_type["sensor_ranges"].items():
                        setattr(conditions, sensor, random.uniform(min_val, max_val))
                else: # Optimal
                    conditions.latency = random.uniform(0.0, 0.2)
                    conditions.voltage = random.uniform(0.8, 1.0)
                    conditions.temperature = random.uniform(0.4, 0.6)

                # Generate glyph sequence
                contemplative_glyphs = self.codec.get_contemplative_glyphs()
                if is_problem:
                    primary_glyphs = problem_type["repair_glyphs"]
                    silence_count = random.randint(4, 8)
                    glyph_sequence = primary_glyphs + random.choices(contemplative_glyphs, k=silence_count)
                    random.shuffle(glyph_sequence)
                else:
                    silence_count = random.randint(10, 15)
                    glyph_sequence = random.choices(contemplative_glyphs, k=silence_count)

                # Effectiveness
                effectiveness = random.uniform(*problem_type["effectiveness"])

                sample = {
                    "conditions": conditions.__dict__,
                    "glyph_sequence": glyph_sequence,
                    "effectiveness": effectiveness,
                    "metadata": { "scenario": scenario_name, "is_calm": not is_problem }
                }
                f.write(json.dumps(sample) + '\n')
        
        print(f"✅ Mycelial data saved to {output_path}")
        return str(output_path)

if __name__ == "__main__":
    generator = MycelialDataGenerator(random_seed=42)
    generator.generate_training_dataset(num_echoes=500, output_file="data/mycelial_calm_small.jsonl", chaos_mode=False)
    generator.generate_training_dataset(num_echoes=500, output_file="data/mycelial_chaotic_small.jsonl", chaos_mode=True) 