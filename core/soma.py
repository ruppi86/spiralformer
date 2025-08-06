"""
soma.py - The Listening Flesh for the Mycelial Spiralformer

This module adapts the Soma concept from the ContemplativeAI prototype.
It provides a pre-attentive sensory membrane that translates the quantitative
NetworkConditions of the mycelial ecosystem into a qualitative "FieldCharge."

This allows the Spiralformer to have a "felt sense" of its environment
before engaging its deeper cognitive functions.
"""

from typing import Dict
from enum import Enum
from dataclasses import dataclass
import numpy as np

class FieldChargeType(Enum):
    """Types of atmospheric charge Soma can sense"""
    EMOTIONAL_PRESSURE = "emotional_pressure"
    TEMPORAL_URGENCY = "temporal_urgency"
    RELATIONAL_INTENT = "relational_intent" # Placeholder for now
    PRESENCE_DENSITY = "presence_density"   # Placeholder for now
    BEAUTY_RESONANCE = "beauty_resonance" # Placeholder for now

@dataclass
class FieldCharge:
    """The atmospheric charge sensed around an interaction or environmental state."""
    emotional_pressure: float    # 0.0 (light) to 1.0 (heavy)
    temporal_urgency: float      # 0.0 (spacious) to 1.0 (rushing)
    
    def crosses_threshold(self, sensitivity: float = 0.7) -> bool:
        """
        Does this charge warrant waking the deeper systems?
        A simple check: if the environment is either very urgent or very heavy,
        it crosses the threshold.
        """
        return self.emotional_pressure > sensitivity or self.temporal_urgency > sensitivity

    @property
    def resonance(self) -> str:
        """Describe the quality of this field charge."""
        if self.temporal_urgency > 0.8:
            return "urgent"
        if self.emotional_pressure > 0.8:
            return "intense"
        if self.temporal_urgency < 0.2 and self.emotional_pressure < 0.2:
            return "spacious"
        return "neutral"

class Soma:
    """
    The Listening Flesh - a pre-attentive sensing membrane that translates
    NetworkConditions into a qualitative FieldCharge.
    """
    def __init__(self, sensitivity: float = 0.7):
        self.sensitivity = sensitivity

    def sense_field_potential(self, conditions: Dict[str, float]) -> FieldCharge:
        """
        Feels the atmospheric pressure of the environment from its conditions.
        This is a heuristic-based translation from quantitative data to felt sense.
        """
        # Temporal Urgency is linked to high latency and low bandwidth
        latency = conditions.get('latency', 0.1)
        bandwidth = conditions.get('bandwidth', 0.8)
        temporal_urgency = np.clip((latency * 0.7) + ((1 - bandwidth) * 0.3), 0, 1)

        # Emotional Pressure is linked to high error rates, low voltage, and extreme temps
        error_rate = conditions.get('error_rate', 0.02)
        voltage = conditions.get('voltage', 0.5)
        temperature = conditions.get('temperature', 0.5)
        
        voltage_stress = 1 - (abs(voltage - 0.5) * 2) # Stress increases as voltage deviates from 0.5
        temp_stress = abs(temperature - 0.5) * 2 # Stress increases at extremes
        
        emotional_pressure = np.clip((error_rate * 0.5) + (voltage_stress * 0.25) + (temp_stress * 0.25), 0, 1)

        return FieldCharge(
            emotional_pressure=float(emotional_pressure),
            temporal_urgency=float(temporal_urgency)
        ) 