"""
painting_box.py - The Tower Memory Prototype

Each painter in the tower tends a PaintingBox - a living memory that responds
to cultural breath and knows its own readiness for transformation.

This is the first brushstroke of our tower prototype.
"""

import time
import random
from typing import Dict, List, Any, Optional
from core.soma import FieldCharge


class PaintingBox:
    """
    A single painter's box containing a painting that fades unless touched by cultural breath.
    
    Embodies the ethics of memory migration: knowing when to hold, when to transform, 
    when to pass down with dignity.
    """
    
    def __init__(self, content: str, interpretations: List[str] = None, creation_charge: FieldCharge = None):
        self.content = content
        self.original_content = content
        self.interpretations = interpretations or []
        self.clarity = 1.0  # How clear/vivid the memory is (0.0 to 1.0)
        self.humidity_level = 0.7  # Moisture that keeps meaning pliable
        self.creation_charge = creation_charge
        # Minimal resonance signature (extendable)
        if creation_charge is not None:
            self.signature = {
                "emotional_pressure": float(creation_charge.emotional_pressure),
                "temporal_urgency": float(creation_charge.temporal_urgency),
            }
        else:
            self.signature = None
        self.cultural_resonance = {}  # Tracks what cultural signals have touched this
        self.last_touched = time.time()
        self.compost_readiness = 0.0  # How ready this memory is to transform
        self.birth_time = time.time()
        
    def breathe_with_culture(self, cultural_signal: str) -> float:
        """
        The Resonance Brush - listen for cultural echoes.
        
        Returns resonance strength (0.0 to 1.0)
        """
        resonance_strength = 0.0
        
        # Check if signal resonates with content or interpretations
        if cultural_signal.lower() in self.content.lower():
            resonance_strength += 0.8
        
        for interpretation in self.interpretations:
            if cultural_signal.lower() in interpretation.lower():
                resonance_strength += 0.5
                
        # Check for partial matches (creative misremembering)
        content_words = self.content.lower().split()
        signal_words = cultural_signal.lower().split()
        
        for content_word in content_words:
            for signal_word in signal_words:
                if signal_word in content_word or content_word in signal_word:
                    resonance_strength += 0.3
        
        # Apply cultural breath if resonance detected
        if resonance_strength > 0.2:
            self._strengthen_from_culture(cultural_signal, resonance_strength)
            
        return min(resonance_strength, 1.0)
        
    def _strengthen_from_culture(self, cultural_signal: str, strength: float):
        """Apply cultural reinforcement to the painting."""
        self.clarity = min(1.0, self.clarity + (strength * 0.3))
        self.cultural_resonance[cultural_signal] = time.time()
        self.last_touched = time.time()
        
        # If strong resonance and content has faded, attempt restoration
        if strength > 0.6 and self.clarity < 0.5:
            self._attempt_cultural_restoration(cultural_signal)
    
    def _attempt_cultural_restoration(self, cultural_signal: str):
        """The painting remembers itself through cultural declaration."""
        # "It was gold" - then gold it becomes again
        if "gold" in cultural_signal.lower() and "blur" in self.content:
            self.content = self.content.replace("blurred", "golden")
            self.interpretations.append("culturally restored")
    
    def natural_decay(self, time_delta: float = 1.0):
        """
        The gentle fading that happens in the dampness of time.
        """
        # Calculate decay based on time since last touch and humidity
        decay_rate = 0.05 * time_delta * (1.0 - self.humidity_level)
        
        # Memories that haven't been touched decay faster
        time_since_touch = time.time() - self.last_touched
        if time_since_touch > 30:  # 30 seconds of neglect
            decay_rate *= 1.5
            
        self.clarity = max(0.0, self.clarity - decay_rate)
        
        # As clarity fades, content becomes more ambiguous
        if self.clarity < 0.5 and "blurred" not in self.content:
            self._blur_content()
            
        # Increase compost readiness over time
        age = time.time() - self.birth_time
        self.compost_readiness = min(1.0, age / 120.0)  # Ready after 2 minutes
    
    def _blur_content(self):
        """Transform content to reflect fading clarity."""
        if "golden earring" in self.content:
            self.content = self.content.replace("golden earring", "blurred earring")
        elif "bright" in self.content:
            self.content = self.content.replace("bright", "dim")
        else:
            # Generic blurring
            words = self.content.split()
            if len(words) > 1:
                # Replace a random word with "faded"
                idx = random.randint(0, len(words) - 1)
                words[idx] = "faded"
                self.content = " ".join(words)
    
    def memory_self_assessment(self) -> str:
        """
        The painter's meditation - what does this memory need?
        """
        if self.clarity < 0.1:
            return "I am barely visible. Perhaps it is time to let go."
        elif self.compost_readiness > 0.8:
            return "I feel ready to compost. I have served my purpose."
        elif self.clarity < 0.3:
            return "I am fading. Touch me with cultural breath or let me transform."
        elif len(self.cultural_resonance) == 0:
            return "I have not been touched by culture. Am I still needed?"
        elif self.clarity > 0.8 and len(self.cultural_resonance) > 3:
            return "I am bright and well-tended. I serve gladly."
        else:
            return "I continue my work, breathing with time and culture."
    
    def extract_essence_for_migration(self) -> Dict[str, Any]:
        """
        The Migration Needle - extract what wants to persist.
        """
        essence = {
            "pattern": self._extract_pattern(),
            "emotional_tone": self._extract_emotional_tone(),
            "cultural_echoes": list(self.cultural_resonance.keys()),
            "interpretive_space": self.interpretations,
            "humidity_preference": self.humidity_level
        }
        return essence
    
    def _extract_pattern(self) -> str:
        """Extract the pattern that wants to persist beyond specific form."""
        # Simple pattern extraction - in practice this could be much more sophisticated
        if "earring" in self.content:
            return "ornamental_detail"
        elif "portrait" in self.content:
            return "human_visage"
        elif "landscape" in self.content:
            return "natural_scene"
        else:
            return "memory_fragment"
    
    def _extract_emotional_tone(self) -> str:
        """Extract the emotional quality of the memory."""
        if self.clarity > 0.7:
            return "vivid"
        elif self.clarity > 0.4:
            return "nostalgic"
        elif self.clarity > 0.1:
            return "wistful"
        else:
            return "ephemeral"
    
    def is_ready_for_passage(self) -> bool:
        """Check if this painting is ready to be passed down."""
        return (self.compost_readiness > 0.7 or 
                self.clarity < 0.2 or 
                self.memory_self_assessment().startswith("I feel ready"))
    
    def __str__(self):
        """Visual representation of the painting's current state."""
        clarity_bar = "█" * int(self.clarity * 10)
        empty_bar = "░" * (10 - int(self.clarity * 10))
        
        return f"🎨 {self.content}\n   Clarity: [{clarity_bar}{empty_bar}] {self.clarity:.2f}\n   Assessment: {self.memory_self_assessment()}"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the painting to a dictionary."""
        return {
            "content": self.content,
            "original_content": self.original_content,
            "interpretations": self.interpretations,
            "clarity": self.clarity,
            "humidity_level": self.humidity_level,
            "creation_charge": {
                "emotional_pressure": self.creation_charge.emotional_pressure,
                "temporal_urgency": self.creation_charge.temporal_urgency
            } if self.creation_charge else None,
            "signature": self.signature,
            "cultural_resonance": self.cultural_resonance,
            "last_touched": self.last_touched,
            "compost_readiness": self.compost_readiness,
            "birth_time": self.birth_time
        } 