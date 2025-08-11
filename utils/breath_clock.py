from dataclasses import dataclass
from typing import Optional

@dataclass
class BreathPhase:
    name: str
    duration: float  # seconds

class BreathClock:
    """Generates a repeating inhale / hold / exhale / pause rhythm."""

    def __init__(self, inhale: float = 4.0, hold: float = 2.0, exhale: float = 4.0, pause: float = 2.0):
        self.phases = [
            BreathPhase("inhale", inhale),
            BreathPhase("hold", hold),
            BreathPhase("exhale", exhale),
            BreathPhase("pause", pause),
        ]
        self._cycle = sum(p.duration for p in self.phases)
        self._time = 0.0

    def tick(self, dt: float = 1.0, *, uncertainty: Optional[float] = None, ethical_risk: Optional[bool] = None):
        """Advances the clock; optionally adapt hold/pause when uncertain or at ethical risk."""
        # Simple adaptive policy: if very uncertain or ethical risk, slow down (extend hold slightly)
        if uncertainty is not None and uncertainty > 0.85:
            dt *= 0.5
        if ethical_risk:
            dt *= 0.5
        self._time += dt

    def phase_at(self, t: Optional[float] = None) -> BreathPhase:
        """Gets the phase at a specific time t, or the current internal time if t is None."""
        if t is None:
            t = self._time
        
        t_mod = t % self._cycle
        acc = 0.0
        for p in self.phases:
            acc += p.duration
            if t_mod < acc:
                return p
        return self.phases[-1]

    def weight_for_phase(self, phase: BreathPhase) -> float:
        mapping = {
            "inhale": 1.0,
            "hold": 0.5,
            "exhale": 0.25,
            "pause": 0.0,
        }
        return mapping.get(phase.name, 1.0) 