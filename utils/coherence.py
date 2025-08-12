import math
from typing import List

class CoherenceResonator:
    """Lightweight multi-oscillator coherence estimator.
    Maintains a small bank of phases; coherence is mean cos phase alignment.
    """
    def __init__(self, freqs_hz: List[float] = None):
        # Arbitrary small set spanning fast/slow bands (model-time units)
        self.freqs = freqs_hz or [1.0, 2.5, 7.0]
        self.phases = [0.0 for _ in self.freqs]
        self.decoherence = 0.0  # 0=no decoherence, 1=fully decohered

    def step(self, dt: float = 1.0) -> float:
        # Advance phases
        for i, f in enumerate(self.freqs):
            self.phases[i] = (self.phases[i] + 2 * math.pi * f * dt) % (2 * math.pi)
        # Coherence metric: average |cos(phase)| (proxy for alignment)
        coh = sum(abs(math.cos(p)) for p in self.phases) / len(self.phases)
        # Apply decoherence attenuation
        coh *= (1.0 - 0.9 * self.decoherence)
        return max(0.0, min(1.0, coh)) 