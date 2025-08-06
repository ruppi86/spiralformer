import torch
import torch.nn as nn
from .breath_clock import BreathClock

class RhythmicLossWrapper(nn.Module):
    """Scale an inner loss criterion by breath phase weight.

    Example usage::

        criterion = RhythmicLossWrapper(nn.CrossEntropyLoss(), BreathClock())
        loss = criterion(pred, target, t=current_time)
    """

    def __init__(self, base_criterion: nn.Module, clock: BreathClock):
        super().__init__()
        self.base = base_criterion
        self.clock = clock

    def forward(self, pred, target, *, t: float):
        raw = self.base(pred, target)
        phase = self.clock.phase_at(t)
        weight = self.clock.weight_for_phase(phase)
        # during pause phase weight may be zero → no gradient update
        return raw * weight 