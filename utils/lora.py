"""Minimal LoRA utilities for SpiralFormer experiments.

Note: This is a *very* lightweight implementation that injects low-rank adapters
into nn.Linear layers.  It is *not* feature-complete; sufficient for research
in the context of breath-synchronised rank scheduling.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import math

class LoRALinear(nn.Module):
    """Wrap an existing Linear with trainable low-rank adapters."""

    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        if r > 0:
            self.A = nn.Parameter(torch.zeros((r, base.in_features)))
            self.B = nn.Parameter(torch.zeros((base.out_features, r)))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        else:
            # deactivated adapter
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def set_rank(self, r: int):
        """Dynamically change rank (re-initialises if size changes)."""
        if r == self.r:
            return
        device = self.base.weight.device
        self.r = r
        if r == 0:
            self.register_parameter("A", None)
            self.register_parameter("B", None)
            return
        self.A = nn.Parameter(torch.zeros((r, self.base.in_features), device=device))
        self.B = nn.Parameter(torch.zeros((self.base.out_features, r), device=device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        out = self.base(x)
        if self.r > 0:
            lora_out = (self.B @ (self.A @ x.transpose(-1, -2))).transpose(-1, -2)
            out = out + lora_out * self.alpha / self.r
        return out


def attach_lora(model: nn.Module, target_substrings=("q_proj", "v_proj"), r: int = 4) -> Dict[str, Tuple[nn.Module, LoRALinear]]:
    """Replace Linear layers whose names contain any of `target_substrings` with LoRA adapters.

    Returns mapping original_name -> (old_linear, new_lora_linear)
    """
    replaced = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(s in name for s in target_substrings):
            parent = model
            *path, last = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            old = getattr(parent, last)
            lora_layer = LoRALinear(old, r=r)
            setattr(parent, last, lora_layer)
            replaced[name] = (old, lora_layer)
    return replaced 