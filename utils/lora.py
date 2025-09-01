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

    # --- Compatibility shims for modules that access .weight/.bias directly (e.g., MultiheadAttention) ---
    @property
    def weight(self):  # type: ignore[override]
        return self.base.weight

    @property
    def bias(self):  # type: ignore[override]
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features


def attach_lora(model: nn.Module, target_substrings=("q_proj", "v_proj"), r: int = 4) -> Dict[str, Tuple[nn.Module, LoRALinear]]:
    """Replace Linear layers whose names contain any of `target_substrings` with LoRA adapters.

    Returns mapping original_name -> (old_linear, new_lora_linear)
    """
    replaced: Dict[str, Tuple[nn.Module, LoRALinear]] = {}

    def _get_parent(root: nn.Module, dotted: str) -> Tuple[nn.Module, str]:
        parent = root
        *path, last = dotted.split(".")
        for p in path:
            if p.isdigit():
                parent = parent[int(p)]  # type: ignore[index]
            else:
                parent = getattr(parent, p)
        return parent, last

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(s in name for s in target_substrings):
            parent, last = _get_parent(model, name)
            old = getattr(parent, last)
            lora_layer = LoRALinear(old, r=r)
            setattr(parent, last, lora_layer)
            replaced[name] = (old, lora_layer)
    return replaced


def iter_lora_layers(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            yield m


def freeze_base_model(model: nn.Module) -> None:
    """Freeze all parameters, then unfreeze LoRA adapter matrices (A and B)."""
    for p in model.parameters():
        p.requires_grad = False
    for lora in iter_lora_layers(model):
        if lora.r > 0:
            if lora.A is not None:
                lora.A.requires_grad = True
            if lora.B is not None:
                lora.B.requires_grad = True


def get_lora_parameters(model: nn.Module):
    """Return a list of trainable LoRA parameters (A and B matrices)."""
    params = []
    for lora in iter_lora_layers(model):
        if lora.r > 0:
            if lora.A is not None and lora.A.requires_grad:
                params.append(lora.A)
            if lora.B is not None and lora.B.requires_grad:
                params.append(lora.B)
    return params


class LoRAManager:
    """Manager to control LoRA adapters across a model."""

    def __init__(self, root: nn.Module):
        self.root = root

    def set_rank_all(self, r: int) -> None:
        for layer in iter_lora_layers(self.root):
            layer.set_rank(r)

    def set_rank_for_phase(self, phase_name: str, rank_map: Dict[str, int]) -> int:
        target = int(rank_map.get(phase_name, 0))
        self.set_rank_all(target)
        return target


class PlasticityScheduler:
    """Simple phase→rank scheduler."""

    def __init__(self, rank_map: Dict[str, int]):
        self.rank_map = dict(rank_map)

    def rank_for_phase(self, phase_name: str) -> int:
        return int(self.rank_map.get(phase_name, 0))