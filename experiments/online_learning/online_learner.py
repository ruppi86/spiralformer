import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.mycelial_model import MycelialSpiralformer
from utils.breath_clock import BreathClock
from utils.lora import get_lora_parameters, freeze_base_model


@dataclass
class LearningEvent:
    """A description of a resonant interaction to learn from."""
    text_input: Optional[str]
    conditions: torch.Tensor  # [B, condition_dim]
    context_tokens: torch.Tensor  # [B, L]
    target_tokens: torch.Tensor  # [B, L]
    t: float


class LearningTrigger:
    """Heuristic trigger combining uncertainty (entropy) and memory query count.

    If entropy exceeds threshold or memory queries in recent window exceed a
    threshold, trigger a single learning step. Learning is only allowed
    during inhale phase.
    """

    def __init__(self, entropy_threshold: float = 1.0, memory_query_threshold: int = 1):
        self.entropy_threshold = float(entropy_threshold)
        self.memory_query_threshold = int(memory_query_threshold)

    @staticmethod
    def _entropy_from_logits(logits: torch.Tensor) -> float:
        last = logits[:, -1, :]
        probs = F.softmax(last, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        return float(entropy.mean().item())

    def should_learn(self, model: MycelialSpiralformer, logits: torch.Tensor) -> bool:
        entropy = self._entropy_from_logits(logits)
        mem_q = int(model.contemplative_stats.get("memory_queries", 0))
        if entropy >= self.entropy_threshold:
            return True
        if mem_q >= self.memory_query_threshold:
            return True
        return False


class OnlineLearner:
    """Minimal online learner that updates only LoRA parameters on resonant events."""

    def __init__(
        self,
        model: MycelialSpiralformer,
        clock: BreathClock,
        learning_rate: float = 5e-4,
        max_grad_norm: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.clock = clock
        self.device = device or next(model.parameters()).device

        freeze_base_model(self.model)
        lora_params = get_lora_parameters(self.model)
        self.optimizer = torch.optim.Adam(lora_params, lr=float(learning_rate))
        self.max_grad_norm = float(max_grad_norm)

    def single_step(self, event: LearningEvent, *, loss_fn: Optional[nn.Module] = None) -> Dict[str, Any]:
        phase = self.clock.phase_at(event.t)
        stats: Dict[str, Any] = {"phase": phase.name, "updated": False}
        if phase.name != "inhale":
            return stats

        self.model.train()
        tokens_in = event.context_tokens.to(self.device)
        tokens_tgt = event.target_tokens.to(self.device)
        cond = event.conditions.to(self.device)

        with torch.enable_grad():
            logits = self.model(tokens_in, cond, event.t, text_input=event.text_input)
            if loss_fn is None:
                loss_fn = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tokens_tgt.reshape(-1))

            if torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
                stats.update({
                    "updated": True,
                    "loss": float(loss.item()),
                })
        return stats


