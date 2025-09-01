import torch
import torch.nn as nn
from typing import Optional, List, Dict
import random

from utils.breath_clock import BreathClock
from utils.positional import SinusoidalPositionalEmbedding
from core.spiral_attention import build_spiral_attention_mask
from core.dynamic_mask import build_glyph_conditioned_mask
from spiralbase import TowerMemory
from core.soma import Soma, FieldCharge
from utils.glyph_codec import GlyphCodec
from utils.lora import attach_lora, LoRAManager, PlasticityScheduler
from utils.coherence import CoherenceResonator

class MycelialSpiralformer(nn.Module):
    """
    An evolution of Spiralformer that is environmentally aware.
    It accepts NetworkConditions as an input to its forward pass, allowing
    its internal state and responses to be influenced by the simulated ecosystem.
    """

    def __init__(self, d_model=128, n_heads=4, seq_len=32, num_layers=4, vocab_size=68, condition_dim=5, padding_idx=0, lora_config: Optional[Dict] = None):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos = SinusoidalPositionalEmbedding(d_model, max_len=seq_len)

        # The Soma is the sensory organ
        self.soma = Soma()
        
        # The TowerMemory is the long-term, living memory
        self.memory = TowerMemory(max_painters=15)
        # Enable anti-repetition hygiene in memory
        if hasattr(self.memory, 'set_anti_repetition'):
            self.memory.set_anti_repetition(enabled=True, recent_max=12, cooldown_sec=1.0)

        # Projection for the condition vector to match d_model
        self.condition_proj = nn.Linear(condition_dim, d_model)

        self.layers = nn.ModuleList([
            _MycelialSpiralBlock(d_model, n_heads) for _ in range(num_layers)
        ])

        self.out = nn.Linear(d_model, vocab_size)

        self.breath = BreathClock()
        base_mask = build_spiral_attention_mask(seq_len)
        self.register_buffer("base_mask", base_mask, persistent=False)
        # Contemplative statistics for observing wisdom
        self.contemplative_stats = {"hold_phases": 0, "memory_queries": 0}
        # Coherence resonator (lightweight)
        self.coherence = CoherenceResonator()
        self.decoherence = 0.0  # can be set by probe to simulate anesthesia
        # Query policy
        self.coherence_gate_threshold: float = 0.25
        self.memory_query_probability: float = 0.35

        # --- LoRA integration (optional) ---
        self._lora_enabled = False
        self.lora_config: Dict = lora_config or {}
        # Provide safe defaults if enabled without full config
        if self.lora_config.get("enabled", False):
            self._lora_enabled = True
            self.lora_config.setdefault("max_rank", 4)
            self.lora_config.setdefault("target_substrings", ["attn.out_proj", "ff."])
            self.lora_config.setdefault("breath_rank_map", {"inhale": 8, "hold": 4, "exhale": 2, "pause": 0})
            # Inject adapters across attention out_proj and feed-forward linears
            self._apply_lora_adapters()
            self._lora_manager = LoRAManager(self)
            self._plasticity = PlasticityScheduler(self.lora_config["breath_rank_map"])
            # Observability of plasticity over time (for probes)
            self.last_plasticity_phase_name: str = "unknown"
            self.last_plasticity_rank: int = int(self.lora_config["breath_rank_map"].get("pause", 0))
            self.plasticity_log: List[Dict[str, float]] = []  # entries: {"t": t, "phase": str, "rank": int}

    def forward(self, tokens: torch.Tensor, conditions: torch.Tensor, t: float, text_input: Optional[str] = None):
        # Synchronize LoRA plasticity with breath phase
        if getattr(self, "_lora_enabled", False):
            self.synchronize_plasticity(t)
        # 1. The Soma feels the environment first
        field_charge = self.soma.sense_field_potential(self._tensor_to_dict(conditions))
        # Step coherence resonator; store last value
        self.coherence.decoherence = float(self.decoherence)
        self.last_coherence = self.coherence.step(dt=1.0)
        
        # 2. Add raw condition data to token embeddings
        x = self.embed(tokens)
        cond_embedding = self.condition_proj(conditions).unsqueeze(1)
        x = x + cond_embedding

        x = self.pos(x)
        
        # 3. Create a query from the initial context for memory recall
        # This simulates the mind forming an "intention" before deep processing
        # We will use the raw text of the glyphs for a simple query
        initial_query = self._tokens_to_text(tokens)
        
        # Correctly slice the base_mask to match the current batch's sequence length
        current_seq_len = tokens.size(1)
        sliced_base_mask = self.base_mask[:current_seq_len, :current_seq_len]

        attn_mask_batch = build_glyph_conditioned_mask(tokens, sliced_base_mask)
        for layer in self.layers:
            # Pass the field_charge and query controls to the block
            x = layer(
                x,
                t,
                attn_mask_batch,
                self.breath,
                self.memory,
                initial_query,
                self.contemplative_stats,
                field_charge,
                self.last_coherence,
                self.memory_query_probability,
            )
        
        self.last_hidden_state = x.detach() # Store the final hidden state
        return self.out(x)

    # --- LoRA helpers ---
    def _apply_lora_adapters(self) -> None:
        target_substrings = self.lora_config.get("target_substrings", ["attn.out_proj", "ff."])
        # Attach to this model; names include paths like "layers.0.attn.out_proj", "layers.0.ff.0", "layers.0.ff.2"
        attach_lora(self, target_substrings=tuple(target_substrings), r=int(self.lora_config.get("max_rank", 4)))

    def synchronize_plasticity(self, t: float):
        phase = self.breath.phase_at(t)
        rank = self._plasticity.rank_for_phase(phase.name)
        self._lora_manager.set_rank_all(rank)
        # Record for probes/debugging
        if getattr(self, "_lora_enabled", False):
            self.last_plasticity_phase_name = phase.name
            self.last_plasticity_rank = int(rank)
            # Keep a short rolling log (cap length to avoid unbounded growth)
            try:
                self.plasticity_log.append({"t": float(t), "phase": phase.name, "rank": int(rank)})
                if len(self.plasticity_log) > 256:
                    self.plasticity_log = self.plasticity_log[-256:]
            except Exception:
                pass
        return phase.name, rank
    
    def _tokens_to_text(self, tokens: torch.Tensor) -> str:
        """A conceptual placeholder to convert token IDs to a text query."""
        # This is a placeholder. A real implementation would use a vocabulary.
        # For now, we simulate a query based on token values.
        # We'll use just the first few non-padding tokens from the first batch item.
        sample_tokens = tokens[0, :10].tolist()
        return "query based on tokens " + " ".join(map(str, sample_tokens))

    def _tensor_to_dict(self, conditions: torch.Tensor) -> Dict[str, float]:
        """Converts a condition tensor back to a dictionary for the Soma."""
        # Assuming the order is [latency, voltage, temperature, error_rate, bandwidth]
        if conditions.ndim > 1:
             conditions = conditions[0] # Take the first of the batch for this conversion
        
        cond_list = conditions.tolist()
        return {
            'latency': cond_list[0],
            'voltage': cond_list[1],
            'temperature': cond_list[2],
            'error_rate': cond_list[3],
            'bandwidth': cond_list[4],
        }

    def get_current_mood_glyph(self, t: float) -> str:
        """
        Returns a glyph representing the AI's current internal "weather,"
        implementing the 'Transparency of Mood' GEP principle.
        """
        phase = self.breath.phase_at(t)

        if phase.name == "pause":
            return "🕯️" # Stillness, calm
        
        if phase.name == "inhale":
            return "🧭" # Exploring, receptive

        if hasattr(self, 'last_hidden_state'):
            variance = torch.var(self.last_hidden_state).item()
            
            if variance < 0.1:
                return "🔆" # Clarity, focus
            elif variance > 0.5:
                return "⛈️" # Storm, high internal activity
            else:
                return "🌫️" # Fog, confusion or ambiguity
        
        return "⚪" # Default unknown state

class _MycelialSpiralBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # A dedicated layer to blend memory into the context
        self.memory_blender = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, t: float, attn_mask_batch: torch.Tensor, breath: BreathClock, memory: TowerMemory, query: str, model_stats: Dict[str, int], field_charge: FieldCharge, last_coherence: float, query_prob: float):
        phase = breath.phase_at(t)

        # During the 'hold' phase, consult the TowerMemory
        if phase.name == "hold":
            model_stats["hold_phases"] += 1
            # Gate recall by coherence and probability
            if last_coherence < 0.25 or random.random() >= max(0.0, min(1.0, query_prob)):
                weight = breath.weight_for_phase(phase)
                return x if weight == 0.0 else self.norm1(x + 0.0)
            # Use field_charge and signature retrieval
            resonant_painting_fc = memory.retrieve_by_field_charge(field_charge)
            signature = {
                "emotional_pressure": float(field_charge.emotional_pressure),
                "temporal_urgency": float(field_charge.temporal_urgency),
            }
            resonant_painting_sig = memory.retrieve_by_signature(signature) if hasattr(memory, 'retrieve_by_signature') else None
            resonant_painting = resonant_painting_sig or resonant_painting_fc
            if resonant_painting:
                model_stats["memory_queries"] += 1
                model_stats.setdefault("retrieved_paintings", []).append(resonant_painting.content)
                print(f"🧠 Memory Resonated: '{resonant_painting.content}'")
                painting_tensor = torch.randn(1, 1, x.size(-1), device=x.device) # Placeholder
                painting_influence = self.memory_blender(painting_tensor)
                x = x + painting_influence # Blend the memory into the current thought
        
        weight = breath.weight_for_phase(phase)
        
        if weight == 0.0:
            return x

        # The mask is now 2D, so we don't need the batch dimension inversion.
        # PyTorch will broadcast the 2D mask across the batch.
        attn_output, _ = self.attn(x, x, x, attn_mask=~attn_mask_batch.bool(), need_weights=False)
        attn_output = attn_output * weight
        
        x = self.norm1(x + attn_output)
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x 