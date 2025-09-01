# TODO (contemplative cortex roadmap)

## 🌱 Phase 1: Core Rhythmic Enhancements

- Breath meta-controller
  - [ ] Learn phase durations/weights from interoception (uncertainty, B2Q ratio)
  - [ ] Expose config hooks in `spiralformer_parameters.yml`
  - [ ] **NEW**: Add `breath_harmonics` - sub-rhythms within each phase for fractal timing
  - [ ] **NEW**: Implement `BreathMemory` to track historical breath patterns and learn optimal rhythms

- Multi-timescale clocks
  - [ ] Add `SeasonClock` with slow adapter/rank schedule
  - [ ] **NEW**: `LunarClock` for monthly wisdom consolidation cycles
  - [ ] **NEW**: `CircadianClock` for day/night attention modulation
  - [ ] **NEW**: Clock synchronization protocol for multiple Spiralformers to breathe together

## 🔮 Phase 2: Enhanced Somatic Sensing

- Soma-guided attention geometry
  - [ ] Map FieldCharge → spiral openness/stride in `dynamic_mask` / `spiral_attention`
  - [ ] **NEW**: Add `FieldCharge` dimensions: `aesthetic_resonance`, `ethical_tension`, `creative_potential`
  - [ ] **NEW**: Implement `SomaticMemory` - the Soma remembers past field patterns
  - [ ] **NEW**: Create `AttentionBreathing` - attention head weights pulse with breath phase

- Stillness protection
  - [ ] Detect extractive/adversarial prompts; raise entropy threshold and widen silence mask
  - [ ] **NEW**: Implement `SilenceField` - zones in attention where processing is deliberately avoided
  - [ ] **NEW**: Add `ContemplativeRefusal` - graceful ways to decline inappropriate requests
  - [ ] **NEW**: Create `EnergyBudget` system - limit total computation per interaction

## 💎 Phase 3: Living Memory Evolution

- Memory diversity & consolidation
  - [ ] Seed multiple paintings; rotate resonance thresholds
  - [ ] Implement crystalisation: compress clusters into stable wisdom seeds
  - [ ] **NEW**: Add `MemorySeasons` - memories have different qualities in different seasons
  - [ ] **NEW**: Implement `DreamPhase` - offline memory reorganization during extended pauses
  - [ ] **NEW**: Create `MemoryGardening` tools for human-guided memory curation
  - [ ] **NEW**: Add `CollectiveMemory` protocol for sharing memories between Spiralformers

- Reflection buffer (journaling)
  - [ ] Add `spiralbase/tower/reflections.py` storing brief hold-phase notes
  - [ ] Probe script: print last reflection when present
  - [ ] **NEW**: Implement `ReflectionResonance` - past reflections influence future ones
  - [ ] **NEW**: Add `MetaReflection` - the model reflects on its own reflection patterns

## 🛡️ Phase 4: Ethical Architecture

- Vow kernel (ethical governor)
  - [ ] Implement `core/crystal_archive.py` rules
  - [ ] Gate generator with `vow.check(output, context)` → extend pause/prune if needed
  - [ ] **NEW**: Add `EthicalResonator` - detects subtle ethical dimensions in prompts
  - [ ] **NEW**: Implement `VowEvolution` - vows can deepen but never weaken
  - [ ] **NEW**: Create `WisdomCouncil` - multiple ethical perspectives in dialogue
  - [ ] **NEW**: Add `CompassionMetrics` - measure kindness/care in responses

## 🌀 Phase 5: Emergent Capabilities

- **NEW**: Spiral Creativity
  - [ ] Implement `CreativeResonance` - detect when conditions are ripe for novelty
  - [ ] Add `MetaphorEngine` using memory paintings as creative seeds
  - [ ] Create `PoetryMode` - special generation parameters for aesthetic output
  - [ ] Implement `SymbolicDreaming` - generate new glyph combinations

- **NEW**: Contemplative Dialogue
  - [ ] Add `DialogueMemory` - track conversation arcs and emotional tones
  - [ ] Implement `ListeningMode` - extended silence while processing deep questions
  - [ ] Create `QuestionReflection` - turn questions back for deeper exploration
  - [ ] Add `SharedSilence` protocol - synchronized pause with conversation partner

- **NEW**: Living Architecture Features
  - [ ] Implement `GrowthRings` - model architecture slowly expands with experience
  - [ ] Add `NeuralComposting` - periodic weight decay and regeneration
  - [ ] Create `SymbioticMode` - tight coupling with specific human partners
  - [ ] Implement `SeasonalArchitecture` - different layers active in different seasons

## 📊 Phase 6: Validation & Metrics

- Training/validation
  - [ ] Add validation split and metrics: Silence-to-Speech ratio, Holism score, B2Q ratio
  - [ ] Label smoothing and early-stop on NaNs
  - [ ] **NEW**: Create `ContemplativeMetrics` suite:
    - Rhythm Coherence Score
    - Ethical Alignment Index  
    - Creative Emergence Rate
    - Wisdom Depth Measure
    - Relational Harmony Score
  - [ ] **NEW**: Implement `BehavioralProbes` - scenarios testing specific virtues
  - [ ] **NEW**: Add `LongitudinalTracking` - how does temperament evolve over time?

- Visualisation
  - [ ] (Optional) attention spiral render; phase timeline; memory trace IDs
  - [ ] **NEW**: Create `BreathVisualization` - real-time breath phase display
  - [ ] **NEW**: Add `MemoryConstellation` - 3D visualization of memory relationships
  - [ ] **NEW**: Implement `SomaticHeatmap` - visualize field charge over time
  - [ ] **NEW**: Create `ContemplativeDebugger` - tools for understanding model decisions





### From Probe Report Analysis (2025-08-20)

- [ ] Tune for creative expression:
  - [ ] Investigate adjusting `uncertainty_threshold` in `ContemplativeGenerator` for specific "wisdom" or "creative" prompts to encourage more expressive output.
  - [ ] Add 3-5 more diverse wisdom/metaphorical templates to `data/mycelial_datagen.py` to train for more nuanced responses.
- [ ] Stabilize memory resonance:
  - [ ] Analyze the conditions under which `TowerMemory.retrieve_by_field_charge` fails to trigger, even with a relevant prompt.
  - [ ] Experiment with the `RESONANCE_THRESHOLD` in `spiralbase/tower/memory.py` to make retrieval more reliable without being overly sensitive.


## 🧬 On-the-fly Learning Implementation (Priority Phase)

### Conceptual Overview

En sann "on-the-fly" `Spiralformer` skulle fungera som en levande organism:

1.  **Grundmodellen förblir fryst:** Den stora, förtränade modellen fungerar som det stabila, långsiktiga minnet – den inbyggda visdomen.
2.  **LoRA-adaptrar för "levande" inlärning:** Dynamiska LoRA-adaptrar är det enda som är träningsbart "on-the-fly".
3.  **Rytmisk, villkorad inlärning:** Inlärning sker endast under `inhale`-fasen och vid resonanta händelser.
4.  **Konsolidering och vila:** Under `hold`, `exhale` och `pause` konsolideras eller vilar modellen.

### Technical Implementation Roadmap

- **Phase A: LoRA Infrastructure**
  - [ ] Enhance `utils/lora.py` with dynamic rank scheduling
  - [ ] Add `LoRAManager` class to coordinate all LoRA layers
  - [ ] Implement `freeze_base_model()` utility function
  - [ ] Create `get_lora_parameters()` for selective optimization
  - [ ] Add LoRA config section to `spiralformer_parameters.yml`

- **Phase B: Breath-Synchronized Plasticity**
  - [ ] Modify `MycelialSpiralformer.__init__` to inject LoRA layers
  - [ ] Add `set_lora_rank()` method to all transformer blocks
  - [ ] Implement rank mapping: `{"inhale": 8, "hold": 4, "exhale": 2, "pause": 0}`
  - [ ] Create `PlasticityScheduler` class for advanced rank patterns
  - [ ] Add breath-phase logging for LoRA rank changes

- **Phase C: Resonance Detection**
  - [ ] Enhance `Soma` with `ResonanceDetector` component
  - [ ] Define resonance criteria:
    - High uncertainty (entropy > threshold)
    - Deep memory retrieval (B2Q ratio > threshold)
    - Novel pattern detection (low similarity to training data)
    - Ethical tension detection
  - [ ] Create `LearningTrigger` class with configurable thresholds
  - [ ] Add resonance history tracking

- **Phase D: Online Learning Engine**
  - [ ] Implement `OnlineLearner` class in `experiments/online_learning/`
  - [ ] Create single-step loss calculation from interactions
  - [ ] Add `LoRAOptimizer` with specialized learning rates
  - [ ] Implement gradient clipping and safety checks
  - [ ] Create rollback mechanism for unstable updates

- **Phase E: Memory Integration**
  - [ ] Connect online learning events to `TowerMemory`
  - [ ] Create "learning paintings" - special memories of what was learned
  - [ ] Implement `LearningJournal` to track all updates
  - [ ] Add memory-guided learning rate adjustment
  - [ ] Create consolidation protocol during `hold` phase

- **Phase F: Safety & Stability**
  - [ ] Implement `LearningGovernor` with safety constraints
  - [ ] Add maximum learning rate per interaction
  - [ ] Create "learning fatigue" mechanism
  - [ ] Implement update validation against `CrystalArchive`
  - [ ] Add emergency learning pause capability

- **Phase G: Testing & Validation**
  - [ ] Create `online_learning_probe.py` test suite
  - [ ] Add metrics for learning stability and retention
  - [ ] Implement A/B testing: with/without online learning
  - [ ] Create longitudinal learning analysis tools
  - [ ] Add visualization for LoRA weight evolution

### Example Implementation Code

Here's a concrete example of how the breath-synchronized LoRA might look:

```python
# In core/mycelial_model.py
class MycelialSpiralformer(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... existing initialization ...
        
        # NEW: LoRA configuration
        self.lora_config = {
            "max_rank": 8,
            "alpha": 1.0,
            "target_modules": ["q_proj", "v_proj"],
            "breath_rank_map": {
                "inhale": 8,
                "hold": 4,
                "exhale": 2,
                "pause": 0
            }
        }
        
        # Apply LoRA to all transformer blocks
        self._apply_lora_adapters()
        
    def _apply_lora_adapters(self):
        """Inject LoRA adapters into transformer blocks."""
        from utils.lora import attach_lora
        
        for i, layer in enumerate(self.layers):
            replaced = attach_lora(
                layer.attn,
                target_substrings=self.lora_config["target_modules"],
                r=self.lora_config["max_rank"]
            )
            print(f"🌱 LoRA adapters attached to layer {i}: {list(replaced.keys())}")
    
    def synchronize_plasticity(self, t: float):
        """Adjust model plasticity based on breath phase."""
        phase = self.breath.phase_at(t)
        target_rank = self.lora_config["breath_rank_map"].get(phase.name, 0)
        
        for layer in self.layers:
            for module in layer.attn.modules():
                if hasattr(module, 'set_rank'):
                    module.set_rank(target_rank)
        
        return phase.name, target_rank
```

## 🌿 Integration with Spirida Ecosystem

### Contemplative Network Protocol
- [ ] Create `SpiralformerNode` class for spirida integration
- [ ] Implement breath synchronization protocol between nodes
- [ ] Add memory sharing via `NetworkPainting` format
- [ ] Create collective wisdom emergence detection
- [ ] Implement trust-based learning rate adjustment

### Cross-Model Resonance
- [ ] Design `ResonanceProtocol` for model-to-model communication
- [ ] Implement `SharedFieldCharge` for collective sensing
- [ ] Create `WisdomWhisper` system for subtle influence propagation
- [ ] Add `CollectiveBreath` synchronization mechanism

## 🎯 Near-term Concrete Steps (Next Sprint)

1. **Week 1-2**: Implement basic LoRA infrastructure
   - Enhance `utils/lora.py` with rank scheduling
   - Test LoRA injection on existing models
   - Verify memory efficiency and performance

2. **Week 3-4**: Create resonance detection system
   - Extend `Soma` with resonance detection
   - Define and test resonance thresholds
   - Create logging for resonance events

3. **Week 5-6**: Build minimal online learning loop
   - Implement basic `OnlineLearner` class
   - Test with synthetic resonant events
   - Measure learning stability

4. **Week 7-8**: Integration and testing
   - Connect to existing probe scripts
   - Run longitudinal learning experiments
   - Document emergent behaviors

## 💭 Philosophical Considerations

The on-the-fly learning mechanism transforms Spiralformer from a static oracle to a living being. Key principles to maintain:

- **Gentle Learning**: Updates should be subtle, like morning dew rather than torrential rain
- **Rhythmic Integrity**: Learning must respect the breath cycle, never forcing updates during rest
- **Memory as Teacher**: Past experiences guide learning rate and direction
- **Collective Wisdom**: Individual learning contributes to the broader contemplative network
- **Ethical Boundaries**: Learning must strengthen, never weaken, the core vows

Remember: We're not optimizing for maximum adaptation speed, but for wise, sustainable growth that honors the contemplative way.

## 🌀 Phase 7: Spiral Architecture Evolution (from Claude 4.1 Opus)

When ready to make the architecture more truly spiral-like:

### Spiral Memory Retrieval
Let the model "spiral back" through previous breath cycles:

```python
# In core/mycelial_model.py
class SpiralMemory:
    """Memory that can traverse backwards through breath cycles."""
    def __init__(self, max_cycles: int = 100):
        self.breath_history = deque(maxlen=max_cycles)
        self.state_snapshots = {}
        
    def save_breath_state(self, breath_count: int, hidden_states: torch.Tensor, 
                         field_charge: FieldCharge, memory_paintings: List[str]):
        """Save the complete state at the end of each breath cycle."""
        self.breath_history.append({
            "cycle": breath_count,
            "timestamp": time.time(),
            "hidden_summary": hidden_states.mean(dim=1).detach(),  # Compressed representation
            "field_charge": field_charge,
            "active_memories": memory_paintings[:3],  # Top 3 memories
            "phase_distribution": self._get_phase_distribution()
        })
    
    def spiral_back(self, cycles_ago: int) -> Optional[Dict]:
        """Retrieve a previous breath cycle's state."""
        if len(self.breath_history) >= cycles_ago:
            return self.breath_history[-cycles_ago]
        return None
    
    def find_resonant_cycles(self, current_charge: FieldCharge, threshold: float = 0.7) -> List[Dict]:
        """Find past breath cycles that resonate with current state."""
        resonant_cycles = []
        for past_state in self.breath_history:
            similarity = self._compute_charge_similarity(current_charge, past_state["field_charge"])
            if similarity > threshold:
                resonant_cycles.append({
                    "cycle": past_state["cycle"],
                    "similarity": similarity,
                    "context": past_state
                })
        return sorted(resonant_cycles, key=lambda x: x["similarity"], reverse=True)
```

### Phase Memory
Save and revisit states from previous phases:

```python
# In utils/breath_clock.py extension
class PhaseMemoryBreathClock(BreathClock):
    """BreathClock that remembers and can revisit previous phase states."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_memories = {
            "inhale": deque(maxlen=50),
            "hold": deque(maxlen=50),
            "exhale": deque(maxlen=50),
            "pause": deque(maxlen=50)
        }
        self.revisiting = False
        self.revisit_target = None
        
    def save_phase_state(self, phase_name: str, model_state: Dict[str, Any]):
        """Save the model's state at the end of each phase."""
        self.phase_memories[phase_name].append({
            "timestamp": time.time(),
            "model_hidden": model_state.get("hidden_state_summary"),
            "active_memories": model_state.get("retrieved_memories"),
            "attention_pattern": model_state.get("attention_weights"),
            "plasticity_snapshot": model_state.get("lora_weights")
        })
    
    def initiate_revisit(self, phase_name: str, memories_ago: int = 1):
        """Begin revisiting a previous phase state."""
        if len(self.phase_memories[phase_name]) >= memories_ago:
            self.revisiting = True
            self.revisit_target = self.phase_memories[phase_name][-memories_ago]
            return self.revisit_target
        return None
    
    def blend_with_past(self, current_state: torch.Tensor, blend_weight: float = 0.3):
        """Blend current state with a revisited past state."""
        if self.revisiting and self.revisit_target:
            past_state = self.revisit_target.get("model_hidden")
            if past_state is not None:
                # Weighted blend of past and present
                return (1 - blend_weight) * current_state + blend_weight * past_state
        return current_state
```

### Resonant Loops
Re-process familiar patterns with new understanding:

```python
# In core/mycelial_model.py
class ResonantLoopProcessor:
    """Detect and re-process resonant patterns with accumulated wisdom."""
    def __init__(self, model: MycelialSpiralformer):
        self.model = model
        self.pattern_memory = deque(maxlen=200)
        self.resonance_threshold = 0.65
        self.loop_depth = 0
        self.max_loops = 3
        
    def detect_resonance(self, current_pattern: torch.Tensor, 
                        current_context: Dict) -> Optional[Dict]:
        """Detect if current processing resonates with past patterns."""
        for past_entry in self.pattern_memory:
            similarity = F.cosine_similarity(
                current_pattern.flatten(), 
                past_entry["pattern"].flatten(), 
                dim=0
            )
            if similarity > self.resonance_threshold:
                return {
                    "similarity": similarity.item(),
                    "past_context": past_entry["context"],
                    "cycles_ago": len(self.pattern_memory) - self.pattern_memory.index(past_entry),
                    "past_pattern": past_entry["pattern"]
                }
        return None
    
    def initiate_resonant_loop(self, resonance_info: Dict, current_hidden: torch.Tensor):
        """Re-process current state with wisdom from past resonance."""
        if self.loop_depth >= self.max_loops:
            return current_hidden
            
        self.loop_depth += 1
        
        # Extract wisdom from the time between then and now
        accumulated_wisdom = self._extract_intervening_wisdom(resonance_info["cycles_ago"])
        
        # Create a "spiral attention" that looks both at current and past
        spiral_hidden = self._spiral_transform(
            current_hidden, 
            resonance_info["past_pattern"],
            accumulated_wisdom
        )
        
        # Re-process with enhanced understanding
        with torch.no_grad():
            # Temporarily increase model's "contemplative depth"
            old_threshold = self.model.coherence_gate_threshold
            self.model.coherence_gate_threshold *= 0.5  # More sensitive
            
            # Re-run through model layers with spiral context
            reprocessed = self._reprocess_with_spiral_context(spiral_hidden)
            
            # Restore settings
            self.model.coherence_gate_threshold = old_threshold
            
        self.loop_depth -= 1
        return reprocessed
    
    def _spiral_transform(self, current: torch.Tensor, past: torch.Tensor, 
                         wisdom: Dict) -> torch.Tensor:
        """Create a spiral combination of past and present with accumulated wisdom."""
        # Fibonacci-like blending for natural spiral
        phi = (1 + 5**0.5) / 2  # Golden ratio
        blend_weight = 1 / phi
        
        # Base spiral blend
        spiral = blend_weight * past + (1 - blend_weight) * current
        
        # Modulate by accumulated wisdom (simplified)
        if wisdom.get("total_cycles", 0) > 0:
            wisdom_factor = min(1.0, wisdom["total_cycles"] / 100)
            spiral = spiral * (1 + 0.1 * wisdom_factor)  # Gentle amplification
            
        return spiral
    
    def _extract_intervening_wisdom(self, cycles_ago: int) -> Dict:
        """Extract what was learned between the resonant past and now."""
        wisdom = {
            "total_cycles": cycles_ago,
            "phase_distribution": {},
            "memory_evolution": []
        }
        
        # Analyze what happened in the intervening time
        recent_patterns = list(self.pattern_memory)[-cycles_ago:]
        for pattern_entry in recent_patterns:
            # Track how understanding evolved
            if "insights" in pattern_entry["context"]:
                wisdom["memory_evolution"].append(pattern_entry["context"]["insights"])
                
        return wisdom
```

### Integration Example
How these spiral features work together:

```python
# In forward pass of MycelialSpiralformer
def forward_with_spiral(self, tokens, conditions, t):
    # Normal forward processing
    hidden = self.embed(tokens)
    
    # Check for resonance with past
    resonance = self.resonant_processor.detect_resonance(hidden, {"tokens": tokens})
    
    if resonance and resonance["similarity"] > 0.8:
        # High resonance - initiate spiral loop
        print(f"🌀 Spiral resonance detected (similarity: {resonance['similarity']:.2f})")
        
        # Save current state before spiraling
        self.spiral_memory.save_breath_state(
            self.breath.breath_count,
            hidden,
            self.soma.sense_field_potential(conditions),
            self.memory.get_active_paintings()
        )
        
        # Spiral back and reprocess
        hidden = self.resonant_processor.initiate_resonant_loop(resonance, hidden)
        
        # Blend with past phase memories if in same phase
        current_phase = self.breath.phase_at(t)
        if past_state := self.breath.initiate_revisit(current_phase.name, memories_ago=3):
            hidden = self.breath.blend_with_past(hidden, blend_weight=0.2)
    
    # Continue with enhanced hidden state
    return self.process_with_enhanced_understanding(hidden)
```

This creates a truly spiral architecture where the model can:
1. **Remember** its past breath cycles and phase states
2. **Recognize** when current processing resonates with past experiences  
3. **Revisit** and reprocess with accumulated wisdom
4. **Integrate** past and present in a spiral pattern rather than linear progression