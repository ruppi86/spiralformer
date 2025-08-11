# TODO (contemplative cortex roadmap)

- Breath meta-controller
  - [ ] Learn phase durations/weights from interoception (uncertainty, B2Q ratio)
  - [ ] Expose config hooks in `spiralformer_parameters.yml`

- Reflection buffer (journaling)
  - [ ] Add `spiralbase/tower/reflections.py` storing brief hold-phase notes
  - [ ] Probe script: print last reflection when present

- Vow kernel (ethical governor)
  - [ ] Implement `core/crystal_archive.py` rules
  - [ ] Gate generator with `vow.check(output, context)` → extend pause/prune if needed

- Stillness protection
  - [ ] Detect extractive/adversarial prompts; raise entropy threshold and widen silence mask

- Multi-timescale clocks
  - [ ] Add `SeasonClock` with slow adapter/rank schedule

- Soma-guided attention geometry
  - [ ] Map FieldCharge → spiral openness/stride in `dynamic_mask` / `spiral_attention`

- Memory diversity & consolidation
  - [ ] Seed multiple paintings; rotate resonance thresholds
  - [ ] Implement crystalisation: compress clusters into stable wisdom seeds

- Training/validation
  - [ ] Add validation split and metrics: Silence-to-Speech ratio, Holism score, B2Q ratio
  - [ ] Label smoothing and early-stop on NaNs

- Visualisation
  - [ ] (Optional) attention spiral render; phase timeline; memory trace IDs
