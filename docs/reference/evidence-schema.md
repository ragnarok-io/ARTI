# Evidence Schema

ARTI benchmark evidence is meant to be auditable before it is persuasive.

Core benchmark result JSON files use this top-level schema:

```text
status          string
scope           string
claim_boundary  string
provenance      object
config          object
runs            list
summary         list
metrics         object or list
```

The schema verifier also checks quality-gate reports:

```text
python
gates
passed
runs
```

All checked payloads must avoid `NaN` and `Infinity`.

For selected high-value evidence lines, the schema verifier also checks
mechanism-specific contracts:

```text
qwen_string_recall_refiner_answering_results.json
  resource_profile
  training_objective_contract

recall_refiner_results.json
  resource_profile

half_recall_fair_training_results.json
  resource_profile

pulse_refiner_repair_results.json
  resource_profile

pulse_refiner_multiseed_results.json
  resource_profile
```

These checks are intentionally lighter than the dedicated benchmark verifiers.
They make the evidence bundle reject stale or incomplete resource/objective
reports before a reader treats the result as comparable.

## Verify

```bash
uv run --extra dev python benchmarks/verify_evidence_schema.py
```

The verifier currently checks these core result files:

| Result | Evidence Type |
| --- | --- |
| `benchmarks/results/cuda_scaling_profile.json` | Local CUDA scaling profile. |
| `benchmarks/results/cuda_evidence_packet.json` | Local CUDA evidence packet. |
| `benchmarks/results/qwen_glyph_runtime_adapter_api_results.json` | Public QwenGlyphRuntimeAdapter API smoke. |
| `benchmarks/results/qwen_runtime_vocab_replacement_results.json` | Frozen Qwen runtime candidate-vocab replacement smoke. |
| `benchmarks/results/qwen_runtime_vocab_training_results.json` | Frozen-context Qwen runtime-vocab adapter training. |
| `benchmarks/results/qwen_output_head_replacement_results.json` | Frozen Qwen teacher to ARTI runtime output-head replacement smoke. |
| `benchmarks/results/qwen_literal_output_context_results.json` | Three-seed frozen-Qwen comparison of fixed slots, a parameter-matched terminal literal head, and output-vocabulary conditioning before the ARTI-side body. |
| `benchmarks/results/qwen_unseen_literal_transfer_results.json` | Three-seed frozen-Qwen exact-string transfer to disjoint external glyph vocabularies with confusable stem/stem+r pairs and validation-only checkpoint selection. |
| `benchmarks/results/qwen_literal_segmentation_generation_results.json` | Three-seed frozen-Qwen BPE-input to independently shuffled glyph-character output generation with complete semantic sentence metrics. |
| `benchmarks/results/qwen_input_head_replacement_results.json` | Frozen Qwen teacher to ARTI glyph/text input-head replacement smoke. |
| `benchmarks/results/qwen_closed_loop_replacement_results.json` | Frozen Qwen body with ARTI glyph input and runtime output closed-loop smoke. |
| `benchmarks/results/qwen_autoregressive_closed_loop_results.json` | Frozen Qwen body with ARTI glyph input, runtime output, and short autoregressive writeback. |
| `benchmarks/results/qwen_controlled_open_generation_results.json` | Frozen Qwen body with ARTI runtime vocab over mixed 8-16 step controlled open-generation writeback. |
| `benchmarks/results/qwen_runtime_vocab_pulse_semantic_results.json` | Runtime vocab plus pulse semantic adapter benchmark. |
| `benchmarks/results/qwen_learned_pulse_adapter_results.json` | Frozen-Qwen boundary LearnedPulse adapter benchmark with parameter-matched baseline. |
| `benchmarks/results/qwen_string_recall_refiner_answering_results.json` | String-first Qwen answer probe with generated-answer and coherence diagnostics. |
| `benchmarks/results/qwen_string_pulse_multiseed_results.json` | Three-seed parameter-matched Qwen string comparison with prompt-balanced direct answer-identity loss. |
| `benchmarks/results/qwen_hidden_refiner_half_results.json` | Controlled frozen-Qwen hidden-state repair probe for Half inside RecallRefiner. |
| `benchmarks/results/qwen_runtime_vocab_pulse_strict_surface_probe.json` | Strict held-out external vocab surface benchmark without semantic bridge; negative control for bridge ablation. |
| `benchmarks/results/qwen_runtime_vocab_pulse_metadata_bridge_probe.json` | Strict held-out external vocab surface benchmark with runtime-vocab semantic metadata bridge. |
| `benchmarks/results/qwen_oov_text_vocab_finetune_results.json` | Frozen Qwen OOV visible-word bitmap runtime-vocab finetune smoke. |
| `benchmarks/results/qwen_external_glyph_answering_results.json` | Frozen Qwen external glyph vocabulary answering and writeback benchmark. |
| `benchmarks/results/qwen_multitoken_glyph_decoder_results.json` | Frozen Qwen multi-token external glyph answer decoder benchmark. |
| `benchmarks/results/qwen_oov_dialogue_preservation_results.json` | Frozen Qwen dialogue-logit preservation while training an ARTI-side OOV glyph head. |
| `benchmarks/results/qwen_membrane_routing_training_results.json` | Qwen-shaped membrane routing training benchmark. |
| `benchmarks/results/qwen_membrane_guessing_game_results.json` | Simulated readable membrane guessing-game benchmark. |
| `benchmarks/results/qwen_membrane_guessing_game_real_forward_results.json` | Real Qwen frozen-forward membrane smoke. |
| `benchmarks/results/micro_glyph_distinctness_results.json` | Micro-glyph small-character distinctness and runtime slot benchmark. |
| `benchmarks/results/half_recall_trace_survival_results.json` | Half activation recall-trace survival benchmark. |
| `benchmarks/results/half_recall_fair_training_results.json` | Same-parameter trainable Recall-delta benchmark for Half under direct reconstruction loss. |
| `benchmarks/results/recall_refiner_results.json` | RecallRefiner iterative latent refinement benchmark. |
| `benchmarks/results/pulse_refiner_repair_results.json` | Same-parameter Pulse/Fold plus RecallRefiner latent repair benchmark. |
| `benchmarks/results/pulse_refiner_multiseed_results.json` | Three-seed paired Pulse/Fold stability evidence with explicit selectivity gains and final-MSE tradeoff boundary. |
| `benchmarks/results/fold_compaction_results.json` | Fold tensor compaction benchmark. |
| `benchmarks/results/learned_pulse_compaction_results.json` | LearnedPulse alpha compaction benchmark using Half plus Fold. |

It also checks these gate reports:

| Gate Report | Meaning |
| --- | --- |
| `benchmarks/results/quality_gate_quick.json` | Unit, docs-generation, backend, and schema smoke. |
| `benchmarks/results/quality_gate_mainline.json` | Compact verifier-only gate for the Qwen hidden Half control, combined core-goal evidence line, and evidence schema. |
| `benchmarks/results/quality_gate_mechanism.json` | Controlled mechanism evidence. |
| `benchmarks/results/quality_gate_cuda.json` | CUDA runtime and scaling evidence. |
| `benchmarks/results/quality_gate_qwen.json` | Qwen integration evidence. |

## How To Read A Result

Use the fields in this order:

1. `status`: `completed` means the run produced evidence; `skipped` must state why.
2. `scope`: the positive claim being tested.
3. `claim_boundary`: what the result explicitly does not prove.
4. `provenance`: runtime, command, platform, model, and seed context.
5. `config`: knobs needed to reproduce the run.
6. `runs`: per-model or per-condition records.
7. `summary`: compact rows for comparison and dashboards.
8. `metrics`: primary and secondary metric names.

## Claim Discipline

Passing schema verification means:

- The file can be loaded and audited.
- Required top-level fields are present.
- Checked numeric values are finite.
- Qwen string-first, RecallRefiner, and Pulse repair results expose their
  required resource and training-objective contracts where applicable.
- Current gate reports are present and finite.

It does not mean:

- ARTI is superior on broad downstream benchmarks.
- Qwen or Stable Diffusion base weights were trained.
- A mechanism is production-safe without downstream validation.
- A skipped optional benchmark is equivalent to a completed run.
