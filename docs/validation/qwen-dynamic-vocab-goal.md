# Qwen Dynamic Vocab Goal Evidence

This page maps the current Qwen dynamic runtime-vocab objective to the concrete
evidence files and gates that protect it.

The supported claim is intentionally narrow: a frozen Qwen body can be wrapped
with ARTI-side glyph/text input, runtime-vocab output, and pulse-compatible
adapters so that controlled semantic answers bind to the current external vocab
slot under replaced candidate ranges and shuffled order. This is not a claim of
full tokenizer replacement, broad open-ended dialogue parity, or base-model
vocabulary surgery.

## Primary Gate

```bash
uv run --extra dev python scripts/quality_gate.py qwen --fail-fast --reuse-passing-producers --output benchmarks/results/quality_gate_qwen.json
uv run --extra dev python scripts/check_release_readiness.py --require-qwen
```

For this expensive gate, `--reuse-passing-producers` may reuse existing passing
producer outputs, while current verifier commands still run. This keeps the
Qwen evidence bundle current without repeating every CUDA training probe.

The goal-level verifier is:

```bash
uv run --extra dev python benchmarks/verify_qwen_dynamic_vocab_goal.py
```

It aggregates these specialized verifiers:

- `benchmarks/verify_qwen_input_head_replacement.py`
- `benchmarks/verify_qwen_closed_loop_replacement.py`
- `benchmarks/verify_qwen_runtime_vocab_pulse_semantic.py`
- `benchmarks/verify_qwen_runtime_vocab_metadata_bridge.py`
- `benchmarks/verify_qwen_runtime_vocab_bridge_ablation.py`
- `benchmarks/verify_qwen_oov_dialogue_preservation.py`
- `benchmarks/verify_qwen_string_recall_refiner_answering.py`

The string-first answer verifier is part of the goal-level chain, so dynamic
vocab/glyph evidence also has to keep the rendered-string input boundary,
prefix next-token CE objective, prompt-conditioning loss decomposition, frozen
Qwen boundary, and generated-string quality checks valid.

## Requirement Matrix

| Requirement | Evidence | Acceptance |
| --- | --- | --- |
| Frozen or semi-frozen Qwen body | `qwen_*_results.json`; `verify_qwen_dynamic_vocab_goal.py` | Qwen model id is `Qwen/Qwen3-0.6B`; Qwen trainable parameters are `0` in the frozen evidence chain. |
| Replace input interface | `qwen_input_head_replacement_results.json`; `verify_qwen_input_head_replacement.py` | Glyph/text tensor encoder aligns to frozen Qwen embeddings, lowers MSE, preserves hidden-state cosine, bounds top-k logits KL, and covers readable probes. |
| Replace input plus output in a closed loop | `qwen_closed_loop_replacement_results.json`; `verify_qwen_closed_loop_replacement.py` | Glyph input plus runtime-vocab output trains around frozen Qwen; closed-loop KL beats fixed-slot baseline; OOV visible candidate and permuted-vocab probes solve. |
| Runtime vocab output head | `qwen_runtime_vocab_pulse_semantic_results.json`; `verify_qwen_runtime_vocab_pulse_semantic.py` | Runtime-vocab-only and runtime-vocab+pulse read current glyph vocab tensors and score current local slots. |
| Independent output vocabulary context | `qwen_literal_output_context_results.json`; `verify_qwen_literal_output_context.py` | Frozen Qwen supplies task-query states while an independent shuffled glyph output range is injected before the ARTI-side body. Three paired seeds compare fixed slots, a parameter-matched terminal head, and front output-context conditioning on disjoint candidate-set ranges. |
| Training-unseen literal transfer | `qwen_unseen_literal_transfer_results.json`; `verify_qwen_unseen_literal_transfer.py` | Frozen Qwen copy-query states are bound to independent glyph output vocabularies. Train/validation/heldout strings are disjoint by confusable pair; dynamic heads must return unseen exact strings, preserve identity under local-slot permutation, and distinguish stem/stem+r candidates. |
| Different input/output segmentation | `qwen_literal_segmentation_generation_results.json`; `verify_qwen_literal_segmentation_generation.py` | Frozen Qwen BPE prompt states drive a character-level `LiteralSequenceDecoder`. Visible output characters are glyph-only, EOS has a dedicated control bit, local character slots are independently shuffled, and evaluation decodes full semantic answer sentences. |
| Pulse-compatible adapter | `qwen_runtime_vocab_pulse_semantic_results.json` | `runtime_vocab_pulse` records both `uses_runtime_vocab` and `uses_pulse`; pulse must not break the glyph runtime-vocab path. |
| Variable vocab order | Runtime vocab semantic and closed-loop results | Candidate local slots are shuffled; permutation invariance/equivariance and dynamic slot accuracy are checked. |
| Variable vocab range | Runtime vocab semantic results | Held-out evaluation uses replaced candidate ranges and precomputed runtime vocab banks. |
| Held-out tokenization views | Runtime vocab semantic results | Training variants are `[0, 1]`; held-out variants are `[2, 3]`; tokenization invariance is checked. |
| Held-out surface semantics | `qwen_runtime_vocab_pulse_strict_surface_probe.json`; `qwen_runtime_vocab_pulse_metadata_bridge_probe.json`; bridge ablation verifier | Strict glyph-only held-out surfaces must remain low; metadata bridge strict held-out surfaces must be high. |
| Baseline comparison | Runtime vocab semantic and metadata bridge results | Tokenizer-only, fixed-head, runtime-only, pulse-only, and runtime+pulse controls are present. |
| Semantic correctness | Runtime vocab semantic and metadata bridge results | Held-out semantic accuracy and held-out surface accuracy must pass verifier thresholds. |
| Dynamic slot accuracy | Runtime vocab semantic and metadata bridge results | `dynamic_slot_accuracy` must be high for runtime+pulse. |
| Held-out generalization | Runtime vocab semantic results | `heldout_generalization_accuracy` must be high. |
| Original dialogue route preservation | `qwen_oov_dialogue_preservation_results.json`; `verify_qwen_oov_dialogue_preservation.py` | Ordinary Qwen dialogue logits have zero drift, KL is effectively zero, top-1 is preserved, and the ARTI-side OOV glyph head still learns. |
| String-first answer output | `qwen_string_recall_refiner_answering_results.json`; `qwen_string_pulse_multiseed_results.json`; corresponding verifiers; `verify_qwen_dynamic_vocab_goal.py` | ARTI-side text tensor frontends read rendered strings at a frozen Qwen boundary. Training samples prompts uniformly before prefixes and adds a direct frozen-lm-head first-answer-token loss, preventing the discarded diagnostic condition head from carrying prompt identity alone. Pulse must reproduce all four canonical strings, beat the parameter-matched `mean_wide` exact rate, cut matched conditional NLL by at least half, and pass coherence/resource checks. The three-seed companion requires Pulse to reach 4/4 in every seed with zero mode collapse. |
| GPU/resource discipline | Runtime vocab semantic and metadata bridge results | CUDA run records effective batch, bf16 autocast, gradient accumulation, precomputed contexts/vocab banks, profiling fields, throughput, and peak memory. |

## Current Evidence Boundary

The current evidence is an alpha engineering gate. It shows that Qwen can learn
controlled external runtime-vocab binding through ARTI-side interfaces while
Qwen remains frozen. It also shows an important negative result: raw glyph-only
strict unseen surfaces do not yet solve unseen word semantics. The metadata
bridge probe is the current positive strict-surface path.

Do not use this evidence to claim:

- full tokenizer replacement
- full open-ended Qwen dialogue parity
- base-model vocabulary surgery without adaptation
- raw unseen glyph semantics without a bridge or larger pretraining stage
