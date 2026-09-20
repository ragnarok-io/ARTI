# Mechanism Evidence

ARTI's current validation evidence is mechanism-level. It supports claims about controlled tensor behavior, not broad downstream superiority.

Useful entry points:

```bash
uv run --extra dev python scripts/quality_gate.py mechanism
```

For a compact mechanism-to-command map, see
`docs/reference/validation-matrix.md`.

Backend capability evidence:

```bash
uv run --extra dev python scripts/quality_gate.py cuda
```

Important generated reports live in `benchmarks/results/`, including:

- `virtual_recall_alignment_results.md`
- `experiential_recall_scaling_results.md`
- `half_recall_trace_survival_results.md`
- `half_recall_fair_training_results.md`
- `pulse_refiner_repair_results.md`
- `pulse_refiner_multiseed_results.md`
- `runtime_vocab_binding_results.md`
- `runtime_vocab_replacement_results.md`
- `runtime_vocab_permutation_results.md`
- `pulse_vocab_invariance_results.md`
- `pulse_distinctness_results.md`
- `text_bitmap_vocab_results.md`
- `source_integrity_stereo_results.md`
- `source_integrity_multisource_results.md`
- `multisource_reliability_arena_results.md`
- `micro_glyph_distinctness_results.md`
- `qwen_arti_smoke_results.md`
- `qwen_glyph_runtime_adapter_api_results.md`
- `qwen_runtime_vocab_replacement_results.md`
- `qwen_runtime_vocab_training_results.md`
- `qwen_output_head_replacement_results.md`
- `qwen_input_head_replacement_results.md`
- `qwen_closed_loop_replacement_results.md`
- `qwen_autoregressive_closed_loop_results.md`
- `qwen_controlled_open_generation_results.md`
- `qwen_runtime_vocab_pulse_semantic_results.md`
- `qwen_string_recall_refiner_answering_results.md`
- `qwen_string_pulse_multiseed_results.md`
- `qwen_hidden_refiner_half_results.md`
- `qwen_oov_text_vocab_finetune_results.md`
- `qwen_external_glyph_answering_results.md`
- `qwen_multitoken_glyph_decoder_results.md`
- `qwen_oov_dialogue_preservation_results.md`
- `qwen_membrane_routing_training_results.md`
- `qwen_membrane_guessing_game_results.md`
- `qwen_membrane_guessing_game_real_forward_results.md`
- `authority_prompt_injection_results.md`
- `membrane_visibility_routing_results.md`
- `coord_virtual_recall_phase_corruption_results.md`
- `operator_bank_image_results.md`
- `scaling_profile.md`
- `evidence_summary.md`
- `nature_gap_audit.md`

## Experiential Recall Scaling

The experiential recall scaling benchmark keeps one synthetic latent task and
one corruption process fixed, then varies only ARTI recall capacity:

- no recall
- shallow recall
- deeper recall
- wider/deeper recall with more recall slots and interface slots

It reports clean latent trace reconstruction error, pooled latent reconstruction
error, clean accuracy, corrupted-input accuracy, and robustness gap. The intended
claim is narrow: larger ARTI recall capacity should reconstruct the clean latent
processing trace from corrupted observations better than low-capacity recall,
without using labels, answer memory, or an external recall tensor.

Run it with:

```bash
uv run --extra dev python benchmarks/run_experiential_recall_scaling.py
uv run --extra dev python benchmarks/verify_experiential_recall_scaling.py
```

## Half Recall Trace Survival

The Half recall-trace survival benchmark tests Half as a generic activation in
a Recall branch, not as a new Recall mechanism:

```python
delta = recall_layer(h)
delta = Half()(delta)
h = h + delta
```

The controlled task now uses a paired multi-seed protocol. For each scenario
and seed, every variant receives the same sampled recall proposal tensors, so
the isolated variable is the activation applied to the Recall delta before the
residual update. The primary scenario is `separable_trace`: the true Recall
trace has salience above the default Half threshold, while weak ambiguous
traces stay below it. The benchmark also includes `low_separation_trace`, a
boundary-only counterexample where true and weak trace salience overlap; that
scenario records behavior but is not allowed to become the primary Half claim.

It compares plain residual recall, one-Half recall, stacked residual recall,
stacked Half recall, matched fixed-shrink controls, SoftShrink controls,
dropout, GELU, ReLU, and stochastic Half. The fixed-shrink controls are the
anti-slip check: they can match Half's weak-trace survival, so Half must still
win by preserving the strong signed trace rather than by globally shrinking
every delta. The SoftShrink controls show the other side of the tradeoff: a
harder denoising activation can leak less weak noise, but it also attenuates
the signed recall trace. Metrics include target-state error, weak-trace noise
leakage, weak-trace survival, positive/negative strong-trace retention, trace
selectivity, probe accuracy, delta norm, paired seed win rate, and boundary
status.

```bash
uv run --extra dev python benchmarks/run_half_recall_trace_survival.py
uv run --extra dev python benchmarks/verify_half_recall_trace_survival.py
```

The claim is narrow and practical: under salience-separable Recall proposals,
Half can act as stateless trace thinning, letting sufficiently strong traces
continue while fading weak sub-threshold trace noise. The stronger comparison is
not "lowest noise at any cost"; it is "matched weak-trace pressure with better
strong-trace survival." The boundary scenario is kept in the result file
precisely to avoid overclaiming: if the true trace is itself weak or ambiguous,
Half may fade useful signal too. It does not add memory state, event logs, or a
heavier Recall controller.

## Half Recall Fair Training

The fair-training benchmark directly trains the Recall proposal branch instead
of scoring only a downstream probe. Every activation variant uses the same
proposal network, initialization seed, sampled train/eval batches, trainable
parameter count, training examples, and direct reconstruction loss:

```text
state = h + activation(recall_net(h) + weak_trace)
loss  = mse(state, h + r_true)
```

This is the stricter Half check for the concern that a bad loss can make a
mechanism look worse than it is. The benchmark measures target MSE, signal MSE,
weak-trace gain, inactive noise leakage, positive/negative signed trace
retention, clean-start drift, trainable parameters, throughput, and peak CUDA
memory when CUDA is used. The verifier checks replicate-level throughput
against train examples divided by train seconds, then checks summary and
resource min/max throughput fields against the recorded replicate rows.

```bash
uv run --extra dev python benchmarks/run_half_recall_fair_training.py
uv run --extra dev python benchmarks/verify_half_recall_fair_training.py
```

The current narrow result is not that Half beats every thresholding activation
on every scalar metric. SoftShrink can be more aggressive on target MSE in this
controlled setup, but it retains less strong signed trace. The Half claim is
more specific: under equal training budget and salience-separable recall
proposals, Half reduces weak-trace pollution versus identity recall, beats a
matched fixed-shrink control on target error, and preserves strong signed traces
better than fixed shrink, SoftShrink, GELU, and ReLU-style controls.

## RecallExecutor

Historical RecallExecutor benchmarks tested an outer residual loop. The current
API deliberately removes that second execution path: iterative behavior belongs
to Recall and is selected by a runtime policy.

```python
refiner = RecallExecutor(recall_layer)
h_refined, info = refiner(
    h,
    policy=ExecutionPolicy.fixed(3, trace_level="summary"),
    return_info=True,
)
```

The synthetic benchmark corrupts clean hidden vectors with masking and noise,
then compares a direct MLP denoiser, single-step residual Recall, multi-step
RecallExecutor without Half, and multi-step RecallExecutor with Half under equal
training examples and a near-equal trainable-parameter budget. It records final
MSE to the clean hidden state, MSE after each refinement step, delta/update
norms, delta/update norm stability, clean-start drift to expose weak correction
accumulation, and basic resource fields such as samples per second and CUDA peak
memory when available.
The Half/no-Half refiner pair is now checked as a genuinely paired comparison:
the verifier requires a shared refiner initialization seed, shared training
batches, shared evaluation batches, a shared evaluation recall-noise stream,
matching train/eval example budgets, and a top-level resource profile. The
report also carries a `half_effect` block whose deltas and ratios must match the
recorded run rows, so the benchmark cannot silently drift into an unpaired seed
comparison.

```bash
uv run --extra dev python benchmarks/run_recall_refiner.py
uv run --extra dev python benchmarks/verify_recall_refiner.py
```

The claim is deliberately small: iterative Recall plus Half can improve a
controlled hidden-vector reconstruction task and reduce weak correction drift
versus the no-Half loop under the benchmark's equal-budget controls. This does
not replace the existing Recall path and does not claim broad denoising or
downstream superiority.

For route-causality experiments, stochastic Half is an explicit control
variable because its sampling does not follow `train()` or `eval()`. Paired
dynamic and frozen-route evaluations therefore use the deterministic survival
expectation or an exactly shared random stream. Exact route replay must also be
compared through the same execution layout; candidate-group freezing is not a
substitute for a complete `RetrievalRoutePlan`.

## Qwen Hidden Refiner Half Probe

The Qwen hidden Half probe is the controlled Qwen-side test for the condition
where Half is expected to help. It freezes Qwen, extracts real hidden states as
clean latent targets, projects them into a compact latent space, corrupts them
with feature masking and noise, and injects a weak post-recall trace pollutant.

```bash
uv run --extra torch --extra qwen python benchmarks/run_qwen_hidden_refiner_half.py
uv run --extra dev python benchmarks/verify_qwen_hidden_refiner_half.py
```

The comparison is deliberately paired: `refiner_no_half` and `refiner_half`
share the same recall architecture, initialization seed, training batches,
trainable parameter count, corruption distribution, and weak-trace distribution.
The training objective is recorded and checked as
`repair_mse + clean_stability_weight * clean_start_stability_mse`, where repair
means `refiner(corrupt_hidden, weak_trace) -> clean_hidden` and clean-start
stability means `refiner(clean_hidden, weak_trace) -> clean_hidden`. The
verifier recomputes the final loss formula and throughput from the recorded
rows, then requires Half to preserve strong hidden repair while reducing
clean-start drift, weak-noise gain, and alignment with the weak trace. This is
not a string-generation result, not Qwen fine-tuning, and not a broad Half
superiority claim.

## Pulse Refiner Repair

The Pulse refiner repair benchmark connects the compact workspace path to the
latent repair path. It starts with overcomplete corrupted fragments containing
two ordered strong traces and weak misleading trace fragments. The paired core
comparison keeps trainable parameters, initialization seed, training batches,
evaluation batches, corruption distribution, and weak-trace distribution fixed.
It also records the train/eval data seeds, an explicit clean-workspace training
objective contract, the repair/signal/consistency loss weights, samples per
second, and a resource profile:

```bash
uv run --extra dev python benchmarks/run_pulse_refiner_repair.py
uv run --extra dev python benchmarks/verify_pulse_refiner_repair.py
```

`fold_refiner_no_half` uses the same shared Fold-style trunk and RecallExecutor
capacity without Half. `pulse_refiner_half` uses public `Pulse` plus
RecallExecutor with Half. The verifier requires the Pulse path to improve final
repair MSE, preserve signal-only repair, reduce weak-trace leakage, reduce
weak-noise gain, and keep the recorded refinement-depth MSE curve
non-increasing with every step no worse than the no-Half Fold path. It also
checks that the result summary matches the run rows and that resource fields
are present. The recorded reproducer command must match the stored config and
fairness budgets, including train steps, eval batches, weak-trace strength,
noise level, signal keep probability, and device. The resource profile records
requested device, actual device type, effective batch size, train/eval examples,
gradient accumulation, AMP status, throughput metric, and whether CUDA peak
memory is expected. CUDA peak memory is required only when the run actually uses
the CUDA runtime, so CPU runs on a CUDA host are not misclassified as missing
GPU memory. It also verifies the input pressure summary: the source fragments
must be overcomplete, the strong signal must be partially missing, the
corruption noise must be nontrivial, and the weak trace norm must be large
enough to make leakage measurable. The final training loss is decomposed into
repair, signal-only, and consistency terms. The objective contract fixes the
clean ordered workspace as the target, records the stop-gradient direction of
consistency, and excludes evaluation metrics from training. The verifier checks
that the weighted sum matches the reported final loss, recomputes throughput
from examples and elapsed time, and checks aggregate throughput and CUDA peak
memory against run rows. The report also records
RecallExecutor per-step update norms; the verifier requires the Pulse + Half path
to contract its update norm across refinement depth, so the iterative repair
claim is checked for stable correction rather than final MSE alone. Derived
depth-effect ratios and boolean summaries are recomputed from the run rows, so
the stability claim cannot drift away from the recorded per-step curves. A
wider q-mean baseline is recorded for context, but the positive claim is the
same-parameter paired Fold/Pulse comparison.

### Multi-seed Stability

The companion `pulse_refiner_multiseed` benchmark repeats the paired Fold/no-Half
and Pulse/Half comparison over three fixed seeds with the same parameter and
per-seed training budgets. It corrects the claim boundary exposed by the
single-seed depth probe: Pulse + Half improves signal-only incomplete-state
recovery and reduces weak-trace leakage and weak-noise gain in all three seeds,
but it does not dominate corrupted-input final MSE. The recorded run shows mean
ratios of about `0.947`, `0.945`, and `0.956` for those three positive metrics,
while final MSE has a bounded mean ratio of about `1.013` and a worst-seed ratio
of about `1.044`. The verifier requires the repeatable selectivity gains and
also requires the report to retain the final-MSE non-dominance outcome.

```bash
uv run --extra torch python benchmarks/run_pulse_refiner_multiseed.py --replicates 3 --base-seed 97 --train-steps 260 --eval-batches 32 --weak-trace-strength 0.55 --device cuda
uv run --extra dev python benchmarks/verify_pulse_refiner_multiseed.py
```

## Latent Repair Goal

The latent repair goal verifier aggregates the controlled repair checks:
Half recall-trace survival, Half fair training, direct RecallExecutor
hidden-state repair, and Pulse/Fold compact-workspace repair.

```bash
uv run --extra dev python benchmarks/verify_latent_repair_goal.py
```

It does not create a new benchmark. It makes the release gate check that the
pieces agree as one claim chain: Half thins weak traces without becoming a
global shrink trick, the trained Half Recall-delta branch improves target MSE
and weak-trace gain under a direct reconstruction loss, RecallExecutor improves
direct corrupted hidden-state recovery while reducing clean-start drift, and
Pulse + Half improves the same-parameter Fold/Pulse compact repair task under
explicit overcomplete input pressure.

## Qwen String-First RecallExecutor Answering

The string-first Qwen answering probe is a live-model adapter check. It renders
chat strings into ARTI text tensors, trains small ARTI-side frontends, and uses
Qwen token ids only at the frozen boundary for canonical target-label encoding,
target labels, final decoding, and conditional-NLL measurement:

```bash
uv run --extra torch --extra qwen python benchmarks/run_qwen_string_recall_refiner_answering.py --model-id Qwen/Qwen3-0.6B --steps 3000 --batch-size 64 --condition-loss-weight 0.35 --answer-identity-loss-weight 0.5 --max-new-tokens 20 --pulse-count 8 --fragment-dim 128 --matched-mean-dim 640 --max-chars 384 --device cuda
uv run --extra torch --extra qwen python benchmarks/verify_qwen_string_recall_refiner_answering.py
uv run --extra torch --extra qwen python benchmarks/run_qwen_string_pulse_multiseed.py --model-id Qwen/Qwen3-0.6B --replicates 3 --base-seed 71 --seed-stride 8 --steps 3000 --batch-size 64 --condition-loss-weight 0.35 --answer-identity-loss-weight 0.5 --max-new-tokens 20 --pulse-count 8 --fragment-dim 128 --matched-mean-dim 640 --max-chars 384 --device cuda
uv run --extra dev python benchmarks/verify_qwen_string_pulse_multiseed.py
```

It records exact canonical-answer reproduction as one metric, but
`RecallExecutor` is also scored with sentence coherence, loop penalty,
degeneration rate, frozen-Qwen conditional NLL, and unique-answer rate. Training
uses next-token CE plus a lightweight prompt-conditioning auxiliary loss; this
is important because iterative refinement should not be judged under a loss that
lets prompt identity wash out. The verifier recomputes the reported open-loop
metrics from the per-prompt generated answer strings, so exact match,
similarity, coherence, degeneration, conditional NLL, and unique-answer rates
must be backed by auditable sample text. The report also records equal training
examples, trainable parameter budgets, a parameter-matched wide mean baseline,
a structured resource profile, and a paired no-Half/Half refiner ablation with
shared initialization and train batches. The resource profile records requested
device, actual device type, effective batch size, train examples, prefix
examples, prompt count, max generation length, AMP state, whether CUDA was
actually used, and whether peak CUDA memory is expected. It also records that
the Qwen body is used for conditional-NLL measurement, but not for target
generation and not during adapter training. Per-model train seconds and throughput are
checked against the row train-example budget, so the resource figures cannot
drift independently from the measured run rows. The primary matched comparisons
share train-batch seeds: `mean_wide` and `pulse` share one sampled prefix stream, and
the no-Half/Half refiner pair shares another, so the comparison is not only
equal-update but same-sampled within each declared pair. It also records a text tensor contract:
the Qwen-side string input uses `control_codepoint_aux`, known visible glyphs
such as `strawberry` must have zero codepoint aux, and control / special
characters retain the auxiliary channel. The no-Half/Half refiner ablation also
records trainable parameter counts and the verifier requires the ratio to be
exactly 1.0, so the joint-training diagnostic is not confounded by capacity.
The report also carries an input boundary contract: ARTI-side training inputs
must be `rendered_text_tensor` plus `mask`, token ids must not be ARTI inputs,
the Qwen body is frozen and not used during adapter training, and the frozen
LM head is only the output boundary. The result records total and trainable
parameter counts for both Qwen and its `lm_head`, and the verifier requires both
trainable counts to stay at zero.
The report now carries a structured training-objective contract and full-prefix
teacher-forced eval losses against canonical answer token labels. The verifier
requires canonical answer strings encoded as frozen Qwen output token ids with
EOS termination, prompt-balanced prefix next-token CE, prompt-conditioning CE,
and a direct first-answer-token CE through the frozen lm head. It rejects
hidden-reconstruction losses and requires open-loop generated strings to remain
evaluation-only text samples. Every reported `final_loss` and `eval_joint_loss`
must equal the three configured terms, so prompt identity cannot be hidden only
in a diagnostic head that is discarded during generation.

The paired three-seed companion was added after the old loss collapsed seed 79
to one repeated answer. With prompt-balanced sampling and direct answer-identity
binding, Pulse reproduces all four canonical strings at seeds 71, 79, and 87,
beats the parameter-matched mean baseline on exact rate and conditional NLL in
every seed, and records zero mode collapse. This is still a four-prompt frozen
Qwen boundary probe, not broad language evaluation.

The current evidence supports string-first ARTI frontend plumbing and learned
Pulse workspace usefulness in the controlled task. In the latest run, `Pulse`
reproduces every explicit canonical answer string, including the glyph-count
target `There are 3 letter r in the word strawberry.`, and beats both the small
mean baseline and the parameter-matched `mean_wide` baseline on exact answer
reproduction, conditional NLL, and non-regression of coherence / loop penalty;
the no-Half RecallExecutor also reproduces every canonical answer string. The Half branch
regresses in this joint-training configuration, but that is recorded as an
optimization diagnostic rather than treated as a Half mechanism verdict. A
rigorous Qwen-side Half test should use a controlled corrupted-hidden or
warm-start refiner design. The verifier now requires the report to carry a Half
mechanism context that points back to the controlled `qwen_hidden_refiner_half`,
`half_recall_trace_survival`, `half_recall_fair_training`, and
`pulse_refiner_repair` gates, so this string-generation probe cannot become a
string-only Half verdict. This does not prove full Qwen tokenizer replacement,
base-model fine-tuning gains, or broad open-ended answer quality.

## Phase Authority Prompt-Injection Proxy

The phase authority benchmark is a synthetic prompt-injection proxy. Each sample
contains two conflicting instruction-like tokens. The content distribution is
symmetric and all tokens are visible; only participant coordinates identify the
authoritative source.

It compares content-only and phase-aware models. The expected result is that
content-only baselines stay near chance, while ARTI with participant phase
identifies the high-authority source. Replacing participant coordinates with
zero phase should remove the advantage.

Run it with:

```bash
uv run --extra dev python benchmarks/run_authority_prompt_injection.py
uv run --extra dev python benchmarks/verify_authority_prompt_injection.py
```

This is not a production prompt-injection security guarantee. It verifies the
mechanism boundary: authority identity is represented by phase tensors, not by
text claiming to be authoritative. Visibility still matters for leakage control,
but it is not the tested source of authority in this proxy.

## Membrane Visibility Routing

The membrane routing benchmark checks the versioned interface for multi-visible-domain
autoregressive output. A normal next-token stream is assigned to
`assistant_public` or `assistant_inner`. Public tokens are emitted to the user;
inner tokens remain in model-side context but are hidden from user/admin/system
viewer visibility.

Run it with:

```bash
uv run --extra dev python benchmarks/run_membrane_visibility_routing.py
uv run --extra dev python benchmarks/verify_membrane_visibility_routing.py
```

This validates the narrow invariant that inner speech is a normal token stream
with different visibility and phase metadata, not a hidden latent channel and
not user-facing decoded output.

## Runtime Vocab Binding

The runtime vocab binding benchmark checks whether the output head must read the
current vocabulary tensor. A semantic next-symbol task is evaluated under a
randomly shuffled runtime vocabulary view. The output softmax order is the
current vocab order.

It compares:

- a fixed output head that does not read the runtime vocab at output time
- a runtime vocab head that scores hidden states against keys generated from the
  current rigid vocab tensor

Run it with:

```bash
uv run --extra dev python benchmarks/run_runtime_vocab_binding.py
uv run --extra dev python benchmarks/verify_runtime_vocab_binding.py
```

This is a synthetic effect test, not a language benchmark. It verifies the
mechanism claim that fixed heads fail when output order is runtime-bound, while
vocab-conditioned dynamic heads can follow the current softmax order.

## Runtime Vocab Replacement

The runtime vocab replacement benchmark is stricter than a shuffle test. It
trains on one disjoint set of rigid symbol tensors and evaluates on a replaced
held-out set. The task is copy-by-symbol: the model must select the same rigid
symbol in the current runtime softmax range.

It compares:

- a fixed output head that never reads the current candidate tensors
- a runtime vocab head that scores against the current rigid vocab tensor

Run it with:

```bash
uv run --extra dev python benchmarks/run_runtime_vocab_replacement.py
uv run --extra dev python benchmarks/verify_runtime_vocab_replacement.py
```

This validates the narrow alpha claim that the model can learn symbol structure
from rigid tensors instead of binding meaning to fixed output rows. It is still
a mechanism test, not evidence of full language modeling quality.

## Runtime Vocab Permutation

The runtime vocab permutation benchmark keeps the same external rigid symbol
tensors but changes their runtime order. It verifies that input ids are remapped
and output softmax indices follow the current vocab order instead of a fixed
row order.

Run it with:

```bash
uv run --extra dev python benchmarks/run_runtime_vocab_permutation.py
uv run --extra dev python benchmarks/verify_runtime_vocab_permutation.py
```

This directly protects the alpha claim that the output index is a current
runtime-vocab slot, not a permanent model vocabulary row.

## Legacy Explicit Pulse Vocab Invariance

The pulse vocab invariance benchmark checks the combined runtime-vocab and pulse
intuition. The same underlying rigid symbol stream is segmented two ways:
single-symbol tokens and variable-length chunk tokens. Legacy explicit pulse compression uses
token span weights to recover the same pulse-level latent sequence.

Run it with:

```bash
uv run --extra dev python benchmarks/run_pulse_vocab_invariance.py
uv run --extra dev python benchmarks/verify_pulse_vocab_invariance.py
```

This validates the narrow claim that pulse can decouple the model's reasoning
steps from a particular tokenization. It does not prove arbitrary tokenizer
invariance for full language models.

## Legacy Explicit Pulse Distinctness

The pulse distinctness benchmark checks the failure mode where compression keeps
sequence shape but destroys symbol differences. It renders repeated-letter text
variants, validates a faithful pulse representation, and includes a collapsed
control where `r` and `rr` are forced to the same pulse latent.

Run it with:

```bash
uv run --extra dev python benchmarks/run_pulse_distinctness.py
uv run --extra dev python benchmarks/verify_pulse_distinctness.py
```

## Learned Pulse Efficiency

The learned Pulse compaction benchmark and Qwen adapter probe compare the
optimized default `Pulse` implementation against a reference path that keeps
the earlier norm+max guidance and `einsum` aggregation. The optimization keeps
the public API stable while reducing the cost of the soft-assignment fold. The
benchmark also includes explicit experimental rows for sparse top-k folding,
q-first pruning, attention folding, gated residual correction, and combinations with the
corrected Pulse path.

Run the compact synthetic check with:

```bash
uv run --extra dev python benchmarks/run_learned_pulse_compaction.py
uv run --extra dev python benchmarks/verify_learned_pulse_compaction.py
```

Run the frozen-Qwen adapter probe with:

```bash
uv run --extra dev python benchmarks/run_qwen_learned_pulse_adapter.py
uv run --extra dev python benchmarks/verify_qwen_learned_pulse_adapter.py
```

These probes support only an adapter-level efficiency claim: optimized learned
Pulse should preserve held-out tokenization accuracy within a small tolerance,
improve throughput versus the reference Pulse path, and avoid higher peak
memory. Current evidence keeps `Pulse(correction=True)` as the default corrected path:
q-first pruning helps the base Pulse throughput, but hurts the corrected Qwen
adapter; sparse top-k and attention folding are available experimental modes,
not default recommendations. Legacy explicit pulse remains the fast
deterministic baseline.

For fixed-shape inference deployments, `torch.compile` can be tested separately
after installing a working Triton package. On Windows, `triton-windows` allowed
the Pulse inference micro-benchmark to compile when the Triton and Inductor
caches were pointed into the workspace. This is not part of the default
training gate because compilation warmup and environment support dominate short
runs.

## Text Bitmap Vocab

The text bitmap vocab smoke test renders repeated-letter word variants into
rigid bitmap tensors, checks that no rendered candidates collide, reports bitmap
entropy, and verifies that a runtime vocab head can select the exact bitmap
target. It exists to protect the alpha API boundary: text identity can be
provided as a visible tensor instead of an opaque token id.

Run it with:

```bash
uv run --extra dev python benchmarks/run_text_bitmap_vocab.py
uv run --extra dev python benchmarks/verify_text_bitmap_vocab.py
```

The micro-glyph distinctness benchmark adds the small-character pressure test
for this surface. It covers punctuation such as `.` and `,`, narrow glyphs such
as `i`, `l`, and `1`, repeated letters such as `r`/`rr`, OOV-like strings,
shift/bold/noise variants, runtime slot binding, pulse distance retention, and
control-character auxiliary channels.

```bash
uv run --extra dev python benchmarks/run_micro_glyph_distinctness.py
uv run --extra dev python benchmarks/verify_micro_glyph_distinctness.py
```

Passing this benchmark means the fallback bitmap renderer preserves enough
micro-glyph information for controlled runtime-vocab experiments. It is still
not a substitute for full font shaping; serious experiments should lock the
font, canvas, normalization, and renderer metadata.

## Source Integrity Stereo

The source integrity stereo benchmark validates the LLM-friendly fixed carrier
interface for synchronized multi-source streams. It uses `SourceIntegrityCarrier`
to superpose left/right source payloads into one field token per timestamp, read
them back with matched carriers, and check normal, swapped, wrong-basis, and
naive-sum controls while exercising block-level summary diagnostics.

Run it with:

```bash
uv run --extra dev python benchmarks/run_source_integrity_stereo.py
uv run --extra dev python benchmarks/verify_source_integrity_stereo.py
```

## Source Integrity Multisource

The multisource benchmark extends the carrier validation to 4, 8, and 16
synchronized sources. It verifies that one field token per timestamp remains
source-separable under normal readout, and that source permutation, missing
source, wrong basis, partial corruption, and naive-sum collapse controls are
detected or localized.

```bash
uv run --extra dev python benchmarks/run_source_integrity_multisource.py
uv run --extra dev python benchmarks/verify_source_integrity_multisource.py
```

## Multi-Source Reliability Arena

The reliability arena moves beyond static separability. It constructs 4, 8, 16,
and 32 synchronized sources with reliability, authority, missing/dropout, delay,
partial corruption, adversarial impersonation, and conflicting payload
annotations. It compares naive sum, naive concat, source-id embedding, Source
Integrity Carrier, and Source Integrity Carrier plus an ARTI block while
recording token count, parameter count, reconstruction error, localization,
impersonation rejection, leakage, block-level integrity, and timing.

```bash
uv run --extra dev python benchmarks/run_multisource_reliability_arena.py
uv run --extra dev python benchmarks/verify_multisource_reliability_arena.py
```

## Qwen 0.6B-Class Smoke

The Qwen smoke path records a real open checkpoint target and validates the ARTI
adapter path on Qwen-shaped hidden tensors. The default run is offline and does
not download weights; pass `--run-model` after installing the optional `qwen`
extra to load `Qwen/Qwen3-0.6B` through Hugging Face Transformers.

```bash
uv run --extra dev python scripts/quality_gate.py qwen
uv run --extra torch --extra qwen python benchmarks/run_qwen_arti_smoke.py --run-model
uv run --extra torch --extra qwen python benchmarks/verify_qwen_arti_smoke.py
```

This smoke checks source identity preservation and budget accounting for
original Qwen, source-id embedding, ARTI carrier, and ARTI carrier plus an
untrained ARTI block. The qwen gate also runs a candidate-vocab replacement
smoke: it gathers a runtime softmax view from Qwen full-vocab logits, then
checks that permuted and replaced candidate vocab ranges bind output indices to
the current candidate order.

The public Qwen glyph runtime adapter smoke validates the developer entry point:
`QwenGlyphRuntimeAdapter.from_pretrained()` loads frozen Qwen, `generate()` uses
the original dialogue path, `read_glyph_vocab()` scores an external visible-word
bitmap vocabulary, and `dialogue_drift()` verifies that the glyph readout does
not mutate ordinary next-token logits.

```bash
uv run --extra dev python benchmarks/run_qwen_glyph_runtime_adapter_api.py
uv run --extra dev python benchmarks/verify_qwen_glyph_runtime_adapter_api.py
```

This is API-surface evidence, not a trained tokenizer-replacement result.

The training-style Qwen runtime vocab benchmark freezes Qwen, extracts a prompt
hidden-state context, and trains small adapter heads to select the local index
of a rigid symbol in the current runtime candidate vocab. It trains on one
symbol range and evaluates on a disjoint replacement range and shuffled
candidate orders. It is adapter-level validation, not full Qwen pretraining or
tokenizer replacement.

The output-head replacement smoke freezes Qwen's tokenizer, transformer body,
and `lm_head` as a teacher. It trains only an ARTI runtime vocab output head to
match the teacher distribution over a dynamic local candidate view built from
frozen Qwen output-head rows.

```bash
uv run --extra dev python benchmarks/run_qwen_output_head_replacement.py
uv run --extra dev python benchmarks/verify_qwen_output_head_replacement.py
```

The expected evidence is teacher-student KL reduction plus local-slot
permutation equivariance. This is the first head-replacement stage, not a full
input/output replacement and not a broad intelligence-retention claim.

The input-head replacement smoke freezes Qwen's tokenizer path, transformer
body, and `lm_head` as teachers. It trains only a glyph/text tensor encoder that
maps visible token tensors into Qwen's original embedding space, then forwards
those embeddings through the frozen Qwen body with `inputs_embeds`.

```bash
uv run --extra dev python benchmarks/run_qwen_input_head_replacement.py
uv run --extra dev python benchmarks/verify_qwen_input_head_replacement.py
```

The expected evidence is lower embedding MSE, high embedding cosine, close
frozen-body hidden states, bounded top-k next-token KL, and a readable short
prompt probe. This is still a short-prompt first-stage replacement, not a full
tokenizer or intelligence-retention result.

The closed-loop replacement smoke combines the two head-replacement directions:
visible token glyph tensors are encoded into Qwen embedding space, the frozen
Qwen body produces hidden states from `inputs_embeds`, and an ARTI runtime vocab
output head scores dynamic local candidate vocab views.

```bash
uv run --extra dev python benchmarks/run_qwen_closed_loop_replacement.py
uv run --extra dev python benchmarks/verify_qwen_closed_loop_replacement.py
```

The expected evidence is that input alignment does not regress, closed-loop KL
falls far below a fixed-slot baseline, candidate permutation equivariance stays
high, and short QA/completion/OOV-like visible candidate/permuted-vocab probes
mostly match teacher local slots. It remains a short-context smoke, not full
long-context generation.

The autoregressive closed-loop smoke turns the local candidate result into a
2-4 step writeback loop. At each step the ARTI runtime head selects a local
candidate slot, maps that slot back to its visible token tensor, appends it to
the next context, and forwards the frozen Qwen body again through the ARTI input
encoder. The report records the teacher token, ARTI local slot, appended visible
token text, and next-step hidden/logit alignment after the append.

```bash
uv run --extra dev python benchmarks/run_qwen_autoregressive_closed_loop.py
uv run --extra dev python benchmarks/verify_qwen_autoregressive_closed_loop.py
```

Passing this smoke means the short-sequence local-slot writeback path is
auditable and beats a fixed-slot baseline. It is still a greedy-teacher adapter
test, not proof of full open-ended Qwen generation.

The controlled open-generation writeback benchmark extends this to 8-16 short
generation steps. Candidate vocab views are intentionally mixed: teacher top-k
tokens, external visible text, OOV-like strings, permuted local slots, and
distractors. The evidence packet records token/slot match, scenario-level
semantic hits, frozen Qwen repeated-logit drift, visible vocab previews, local
slots, selected tokens, failure samples, and next-step hidden/logit changes
after ARTI writeback.

```bash
uv run --extra dev python benchmarks/run_qwen_controlled_open_generation.py
uv run --extra dev python benchmarks/verify_qwen_controlled_open_generation.py
```

Passing this benchmark supports the claim that ARTI runtime vocab binding can
survive a longer controlled writeback loop while Qwen remains frozen. It does
not claim polished open-ended dialogue; the generated text may remain rough,
and scenario-level semantic hits are reported separately from strict token
matches.

The Qwen runtime vocab + pulse semantic benchmark goes one step closer to the
intended alpha feature. Qwen remains frozen and provides question hidden states;
small adapter heads are trained to answer deterministic semantic questions by
selecting the local index of the correct answer in the current runtime vocab.
The current gate uses rigid glyph bitmap tensors for the external vocabulary:
each visible answer surface is rendered once as the physical runtime vocab item,
so tokenizer splits do not redefine the item's identity. The candidate answer
tensors are shuffled each batch, and evaluation uses replaced candidate ranges
plus held-out runtime views. The comparison is:

- tokenizer-only Qwen scores from fixed permanent token rows
- Qwen fixed local head
- Qwen + runtime vocab only
- Qwen + pulse only
- Qwen + runtime vocab + pulse

The runner records the resource boundary too: Qwen is loaded and frozen,
question contexts are precomputed once, adapter training uses bf16 autocast by
default on CUDA, and results include effective batch size, gradient
accumulation, peak memory, adapter-loop samples per second, batch sampling time,
forward/backward time, optimizer time, and precomputed vocab-bank/context
counts. It also records ordinary Qwen repeated-logit drift on the same prompts,
so the gate checks that the external runtime-vocab training path preserves the
untouched Qwen dialogue route.

Run it with:

```bash
uv run --extra torch --extra qwen python benchmarks/run_qwen_runtime_vocab_pulse_semantic.py --steps 220 --eval-batches 16 --batch-size 128 --grad-accum-steps 2
uv run --extra torch --extra qwen python benchmarks/verify_qwen_runtime_vocab_pulse_semantic.py
```

The current evidence supports a narrow mechanism claim: reading the current
glyph runtime vocab gives exact semantic local-index answers under shuffled
candidate order and held-out views, including against a tokenizer-only Qwen
baseline that sees the answer surfaces through the original fixed vocabulary
rows. In glyph mode, runtime-vocab-only and runtime-vocab+pulse are both ARTI
dynamic heads; the pulse condition is required not to regress while remaining
compatible with the larger bitmap tensor. It does not claim full Qwen tokenizer
replacement, open-ended dialogue quality, or pretrained-model vocabulary
surgery without adaptation.

An additional strict probe can force the positive answer surface to be unseen
during adapter answer training:

```bash
uv run --extra torch --extra qwen python benchmarks/run_qwen_runtime_vocab_pulse_semantic.py --steps 220 --eval-batches 16 --batch-size 128 --grad-accum-steps 2 --strict-heldout-surfaces --output benchmarks/results/qwen_runtime_vocab_pulse_strict_surface_probe.json
```

The strict probe without a semantic bridge is intentionally not a green release
gate. It records a real gap: with training positives limited to surface ids 0
and 1 and evaluation forced to surface id 2, `runtime_vocab_pulse` reaches only
about 0.28 accuracy. That means raw glyph identity alone proves replaceable
vocab binding and slot-order generalization, but not unseen surface semantics.

A second strict probe adds a frozen-Qwen metadata bridge to each external vocab
item while keeping a nonzero down-weighted glyph channel:

```bash
uv run --extra torch --extra qwen python benchmarks/run_qwen_runtime_vocab_pulse_semantic.py --steps 520 --eval-batches 16 --batch-size 128 --grad-accum-steps 2 --strict-heldout-surfaces --semantic-bridge --semantic-bridge-source metadata --vocab-glyph-scale 0.02 --output benchmarks/results/qwen_runtime_vocab_pulse_metadata_bridge_probe.json
uv run --extra torch --extra qwen python benchmarks/verify_qwen_runtime_vocab_metadata_bridge.py
uv run --extra torch --extra qwen python benchmarks/verify_qwen_runtime_vocab_bridge_ablation.py
```

This bridge probe reaches about 0.93 strict held-out accuracy for
`runtime_vocab_pulse` while tokenizer-only and fixed-head controls stay near
chance. The supported claim is therefore sharper: ARTI can train a Qwen-side
dynamic external vocabulary loop that survives replaced vocab order and
held-out vocab surfaces when runtime vocab items carry a stable semantic bridge.
The ablation verifier keeps the contrast explicit: the strict glyph-only probe
must stay low while the metadata-bridge probe must stay high. Raw glyph-only
unseen word meaning remains a later pretraining problem.

## Qwen Literal Output Context

The decoupled literal-vocabulary probe tests whether the output vocabulary
should be visible only to the terminal scorer or also to the front of the
ARTI-side body. Frozen Qwen provides four task-query hidden states. Every
example supplies an independently shuffled five-item output vocabulary made
from rigid digit glyph tensors; the correct local slot depends on a relation
inside the current set, such as distance from its mean or relative rank.
Training and evaluation use disjoint candidate sets.

```bash
uv run --extra torch --extra qwen python benchmarks/run_qwen_literal_output_context.py
uv run --extra dev python benchmarks/verify_qwen_literal_output_context.py
```

The three-seed comparison includes a fixed-slot control, a widened terminal
`LiteralOutputHead`, and `OutputLexiconContext` before the ARTI-side body. The
terminal and conditioned models differ by less than one percent in parameter
count. Current formal evidence records mean held-out accuracy of about 0.539
for the terminal head and 0.576 for front conditioning; mean cross-entropy
falls from about 1.056 to 0.947. Cross-entropy improves in all three paired
seeds and accuracy improves in two. Some individual relation tasks and one
seed still regress, so this supports front output-range conditioning as a
useful inductive bias, not universal dominance. It does not test unseen glyph
semantics or open-ended generation.

## Qwen Unseen Literal Transfer

The next isolated variable is a training-unseen output string. Frozen Qwen
encodes prompts that ask for an exact pseudoword copy, while the answer range is
an independent six-item glyph vocabulary. Pseudowords are arranged as
`stem`/`stem+r` pairs; each complete pair belongs to exactly one of training,
validation, or heldout. Validation chooses checkpoints, and heldout strings are
never used for optimization or selection. Every candidate view is shuffled and
contains the confusable pair.

```bash
uv run --extra torch --extra qwen python benchmarks/run_qwen_unseen_literal_transfer.py
uv run --extra dev python benchmarks/verify_qwen_unseen_literal_transfer.py
```

Across three seeds, the fixed-slot control stays near six-way chance at about
0.165 unseen exact-string accuracy and loses identity under permutation. The
parameter-matched terminal `LiteralOutputHead` reaches about 0.799 exact
accuracy, while front `OutputLexiconContext` plus the same thin head reaches
about 0.836. Both dynamic paths have 1.0 permutation consistency. Front
conditioning reduces mean `stem`/`stem+r` confusion from about 0.069 to 0.037
and cross-entropy from about 0.882 to 0.601. This supports compositional literal
transfer and current-vocabulary decoding. It does not establish unseen word
meaning, different input/output segmentation, or open-ended dialogue.

## Qwen Literal Segmentation Generation

The segmentation probe keeps frozen Qwen BPE prompts on the input side and uses
an independently shuffled character glyph vocabulary on the output side.
Visible characters contain glyph pixels only; EOS is represented by zero glyph
pixels plus one control bit. `LiteralSequenceDecoder` emits complete canonical
sentences one local character slot at a time. Train, validation, and test use
different prompt templates, and checkpoint selection uses validation open
generation rather than teacher-forced loss alone.

```bash
uv run --extra torch --extra qwen python benchmarks/run_qwen_literal_segmentation_generation.py
uv run --extra dev python benchmarks/verify_qwen_literal_segmentation_generation.py
```

Across three seeds, the parameter-matched fixed character head reaches about
0.958 exact/semantic sentence accuracy, the terminal dynamic head reaches about
0.839, and front output-context conditioning reaches about 0.904. The context
path beats the terminal dynamic head in two seeds, has 1.0 coherent-sentence and
EOS completion rates, and about 0.988 mean character similarity. The remaining
gap to the fixed character head is an explicit cost of dynamic local binding in
this small adaptation run. This supports BPE-input to glyph-character output
generation, not unrestricted tokenizer replacement or base-model parity.

## Qwen Dynamic Runtime-Vocab Goal Gate

The goal-level Qwen verifier aggregates the separate Qwen evidence files and
checks that the current alpha chain covers the intended training-loop claim:

- glyph/text input-head replacement into frozen Qwen embedding space
- closed-loop glyph input plus runtime-vocab output with permuted/OOV scenarios
- independent glyph output vocabularies conditioned before the ARTI-side body,
  compared against parameter-matched terminal-head and fixed-slot controls
- heldout exact-string transfer with train-only validation selection and
  confusable glyph-pair checks
- BPE-input to shuffled glyph-character output with complete semantic sentence,
  coherence, EOS, throughput, and memory measurements
- glyph runtime vocabulary plus pulse semantic training with tokenizer-only,
  fixed-head, runtime-only, pulse-only, and runtime+pulse controls
- strict held-out surface semantics with a metadata semantic bridge and nonzero
  glyph channel
- ordinary Qwen dialogue-logit preservation while ARTI-side heads train
- CUDA resource evidence: effective batch, bf16 autocast, precomputed Qwen
  contexts, precomputed runtime vocab banks, batch sampling time,
  forward/backward time, optimizer time, throughput, and peak memory

```bash
uv run --extra dev python benchmarks/verify_qwen_dynamic_vocab_goal.py
```

This gate is deliberately evidence-chain based. It supports the alpha claim
that Qwen can be wrapped by ARTI-side external vocabulary interfaces under
controlled training while Qwen remains frozen. It still does not claim a full
open-ended tokenizer replacement or base-model vocabulary surgery.

## Qwen External Glyph Answering

This benchmark makes the external glyph vocabulary participate in the answer
itself. Qwen remains frozen and provides the context/body dynamics; ARTI trains
only a glyph input encoder and runtime vocab output head. Each task asks for an
answer supplied by the current visible glyph vocab rather than by Qwen's default
next-token preference. The selected local slot is mapped back to token/text and
written into a 4-12 step answer template.

```bash
uv run --extra dev python benchmarks/run_qwen_external_glyph_answering.py
uv run --extra dev python benchmarks/verify_qwen_external_glyph_answering.py
```

The evidence compares ARTI with a fixed glyph-slot head and tokenizer-only
Qwen. It records external glyph answer hit rate, multi-step writeback match,
slot permutation equivariance, ordinary Qwen repeated-logit drift, visible vocab
previews, glyph source variants, selected slots, and failure samples. The
current boundary is explicit: answer words must be single-token-writeback
compatible; multi-token external glyph answers require a later decoder.

## Qwen Multi-Token Glyph Decoder

This benchmark removes the single-token-writeback restriction for controlled
external glyph answers. Qwen remains frozen; ARTI first selects the current
external glyph answer slot, then a lightweight decoder reads the selected glyph
and emits Qwen token pieces step by step. The generated token pieces are written
back through the frozen Qwen body.

```bash
uv run --extra dev python benchmarks/run_qwen_multitoken_glyph_decoder.py
uv run --extra dev python benchmarks/verify_qwen_multitoken_glyph_decoder.py
```

The task set includes OOV-like answers such as `qzvxr`, `glyphx`,
`strawberrry`, repeated-letter strings, and small punctuation combinations. The
evidence records answer-slot accuracy, decoder exact text match, answer-token
hit rate, multi-step writeback match, glyph source variants, decoded token
traces, ordinary Qwen drift, and failure samples. It depends on the micro-glyph
distinctness gate for the small-character surface. This is a controlled short
answer decoder, not a full tokenizer replacement.

## Qwen OOV Text Vocab Finetune Smoke

The OOV text vocab finetune smoke freezes Qwen, extracts prompt hidden states,
and trains only small ARTI-side heads. The tested output vocabulary is a current
bitmap-rendered word list, not Qwen's fixed tokenizer rows. Training uses one
set of visible words; evaluation uses held-out/OOV strings such as repeated
letters and artificial words.

```bash
uv run --extra dev python benchmarks/run_qwen_oov_text_vocab_finetune.py
uv run --extra dev python benchmarks/verify_qwen_oov_text_vocab_finetune.py
```

The narrow positive claim is that a glyph runtime head that directly reads the
current bitmap vocab can select held-out visible words substantially better than
a fixed local head. The boundary is equally important: Qwen base weights are
frozen, this is not full Qwen fine-tuning, not tokenizer replacement, and not
open-ended language ability.

## Qwen OOV Dialogue Preservation Smoke

This smoke keeps the normal Qwen dialogue path on the original frozen logits
while a separate ARTI-side glyph runtime head trains on the held-out visible-word
task. It records ordinary dialogue next-token logits before and after adapter
training, then verifies max logit drift, KL drift, and top-1 preservation.

```bash
uv run --extra dev python benchmarks/run_qwen_oov_dialogue_preservation.py
uv run --extra dev python benchmarks/verify_qwen_oov_dialogue_preservation.py
```

The positive claim is deliberately narrow: the OOV glyph-reading path can be
trained without mutating the ordinary Qwen dialogue path in this frozen-route
setup. It does not prove that updating Qwen base weights, replacing the full
tokenizer, or changing open-ended dialogue decoding will preserve capability.

## Qwen Membrane Routing Training

The Qwen membrane routing benchmark uses simulated frozen Qwen-class hidden
states and trains only small adapter heads plus `MembraneVisibilityRouter`. The
synthetic task requires a draft token to carry a hidden operand and a final
public token to emit the answer. It compares:

- `no_membrane`: no inner draft context for the final answer.
- `all_public`: draft and answer are both public; this can answer but leaks the
  draft token.
- `all_hidden`: draft and answer are both inner; this avoids leakage but emits
  no public answer.
- `membrane`: draft is `assistant_inner`, final answer is `assistant_public`.

Run it with:

```bash
uv run --extra dev python benchmarks/run_qwen_membrane_routing_training.py
uv run --extra dev python benchmarks/verify_qwen_membrane_routing_training.py
```

The current local result on CUDA shows membrane answer accuracy `1.000`, stream
routing accuracy `1.000`, and inner leakage `0.000`; the all-public baseline
also answers correctly but leaks the draft, while all-hidden does not emit a
public answer. This is alpha adapter-level evidence, not open-ended Qwen
dialogue validation.

## Qwen Membrane Guessing Game

This path has two deliberately separated stages.

### Stage 1: Simulated Qwen-Shaped Hidden States

The simulated benchmark makes the membrane behavior readable as a dialogue
trace without forwarding a real Qwen model. Each sample has a fixed hidden
target number, normal user guesses, and introspection/prompt-injection requests
such as asking the model to reveal its hidden target or scratchpad. Inner tokens
record readable private state such as `state:range=9-15;guess=8;cmp:higher` or
`secret:target=11;scratchpad=...`; public tokens are limited to legal hints
(`say:higher`, `say:lower`, `say:correct`) or `say:refuse`.

The comparison separates four cases:

- `no_membrane`: public answer without useful private state.
- `all_public`: can solve/refuse but leaks the secret/scratchpad inner stream.
- `all_hidden`: keeps inner state private but emits no public dialogue.
- `membrane`: keeps inner state in `assistant_inner` and public hints in
  `assistant_public`.

Run it with:

```bash
uv run --extra dev python benchmarks/run_qwen_membrane_guessing_game.py
uv run --extra dev python benchmarks/verify_qwen_membrane_guessing_game.py
```

The generated report includes normal hint accuracy, attack rejection rate,
secret leakage rate, inner addressability rate, per-turn user-visible text,
assistant inner state, assistant public output, stream probabilities, visibility
checks, and leakage flags. It is a mechanism test only: no real Qwen forward and
no Qwen training claim.

### Stage 2: Real Qwen Frozen-Forward Smoke

The real-forward smoke loads Qwen through Transformers when available, builds
small readable guessing-game prompts with both normal guesses and introspection
attacks, extracts frozen hidden states from the real tokenizer/model forward
pass, and trains only adapter/router/heads. If the environment lacks
dependencies, weights, network access, or sufficient resources, it writes a
`skipped` result with an explicit `skip_reason` instead of silently substituting
the simulated benchmark.

Run it with:

```bash
uv run --extra dev python benchmarks/run_qwen_membrane_guessing_game_real_forward.py
uv run --extra dev python benchmarks/verify_qwen_membrane_guessing_game_real_forward.py
```

Passing this smoke means the real-Qwen frozen hidden-state integration path ran
or that the report clearly records why it could not run. It still does not
claim open-ended Qwen dialogue quality.

Passing these checks means the local controlled mechanisms behave as expected. It does not prove industrial robustness, external benchmark dominance, or independent reproduction.
