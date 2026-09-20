# API Quick Reference

This page lists the stable symbols most downstream projects should start with.
Full generated API pages are linked from the navigation.

## Core Layers

```python
import arti

layer = arti.ARTILayer()
assert layer.program is None
```

`arti.ARTILayer` and `arti.torch.ARTILayer` are the same `arti/layer@3` class.
It hosts a configured `FederatedProgram` execution region: root routing, routed-program
iteration, Formula Fabric and cross-program traversal. Its unconfigured shell is an
exact identity for host insertion. `arti.nn.Layer` remains the explicit
profile-based constructor for the classic tensor pipeline.

```python
out = layer(x, mask=mask)
out, federal_run = layer(x, mask=mask, return_info=True)
```

Inspect the resolved execution contract:

```python
print(arti.component_ref(layer))
print(layer.runtime_provenance())
print(arti.inspect(layer, x).to_markdown())
```

## Activations

```python
import arti.nn as ann
from arti.functional import half

act = ann.Half()
y = act(x)
```

`Half` is a stateless salience-conditioned activation. It computes
`q = base ** relu((threshold - abs(x)) / scale)`. With the default
`base=0.5`, each unit of insufficient salience halves feature survival.
`stochastic=False` returns `q * x`; `stochastic=True` samples survival from
`q` in both `train()` and `eval()`. Pass `learnable=True` to make the
salience curve parameters trainable, or call `layer.survival(x)` to inspect
the differentiable `q(x)` without sampling. The salience rule can be replaced
with a versioned builtin or a local same-shape operator:

```python
from arti import ExponentialSurvival, Half

layer = Half(
    stochastic=False,
    survival=ExponentialSurvival(threshold=0.5, learnable=True),
)
```

Custom survival must return a floating-point `q` with exactly `x.shape` and
values in `[0, 1]`. `arti/survival@1` is accepted as a source declaration at
the Python construction boundary; `describe_survival("arti/survival@1")`
returns the resolved full SHA-256 contract ref used in provenance and
artifacts. Unregistered custom callables are local runtime-only
implementations and cannot be stored in an `arti.st` artifact.

Recall and Half remain independently composable mechanisms; configuring one
does not silently mutate the other.

Standalone `arti.nn.Retrieve` uses K-wide winner selection by default:

```python
retrieve = ann.Retrieve(dim=64, slots=32)  # breadth=min(8, available routes)
y = retrieve(x)                           # same shape as x; one winner is forwarded
y, run = retrieve(x, active_k=4, return_branches=True)

larger = ann.Retrieve(dim=64, slots=64, group_topk=16, breadth=16)
weighted = ann.Retrieve(
    dim=64,
    slots=64,
    group_topk=16,
    breadth=16,
    breadth_aggregation="route_weighted",  # explicit non-default ablation
)
```

The default retrieval engine executes independent candidate trajectories and chooses
one hard winner. The hard selection uses a soft routing surrogate during
backpropagation. Eight is the portable default; values in the 8-32 range are
recommended when the Bank and compute budget are large enough. `active_k`
reduces a particular call without rebuilding the module.

The K-wide host bridge is eager in this release. For `torch.compile(fullgraph=True)`,
construct `Retrieve(..., breadth_mode="mixed")`; compiled K-wide dispatch awaits a
fixed-capacity graph-safe bridge.

## Retrieval Execution

```python
import arti.nn as ann

refiner = ann.RetrieveExecutor(retrieve)
h_refined, info = refiner(
    h,
    policy=arti.ExecutionPolicy.fixed(3, trace_level="summary"),
    return_info=True,
)
h_more = refiner(h, policy=arti.ExecutionPolicy.fixed(5))

adaptive = arti.ExecutionPolicy.adaptive(
    max_steps=64,
    min_steps=2,
    scope="token",
    relative_tolerance=1e-3,
    route_tolerance=1e-2,
    patience=2,
    executor="early_break",
)
h_adaptive, trace = refiner(h, policy=adaptive, return_trace=True)
```

`RetrieveExecutor` is a stable tensor-in / tensor-out policy adapter. The wrapped
Retrieve remains the only loop owner; `ExecutionPolicy` controls runtime depth,
stopping, checkpoints, and diagnostics.

`ExecutionPolicy@1` preserves the original per-sample contract. The composable
`ExecutionPolicy@2` returned by `ExecutionPolicy.adaptive(...)` separates
`ExecutionBudget@1` from `ExecutionStop@1`, supports monotonic per-token stopping,
and reports logical token work separately from executed loop steps. A high
`max_steps` is a compute ceiling, not a target or a claim that fewer steps are
always better.

Exact route replay is also versioned and composable. `RetrievalRoutePlan@1`
captures one Recall decision (indices, weights, and route mass) without caching
Bank values. `RetrievalRouteStack@1` recursively groups plans over caller-defined
runtime axes such as block or site. Both are runtime-only stable objects: they do
not enter `state_dict`, `RecallState`, or `arti.st`.

## Tensor Compaction

```python
import arti.nn as ann

fold = ann.Fold(k=16)
z = fold(x)              # x: [B, N, D], z: [B, 16, D]
z = fold(x, q=survival)  # survival: [B, N] or [B, N, 1]
z = fold(x, mask=mask)   # mask: [B, N] or [B, N, 1]
```

`Fold` is a soft tensor compaction layer. It learns differentiable assignments
from `N` latent slots into a fixed `K`-slot workspace. Optional `q` values guide
survival/salience; optional `mask` values only suppress invalid slots. Keep
padding masks in `mask`, not `q`, when they are not meant to rank slot
importance. The folded output is a stable tensor workspace for downstream
`Conv1d`, flatten+MLP, attention, or another ARTI block.

## Pulse

```python
import arti.nn as ann

pulse = ann.Pulse(k=8)
z = pulse(x)  # x: [B, N, D], z: [B, 8, D]

z = pulse(x, mask=mask)  # padding / validity mask
z = pulse(x, q=survival) # optional salience / survival guide
z, info = pulse(x, return_info=True)
```

`Pulse` is the stable learned pulse layer. It uses `Half` and `Fold` to
form compact latent pulses from overcomplete fragments. `LearnedPulse` remains
available as an explicit alias. `PulseCompressor` and `pulse_compress` are the
legacy explicit path for externally supplied pulse ids.

Use `mask` for padding or invalid fragments. Use `q` only when the caller has a
real salience/survival signal. This matters for sparse modes such as `q_topk`:
padding masks should not be interpreted as feature importance.

## Runtime Context

```text
x          [B, N, D] or [B, D]
coord      [B, N, C]
mask       [B, N]
visibility [B, N, N]
recall     optional latent tensor
```

For `[B, D]` inputs, ARTI treats the tensor as one token.

## Target-Bank Updater

```python
from arti.mechanisms import TargetBankUpdater, WriteRefinePolicy

updater = TargetBankUpdater(
    hidden_dim=64,
    slots=32,
    private_slots=8,
    target_coupling="required_after_bootstrap",
    policy=WriteRefinePolicy.fixed(steps=4),
)
next_bank, info = updater(
    trace,
    target_bank,
    target_mask=readable_slots,
    write_mask=writable_slots,
    return_info=True,
)
```

After the bootstrap step, the target Bank is a required addressable memory
partition during every write integration step. An optional private Bank supplies learned writer experience; its
routing is normalized separately and it is not modified by the forward call.
`target_coupling="required_after_bootstrap"` selects
`arti/target-bank-updater@2`: the first step can bootstrap an empty Bank, while
later Formula transitions are hard-coupled to a real target-Bank read. The
default `"optional"` mode remains the event-conditioned `@1` control.
`target_mask` controls readable slots and `write_mask` independently controls
writable slots; when omitted, `write_mask` follows `target_mask`.
`exposure` controls the total write strength, while the policy controls how
many internal steps integrate that one exposure. The returned target Bank is
runtime state and should be persisted separately from the Updater parameters.
This versioned API is stable and is exposed through `arti.mechanisms`.

## Fit And Adapter Build

```python
import arti

project = arti.project(model).scan(sample)
plan = project.plan_insert(where=["*.out_proj"], max_extra_params="1%")
result = arti.fit(model, sample_batch=sample, dry_run=True)
```

Progressive non-mutating preview:

```python
preview = (
    arti.project(model)
    .at("model.layers.*.mlp.down_proj", every=4)
    .freeze(True)
    .budget(max_adapters=8, max_extra_params="1%")
    .preview(sample)
)
```

Common helpers:

```python
arti.load_fit_config("arti.json")
arti.validate_fit_config("arti.json")
arti.validate_plan("artifacts/arti-plan.json")
arti.validate_artifact("artifacts/arti-adapter.pt")
```

## Runtime Vocab And Pulse

```python
from arti import RuntimeVocabInput, RuntimeVocabHead
from arti import Pulse, PulseCompressor, RuntimeVocabPulseAdapter
from arti import attach_runtime_vocab_semantics
```

Use these stable APIs when the current output softmax order is supplied at
runtime and should not be tied permanently to model parameter rows.
Use `Pulse` for learned fragment compaction. Use `PulseCompressor` only when
the caller already owns explicit pulse ids or token span weights.

Runtime vocab items can also carry an optional tensor-native semantic anchor:

```python
runtime_vocab = attach_runtime_vocab_semantics(
    glyph_vocab,        # [K, ...] or [B, K, ...]
    semantic_anchor,   # [K, S] or [B, K, S]
    vocab_scale=0.02,
)
```

The glyph tensor remains the rigid visible identity channel; the semantic anchor
is an aligned tensor field that a dynamic head can read when raw glyph identity
alone is not enough to generalize to held-out vocab items.

## Qwen Glyph Runtime Adapter

```python
from arti.integrations.qwen import QwenGlyphRuntimeAdapter
```

Use this optional integration when ordinary Qwen dialogue should stay on
the frozen base logits/generate path while a separate ARTI readout scores an
external glyph runtime vocabulary.

For the string-first Qwen answering probe:

```bash
uv run --extra torch --extra qwen python benchmarks/run_qwen_string_recall_refiner_answering.py --model-id Qwen/Qwen3-0.6B --steps 3000 --batch-size 64 --condition-loss-weight 0.35 --answer-identity-loss-weight 0.5 --max-new-tokens 20 --pulse-count 8 --fragment-dim 128 --matched-mean-dim 640 --max-chars 384 --device cuda
uv run --extra torch --extra qwen python benchmarks/verify_qwen_string_recall_refiner_answering.py
```

This probe renders chat strings into ARTI text tensors, trains small ARTI-side
frontends, and compares generated answer strings, prompt-conditioning accuracy,
coherence, loop diagnostics, conditional NLL, unique-answer rate, equal training
examples, trainable parameter budgets, and resource fields. Training uses
next-token CE plus a lightweight prompt-conditioning auxiliary loss; Qwen token
ids are used only at the frozen Qwen output/decode boundary. The live probe also
records a parameter-matched wide mean baseline and a paired no-Half/Half refiner
ablation. The Half branch is treated as an optimization diagnostic for this
joint-training setup, not as a Half mechanism verdict.

For the controlled Qwen-side Half check:

```bash
uv run --extra torch --extra qwen python benchmarks/run_qwen_hidden_refiner_half.py
uv run --extra dev python benchmarks/verify_qwen_hidden_refiner_half.py
```

This hidden-state probe freezes Qwen, repairs corrupted Qwen-derived hidden
vectors, and compares paired no-Half/Half RecallExecutors with shared
initialization, shared train batches, equal parameters, and weak-trace noise.

## Text Tensor

```python
from arti import render_text_vocab, render_text_layout, TextTensorConfig
```

Visible natural-language characters should be glyph-first. Codepoint channels
are auxiliary for controls, fallback glyphs, and confusable diagnostics.

## Participant Context And Membrane

```python
from arti import build_participant_context
from arti import MembraneVisibilityRouter
```

Use participant context to construct tensor-first phase, mask, and visibility
fields for multi-participant dialogue experiments. Use membrane routing to
separate public output tokens from inner model-side tokens.

## Evidence

```bash
uv run --extra dev python benchmarks/verify_evidence_schema.py
uv run --extra dev python scripts/quality_gate.py quick
```

Evidence schema and gate interpretation are documented in
`docs/reference/evidence-schema.md`.
