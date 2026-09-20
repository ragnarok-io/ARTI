# Fit Adaptation

`arti.fit()` turns ARTI from a single latent layer into a small adaptation build system for PyTorch models.

The workflow is intentionally Gradle-like:

```text
project -> scan -> insert -> fit -> validate -> export
```

## Quick Fit

```python
import torch
import torch.nn as nn
import arti

model = nn.Sequential(
    nn.Linear(32, 64),
    nn.GELU(),
    nn.Linear(64, 10),
)

sample = torch.randn(4, 32)

result = arti.fit(
    model,
    objective=["preserve-output", "task-fit", "validate"],
    calibration_loader=calibration_loader,
    calibration_steps=100,
    train_loader=train_loader,
    val_loader=val_loader,
    validation_steps=20,
    steps=1000,
    sample_batch=sample,
    target_modules=["0", "2"],
    profile="latent-adapt",
    scale="small",
    freeze_base=True,
    max_adapters=1,
    max_extra_params="1%",
)

adapted_model = result.model
result.write_report("arti-fit-report.md")
result.export("arti-adapter.pt")
```

## Objective Plan

`arti.fit()` can run as a declarative task plan. This is the first Gradle-like layer above the low-level PyTorch modules:

```python
result = arti.fit(
    model,
    objective=["preserve-output", "task-fit", "validate"],
    calibration_loader=calibration_loader,
    calibration_steps=100,
    train_loader=train_loader,
    steps=1000,
    val_loader=val_loader,
)
```

Supported objectives are:

```text
preserve-output  -> calibrate inserted adapters against the frozen base output
task-fit         -> train adapters on the supplied task batches
validate         -> append validation metrics to the fit report
```

If `objective` is omitted, ARTI infers the same plan from supplied loaders and step counts. The resolved plan is stored in the exported report as `objective_plan`, so an adapter artifact records not only what was inserted, but also which adaptation tasks produced it.

Reports include a `build_plan` with the Gradle-like task graph:

```text
scan -> insert -> preserve-output -> task-fit -> validate
```

Each build task records:

```text
name
kind
depends_on
enabled
```

Each executed objective task also writes a structured `task_history` row:

```text
name
status
steps
metric_name
metric_value
```

This keeps the artifact auditable: `build_plan` is the full adaptation graph, `objective_plan` is the requested training/validation objective sequence, and `task_history` is the observed execution summary.

Reports also include a resolved `mechanism` summary. This expands the chosen profile and scale into the actual ARTI adapter settings:

```text
observer_phase
coord_dim
coord_frame_mode
operator_count
interface_slots
recall_slots
recall_steps
recall_min_steps
recall_tolerance
recall_bank_fraction
recall_routing
recall_key_dim
recall_group_size
recall_group_topk
hidden_multiplier
```

This is useful when comparing artifacts created from different build profiles, because `profile="observer-phase"` and `scale="base"` are recorded together with the concrete phase, operator, virtual interface, and private recall settings. Each inserted adapter also records its resolved `hidden_dim`, `recall_slots`, Recall-bank parameter count, actual bank ratio, and routing settings. `recall_bank_fraction` is a budget request; the per-adapter fields are the authoritative resolved allocation.

Apply the adapter-only artifact to another instance with the same module structure:

```python
fresh_model = nn.Sequential(
    nn.Linear(32, 64),
    nn.GELU(),
    nn.Linear(64, 10),
)

applied = arti.apply_adapter(
    fresh_model,
    "arti-adapter.pt",
    sample_batch=sample,
)
```

Continuation jobs may explicitly remove an artifact's old identity gate and
upgrade shape-stable Recall runtime settings while preserving the remaining
adapter tensors:

```python
applied = arti.apply_adapter(
    fresh_model,
    "arti-adapter.pt",
    sample_batch=sample,
    identity_gate=False,
    mechanism_overrides={
        "recall_steps": 3,
        "recall_min_steps": 1,
        "recall_tolerance": 1e-2,
    },
)
```

Applied migrations are recorded in `report.applied_artifact["migrations"]`.
Overrides that change parameter shapes remain subject to strict state loading
and fail instead of silently dropping incompatible tensors.

Recall attachments do not use an output bridge. Their value bank is allocated
directly in the host feature dimension, so each retrieved value is the residual
that is added to the host tensor:

```text
weights = route(host, recall_keys)
delta = Half(weights @ residual_bank)
host = host + residual_budget * delta
```

The query and keys may use a smaller routing dimension, but the residual bank
does not require a learned decoder. With `zero_init_output=True`, the residual
bank starts at zero, preserving the base model exactly while still receiving
gradient on the first optimizer step.

For non-Recall latent transformation attachments, `bridge_mode="radial"`
remains the default shape adapter. `bridge_mode="dense"` is retained for
explicit compatibility with historical non-Recall artifacts.
`identity_gate=True` remains a separate legacy option and is mutually
exclusive with `zero_init_output=True`.

When a host block already receives its padding mask, name that forward
parameter explicitly so ARTI uses the same validity contract even when the
host passes it positionally:

```python
project.insert(
    where="repeated-composites",
    zero_init_output=True,
    boundary_mask_key="attention_mask",
)
```

## Project DSL

Preview a build before ARTI wraps or freezes any module:

```python
preview = (
    arti.project(model)
    .plugin("transformers")
    .profile("latent-adapt")
    .at(["*.self_attn.o_proj", "*.mlp.down_proj"], every=4)
    .freeze(True)
    .budget(max_adapters=8, max_extra_params="1%")
    .preview(sample)
)

print(preview.insertion_plan.to_dict())
```

The preview includes every scanned candidate, selected adapter, budget skip,
resolved parameter count, insertion pattern, freezing policy, plugin strategy,
and runtime-field contract. It does not mutate the model or alter
`requires_grad`. The equivalent one-shot path is `arti.fit(..., dry_run=True)`.

```python
project = (
    arti.project(model)
    .plugin("torch")
    .profile("latent-adapt")
    .scale("small")
    .runtime(causal=True)
    .scan(sample, positions=("input", "output"))
    .insert(
        where=["*.out_proj", "*.mlp.fc2"],
        exclude=["*.norm", "*.lm_head"],
        positions="output",
        scale_pattern={
            "encoder.layers.0.*": "tiny",
            "encoder.layers.8.*": "base",
        },
        every=2,
        freeze_base=True,
        max_adapters=4,
        max_extra_params=1_000_000,
    )
)

plan = project.plan_insert(where=["*.out_proj", "*.mlp.fc2"], max_extra_params="1%")
print(plan.to_dict())

result = project.fit(train_loader, steps=100)
metrics = project.validate(val_loader)
```

With a `sample_batch`, `scan()` observes every executed `nn.Module`; module class and model family are not eligibility gates. Any module that emits a floating tensor with a recoverable batch and feature axis becomes a candidate, including custom and parameterless layers. Direct tensors and nested tuple/list/mapping outputs are supported. `TensorLayout` moves the inferred batch and feature axes into a temporary `[B,D]` or `[B,N,D]` latent view, folds all other axes into tokens, and restores the original values, order, rank, and output tree after ARTI runs. Channel-first convolutional tensors, time-first recurrent tensors, last-feature transformer tensors, and higher-rank latent fields therefore use the same wrapper.

Runtime scanning defaults to `positions="output"`. Pass `positions="input"` or `("input", "output")` to discover exact input-side boundaries too. Output boundary ids remain the module path, such as `encoder.layers.3`; input boundary ids use `encoder.layers.3::input`. Reports record the module path, side, tensor path, layout axes, feature dimension, dtype, device, resolved scale, and planned parameter cost. This makes the insertion contract reproducible without a model-family whitelist. One insertion pass accepts at most one side of a module.

For LoRA-style control, combine `where`, `exclude`, and `scale_pattern`. Patterns are shell-style globs and match either the boundary id or module path. `scale_pattern` is ordered; the last matching rule wins and the resolved preset is written into the plan:

```python
preview = (
    arti.project(model)
    .at(
        "encoder.layers.*",
        exclude=["*.norm", "*.lm_head"],
        positions="output",
        scale_pattern={
            "encoder.layers.[0-3].*": "tiny",
            "encoder.layers.8.*": "base",
        },
    )
    .budget(max_adapters=12, max_extra_params="1%")
    .preview(sample)
)
```

Without a sample forward, static discovery remains conservative because arbitrary Python modules do not declare their output contract. Model-family plugins provide convenient name filters only; they never grant or deny ARTI compatibility. The default plain-PyTorch strategy selects observed tensor-producing leaves. Use `where="all-tensor-boundaries"` only when intentionally selecting both composite modules and their children.

Last-feature tensors and channel-first modules with declared channel dimensions
are inferred automatically. For an ambiguous custom layout, declare axes during
the scan rather than creating a model-specific adapter:

```python
project.scan(
    sample_batch,
    batch_axis={"custom.*": 0},
    feature_axis={"custom.channel_first": 1},
)
```

```python
class MyTensorOperation(nn.Module):
    def forward(self, x):
        return {"metadata": x.shape, "payload": [{"state": x.sin()}]}

project = arti.project(model).scan(sample_batch)
project.insert(where="custom_block.my_operation")
```

`insert()` wraps selected modules with shape-stable ARTI residual adapters and preserves non-latent fields in structured outputs. With `freeze_base=True`, the original model parameters are frozen and only inserted ARTI adapters are trainable.

If a module is reused multiple times during a single forward pass, the scanner keeps the first observed candidate for that module name so plans remain stable. The JSON and Markdown scan reports record `scanned_modules`, `candidate_count`, `candidate_events`, and `duplicate_events`, which makes reused-module deduplication visible in CI reports instead of hiding it inside the scanner.

Each insertion candidate records:

```text
name
module_path
position      # input or output
module_type
output_shape
dim
parameters
source        # forward or static
tensor_rank
path_depth
output_path
tensor_path
batch_axis
feature_axis
device
dtype
is_leaf
```

`calibrate()` can first preserve the pretrained model's original behavior:

```python
project.calibrate(calibration_loader, steps=100, objective="preserve-output")
```

For each batch, ARTI temporarily disables inserted adapters, records the base output, re-enables adapters, and trains adapter parameters to match the base output. This gives pretrained models a no-surprise warmup before task fine-tuning.

`fit()` records per-step loss in the report. `validate()` appends metric summaries to validation history, so exported artifacts carry training and validation diagnostics:

```python
print(project.report().calibration_history[-1])
print(result.report.loss_history[-1])
print(project.report().validation_history)
```

Use `profile_forward()` to capture a lightweight runtime profile:

```python
profile = project.profile_forward(sample, warmup=1, repeats=5)
print(profile.to_dict())
```

The profile records mean/min/max forward time in milliseconds, output shape, dtype, and device. Results are appended to `report.forward_profiles` and exported with the adapter artifact.

`where` accepts a single shell-style pattern or a list of patterns. `exclude`, `positions`, `scale_pattern`, `every`, `max_adapters`, and `max_extra_params` act like build constraints: they select a reproducible adapter plan before training starts.

`max_extra_params` accepts either an integer parameter budget or a string percentage such as `"1%"`, measured against the scanned base model parameter count.

Use `plan_insert()` to dry-run insertion without mutating the model. The returned `AdapterInsertionPlan` contains:

```text
selected
skipped_budget
excluded
adapter_parameters
spec
```

This is useful for CI checks and budget review before wrapping modules.

The convenience API also supports the same planning mode:

```python
dry = arti.fit(
    model,
    sample_batch=sample,
    target_modules=["*.out_proj", "*.mlp.fc2"],
    max_adapters=4,
    max_extra_params="1%",
    dry_run=True,
)

print(dry.report.insertion_plan.to_dict())
```

`dry_run=True` scans the model, resolves plugins, profiles, scale, objectives, and budget constraints, then returns a report with `insertion_plan`. It does not wrap modules, freeze parameters, run calibration, train, or validate.

For CI or review workflows, write the dry-run plan as an auditable build artifact:

```python
project = (
    arti.project(model)
    .plugin("transformers")
    .profile("latent-adapt")
    .scale("small")
    .objectives(["preserve-output", "task-fit", "validate"])
    .scan(sample)
)

project.write_plan(
    "artifacts/arti-plan.json",
    where=["*.out_proj", "*.mlp.fc2"],
    max_adapters=8,
    max_extra_params="1%",
)
project.write_plan("artifacts/arti-plan.md")
```

The JSON payload has `kind: "fit-plan"` and embeds the same report schema used by adapter artifacts, including `scanned`, `build_plan`, `objective_plan`, `parameters`, and `insertion_plan`.
Plans generated through the CLI also include top-level `provenance` with the model reference, sample source, config path, insertion targets, mechanism overrides, budget limits, and runtime flags used to create the plan. The sibling `provenance_fingerprint` is a stable hash of that provenance block, and `arti validate plan` rejects plans whose provenance no longer matches the fingerprint. Markdown plans include the same data in a `Plan Provenance` section.

The installed CLI can also create a dry-run plan from an importable model factory. This keeps plan generation out of notebooks and lets CI review insertion budgets before training starts:

```bash
arti plan my_project.models:make_model artifacts/arti-plan.json \
  --sample-shape 2,16,768 \
  --config arti.json \
  --target-modules "*.out_proj,*.mlp.fc2" \
  --max-adapters 8

arti validate plan artifacts/arti-plan.json --expect-config arti.json
```

After reviewing the dry-run plan, build an adapter artifact from the same importable model factory:

```bash
arti build my_project.models:make_model artifacts/arti-adapter.pt \
  --config arti.json \
  --sample-json sample-batch.json \
  --target-modules "*.out_proj" \
  --max-adapters 8 \
  --report artifacts/arti-report.json \
  --lock-output artifacts/arti.lock.json \
  --task-graph-output artifacts/build-task-graph.json \
  --expect-plan artifacts/arti-plan.json
```

`arti build` runs the same scan and insertion resolver as `arti.fit(...).export(...)`, then writes an adapter artifact and optional fit report. With `--expect-plan`, the built artifact must match the reviewed dry-run plan's selected adapter names, adapter parameter total, profile, scale, and config fingerprint. The exported artifact also stores a `build` metadata block with the expected plan path, provenance fingerprint, selected adapter names, profile, scale, and planned adapter parameter total, so release checks can trace which reviewed plan produced the adapter. Pass `--lock-output` to write and validate the build lock in the same build task after the artifact has passed its checks. It is intentionally conservative: data loading, losses, and long training loops stay in Python code for now, while CI can still create and validate a frozen-base adapter artifact from an importable model.

The JSON summary printed by `arti build` includes a machine-readable `task_graph` with task nodes such as `scan`, `insert`, `export-artifact`, `write-report`, and `write-lock`, plus the produced artifact paths. Pass `--task-graph-output` to save that graph as a JSON artifact, and validate it with `arti validate task-graph artifacts/build-task-graph.json --expect-kind build --require-existing-artifacts`. This lets CI or a higher-level build runner treat ARTI as a small task graph instead of scraping human logs.

Python callers can use the same artifact helpers directly: `arti.create_task_graph_payload(...)`, `arti.write_task_graph_artifact(...)`, and `arti.validate_task_graph(...)`.

For CI jobs that should not materialize a config file, mechanism dimensions can be overridden directly on the plan command:

```bash
arti plan my_project.models:make_model artifacts/observer-plan.json \
  --sample-shape 2,16,768 \
  --target-modules "*.out_proj" \
  --profile observer-phase \
  --phases 16 \
  --mechanism-coord-dim 16 \
  --mechanism-coord-frame-mode operator_bank \
  --mechanism-observer-phase \
  --mechanism-virtual-recall \
  --mechanism-operator-count 8 \
  --mechanism-interface-slots 16 \
  --mechanism-recall-slots 8 \
  --mechanism-recall-steps 1 \
  --mechanism-hidden-multiplier 2.0 \
  --mask-key token_mask \
  --coord-key phase_coord \
  --observer-coord-key next_token_phase \
  --frame-operators-key inverse_operator_bank
```

The mechanism flags feed the same resolver as the `mechanism` config block. Runtime field flags feed the same adapter context resolver as the `runtime` config block, so dict batches can keep model-native fields separate from ARTI-only phase, visibility, and inverse-frame tensors. Both groups are recorded in the normalized `fit_config` and plan provenance.

If the factory requires constructor arguments, pass them as JSON:

```json
{
  "model_name": "local-or-hub-model-id",
  "num_labels": 2
}
```

```bash
arti plan my_project.models:make_model artifacts/arti-plan.json \
  --model-kwargs-json model-kwargs.json \
  --sample-shape 2,16,768
```

For dict-style pretrained batches, use `--sample-json` instead of `--sample-shape`:

```json
{
  "input_ids": {"shape": [2, 16], "dtype": "long", "kind": "randint", "low": 0, "high": 32000},
  "attention_mask": {"shape": [2, 16], "dtype": "long", "kind": "ones"}
}
```

```bash
arti plan my_project.models:make_model artifacts/arti-plan.json \
  --sample-json sample-batch.json \
  --profile transformer \
  --max-adapters 8
```

## Capability Discovery

ARTI exposes its fit profiles, scale presets, and optional plugins as machine-readable metadata:

```python
caps = arti.capabilities(phases=16)
print(caps["profiles"])
print(caps["scales"])
print(caps["plugins"])
```

The same information is available from the CLI:

```bash
arti inspect
arti inspect profiles --phases 16
arti inspect scales
arti inspect plugins
```

This lets CI, notebooks, and downstream adapters discover supported mechanism profiles and budget scales without importing private registries.

## Declarative Config

For repeatable build jobs, store mechanism and insertion settings in a JSON or TOML config:

```bash
arti init-config arti.json
arti init-config arti.toml --profile virtual-recall --scale base
arti schema fit-config generate --output docs/reference/fit-config.schema.json
arti schema task-graph generate --output docs/reference/task-graph.schema.json
```

```json
{
  "fit": {
    "plugins": ["transformers"],
    "profile": "observer-phase",
    "phases": 16,
    "scale": "small",
    "objectives": ["preserve-output", "task-fit", "validate"]
  },
  "mechanism": {
    "coord_dim": 16,
    "coord_frame_mode": "operator_bank",
    "observer_phase": true,
    "virtual_recall": true,
    "operator_count": 8,
    "interface_slots": 16,
    "recall_steps": 1,
    "recall_bank_fraction": 0.6,
    "recall_routing": "grouped",
    "recall_key_dim": 32,
    "recall_group_size": 128,
    "recall_group_topk": 2,
    "hidden_multiplier": 2.0
  },
  "runtime": {
    "causal": true,
    "mask_key": "attention_mask",
    "coord_key": "phase_coord",
    "observer_coord_key": "next_token_phase",
    "frame_operators_key": "inverse_operator_bank"
  },
  "insertion": {
    "where": ["*.out_proj", "*.mlp.fc2"],
    "exclude": ["*.norm", "*.lm_head"],
    "positions": ["output"],
    "scale_pattern": {
      "encoder.layers.0.*": "tiny",
      "encoder.layers.8.*": "base"
    },
    "max_adapters": 8,
    "max_extra_params": "1%"
  }
}
```

`profile` and `scale` choose a preset. The optional `mechanism` block overrides the resolved tensor mechanism dimensions for builds that need explicit control over observer phase, coordinate frame inversion, virtual interface slots, private recall capacity, dynamic operator count, or adapter hidden width. With Recall enabled, `recall_bank_fraction` asks the planner to allocate a majority of the adapter budget to independent bank values and their sparse grouped routing assets. Omit `recall_slots` when using this automatic mode; setting explicit slots instead opts into a fixed shape. An explicit `hidden_multiplier` acts as the controller-width ceiling when a bank fraction is also supplied. The `runtime` block controls causal visibility and can declare custom batch field names for ARTI-only context tensors. Standard model inputs such as `input_ids` or `attention_mask` can keep their native names, while ARTI can read coordinates, observer phase, or inverse operator banks from separate fields. These values are written into the report's `mechanism` and normalized `fit_config` sections and are covered by the config fingerprint.

The generated `fit-config.schema.json` mirrors the public config surface for editor completion and CI checks. It is also bundled into wheels as a package resource and can be read with `arti.packaged_fit_config_schema_json()`. Regenerate it after changing config fields, profile names, scale names, plugin names, runtime field names, or insertion budget options.
JSON files written by `arti init-config` include a `$schema` pointer to this generated schema. The pointer is editor metadata only and is not included in the normalized `config_fingerprint`.

The generated `task-graph.schema.json` mirrors the machine-readable build/apply task graph artifact. It is bundled with wheels and can be read with `arti.packaged_task_graph_schema_json()`.

Apply it to a project:

```python
config = arti.load_fit_config("arti.json")
project = arti.project(model).configure(config).scan(sample)
project.write_plan("artifacts/arti-plan.json")
```

Or pass it directly to the convenience API:

```python
dry = arti.fit(model, config="arti.json", sample_batch=sample, dry_run=True)
print(dry.report.insertion_plan)
```

Validate it from CI:

```bash
arti validate config arti.json
arti validate config arti.json --expect-profile observer-phase --expect-scale small
arti validate config arti.json --expect-mechanism operator_count=8 --expect-mechanism coord_dim=16
arti validate config arti.json --expect-runtime-field mask_key=attention_mask
arti schema fit-config check --output docs/reference/fit-config.schema.json
```

Config validation checks JSON/TOML syntax, known profile names, scale presets, resolved mechanism dimensions, plugins, objectives, runtime field contracts, and insertion budget values. It can also enforce expected profile, scale, concrete mechanism settings, and runtime field names before any model is scanned. The config intentionally describes ARTI build mechanics only. Data loading, labels, losses, and business-specific adapters stay outside the core package.

Every report now includes the normalized `fit_config` and a stable `config_fingerprint`. Adapter artifacts copy that fingerprint into the manifest, and `validate_plan()` / `validate_artifact()` check that the fingerprint still matches the embedded report. This gives CI a compact way to prove that a plan, report, and exported adapter came from the same effective build configuration.

## Transformer Strategy

For transformer-like models, ARTI can expand a semantic strategy into common attention and MLP output names:

```python
result = arti.fit(
    model,
    sample_batch=sample,
    profile="transformer",
    max_adapters=4,
    max_extra_params="1%",
)
```

This uses the `transformers` plugin strategy and searches names such as:

```text
*.attn.out_proj
*.attention.output.dense
*.self_attn.out_proj
*.mlp.fc2
*.mlp.down_proj
*.output.dense
```

The explicit form is:

```python
project = arti.project(model).plugin("transformers").scan(sample).insert(max_adapters=4)
```

Plugin metadata is inspectable without importing the optional dependency:

```python
plugin = arti.get_plugin("transformers")

print(plugin.default_strategy)
print(plugin.available)
print(plugin.capabilities)
```

Fit reports include plugin details, including optional dependency availability and declared capabilities.

## Timm / Vision Transformer Strategy

For timm-style Vision Transformer modules, use the `timm` plugin or the `profile="timm"` shortcut:

```python
result = arti.fit(
    model,
    sample_batch=sample,
    profile="timm",
    max_adapters=4,
    max_extra_params="1%",
)
```

The timm plugin uses the `vision-transformer` strategy and searches common names such as:

```text
blocks.*.attn.proj
blocks.*.mlp.fc2
blocks.*.norm1
blocks.*.norm2
norm
fc_norm
head
```

Adapter-only artifacts preserve the concrete inserted module paths, so `arti.apply_adapter()` can rehydrate the same wrappers on another model instance with the same structure.

## CNN Strategy

For convolutional vision models, use the metadata-only `vision-cnn` plugin or the `profile="cnn"` shortcut:

```python
result = arti.fit(
    model,
    sample_batch=image_batch,
    profile="cnn",
    max_adapters=4,
)
```

The CNN strategy searches common convolutional names such as:

```text
conv
conv1
*.conv1
*.bn1
*.features.*
*.layer*.*.conv*
*.layer*.*.bn*
*.downsample.*
*.stem.*
```

Conv2d outputs are adapted through the spatial-token bridge described above, so downstream modules still receive `[B, C, H, W]`.

## Recurrent Strategy

For recurrent sequence models, use the metadata-only `recurrent` plugin or the `profile="recurrent"` shortcut:

```python
result = arti.fit(
    model,
    sample_batch=sequence_batch,
    profile="recurrent",
    max_adapters=2,
)
```

The recurrent strategy searches common names such as:

```text
rnn
lstm
gru
*.encoder.rnn
*.encoder.lstm
*.encoder.gru
```

RNN/LSTM/GRU tuple outputs preserve their hidden state payloads. If the recurrent layer uses PyTorch's default time-first layout `[T, B, H]`, ARTI adapts it as `[B, T, H]` internally and restores the original layout afterward.

## Batch Schema

`scan(sample_batch)` records common pretrained-model batch fields without requiring a hard dependency on Hugging Face:

```python
batch = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "labels": labels,
}

report = arti.project(model).plugin("transformers").scan(batch).report()
print(report.scanned.batch_schema)
```

ARTI identifies token, mask, and label fields where possible:

```text
input_ids       -> token_key
attention_mask  -> mask_key
labels          -> label_key
```

The mask bridge is available directly:

```python
visibility = arti.attention_mask_to_visibility(attention_mask, causal=True)
```

When an inserted adapter runs inside `arti.fit()` / `arti.project()` execution on a dict batch, ARTI temporarily exposes the detected `attention_mask` to adapter wrappers. If the wrapped tensor has shape `[B, N, D]` and the mask has shape `[B, N]`, the inserted ARTI block receives:

```text
mask       : [B, N]
visibility : [B, N, N]
```

For observer-phase profiles, dict batches may also provide:

```text
coord / arti_coord
observer_coord / arti_observer_coord / next_token_coord
frame_operators / arti_frame_operators / inverse_bank
```

These are passed to inserted ARTI adapters when the wrapped tensor shape matches `[B, N, D]`. Missing `frame_operators` for `operator_bank` observer adapters raises a clear error instead of silently disabling the phase mechanism.

## Profiles And Scales

Profiles describe the mechanism family:

```text
latent-adapt
virtual-recall
observer-phase
autoregressive-observer
transformer
```

Scales describe adapter size presets:

```text
tiny
small
base
large
```

The first implementation keeps inserted adapters shape-stable so they can safely wrap pretrained modules. Wider bottleneck variants can be added without changing the public `project -> scan -> insert -> fit` contract.

## Artifact

`ARTIFitResult.export(path)` writes an adapter-only torch artifact by default:

```text
manifest
report
adapter_state_dict
```

The `report` contains scan metadata, insertion constraints, build plan, resolved mechanism settings, objective plan, task history, losses, validation summaries, and adapter parameter counts.

Reports also include a compact `summary` for CI and dashboards:

```text
candidate_count
inserted_count
adapter_parameters
total_parameters
adapter_parameter_ratio
frozen_base
budget_limit
budget_used
budget_exhausted
last_loss
last_calibration_loss
last_validation_metric
```

For freeze and fine-tuning audits, reports include a `parameters` summary:

```text
total_parameters
trainable_parameters
adapter_parameters
trainable_adapter_parameters
base_parameters
trainable_base_parameters
frozen_base
```

When `freeze_base=True`, `trainable_base_parameters` should be `0`; when `freeze_base=False`, ARTI reports the remaining trainable base parameters explicitly.

The `manifest` contains stable artifact metadata:

```text
format_version
package_name
package_version
backend
include_base
adapter_key_count
adapter_parameters
profile
scale
adapter_state_sha256
report_sha256
```

Pass `include_base=True` to include the full patched model state dict as well.

Use `validate_artifact()` in CI or application startup checks:

```python
payload = arti.validate_artifact("arti-adapter.pt")
print(payload["manifest"])
```

Validation checks the artifact format version, package identity and version field, backend, manifest field types, manifest/report consistency, adapter key count, adapter tensor SHA256 fingerprint shape and value, report SHA256 fingerprint shape and value, report summary consistency, optional `build` metadata shape, and whether `include_base=True` artifacts actually include a full `state_dict`. The report hash locks the exported scan, insertion plan, budget usage, and diagnostics to the artifact that was reviewed.

Use `validate_plan()` for dry-run build plans before any training job starts:

```python
plan = arti.validate_plan("artifacts/arti-plan.json")
print(plan["report"]["insertion_plan"])
```

The same checks are available from the command line:

```bash
arti validate plan artifacts/arti-plan.json
arti validate plan artifacts/arti-plan.json --expect-config arti.json
arti validate plan artifacts/arti-plan.json --expect-provenance-fingerprint APPROVED_SHA256
arti validate plan artifacts/arti-plan.json --expect-profile observer-phase --expect-scale small
arti validate plan artifacts/arti-plan.json --expect-mechanism operator_count=8 --expect-mechanism coord_dim=16
arti validate plan artifacts/arti-plan.json --expect-runtime-field mask_key=token_mask --expect-runtime-field coord_key=phase_coord
arti validate plan artifacts/arti-plan.json --max-adapters 8 --max-extra-params 500000
arti validate artifact artifacts/arti-adapter.pt
arti validate artifact artifacts/arti-adapter.pt --expect-plan artifacts/arti-plan.json
arti validate artifact artifacts/arti-adapter.pt --expect-config arti.json
arti validate artifact artifacts/arti-adapter.pt --expect-adapter-state-sha256 APPROVED_SHA256
arti validate artifact artifacts/arti-adapter.pt --expect-profile observer-phase --expect-scale small
arti validate artifact artifacts/arti-adapter.pt --expect-mechanism recall_slots=8 --expect-mechanism observer_phase=true
arti validate artifact artifacts/arti-adapter.pt --expect-runtime-field mask_key=token_mask
arti validate artifact artifacts/arti-adapter.pt --max-adapters 8 --max-extra-params 500000
```

Plan validation checks the fit-plan schema, embedded report shape, insertion-plan fields, adapter parameter totals, declared `max_adapters`, declared `max_extra_params`, and provenance fingerprint consistency. Artifact validation checks manifest/report consistency, adapter tensor SHA256, and report SHA256. With `--expect-plan`, artifact validation also replays the dry-run plan comparison and requires the artifact `build` metadata to match that reviewed plan. The CLI can also enforce external `--max-adapters`, `--max-extra-params`, `--expect-profile`, `--expect-scale`, `--expect-mechanism`, `--expect-runtime-field`, `--expect-provenance-fingerprint`, and `--expect-adapter-state-sha256` gates for repository-level policy. This makes budget, source-context, concrete mechanism settings, runtime batch contracts, adapter-weight, and build-report review independent from training side effects.

For release workflows, write a build lockfile after the plan and adapter artifact have been reviewed:

```bash
arti lock arti.lock.json \
  --plan artifacts/arti-plan.json \
  --artifact artifacts/arti-adapter.pt \
  --config arti.json

arti validate lock arti.lock.json \
  --expect-config arti.json \
  --expect-plan artifacts/arti-plan.json \
  --expect-provenance-fingerprint APPROVED_PROVENANCE_SHA256 \
  --expect-adapter-state-sha256 APPROVED_ADAPTER_SHA256 \
  --expect-report-sha256 APPROVED_REPORT_SHA256 \
  --expect-profile observer-phase \
  --expect-scale small \
  --expect-mechanism operator_count=8 \
  --expect-mechanism recall_slots=8 \
  --expect-runtime-field mask_key=token_mask \
  --max-adapters 8 \
  --max-extra-params 500000
```

The lockfile binds the approved artifact path, adapter tensor SHA256, report SHA256, config fingerprint, adapter parameter count, adapter key count, inserted adapter count, resolved mechanism summary, runtime field contract, optional plan provenance fingerprint, and any artifact `build` metadata produced by `arti build --expect-plan`. `arti validate lock --expect-plan` compares that locked build metadata against the reviewed plan. Commit it alongside the reviewed adapter metadata when CI or deployment needs to prove that the applied adapter still matches the approved build.

`arti.apply_adapter(model, artifact, sample_batch=...)` rebuilds the adapter wrappers from the report and loads only the adapter weights. The returned report includes `applied_artifact` with the artifact path, manifest summary, and `adapter_state_sha256`. The target model must expose the same module paths selected by the original plan; otherwise ARTI raises a structure mismatch error with the missing and unexpected adapter key counts.

The same compatibility check is available from the CLI:

```bash
arti apply my_project.models:make_model artifacts/arti-adapter.pt artifacts/applied-report.json \
  --sample-json sample-batch.json \
  --lock arti.lock.json \
  --expect-config arti.json \
  --expect-adapter-state-sha256 APPROVED_SHA256 \
  --max-adapters 8 \
  --save-state-dict artifacts/patched-state.pt \
  --deployment-output artifacts/deployment.json \
  --task-graph-output artifacts/apply-task-graph.json
```

This applies the adapter in memory and writes a JSON or Markdown application report. Pass `--save-state-dict` to also write the patched model state dict after every apply gate has passed; the CLI summary includes `saved_state_dict_sha256` for deployment logs. Pass `--deployment-output` with `--lock` and `--save-state-dict` to write and validate the deployment manifest in the same apply task. When `--lock` is provided, ARTI first validates the lockfile and refuses to apply a different artifact than the one recorded in the approved build.

The `arti apply` JSON summary also includes `task_graph`, typically `apply-adapter -> write-apply-report`, with optional `write-state-dict` and `write-deployment-manifest` nodes when those outputs are requested. Use `--task-graph-output` and `arti validate task-graph artifacts/apply-task-graph.json --expect-kind apply --require-existing-artifacts` to keep that graph as a reviewed CI artifact.

Validate the final deployment checkpoint with the same hashing scheme:

```bash
arti validate state-dict artifacts/patched-state.pt \
  --expect-state-dict-sha256 APPROVED_PATCHED_STATE_SHA256
```

For deployment handoff, the apply task can write the manifest directly. You can also generate it explicitly when the artifacts were produced by separate jobs:

```bash
arti deployment-manifest artifacts/deployment.json \
  --lock arti.lock.json \
  --artifact artifacts/arti-adapter.pt \
  --applied-report artifacts/applied-report.json \
  --state-dict artifacts/patched-state.pt

arti validate deployment artifacts/deployment.json \
  --expect-plan artifacts/arti-plan.json \
  --expect-adapter-state-sha256 APPROVED_ADAPTER_SHA256 \
  --expect-state-dict-sha256 APPROVED_PATCHED_STATE_SHA256 \
  --expect-profile observer-phase \
  --expect-scale small \
  --expect-mechanism operator_count=8 \
  --expect-mechanism recall_slots=8 \
  --expect-runtime-field mask_key=token_mask \
  --max-adapters 8 \
  --max-extra-params 500000
```

Deployment manifests cross-check adapter hashes, adapter parameter count, adapter key count, inserted adapter count, the resolved mechanism summary, runtime field contract, and artifact `build` metadata against both the adapter artifact and build lock. `arti validate deployment --expect-plan` repeats the reviewed-plan metadata check at the final handoff, so a checkpoint can be rejected when its concrete ARTI mechanism settings, batch-context contract, adapter structure, or reviewed-plan provenance differ from the approved build.

`ARTIFitResult.write_report(path)` writes either JSON or Markdown depending on the file suffix.

## Compiled adapter hot paths

For repeated CUDA training or inference, compile attached ARTI adapters without
compiling the host model:

```python
compiled = arti.compile_adapter_hotpaths(model)
```

Compilation is a runtime optimization only. It does not change adapter
parameters, `state_dict` keys, or `arti.st` artifacts. Compatible grouped
product Recall layers keep discrete routing eager and share one compiled
read/write tail across all insertion points; unsupported configurations retain
the generic safe path. Compile once after attaching or loading adapters and
before the measured training loop.
