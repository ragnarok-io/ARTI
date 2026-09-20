# Pretrained Workflows

ARTI can adapt an existing Transformers, PEFT, Diffusers, or plain PyTorch
model through one reviewed lifecycle:

```text
scan -> plan -> apply -> fit -> export -> fresh reload -> generate -> detach
```

The provider loads and inspects an ecosystem object. `ARTIPlan` is the stable
boundary between discovery and mutation. Applying a plan uses its exact module
paths and refuses a model whose structure fingerprint has changed.

## One-Call Setup

```python
import arti

workflow = arti.from_pretrained(
    "Qwen/Qwen3-0.6B",
    provider="transformers",
    task="causal-lm",
    revision="<immutable-commit>",
    loader_kwargs={"dtype": "auto"},
    features={
        "recall": {"enabled": True, "steps": 2, "slots": 4},
        "half": {"enabled": True},
    },
    where="mlp",
    scale="tiny",
    max_adapters=4,
    training={
        "engine": "accelerate",
        "steps": 100,
        "mixed_precision": "bf16",
        "gradient_accumulation_steps": 4,
        "device": "auto",
    },
)

text = workflow.generate(**tokenizer("Hello", return_tensors="pt"))
result = workflow.fit(train_loader)
result.export("arti.st")
```

For a reversible integration smoke test, the workflow can be reopened on a
fresh instance of the same base model and then detached in place:

```python
restored = arti.pretrained(model, provider="transformers")
restored.scan(sample_batch).plan(where="mlp", scale="tiny")
restored.apply()
restored.load_weights("arti.st")
tokens = restored.generate(**inputs)
restored.detach()
```

`detach()` unwraps only the modules inserted by that workflow, restores their
pre-attachment `requires_grad` values, and leaves the native model class and
methods available. It is a structural lifecycle operation; it does not delete
the exported `arti.st` artifact or alter the base checkpoint.

`workflow.model` remains the native model or pipeline. Applying ARTI does not
replace `generate()`, KV cache handling, PEFT methods, `save_pretrained()`, or a
Diffusers Pipeline. Recall insertions store host-dimensional state writes.
Routing queries the Recall bank, `Half` applies survival pressure, and each
surviving write modifies the current state before the next Recall query. The
adapter returns that modified state directly: it does not expose a residual
branch or use a learned output bridge. The default `single` value composition
queries one complete host-dimensional write. The optional `product`
composition independently queries bounded scale and shift values and applies
an in-place affine write. A zero-initialized value bank is an exact no-op and
receives gradient immediately in either mode.

Non-Recall latent transformation insertions may still use the radial host
bridge when a shape adapter is required. The older scalar `identity_gate` and
historical `bridge_mode="dense"` remain available only for those explicit
compatibility paths.

## Reviewed Plan

For production code, keep discovery and mutation separate:

```python
workflow = arti.pretrained(model, provider="peft")
workflow.scan(sample_batch)
plan = workflow.plan(
    features={"recall": {"enabled": True, "steps": 1}},
    where="mlp",
    max_extra_params="2%",
)
plan.write("arti.plan.json")

reviewed = arti.ARTIPlan.read("arti.plan.json")
workflow.apply(reviewed)
```

The plan records the provider, model id, requested and resolved revision,
component structure hashes, exact selected modules, parameter budget,
Recall/Half/Phase settings, training engine, dependency versions, and native
capabilities. An unmatched selector fails during planning and reports candidate
paths; it does not silently produce an empty training job.

`Half` is the default activation when Recall is enabled. For an ablation only,
set `features={"recall": {"enabled": True, "steps": 1}, "half": {"enabled": False}}`;
this retains the same Recall structure while replacing trace survival with an
identity activation. Production plans should leave it enabled unless their own
controlled benchmark establishes a reason to change it.

## Providers

Install only the ecosystems in use:

```bash
uv sync --extra qwen       # Transformers and Accelerate
uv sync --extra peft       # PEFT, Transformers, Accelerate
uv sync --extra sd         # Diffusers, PEFT, Transformers, Accelerate
```

For externally distributed Diffusers checkpoints, set
`loader_kwargs={"use_safetensors": True}` when the repository provides that
format. Compatibility with a legacy checkpoint must be an explicit project
decision.

Provider behavior:

| Provider | Root object | ARTI components |
| --- | --- | --- |
| `torch` | `nn.Module` | the model |
| `transformers` | `PreTrainedModel` | the model |
| `peft` | `PeftModel` | the PEFT-wrapped model |
| `diffusers` | `DiffusionPipeline` | selected `unet`, `transformer`, text encoder, or VAE modules |

Custom ecosystems can subclass `PretrainedProvider` and call
`register_provider()`.

## Phase Runtime Context

External Phase is a runtime requirement, not an optional feature channel:

```python
workflow = arti.from_pretrained(
    model,
    provider="transformers",
    sample_batch=sample,
    features={
        "phase": {
            "enabled": True,
            "mode": "external",
            "coord_dim": 8,
            "frame_mode": "operator_bank",
        }
    },
    where="mlp",
)

tokens = workflow.generate(
    input_ids,
    arti_context={
        "coord": coord,
        "observer_coord": observer_coord,
        "frame_operators": inverse_operator_bank,
        "mask": attention_mask,
    },
)
```

The equivalent context-manager form preserves every native call surface:

```python
with workflow.context(coord=coord, frame_operators=operators):
    output = workflow.model.generate(input_ids)
```

If an external-phase plan is called without the required context, ARTI fails
instead of silently substituting zero coordinates.

## Training Engines

`TrainingSpec.engine` selects one of:

- `torch`: compact native loop for ordinary tensor batches.
- `accelerate`: device placement, mixed precision, accumulation, and launched
  distributed execution.
- `transformers`: native `Trainer` and `TrainingArguments` over the mutated
  model.

Set `distributed=true` only under `accelerate launch`; a single-process launch
is rejected. TrainingArguments fields such as `per_device_train_batch_size`
may be supplied in `trainer_kwargs`, while actual Trainer constructor options
remain available alongside them.

Optimizer, scheduler, loss history, engine, step count, and plan fingerprint
are stored in the `arti.st` checkpoint sidecars. To resume exactly, construct
the same optimizer and scheduler, then pass them to `load_weights()` before the
next `fit()` call.

## Declarative CLI

Example `arti-pretrained.toml`:

```toml
[model]
id = "Qwen/Qwen3-0.6B"
provider = "transformers"
task = "causal-lm"
revision = "<immutable-commit>"

[loader]
local_files_only = true

[features.recall]
enabled = true
steps = 2
slots = 4

[insertion]
where = "mlp"
scale = "tiny"
freeze_base = true
max_adapters = 4

[training]
engine = "accelerate"
steps = 100
learning_rate = 0.0003
mixed_precision = "bf16"
gradient_accumulation_steps = 4
device = "auto"

[sample]
factory = "my_project.data:qwen_sample"

[data]
factory = "my_project.data:train_loader"
```

Run each lifecycle stage explicitly:

```bash
arti pretrained doctor
arti pretrained scan arti-pretrained.toml --output scan.json
arti pretrained plan arti-pretrained.toml --output arti.plan.json
arti pretrained apply arti-pretrained.toml --plan arti.plan.json --weights arti.st
arti pretrained fit arti-pretrained.toml --plan arti.plan.json --weights arti.st
arti pretrained validate-lock --lock arti.pretrained.lock.json
```

`model.factory` can replace `model.id` for a locally constructed model. Factory
paths use explicit `package.module:callable` syntax; domain datasets and schemas
remain outside ARTI.

## Export And Lock

Export produces the normal `arti.st` package plus:

```text
arti.plan.json               reviewed ARTIPlan
arti.pretrained.lock.json    source, environment, structure, plan, and weight hashes
```

The lock binds the immutable model revision, package versions, component
structure fingerprints, training settings, plan hash, `arti.st` hash, and its
integrity lock hash. SHA-256 detects drift and corruption; it is not a publisher
signature.

The minimum adoption acceptance path is therefore: one standard PyTorch model
and one local Transformers configuration must each support explicit insertion,
one or more training steps, `arti.st` export, fresh-process reload, native
`forward`/`generate`, and `detach`. This is an API and lifecycle contract, not a
quality benchmark for a downstream task.

## Validation Scope

The repository's pretrained ecosystem smoke uses local real checkpoints for
Qwen3-0.6B, a tiny ViT, and tiny Stable Diffusion. It checks PEFT, generation,
KV cache, Accelerate BF16 training, Transformers Trainer, Diffusers Pipeline,
identity application, `arti.st` restoration, and lock validation. This is
integration and reproducibility evidence, not a claim of downstream quality
superiority. A separate two-process gloo smoke verifies the Accelerate
distributed path and identity-gated DDP behavior.
