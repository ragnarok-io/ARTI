# Unified Model Attachment

`ARTI.attach` installs `ARTILayer@3` at selected PyTorch module boundaries
without changing the host model class. A configured layer runs a FederatedProgram
graph; attachment only requires that its terminal `value` restores the packed
host-boundary shape.

## Attach

```python
import arti

# ``program`` is a configured arti.mechanisms.FederatedProgram execution region.
layer = arti.ARTILayer(program)

model = arti.ARTI.attach(
    model,
    layer,
    layers="model.layers.*",
)
print(model.arti.summary())
```

Omit `layers` for recognized Transformer architectures or shape-preserving
`nn.Sequential` models. Use `ARTI.discover()` and `ARTI.preview()` to inspect
the same path selection without mutating the model.

The default `ARTILayer()` is an explicit identity Federal shell and has no
trainable parameters. Training requires a configured Federation with
trainable Bank operands or Formula parameters.

## Declarative Placement

TOML owns host placement, not mechanism mathematics:

```toml
[arti]
format_version = 2

[layer]
layers = ["model.layers.*"]
freeze_backbone = true

[training]
engine = "torch"
objective = "tensor_alignment"
learning_rate = 0.001
steps = 100
gradient_accumulation_steps = 1
mixed_precision = "no"
corruption_probability = 0.15
```

```python
model = arti.ARTI.attach(model, layer, config="arti-attach.toml")
model.arti.write_lock("arti.attach.lock.json")
model.arti.validate_lock("arti.attach.lock.json")
```

The lock binds the source config, resolved paths, host structure, and Federal
runtime contract fingerprint.

## Training

The backbone is frozen by default. `model.arti.parameters()` returns only
trainable parameters owned by attached Federal graphs.

```python
session = model.arti.trainer(
    engine="accelerate",
    objective="model_loss",
)
result = session.fit(loader, checkpoint_path="run.arti.st")
```

`tensor_alignment` compares corrupted execution against clean host-boundary
tensors. `model_loss` uses the host model's native scalar loss. A custom
objective is any callable returning a scalar Tensor.

## Save And Reload

```python
model.arti.save("assistant.arti.st")

fresh_model = load_base_model()
fresh_model = arti.ARTI.load(
    fresh_model,
    "assistant.arti.st",
    layer=layer,
    map_location="cuda",
)
```

Attachment artifacts end in `.arti.st`. A configured Federation must be
supplied again on load; ARTI verifies its Federal contract before writing
weights. This avoids
reconstructing mechanism semantics from JavaScript-like metadata or class-name
guesses.

## Runtime Control

```python
model.arti.disable(paths=("model.layers.0",))
model.arti.enable(paths=("model.layers.0",))
model.arti.set_capture(True)
diagnostics = model.arti.diagnostics()
model = model.arti.detach()
```

Enable and disable operate on complete ARTILayer graphs. AdaptivePulse remains
an independent versioned module that may appear inside a Bank program, but it
is not the attachment's outer runtime.

## Hub Bundle

```python
model.arti.save_pretrained(
    "my-arti-model",
    base_model="Qwen/Qwen3-0.6B",
)

model = arti.ARTI.from_pretrained(
    "my-arti-model",
    layer=layer,
)
```

The bundle stores `model.arti.st`, `arti-attach.toml`, the deterministic lock,
and a base-model reference. It does not copy base model weights.
