# Unified Model Attachment

`ARTI.attach` installs `ARTILayer@2` at selected PyTorch module boundaries
without changing the host model class. The attached layer is backed by one
composable `AdaptivePulse` graph.

## Attach

```python
import arti

pulse = arti.mechanisms.AdaptivePulse(
    half=arti.Half(stochastic=False, learnable=True),
)
layer = arti.ARTILayer(pulse)

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

The default `ARTILayer()` contains an identity Pulse graph and therefore has
no trainable parameters. Training requires at least one trainable stage.

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

The lock binds the source config, resolved paths, host structure, and Pulse
manifest fingerprint.

## Training

The backbone is frozen by default. `model.arti.parameters()` returns only
trainable parameters owned by attached Pulse graphs.

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

Attachment artifacts end in `.arti.st`. A non-identity graph must be supplied
again on load; ARTI verifies its manifest before writing weights. This avoids
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

Enable and disable operate on complete ARTILayer graphs. Individual Pulse
stages remain ordinary versioned modules and are configured when the graph is
constructed.

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
