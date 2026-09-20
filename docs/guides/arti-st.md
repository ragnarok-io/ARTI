# `arti.st` Weight Protocol

`arti.st` is ARTI's default SafeTensors weight protocol. The name reads as
"artist", but the file contract is intentionally plain: `.st` contains tensors
only, and everything with different ownership lives in a separate sidecar.

## Save And Load

```python
import arti
from torch import nn

saved = arti.save(model, "arti.st")

restored = build_the_same_model()
loaded = arti.load("arti.st", model=restored, map_location="cpu")
```

ARTI does not import or instantiate the class named in the manifest. Callers
construct the model explicitly, then `load()` verifies its class before loading
weights. Passing no model returns the validated state dict for a caller-owned
construction path:

```python
loaded = arti.load("arti.st")
state_dict = loaded.state_dict
config = loaded.manifest["architecture"]["config"]
```

## Files

A complete package can contain:

```text
arti.st                  learned model state tensors
arti.json                protocol, package version, architecture, file records
arti.lock.json           manifest and member SHA-256 records
arti.glyphs.st           optional rigid glyph tensors
arti.vocab.json          optional external vocabulary metadata
arti.checkpoint.st       optional optimizer/scheduler/training tensors
arti.checkpoint.json     optional non-tensor checkpoint tree
```

The primary `.st` metadata contains only protocol strings. Architecture config,
glyph tensors, vocabulary rows, and training state are never hidden inside the
learned weight file.

## Rigid Resources

Pass physical symbol resources explicitly:

```python
arti.save(
    model,
    "arti.st",
    glyph_tensors={"visible": glyph_bank, "controls": control_bank},
    vocab_metadata={"items": visible_strings, "font": "Inter-Regular.ttf"},
)
```

`glyph_tensors` are written to `arti.glyphs.st`; `vocab_metadata` is written to
`arti.vocab.json`. They are returned separately as `loaded.glyph_tensors` and
`loaded.vocab_metadata`. This preserves the distinction between learned model
state and externally defined literal constants.

## Training Resume

```python
arti.save(
    model,
    "arti.st",
    optimizer=optimizer,
    scheduler=scheduler,
    training_state={"step": step, "best_loss": best_loss},
)

loaded = arti.load(
    "arti.st",
    model=restored_model,
    optimizer=restored_optimizer,
    scheduler=restored_scheduler,
    map_location="cuda",
)
```

Optimizer and scheduler tensors use SafeTensors. Their nested non-tensor state
uses a tagged JSON tree, so ARTI does not need pickle to restore a training
checkpoint. Arbitrary Python objects are rejected.

## Weight Scope

The default `scope="all"` stores the complete model state. For a frozen base
model with a small ARTI adapter:

```python
arti.save(adapter, "arti.st", scope="trainable")
```

On load, missing frozen-base keys are expected, but saved keys that do not exist
in the target model are rejected. This is useful for Qwen integrations where
the pretrained base checkpoint is managed separately.

An `ARTIFitResult` uses the same protocol directly:

```python
result = arti.fit(model, sample_batch=sample)
result.export_st("arti.st")                 # trainable adapters only
result.export_st("arti.st", include_base=True)
```

Fit exports embed a hash-bound insertion report in the JSON sidecar. This lets
`arti.apply_adapter(fresh_model, "arti.st", sample_batch=...)` reconstruct and
validate the exact tensor boundaries before loading adapter tensors. The
default adapter-only export requires the base model to be frozen so unrelated
trainable weights cannot be included accidentally.

Lazy deployment runtimes that cannot execute a representative forward during
model loading may opt in to the hash-bound artifact contract with
`arti.apply_adapter(model, path, trust_artifact_contract=True)`. This mode still
checks module paths, declared feature dimensions, state keys, and artifact
integrity; ordinary Python integrations should continue passing `sample_batch`
for runtime boundary validation.

The older `result.export("adapter.pt")` remains available for compatibility.

## Composite Components

ARTI keeps ordinary composition as ordinary PyTorch composition. A model can
contain nested ARTI modules without inheriting a special composite base class:

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.recall = arti.Retrieve(dim=64, slots=8)
        self.half = arti.Half()

    def forward(self, x):
        return self.half(self.recall(x))
```

The manifest also contains an optional `component_graph` section. It records
component instances, mount paths, typed bindings, and shared parameter groups.
The graph has its own closure fingerprint; the existing component provenance
fingerprint and state-schema fingerprint remain separate compatibility checks.

Inspect the graph without loading weights into a model:

```python
graph = arti.component_graph(model)
arti.validate_component_graph(graph)
```

Repeated mounts of the same module instance are represented by one graph node
with multiple mount paths. Equal tensor values are not treated as shared
parameters; sharing is recorded only when the same PyTorch parameter or buffer
object is actually reused. The loader does not import classes from a manifest,
and graph or parameter-sharing changes require an application-owned migration.

## Integrity And Compatibility

`load()` checks the manifest hash first, then every referenced file hash, the
SafeTensors protocol metadata, component graph fingerprint, state-schema
fingerprint, tensor count, ARTI package compatibility, model class, and exact
state-dict schema compatibility. Alpha `0.x` packages reject manifests from a
newer minor release. A package missing the current state contract is rejected;
it must be explicitly re-exported or migrated by an application-owned tool.

SHA-256 detects corruption and accidental replacement. It is not a signature
and does not prove publisher identity; signed release channels can sign
`arti.lock.json` externally.

## Legacy Weights

The current public API does not import or migrate legacy `.pt` files. Load an
old tensor-only checkpoint in an application-owned conversion script, verify
its tensors, construct the current module explicitly, and export a fresh
`arti.st` package.

Run the complete example:

```bash
uv run --extra torch python examples/arti_st_roundtrip.py
```
