# Progressive ARTI API

ARTI mechanisms are optional capabilities, not a bundle that must be enabled
together. Start with the smallest layer whose input contract matches the data.

## Ordinary tensors without phase

```python
layer = arti.nn.Layer(dim=768)
out = layer(hidden, mask=mask)
```

The `minimal` default has no phase, coordinate fallback, pairwise context,
virtual interface, Recall, or virtual Recall parameters. It does not invent
phase for data that has none.

## Recall only

```python
layer = arti.nn.Layer(dim=768, profile="recall")
out = layer(hidden, mask=mask)
loss = task_loss(out.y, target) + recall_loss(out.recall_prediction, clean_trace)
```

This profile enables Recall, default `Half` trace survival, and the virtual
Recall training output. Phase and shared interfaces remain absent.

## Multi-participant context

Use participant helpers to construct coordinates, masks, and visibility, then
require visibility at the layer boundary:

```python
context = arti.build_participant_context(...)
layer = arti.nn.Layer(dim=768, profile="multisource", coord_dim=context.coord.shape[-1])
out = layer(
    hidden,
    coord=context.coord,
    mask=context.mask,
    visibility=context.visibility,
    frame_operators=inverse_operators,
)
```

See [Participant Context](participant-context.md) and
[Membrane Routing](membrane-visibility-routing.md).

## Multi-sensor phase

For sensor data already transformed by a known observation operator, use a
coordinate inverse. ARTI applies the supplied inverse operator; it does not
append phase as an identity label:

```python
selected = arti.features(
    phase=True,
    coord_dim=sensor_count,
    coord_frame_mode="operator_bank",
    visibility=True,
    virtual_interface=True,
)
layer = arti.nn.Layer(dim=dim, features=selected)
```

If the data has no meaningful phase, keep `phase=False` and `coord_dim=0`.

## Transformer insertion

```python
preview = (
    arti.project(model)
    .plugin("transformers")
    .at("model.layers.*.mlp.down_proj", every=4)
    .freeze(True)
    .budget(max_extra_params="1%")
    .preview(sample_batch)
)
```

Inspect `preview.insertion_plan` before calling `insert()`, or use
`arti.fit(..., dry_run=True)`. See [Fit Adaptation](fit-adaptation.md).

## VisualScan

`VisualScan` is a separate registered visual workspace. Continuous shifts are
inverted by a deterministic operator before its carrier and Pulse residual are
formed. See [Activation And Workspace](activation-workspace.md) and the
[pixel-shift validation](../validation/visual-scan-superresolution.md).

## Dynamic literal vocabulary

Use rigid glyph tensors and a runtime-bound output lexicon when input and output
vocabularies are not fixed to the same slot order. See
[Runtime Vocab Binding](runtime-vocab-binding.md) and [Text Tensor](text-tensor.md).

## Save and inspect

```python
report = arti.inspect(layer, hidden, mask=mask)
print(report.to_markdown())

arti.save(layer, "adapter.arti.st", config={"arti": layer.config.to_dict()})
restored = arti.nn.Layer(dim=768, features=layer.features.to_dict())
arti.load("adapter.arti.st", model=restored)
```

`config.explain()` shows enabled mechanisms, required and accepted inputs,
allocated capacities, and whether synthetic fallback context exists.
`config.diff(other)` returns only execution fields that differ.

## Profiles are transparent

```python
selected = arti.profile("recall")
print(selected.to_dict())
print(selected.compile(input_dim=768).explain())
```

Profiles are immutable feature declarations. They do not inspect data, download
models, choose hidden dimensions, or silently enable unrelated mechanisms.
