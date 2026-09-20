# Layered Recall

`LayeredRecallModel` is a legacy placement helper that attaches independent
candidate-trace Recall branches to named Transformer-style layers while keeping
the backbone frozen. Prefer the explicit config form when the same placement is
used by training, evaluation,
and artifact tooling:

```python
from arti.legacy import LayeredRecallConfig, LayerRecallSpec, LayeredRecallModel

config = LayeredRecallConfig(
    layer_paths=(
        "model.layers.4",
        "model.layers.12",
        "model.layers.20",
    ),
    rank=16,
    slots=8,
    use_half=True,
)
layered = LayeredRecallModel.from_config(
    model,
    config,
    sample_batch=sample,
)
```

This legacy helper is retained for loading and inspecting older layered Recall
workflows. New attachments use `ARTI.attach(...)` with the FederatedProgram-hosted
`ARTILayer`.

The API does not prescribe three layers, equal sizes, or early/middle/late
placement. Declare any number of branches with independent dimensions and
mechanisms:

```python
config = LayeredRecallConfig(
    layers=(
        LayerRecallSpec("model.layers.2", rank=8, slots=32),
        LayerRecallSpec(
            "model.layers.9",
            rank=24,
            slots=128,
            copies=3,
            combine="mean",
            use_half=False,
            recognition_mode="none",
        ),
        LayerRecallSpec(
            "model.layers.21",
            dim=1024,
            rank=12,
            slots=64,
            recognition_mode="explicit",
        ),
    )
)
```

`dim` is optional and otherwise inferred. Layer order, count, rank, bank size,
Half, recognition mode, threshold, and temperature are ordinary hyperparameters.
`copies` creates genuinely independent Recall lines at the same physical layer;
their deltas are combined by `sum` or `mean`. This is not equivalent to merely
increasing one line's rank or slots, so repeated-key topologies can be tested
against wider single-line controls under the same parameter budget.
The three-layer examples in validation documents are controlled comparisons,
not architectural defaults.

Tensor outputs, tuple outputs whose first item is hidden state, and common
mapping outputs are preserved. Runtime scanning can infer hidden dimensions for
explicit paths, including Qwen-style `model.layers.N` paths. Pass `dims` when a
model cannot be safely executed for inspection.

Each branch applies exactly:

```text
delta_l = Half(Recall_l(h_l))
h'_l = h_l + delta_l
```

Recall branches own separate banks and layer paths. They do not contain a
trainable strength controller, share answer targets, or read future tokens. A
recognizer exists only when `recognition_mode="alignment"` is explicitly
selected.

## Label-free trajectory repair

Run the same sample through clean and corrupted views:

```python
result = legacy.layered_recall_trajectory_loss(
    layered,
    clean_inputs,
    corrupt_inputs,
    mask=mask,
)

result.loss.backward()
```

The clean pass disables every Recall branch. At each named layer, the corrupt
pass learns toward the detached clean hidden state:

\[
\mathcal L_l = \|h_l^{corrupt} +
\operatorname{Half}(R_l(h_l^{corrupt})) -
\operatorname{stopgrad}(h_l^{clean})\|^2.
\]

This positive-only objective is the default acquisition stage. Pass
`unseen_inputs` and a nonzero `unseen_weight` only in a later selectivity stage
when the application needs Recall abstention.

The optional unseen term drives candidate deltas toward zero on unrelated
inputs. Labels, answers, query targets, and future tokens are not accepted by
this objective.

### Normalized multi-layer loss

Absolute hidden MSE can differ greatly by depth. Calibrate the no-Recall
corruption at each layer before comparing or jointly training branches:

```python
calibration = layered.calibrate(
    clean_inputs,
    corrupt_inputs,
    mask=mask,
)

result = legacy.layered_recall_trajectory_loss(
    layered,
    clean_inputs,
    corrupt_inputs,
    mask=mask,
    unseen_inputs=unrelated_inputs,
    unseen_weight=1.0,
    calibration=calibration,
)
```

`LayeredRecallCalibration.scales` contains detached baseline corruption MSE
for each named path. The normalized objective starts near one per layer when
the residual branches are identity-initialized. This prevents a high-variance
late layer from silently dominating the multi-layer objective. Calibration is
explicit because different corruption families may need different scales.
It is not a fixed point, learned law, or persistent property of a layer. Refit
or omit it whenever the data distribution, corruption process, tensor shape,
or training objective changes.

## Layer control

```python
layered.set_enabled(False)
layered.set_enabled(True, paths=("model.layers.20",))
parameters = tuple(layered.recall_parameters())

with layered.enabled_layers(("model.layers.4", "model.layers.12")):
    partial = layered(**batch)

with layered.disabled():
    baseline = layered(**batch)
```

Disabling a branch leaves the base layer unchanged. This supports last-layer,
leave-one-layer-out, and depth-curve ablations without rebuilding the model.
Both context managers restore the exact previous enabled state, including when
the wrapped forward raises.

Enable capture on the wrappers when detailed traces are needed, then read a
stable layer-keyed report:

```python
for wrapper in layered.wrappers.values():
    wrapper.capture = True

layered(**batch)
diagnostics = layered.diagnostics()
# raw_delta, delta, recognition, and Half survival per layer
```

## Artifact boundary

Layered Recall is a legacy placement surface and no longer owns a
second layer-local artifact format. Use the stable attachment API's
`RecallBankContract`, `save_bank`, `load_recall_bank`, and `RecallBankAssembly`
for strict bank assets and composition. This keeps one artifact schema and one
compatibility check for all bank-backed integrations.

## Practical placement

Begin with three separated paths rather than every layer. Match total Recall
parameters against a single larger branch and an ordinary low-rank adapter.
Measure every single-layer location before claiming that depth matters. See the
[Layered Recall validation](../validation/layered-recall-trajectory.md) for the
current controlled evidence and its limits.
