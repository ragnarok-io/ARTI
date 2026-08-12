# Recall Artifacts

Recall is a tensor mechanism: a fixed Query reads candidate factors from a
Bank, and a Recall Formula maps the current state and recalled factors to the
next state. Pretraining, fine-tuning, and test-time optimization are caller
choices; they do not change Recall's forward semantics and do not require a
Recall-specific training session.

`*.recall.arti.st` stores a replaceable Recall expert. It is not a complete
`arti.st` checkpoint. The artifact contains expert weights plus strict host and
injection fingerprints so loading cannot silently target a merely
shape-compatible model.

```python
import arti

spec = arti.RecallArtifactSpec(
    capability="latent-reconstruction",
    base_model_fingerprint=arti.module_structure_fingerprint(model),
    injection_fingerprint=arti.module_structure_fingerprint(recall_expert),
    visibility_policy="caller-supplied",
    training_metadata={"objective": "complete-trace alignment"},
)

arti.export_recall_artifact(
    recall_expert,
    "latent-reconstruction.recall.arti.st",
    spec,
)
```

Artifact metadata records provenance; it does not prescribe an optimizer,
loss, support schema, or training schedule.

`RecallCapacityPlan` is optional storage metadata. It deterministically reports
how many items fit across bounded expert Banks, but it does not inspect tensors,
route a forward pass, or control training.

## Loading and rollback

Use `RecallExpertRegistry` when expert selection is independent from model
construction. Activation validates both fingerprints before changing the
active expert, and a failed activation restores the prior state.

```python
registry = arti.RecallExpertRegistry(recall_expert, base_model=model)
registry.activate("latent-reconstruction.recall.arti.st")
registry.rollback()
```

## Simultaneous loading

`RecallExpertPool` keeps several compatible artifacts resident. Routing stays
explicit: choose one expert or provide non-negative mixture weights.

```python
import torch

pool = arti.RecallExpertPool(recall_expert, base_model=model)
pool.load_expert("first", "first.recall.arti.st", map_location="cuda")
pool.load_expert("second", "second.recall.arti.st", map_location="cuda")

exact = pool(h, expert="first")
weights = torch.tensor([[0.9, 0.1], [0.2, 0.8]], device=h.device)
mixed = pool(h, mixture_weights=weights)
```

The pool does not infer semantic routing. Tensor, tuple/list, and mapping
outputs retain their structure, and every loaded expert participates normally
in `state_dict()` and device movement.

For native `ARTILatentRecallField` artifacts, concatenate compatible Banks to
create one larger address space:

```python
recall = pool.concatenate()
```

This is Bank composition, not output remixing. Shared Query, routing,
recognition, Formula, and projection state must be identical; only Bank values
may differ.

Fit-attached Recall fields expose the same composition path:

```python
arti.concatenate_adapter_banks(
    model,
    ("first.recall.arti.st", "second.recall.arti.st"),
    bank_names=("first", "second"),
    weights={"first": 2.0, "second": 1.0},
)

arti.set_adapter_bank_weights(model, {"first": 4.0, "second": 1.0})
```

Bank weights modify routing priors, not recalled values. `1.0` is neutral and
`0.0` disables a Bank; at least one valid route must remain enabled.

Signed influence is a separate runtime control:

```python
arti.set_adapter_bank_influences(model, {"first": -1.0, "second": 1.0})
```

Influence changes write direction and strength without changing source
artifacts. Recall refinement depth still controls how often the updated state
is queried again.

## Training boundary

Recall parameters use normal PyTorch autograd. A typical reconstruction setup
compares the internal trace produced from a corrupted or incomplete view with
the detached trace produced from the complete view of the same processed
signal. The project trainer owns the optimizer, schedule, validation split,
and parameter selection.

Recall Formula contracts and parameter tags describe tensor behavior and
parameter ownership. They deliberately do not inject an optimizer or create a
second training runtime. See
[Custom Recall Formulas](custom-recall-formulas.md).
