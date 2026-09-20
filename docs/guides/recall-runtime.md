# Recall Runtime

`RecallRuntime` is the explicit composition boundary for a forward-updated
Recall Bank. It keeps the trained writer, the dynamic values-only state, and
the Formula reader separate.

```python
import torch
import arti
from arti.mechanisms import NormalizedDeltaRecallValueUpdater, RecallRuntime

writer = NormalizedDeltaRecallValueUpdater(
    hidden_dim=64,
    slots=32,
    workspace_dim=16,
    factors=2,
)
reader = arti.Retrieve(64, 32, formula="arti/delta@1")
runtime = RecallRuntime(writer, reader)

hidden = torch.randn(2, 8, 64)
trace = torch.randn(2, 16, 64)
state = runtime.initial_state(2, reference=hidden)

state = runtime.update(
    trace,
    state,
    trace_mask=torch.ones(2, 16, dtype=torch.bool),
    detach_state=True,
)
y = runtime.read(
    hidden,
    state,
    mask=torch.ones(2, 8, dtype=torch.bool),
)
```

The writer parameters are ordinary trainable module parameters. `RecallState`
is the caller-owned forward state and is not an optimizer parameter. Use
`state.fork()` for an independent branch, `state.snapshot()` for a detached
checkpoint, and `state.state_dict()` for a tensor-only persistence payload.

Each runtime exposes a `RecallRuntimeContract` and a stable
`contract_fingerprint`. The contract explicitly records the reader, Formula,
Updater, values-only Bank layout, and `arti/recall-state@1` state descriptor.
States created by `runtime.initial_state()` carry that fingerprint and the
state schema version. A state from another Formula, Updater, slot layout, or
hidden dimension is rejected before `read()` or `update()`; a raw tensor or an
old unbound state must be migrated explicitly:

```python
legacy = {"value": old_value, "step": torch.tensor(4)}
state = runtime.migrate_state(legacy)
```

The fingerprint records component references, configuration, parameter layout,
and Formula behavior without depending on current `requires_grad` flags.

Use `runtime.scan()` when several trace chunks must update one state in a
defined order. Its order axis is sequential; it is not averaged as a normal
batch dimension. `runtime.read()` passes the resulting state to the reader as
external `memory`; the reader's Formula remains responsible for transforming
that state into a host next-state tensor.

`detach_state=False` keeps the complete transition differentiable for offline
training. Forward-only applications should pass `detach_state=True` or call
`state.snapshot()` before persistence.
