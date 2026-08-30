# Operable Tensor Port

The alpha tensor-operation surface provides a stable tensor port whose backing
can be replaced between calls without changing model parameters or graph
structure. The port always resolves to a real tensor: its runtime-owned default
backing or an explicitly mounted external backing.

```python
import torch

from arti import alpha

spec = alpha.PortSpec(
    canvas_tokens=64,
    port_slots=16,
    dim=128,
    port_to_canvas=tuple(range(16)),
)
port = alpha.OperableTensorPort(spec, batch_size=1, device="cuda")
fold = alpha.SharedCanvasFold(spec)

snapshot = port.resolve()
canvas = fold(world, snapshot, world_mask=world_mask)
```

`SharedCanvasFold` returns a world-shaped `[B, N, D]` tensor. It overlays
visible backing slots at their declared canvas coordinates; it does not append
a modality or feature axis. `canvas.source_plane` and `canvas.source_index`
retain provenance for each visible value.

## Typed Edits

The first edit Formula accepts one hard instruction per batch row:

- `KEEP` preserves the backing.
- `COPY` copies one pre-transition value from the world or backing into a
  backing slot.
- `CLEAR` writes the declared empty value and clears that slot's visibility.

```python
instruction = alpha.TensorEditInstruction(
    operation=torch.tensor([int(alpha.EditOperation.COPY)], device="cuda"),
    source_plane=torch.tensor([int(alpha.CanvasSource.WORLD)], device="cuda"),
    source_index=torch.tensor([12], device="cuda"),
    destination_index=torch.tensor([3], device="cuda"),
    active=torch.tensor([True], device="cuda"),
)

edited = alpha.TensorEditFormula(spec)(canvas, snapshot, instruction)
port.advance(edited.value, edited.mask)
```

The Formula reads one immutable snapshot and returns a separate next-backing
proposal. It never writes the world tensor. `port.advance(...)` makes an edit
visible to a later call; it cannot alter the canvas already consumed by the
current call.

## Learned Operations

`TensorOperationSelector` uses a fixed, deterministic Query to select a hard
edit from a trainable `TensorOperationBank`. All optimizer-owned selection
values live in the Bank.

```python
bank = alpha.TensorOperationBank(
    spec,
    candidate_count=32,
    key_dim=64,
)
selector = alpha.TensorOperationSelector(spec, bank)
operation = alpha.TensorOperation(
    spec,
    selector,
    surrogate=alpha.TensorEditSurrogate(spec),
)
loop = alpha.TensorOperationLoop(operation)

proposal = loop(
    world,
    port.resolve(),
    schedule=alpha.TensorOperationSchedule(operation_steps=8),
    world_mask=world_mask,
)

port.advance(proposal.value, proposal.mask)
```

Every operation iteration folds the latest internal shadow backing, runs a fresh
Query, selects a typed instruction, and applies one Formula transition. The
loop never mutates the live port. Its schedule may be a scalar or an `int64
[B]` tensor, so batch rows can request different operation depths. Optional
post-transition stable stopping is configured with
`TensorOperationStopPolicy`.

For compiled or accelerator execution, a tensor-valued schedule must also set
`max_steps`; this provides the fixed execution and trace capacity without
reading the requested depth back to the host.

The decoded instruction and training logits are available through each
operation result and trace. Free-running inference always executes a hard
`KEEP`, `COPY`, or `CLEAR`; the Formula never applies a weighted mixture of
edits. The Query remains fixed while its input changes after every shadow
transition.

`TensorEditSurrogate` is optional. It leaves the hard Formula result unchanged
while attaching a continuous backward path to the selected Bank fields. Omit it
for a purely discrete operation path. The fixed Query is never made trainable by
enabling the surrogate.

`TensorOperationLoop` defaults to `executor="static_masked"`: it executes the
declared capacity with device-side masks, so accelerator execution does not
perform a per-step host read. Logical early stop still appears in the trace, but
does not claim row-level physical savings. `executor="early_break"` performs a
real short circuit and is intentionally limited to eager CPU execution.

## Independent Iteration Axes

`TensorInvocation` coordinates Reader Refine and tensor operations from the
same call-boundary snapshot. The axes have independent schedules and do not
feed intermediate state into each other during the current call.

```python
invocation = alpha.TensorInvocation(spec, reader_module, loop)
result = invocation(
    world,
    port.resolve(),
    reader_schedule=alpha.ReaderRefineSchedule(reader_steps=6),
    operation_schedule=alpha.TensorOperationSchedule(operation_steps=12),
    world_mask=world_mask,
)

current_output = result.output
next_backing = result.operation
```

An external backing can be selected at call boundaries:

```python
port.mount(external_a, external_a_mask)
port.replace(external_b, external_b_mask)
port.detach_to_default()
```

Detaching returns to the retained default backing. Mounting does not register
the external tensor as a parameter or model buffer and does not merge it into
the default state.

This alpha slice establishes exact port, shared-canvas Fold, fixed-Query Bank
selection, typed hard edits, independently scheduled multi-step operation, and
next-call proposal semantics. It does not claim geometric edit macros,
persistence infrastructure, transactions, row-level sparse execution, or
physical performance gains.

## Long-Lived Validation

A one-call sample only proves that the operation path is connected. It does
not validate the value of a persistent backing. Long-lived experiments must
reuse the same committed root across calls:

```text
snapshot(B_t)
  |-- Reader Refine(world_t, B_t) --> current output
  `-- TensorOperation(world_t, B_t) --> proposal

call boundary: commit(proposal) -> B_(t+1)
```

The current output and operation branch read the same snapshot. A proposal is
visible only to later calls. Evaluate persistent behavior over a sequence of
different events, with committed checkpoints and matched `frozen`, `reset`,
shuffled-state, and event-order controls. Reload and fork checks must begin
from a committed root; an uncommitted proposal is not persistent state.

Lifecycle checks establish state transport and causal ordering only. A
downstream application must separately evaluate whether the accumulated state
improves its task; a single changed output is not quality evidence.
