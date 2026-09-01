# Operable Tensor Port

The stable tensor-operation surface provides a tensor port whose backing
can be replaced between calls without changing model parameters or graph
structure. The port always resolves to a real tensor: its runtime-owned default
backing or an explicitly mounted external backing.

```python
import torch

from arti import mechanisms

spec = mechanisms.PortSpec(
    canvas_tokens=64,
    tensor_shape=(16, 16),
    dim=128,
    tensor_to_canvas=tuple(range(16)),
    folded_tensor_coordinates=tuple((0, column) for column in range(16)),
)
port = mechanisms.OperableTensorPort(spec, batch_size=1, device="cuda")
fold = mechanisms.SharedCanvasFold(spec)

snapshot = port.resolve()
canvas = fold(world, snapshot, world_mask=world_mask)
```

The backing is a logical tensor with shape `[B, *tensor_shape, D]`; it is not a
sequence of Bank slots. `SharedCanvasFold` returns a world-shaped `[B, N, D]`
view. It overlays only the backing coordinates named by
`folded_tensor_coordinates` at the corresponding `tensor_to_canvas` positions;
it does not append a modality or feature axis.
The Reader therefore receives a bounded Fold view even when the complete
backing is larger. `canvas.source_plane` and `canvas.source_index` retain
provenance for each exposed value.

## Typed Operation Fields

`TensorEditFormula@3` accepts one bounded operation field per batch row. Every
field has shape `[B, M]`, so one selected Bank member can act on any number of
tensor elements up to its declared support size:

- `KEEP` preserves the backing.
- `COPY` copies a pre-transition value from the world or backing into a tensor
  coordinate.
- `ERASE` writes the declared empty value and clears that coordinate's visibility.

```python
instruction = mechanisms.TensorEditInstruction(
    operation=torch.tensor([[1, 1, 2]], device="cuda"),
    source_plane=torch.tensor([[0, 0, -1]], device="cuda"),
    source_offset=torch.tensor([[12, 37, -1]], device="cuda"),
    destination_offset=torch.tensor([[3, 9, 11]], device="cuda"),
    active=torch.tensor([[True, True, True]], device="cuda"),
)

edited = mechanisms.TensorEditFormula(spec)(canvas, snapshot, instruction)
port.advance(edited.value, edited.mask)
```

Offsets are a compiled execution representation. Developers can construct
them from logical coordinates or slices with `spec.ravel_coordinate(...)` and
`spec.region_offsets(...)`; the persistent tensor retains its original shape.
The Formula gathers every source from one immutable pre-state snapshot, then
applies all destinations synchronously. This permits swaps, moves expressed as
`COPY + ERASE`, rotations, sparse ranges, and other multi-position edits
without sequential read-after-write artifacts. If multiple active elements target
the same destination, the later support element wins deterministically. The
Formula never writes the world tensor. `port.advance(...)` makes the proposal
visible to a later call; it cannot alter the canvas already consumed by the
current call.

## Learned Operations

`TensorOperationSelector` uses a fixed, deterministic Query to select one
complete hard operation field from a trainable `TensorOperationBank`. The
Query reads the complete world snapshot and complete backing independently;
it is not limited to the Reader's folded view. All optimizer-owned selection
and field values live in the Bank.

```python
bank = mechanisms.TensorOperationBank(
    spec,
    candidate_count=32,
    key_dim=64,
)
selector = mechanisms.TensorOperationSelector(spec, bank)
operation = mechanisms.TensorOperation(
    spec,
    selector,
    surrogate=mechanisms.TensorEditSurrogate(spec),
)
loop = mechanisms.TensorOperationLoop(operation)

proposal = loop(
    world,
    port.resolve(),
    schedule=mechanisms.TensorOperationSchedule(operation_steps=8),
    world_mask=world_mask,
)

port.advance(proposal.value, proposal.mask)
```

Every operation iteration folds the latest private shadow backing, runs a fresh
Query, selects a typed instruction, and applies one Formula transition. The
loop never mutates the live port. Its schedule may be a scalar or an `int64
[B]` tensor, so batch rows can request different operation depths. Optional
post-transition stable stopping is configured with
`TensorOperationStopPolicy`.

For compiled or accelerator execution, a tensor-valued schedule must also set
`max_steps`; this provides the fixed execution and trace capacity without
reading the requested depth back to the host.

The decoded field and training logits are available through each operation
result and trace. Free-running inference always executes one hard Bank member,
whose index map contains hard `KEEP`, `COPY`, or `ERASE` operations. It never combines
parts from different candidate members. The Query remains fixed while its input
changes after every shadow transition.

## Concatenating Operation Banks

Operation Banks compose along the candidate axis. Concatenation preserves each
member's complete field, member ID, source Bank ID, local route normalization,
and explicit Bank influence:

```python
combined = mechanisms.TensorOperationBank.concat(
    (navigation_bank, layout_bank, cleanup_bank),
    name="project-operations",
    influences=(1.0, 0.75, 1.0),
)
selector = mechanisms.TensorOperationSelector(spec, combined)
```

No operand is remixed during concatenation. The default hard route still
selects exactly one complete member from the expanded candidate set. Setting a
Bank influence to zero disables that Bank without changing the remaining
members; positive influences control cross-Bank competition while local member
probabilities remain normalized inside each source Bank.

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
invocation = mechanisms.TensorInvocation(spec, reader_module, loop)
result = invocation(
    world,
    port.resolve(),
    reader_schedule=mechanisms.ReaderRefineSchedule(reader_steps=6),
    operation_schedule=mechanisms.TensorOperationSchedule(operation_steps=12),
    world_mask=world_mask,
)

current_output = result.output
next_backing = result.operation
```

Training should supply only a final next-call task loss. Tensor
differences between internal steps are not treated as a stream or as labels;
the operation path does not require action, route, or intermediate-state
supervision.

An external backing can be selected at call boundaries:

```python
port.mount(external_a, external_a_mask)
port.replace(external_b, external_b_mask)
port.detach_to_default()
```

Detaching returns to the retained default backing. Mounting does not register
the external tensor as a parameter or model buffer and does not merge it into
the default state.

This stable slice establishes an arbitrary-rank logical tensor port,
shared-canvas Fold, fixed-Query Bank selection, complete hard index-map fields,
concat-native Bank composition,
independently scheduled multi-step operation, and next-call proposal semantics.
It does not claim persistence infrastructure, transactions, row-level sparse
execution, or physical performance gains.

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

Lifecycle checks should cover repeated calls, checkpoint/reload, fork isolation,
reset, bounded state storage, and matched frozen-state controls. A single output
does not establish the value of a long-lived tensor workspace.
