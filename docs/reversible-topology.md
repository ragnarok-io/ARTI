# Reversible Topology

`arti.alpha.Fold` and `arti.alpha.UnFold` are the versioned `Fold@2` and
`UnFold@2` operations. They change which tensor instances are exposed to an
intermediate block without summarizing, reconstructing, duplicating, or
discarding the other instances.

```python
import torch
from arti import alpha

topology = alpha.ReversibleTopology(
    active_count=2,
    policy=alpha.FixedTopologyPolicy(order=[2, 0, 3, 1]),
)
fold, unfold = topology.operations()

x = torch.randn(4, 4, 64)
state = fold(x)
active = some_block(state.active)
result = unfold(state.replace(active=active))
y = result.value
```

The fold result contains three required parts:

- `active`: instances that continue through the intermediate computation.
- `folded`: original value payloads that bypass that computation.
- `record`: the recorded permutation, mask, shape, and component contract.

UnFold consumes the original record. It never predicts a second layout and it
does not own a decoder. With no intermediate value change, the same-device and
same-dtype round trip is exact:

```python
state = fold(x, mask=mask)
result = unfold(state)
assert torch.equal(result.value, x)
assert torch.equal(result.mask, mask)
```

## Component Identities

| Component | Reference | Contract |
| --- | --- | --- |
| reversible Fold | `arti/fold@2` | recorded permutation and partition |
| reversible UnFold | `arti/unfold@2` | exact inverse from the Fold record |
| shared topology | `arti/reversible-topology@1` | active budget and policy |
| fixed policy | `arti/fixed-topology-policy@1` | deterministic full index order |
| topology action | `arti/topology-action@1` | continuous operands for one topology decision |
| hard partition | `arti/stable-priority-partition@1` | valid-first stable priority ordering |
| learned policy | `arti/learned-topology-policy@1` | lightweight scorer baseline |
| Bank Formula policy | `arti/bank-formula-topology-policy@1` | fixed Query, operand Banks, and versioned Formula |
| topology operand Bank | `arti/topology-operand-bank@1` | fixed addresses and trainable topology operands |
| topology Formula | `arti/topology-priority-formula@1` | interprets operands as priority and confidence |
| topology surrogate | `arti/topology-surrogate@1` | backward-only estimator for hard selection |
| inverse contract | `arti/inverse-topology-contract@1` | validates and applies the recorded inverse without retaining the learner |
| record | `arti/fold-record@1` | topology-only runtime record |
| state | `arti/fold-state@1` | active and folded value payloads |

The previously published `arti/fold@1` and `arti/unfold@1` remain separate
alpha mechanisms for soft compaction and learned expansion. Exact references
never silently migrate between these contracts.

## Topology Learning Is Bank Formula Evaluation

The main learned topology path does not hide topology knowledge in Fold or
UnFold. It evaluates a versioned Formula over operands read from one or more
Banks:

```text
fixed Query -> topology operand Bank -> topology Formula -> TopologyAction
TopologyAction -> stable hard operator -> permutation and partition
```

For an instance `i`, the path can be written as:

\[
q_i = Q_{fixed}(stopgrad(x_i)),\qquad
o_i = Read(B, q_i),\qquad
p_i = \Phi_{topology}(o_i)
\]

The fixed topology operator then computes the valid-first stable ordering from
`p` and the mask. In this sense, topology learning happens in Bank operand
space and is expressed through Formula evaluation. The Formula does not own
the Bank, and the Bank does not own topology semantics.

This separation is a hard contract:

- Bank stores reusable topology operands.
- Formula interprets typed operands as priority and optional confidence.
- Policy composes Query, Bank, and Formula outputs into a priority proposal.
- ReversibleTopology owns the active budget and the optional training surrogate.
- Operator alone applies mask validity, stable tie-breaking, and constructs a legal full permutation.
- Fold transports real tensor instances through that permutation.
- UnFold consumes the recorded inverse and never queries the Bank again.

Formula output is therefore a proposal, not an arbitrary permutation program.
It cannot summarize values, mix soft values into the hard forward path, change
the active budget, or bypass the recorded inverse. The exact round trip remains
a gather/scatter property and does not depend on how priorities were learned.

Independent topology Banks are read and normalized independently before their
Formula outputs are composed with explicit weights and confidence. They are not
placed under one global softmax, so adding a Bank does not change how an
existing Bank addresses its own slots. The composed priority field and final
hard topology can still change when Banks contribute conflicting priorities;
Phase B does not claim arbitrary concat closure or topology isolation.

`LearnedTopologyPolicy@1` remains a useful direct-scorer baseline.
`BankFormulaTopologyPolicy@1` is the principal composable mechanism: fixed
Query, trainable Bank values, a versioned Formula, and a fixed hard operator.
Custom Query or Formula modules can satisfy the runtime contract directly. To
participate in verified `arti.st` save/load, a custom component must also be
registered with a stable canonical reference and deterministic config builder.

## Learning Boundary

Hard sorting is discrete. The real forward path always gathers the selected
original instances. `TopologySurrogate@1` is owned by `ReversibleTopology@1`
and exists only to estimate gradients for the priority-producing parameters;
it is not a soft Fold and it must never enter the stored `active` or `folded`
values. Evaluation and `no_grad` execution skip this training-only path.

`UnFold@2` does not retain the scorer, Query, Banks, Formula, Operator, or
surrogate. It holds only `InverseTopologyContract@1`, validates the Fold record,
and applies its recorded inverse permutation.

Bank routing diagnostics are opt-in. `diagnostics="summary"` reduces every
route to a bounded per-slot mean and every confidence to one scalar before
copying the latest detached summaries to CPU; `clear_diagnostics()` releases
them. `diagnostic_slot_limit` rejects summaries that exceed the declared host
budget. The default forward path retains and downloads no route tensors.

## Transport Boundaries

The current contract supports the penultimate instance axis, fixed topology,
a lightweight learned scorer, and Bank Formula topology. It does not include
Half-specific behavior, Refine scheduling, a learned value source, or a custom
performance kernel. Masked instances never displace valid active instances,
and the same recorded permutation transports values and masks.

The preserved payload must remain available until UnFold. Fold can reduce the
physical sequence length seen by the intermediate block, but it does not claim
to reduce storage for the preserved payload itself.

Without coordinates or another stable lineage identity, permutation
equivariance holds when priorities establish the same strict ordering. Exact
priority ties deliberately use original host index as the deterministic stable
tie-break. Therefore the contract does not claim unconditional equivariance at
ties, including ties introduced by reduced precision.

`FoldRecord.producer_provenance_fingerprint` binds the producer configuration
and ordered Bank layout, while the record's permutation binds the topology that
actually executed. Exact trained tensor hashes belong to the saved `arti.st`
component state contract. Fold does not hash an entire mutable Bank during each
training forward.

## Validation Boundary

The public contract guarantees exact transport, complete lineage, bounded
active workspace size, explicit surrogate gradients, and fail-closed record
validation. It does not claim task-quality improvement or training-time speedup.
Applications should measure the complete scorer, transport, preserved-payload,
intermediate-block, and inverse costs for their own workload.
