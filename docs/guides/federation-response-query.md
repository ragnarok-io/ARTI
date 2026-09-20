# Execution-derived federation queries

`arti/formula-program-query@6` is an alpha composition of existing ordinary
Bank/Fabric members, named SSA outputs, calls and predecessor self-effects.
It does not create a Query Bank, a Query slot containing a network, or an
external neural scorer. Query describes the behavior of the executing
federation, not a new owner of its parameters.

## Composition

```python
from arti.mechanisms import FormulaProgramQueryV6

federation = FormulaProgramQueryV6(
    slot_ids=slot_ids,
    candidates=members,
    terminal_slots={"result": "result"},
    entry_candidates=("observe",),
    continuations={
        "observe": {"combine": "observation_response"},
        "combine": {
            "branch_a": "response_a",
            "branch_b": "response_b",
        },
    },
    max_steps=16,
)
execution = federation(inputs, bank_state=state)
```

Here `members` are ordinary Formula candidates or child calls. Each referenced
response slot must be a named output of its declared member, with floating
shape `[B]` or `[B,1]`. The member may use that same computation for task data,
calling another federation, and generating subsequent responses. It is not
assigned a special Query role or required to have a separate scoring head.

The input port activates the declared entry candidates at step zero with equal
scores. Once a member has actually executed, its response can open a successor.
Shape, dependency and local step admission still apply. A response does not
make an otherwise unavailable instruction executable.

Version 6 sums declared logit contributions to each local action. This is an
explicit local response-combination rule, not a weighted merge of recalled
Values. Log-softmax is over the currently legal local frontier; scores from
different scopes must not be interpreted as globally calibrated raw logits.
STOP has a zero baseline plus any declared contributions, and still requires
all terminal outputs and minimum steps. Hard budget exhaustion is not learned
convergence. An entry list and finite candidate graph are model wiring, not
task-specific semantic labels.

Contribution order is canonical by producer/action name, independent of mapping
insertion order. Accumulation uses at least float32 and preserves float64
responses. Non-finite legal aggregates are rejected, not clipped into valid
preferences. This applies to native and device-side scoring.

## State and iteration

Reading responses never executes their producers again. The tensors retain
their actual numerical and Bank ancestry. Self-effects preserve their current
data input and modify the real predecessor Bank. Later execution can therefore
produce different intermediate values and successor preferences.

Already produced SSA values do not change retroactively when their Bank changes.
To observe the successor Bank in the same invocation, declare a later ordinary
occurrence that reads that shared owner. A child returns its actual named
outputs and original Bank lineage; completion identifies the returned response
without replacing its self-effect target.

Fixed relevant state, wiring, inputs and random conditions give stable replay.
Repeated calls that carry changed state need not select the same paths. An
untouched and unreachable member need not be scanned to establish this property.

## Training and search

The private recursive K-wide executor consumes these same response tensors.
All retained-path write participation and shared-prefix composition retain
their existing explicit policies; the direct `forward` convenience call is
still a single hard path. Version 6 does not silently change single-path
`commit_` into a multi-path state merger.

Routing credit must flow through the computation that emitted each response,
including earlier writes. Enumerating only external `network` parameters or
detaching all ordinary computation erases that path. The existing recorded
replay uses `query_logits(arena)` and constructs each actual effect once per
replay, rather than rerunning a hidden Query subprogram.

Numerical derivatives along a recorded execution remain ordinary autograd.
Hard Top-K and other discrete choices still require an explicitly chosen
training estimator; changing the representation does not make them continuous.

The prepared device executor consumes the same emitted responses and keeps
CALL-response origin separate from original Bank lineage. The private captured
search supports complete initial invocations, including scalar response pools
and half-precision data with float32 routing scores. A resumed partial invocation
uses native execution; it does not silently reconstruct missing execution state.
Mixed float64 response pools with lower-precision entry data also select native
execution before capture, preserving per-response dtype promotion.

## Scope and compatibility

This version supports a declared finite candidate/call organization with
state-dependent numerical responses. It is not a claim of arbitrary runtime
opcode generation or mutable cyclic call graphs. Existing local iteration and
call budgets remain distinct.

`FormulaProgramQueryV4` and `V5` keep their external-policy semantics and saved
assets. A new composition can reuse existing ordinary Banks, Formula programs
and matching parameters. Moving an old scorer's weights into a Bank alone
does not implement execution-derived federation querying; changing the graph
must not be called an identical-checkpoint continuation.
