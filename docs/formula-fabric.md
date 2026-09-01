# Formula Fabric

Formula Fabric is a stable, fixed-capacity tensor executor. A program declares
a bounded set of Formula cells, while a route source selects their operands.
The executor remains the only implementation of Formula mathematics.

## Typed Atom Basis

`FormulaFabricV2` is a stable typed SSA executor. It is a separate component
version from `FormulaFabric@1`; the original fixed-arena runtime remains
unchanged. A v2 program declares named tensor axes, domains, dtypes, bounded
slots, explicit Input and Bank bindings, and a bounded public atom basis:

- `Contract@1` contracts exactly the named axis pairs.
- `Scale@1` applies a scalar or named-axis factor.
- `Add@1` combines two tensors with the same typed shape.
- `Reduce@1` performs an ordered deterministic sum over one named axis.
- `Reshape@1` changes named shape without changing element count.
- `Permute@1` reorders named axes.
- `Gather@1` selects an explicitly indexed workset.
- `Scatter@1` restores or replaces an explicitly indexed workset.
- `ScalarMap@1` applies a declared elementwise function such as GELU or SiLU.
- `Broadcast@1` explicitly expands named singleton or new axes.
- `Select@1` chooses values with a broadcastable boolean tensor.
- `Lookup@1` reads a typed table with arbitrary-rank integer indices.
- `Slice@1` takes a bounded positive-step slice from one named axis.
- `Concat@1` joins statically sized segments along one named axis.
- `MaskedSoftmax@1` performs stable visible-only normalization and returns
  zeros for a fully masked row.

These atoms are sufficient to express the data movement, nonlinearities,
attention normalization, and table access used by a small Transformer. ARTI
does not provide an opaque Transformer, Attention, MLP, or RoPE Formula atom;
those structures remain ordinary typed SSA programs whose operands can come
from Banks.

`index_fold()` and `index_unfold()` are convenience macros over `Gather@1` and
`Scatter@1`. They do not have Formula atom identities of their own. In
particular, they are not `arti/fold@2` and `arti/unfold@2`: those references
belong exclusively to the reversible-topology components, which preserve the
folded payload and carry a `FoldRecord` for exact inversion.

For example, a rank-r LoRA-shaped operation is expanded into ordinary atoms:

```python
import torch

from arti import mechanisms

program = mechanisms.build_lora_program(
    input_dim=64,
    output_dim=64,
    rank=8,
    source_ref="my-package/adapter-bank@1",
)
fabric = mechanisms.FormulaFabricV2(program)

x = torch.randn(2, 16, 64)
base = torch.randn(2, 16, 64)
values = {
    "lora.A": torch.randn(8, 64),
    "lora.B": torch.randn(64, 8),
}
banks = {
    binding.name: binding.bind(values[binding.name])
    for binding in program.bindings
    if isinstance(binding, mechanisms.BankBinding)
}
result = fabric(
    inputs={"x": x, "base": base, "lora.gain": torch.tensor(0.5)},
    banks=banks,
)
y = result.values[0]
```

The program contains no opaque LoRA or matrix-multiply primitive. `A`, `B`, and
`gain` remain explicit operands, so a Bank may learn and route them as one
bundle. `build_routed_lora_program()` exposes the candidate-axis route as an
ordinary typed input; this keeps Formula execution generic and also permits an
explicit caller-selected mixture. `FormulaOperandBank.route()` is the public
hard one-candidate producer. It uses stable member identities to break exact
ties and can use a straight-through softmax surrogate during optimization
without changing the hard forward value.

The route query is a runtime tensor, not state owned by `FormulaOperandBank`.
Gradients may flow through that tensor to the current hidden state, while the
Bank state dict contains only its keys and Formula operands. In ARTI Refine
compositions, the Query producer remains fixed and must not be added to the
optimizer; a new hidden state still produces a fresh query value at every step.

The numerical policy is also part of the immutable program. Contract and
pointwise accumulation can independently use `"float32"` or `"activation"`.
Contract defaults to float32 because it contains the long reductions;
pointwise Scale/Add default to activation dtype. Using activation accumulation
for Contract is an explicit opt-in for hardware-native low-precision execution.
Admission requires every operand of an instruction to use the same concrete
runtime dtype, including when its declared type is the generic `"floating"`
category. `FormulaLimits` bounds both individual tensors and a conservative
aggregate live working set; these limits are serialized into the program and
therefore participate in its fingerprint. The aggregate includes bound values,
accumulation-dtype operands, the accumulation result, its activation-dtype
cast, and ordered-reduction accumulator overlap. Backend-owned kernel scratch
space remains an execution-provider concern rather than part of this tensor
ABI limit.

Host-side validation remains mandatory before compiled execution:

```python
bindings = fabric.bind_tensors(inputs=inputs, banks=banks)
plan = fabric.execution_plan()
compiled = torch.compile(plan, fullgraph=True)
outputs = compiled(bindings)
```

`bind_tensors()` returns `PreparedFormulaBindings`, a host-admission receipt for
one exact program and binding order. `FormulaExecutionPlanV2` rejects raw tensor
tuples and receipts from another program. This is an accidental-misuse guard,
not caller authentication: identity metadata is a local calling contract. The
compiled graph does not independently rediscover Bank provenance from tensor
bytes. Trace data is diagnostic program provenance; it does not attest the
bytes of externally supplied Bank tensors.

The v2 arithmetic atoms operate on dense floating tensors with positive axis
extents. Gather and Scatter may also transport explicit boolean masks. Masks
and ragged layouts are not implicit executor behavior; a caller must represent
them as typed operands in a compatible program or keep that policy outside
Formula execution. Compiled plans consume bindings that have already passed
host-side shape, dtype, device, identity, and working-set admission.

See `examples/formula_v2_typed_lora.py` for a complete hard-routed Bank example.

## Transformer Composition

Formula Fabric can express a causal Transformer block without introducing an
opaque Transformer atom. Token and position lookup, normalization, Q/K/V
projections, masked attention, residual connections, an MLP, and an output head
can be expanded into ordinary typed SSA instructions. The required operations
come from public atom identities such as:

```text
Lookup, Contract, Reduce, Broadcast, ScalarMap,
MaskedSoftmax, Reshape, Scale, Add
```

The resulting structure remains a composition of public Formula atoms and Bank
values rather than hidden implementation code. Expressibility does not imply a
runtime advantage over optimized Transformer kernels or unrestricted neural
architecture search.

## Bounded Program Query

`FormulaProgramQuery` is an alpha composition for learning a small typed SSA
program from final task loss. Each `FormulaProgramCandidate` must contain
exactly one Formula instruction and explicitly declares its existing input
slots and one previously empty output slot. At each step, candidates whose
inputs are absent, whose output has already been written, or whose Formula
types do not admit the current tensors are removed before normalization.
`requires_empty_slots` may additionally close a branch after another SSA slot
has been produced. It is a serialized program-grammar constraint shared by
training and hard execution; it does not identify a preferred candidate.

```python
query = mechanisms.FormulaProgramQuery(
    slot_ids=("x", "hidden", "activated", "output"),
    candidates=(project, gelu, output),
    terminal_slot="output",
    min_steps=3,
    max_steps=3,
)

loss = mechanisms.ExactFormulaProgramQueryTraining().loss(
    query,
    initial={"x": x},
    target=target,
    task_loss=final_task_loss,
)
```

The exact trainer enumerates only the bounded shape-valid path graph. It uses
the final task loss and supplies no atom, wiring, route, transition, or stop
teacher. Equivalent execution orders are merged by the immutable SSA producer
map, so independent operations such as Q/K/V projection are evaluated once per
unique tensor state rather than once per permutation. This remains an exact
expected-policy loss, not a sampled or mixed-output estimator. Hard execution
currently accepts one sample because different samples
may choose heterogeneous atoms and tensor shapes. It selects one candidate,
never a weighted merge of candidate outputs.

The component is intentionally bounded: the host supplies the candidate catalog,
slot schema, step limits, and final task loss. It does not generate Python code,
invent new atoms, or claim unrestricted architecture search. Training cost can
grow with the number of reachable SSA states, so catalogs should stay small or
use an application-owned search strategy.

## Objective-Controlled Commits

`ObjectiveFormulaFabricCompute` is a stable composition adapter that lets an
`ObjectiveExposureBank` provide bounded commit strength to an existing
`FormulaCommitBlend` executor. It does not choose Formula routes, primitives,
fire masks, commit authority, topology, or execution depth.

```python
from arti import mechanisms

compute = mechanisms.FormulaFabricCompute(
    mechanisms.FormulaCommitBlend(mechanisms.FormulaFabric(program)),
    active_count=8,
)
controlled = mechanisms.ObjectiveFormulaFabricCompute(
    compute,
    mechanisms.ObjectiveExposureBank(slots=16, query_dim=64),
)
pulse = mechanisms.AdaptivePulse(
    fold=fold,
    selective_compute=controlled,
    unfold=unfold,
)

result = pulse.run_tensor(
    x,
    objective_query=current_or_past_query,
    formula_route=route,
)
```

The Objective query is an explicit current/past-only input. Future targets and
losses remain outside the forward graph. Omitting this adapter leaves existing
external-factor and hard-commit Formula paths unchanged.

The query must use the same device and dtype as the folded workspace. Before
Objective routing runs, the adapter admits the complete Objective, diagnostic,
route-source, and Formula byte-work bound against the existing Formula limits.
Supplying `objective_query` to a Pulse without this controlled stage fails
closed instead of being ignored.

## State-Conditioned Re-Routing

`IterativeRoutedFormulaFabricCompute` repeats an existing routed executor. Each
completed Formula program produces the workspace used by the next route query:

```python
from arti import mechanisms

routed = mechanisms.RoutedFormulaFabricCompute(compute, route_source)
refined = mechanisms.IterativeRoutedFormulaFabricCompute(routed, steps=4)

next_workspace, info = refined(workspace, return_info=True)
```

This is different from executing one frozen route four times. The route source
is invoked after every complete program execution, so later operand selections
may change as the workspace changes. Scratch arena values do not persist across
program invocations. `steps` is fixed and bounded in this first version. The
adapter does not implement Formula operations itself and does not accept one
externally frozen `formula_route`.

The canonical component identity is
`arti/iterative-routed-formula-fabric-compute@1`. Diagnostics preserve one
ordered `RoutedFormulaFabricComputeInfo` record per executed iteration.

## Compiled Topology Sources

Caller-owned topology sources may remain outside a Pulse artifact. Before
constructing Pulse and calling `torch.compile(..., fullgraph=True)`, bind their
static contract to Fold:

```python
fold.bind_source_contract(topology_source)
prepared = fold.prepare_source_inputs(payload, (keys, query))
```

Binding records the source reference, input count, instance axes, contract
fingerprint, and producer provenance. It does not copy, register, or save the
source module. `prepare_source_inputs` is a host-side admission step that rejects
the payload and all of its storage aliases. Eager execution checks a supplied
source against the binding; compiled execution accepts only the opaque prepared
inputs and consumes no Python JSON work in the tensor graph. Reconstruct and
bind the caller-owned source before loading an artifact that contains a bound
Fold.
