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

## In-Path Neural Adaptation

`FormulaEffectProgramV3` allows one ordinary Formula data path to contain an
ordered chain of NeuralPlasticity effects. Every effect is an intermediate SSA
instruction: an ordinary tensor instruction must precede it, another ordinary
tensor instruction must consume it, and the public output cannot be the effect
itself.

Each effect returns its local tensor operand unchanged. In a predecessor-owned
program, its target is the actual Bank slot consumed by the arriving ordinary
Formula producer, resolved from execution lineage. The effect owns no separate
memory and accepts no caller-selected target. Multiple effects apply their
transitions in program order without incrementing Bank-local Refine depth.

Visibility is versioned. `FormulaProgramQueryV3` and
`TensorViewFormulaProgram` keep proposals write-only until the selected stopped
execution is committed. `FormulaProgramQueryV4` permits a later execution of
the same producer owner to use the branch-local successor immediately. Neither
version exposes effect state as an extra tensor input to the router.

For the commit-visible version:

```text
ordinary tensor Formula
-> identity-data self effect
-> ordinary tensor Formula
-> identity-data self effect
-> winner Bank-slot commit
-> later invocation re-executes the adapted ordinary Formula
```

The downstream task loss trains the complete path. There is no state target,
update teacher, or requirement to repeat one route. Refine remains responsible
for repeated Bank queries, not for choosing the in-path effect topology. The
number and placement of in-path effects are a separate architecture dimension.

The low-level ordered-effect executor identities are
`arti/formula-effect-program@3` and `arti/formula-fabric@5`. Federation search
uses `arti/bank-local-formula-effect-action@1` inside
`arti/tensor-view-formula-program@2`; the effect owns no state and accepts no
caller-selected target.

`FormulaEffectProgramV3` is only the executor contract for an already selected
path. It is not evidence that the path topology was discovered. Use
`FormulaProgramQueryV3` when NeuralPlasticity nodes themselves belong to the
search space. A normal Formula producer explicitly declares the Bank binding
that it owns and consumes; an effect resolves that binding from runtime producer
lineage rather than accepting a state or target identifier:

```python
producer = mechanisms.FormulaProgramTensorCandidateV2(
    ordinary_formula_candidate,
    plastic_bank_slot="memory",
)

query = mechanisms.FormulaProgramQueryV3(
    slot_ids=("x", "hidden", "adapted", "output"),
    candidates=(
        producer,
        neural_plasticity_candidate,
    ),
    terminal_slot="adapted",
    max_steps=6,
)

loss = mechanisms.ExactFormulaProgramQueryTrainingV3(
    exploration_probability=0.25,
).loss(
    query,
    event1={"x": support},
    event2={"x": query_value},
    event2_candidate_id=producer.candidate_id,
    target=target,
    task_loss=task_loss,
)
```

The candidate catalog declares local typed edges, not a prebuilt complete
effect chain. ProgramQuery chooses the number, order, placement, and wiring of
ordinary and self-effect nodes from the later event's final task loss. Query
summaries have no direct Bank-state input. The effect changes the Bank slot used
by its actual ordinary predecessor, but preserves the current data tensor. The
change becomes observable only after the winning state is committed and that
ordinary predecessor is executed again. Program steps and Bank-local Refine
steps remain separate accounting dimensions.

`FormulaProgramQueryV3` returns the successor Bank state without mutating
parameters in place. `commit_()` accepts only a stopped winner produced by the
same Query. Saved state contains predecessor-owned Formula Bank slots, not an
effect-owned state arena.

### Branch-Visible Producer Re-Execution

`FormulaProgramQueryV4` separates a producer's persistent `bank_owner_id` from
its individual SSA occurrences. Wrap ordinary candidates with
`FormulaProgramTensorCandidateV3(..., plastic_bank_slot="weight",
bank_owner_id="producer")` and effects with `FormulaProgramEffectCandidateV3`.
Repeated occurrences with the same compatible owner share one registered
`FormulaProgramBankOwnerV1` buffer, not multiple trainable copies.

```text
ordinary producer using Bank revision r
-> identity-data effect proposes revision r+1
-> same producer re-executes using revision r+1
-> ordinary output changes
-> query observes that output
-> stopped winner may be committed
```

Pending successors are branch-local. They do not modify the registered Bank
until `query.commit_(execution)`; losing branches do not write persistent
state. An effect does not retroactively change an already computed tensor.
The Bank's influence is expressed by executing its ordinary Formula again.
`state_dict()` retains the shared Bank values and revisions for fresh reload.
Persistent commit detaches the installed buffers. Differentiable cross-event
training passes the returned functional Bank state instead of committing it.
These forward-written buffers cannot also be optimizer-owned parameters.

Six effect families are available: additive/multiplicative, blend, outer,
transport, polynomial, and proximal. An effect candidate may carry a scalar
`execution_count` and an explicit `max_executions` bound. Forward execution
rounds the clamped count to an integer. A trainable generic count evaluates
the selected hard path with gradients, then probes detached successor states
for count credit. The probe stops before including a non-finite successor;
only the finite prefix participates in the straight-through count surrogate.
An invalid selected hard transition is not clipped or repaired. Training may
still evaluate through `max_executions`, so the selected count is not the
physical training cost. Generic counts repeat the transition over
the latest successor, not by multiplying one precomputed delta. `Outer@2`
instead carries its count directly among its Formula operands and computes
`state + count * update`, which is exactly repeated addition of a fixed update
without an actual repetition loop. Do not wrap it in a second count mechanism.
Neither straight-through rule is the true derivative of integer execution.
One proposal advances the logical Bank revision once, regardless of its count.

`FormulaProgramQueryTensorEncoderV1` is an optional content-sensitive input
encoder for routing. It sees ordinary SSA tensors, not a direct Bank-state
read. Frozen routing parameters can therefore still produce different routes
after a producer re-executes with an updated Bank.

The previous alpha `FormulaProgramQueryV2` effect-owned state arena has been
removed. Rebuild its programs with explicit ordinary producer Bank ownership;
its state artifacts are not silently reinterpreted. The alpha effect metadata
now identifies `runtime-predecessor-bank` binding and `effect-operands-only`
access; rebuild older effect contracts rather than treating their former
execution-site labels as current ownership. `FormulaProgramQuery@1`
ordinary program search remains available. The new identities are
`arti/formula-program-query@3` and `arti/formula-program-query@4`.
Hard ProgramQuery execution currently supports one sample per persistent Bank
state. Batched summaries do not imply batched independent state commits or a
compiled whole-search executor.

See `examples/predecessor_bank_plasticity.py` for a complete deterministic
composition and state-dict round trip. Its fixed candidate graph illustrates
execution semantics, not learned architecture-search effectiveness.

### Grouped Candidate Execution

`FormulaProgramQueryV4.execute_many(requests, chunk_size=16)` is a low-level
execution entry point for an existing scheduler. Each request contains a
candidate and its existing branch arena; results preserve request order.
It does not rank candidates, search, select a winner, or commit persistent state.

Under `torch.no_grad()` or `torch.inference_mode()`, compatible pure-tensor
programs share the existing checked Formula execution plan. Chunking happens
before stacking operands, bounding additional input copies without reducing
the search width, Bank size, or execution depth. Unsupported programs and
custom execution hooks retain native execution. Intermediate non-finite values
remain errors, even if a later saturating operation would hide them.

With gradients enabled, each supported request uses a separate checked plan
execution and its own numerical autograd graph. Only Boolean validity flags
are aggregated. This preserves both gradient values and absent gradients for
parameters belonging to unused candidates; materializing zeros in those
positions could change optimizer updates. Unsupported requests remain native,
and `serial=True` explicitly selects the native reference path for all requests.
This is not vectorized training or whole-search compilation.
Immutable program fingerprints are cached for both training and inference;
runtime Tensor admission and live operand metadata are never cached.

See `examples/batched_formula_candidates.py` for independent branches,
predecessor effects, same-owner re-execution and a serial comparison.

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
