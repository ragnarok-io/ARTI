# Formula Fabric

For selection expressed by ordinary Bank/Fabric execution rather than a
separate Query network, use the versioned `FormulaProgramQueryV6` composition.

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
- `ObserveIdentity@1`, `ObserveAffine@1`, and `ObserveFourier@1` execute the
  existing Observation operators over an explicit trajectory. See
  [Adaptive Observation](adaptive-observation.md).

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

### Explicit Numeric Precision

The new alpha atoms retain the existing FormulaProgram@2 schema and execute in
the same checked plan:

- `ReduceAtomV2` / `reduce_tensor()` produce `arti/formula-atom-reduce@2` for
  native `sum`, `mean`, `amax` and stable `logsumexp` over a named axis.
  `reduce_sum()` and `ReduceAtom` still produce the original ordered sum @1.
- `ScalarMapAtomV2` / `scalar_map_v2()` produce `arti/formula-atom-scalar-map@2`.
  In addition to the original six maps, they support `abs`, `exp`, `expm1`,
  `log`, `log1p`, `softplus`, `reciprocal`, `sin` and `cos`.
- `CastAtom` / `cast()` produce `arti/formula-atom-cast@1` for explicit floating
  conversion. Axes and domain are unchanged; output dtype is part of the type.

The new scalar/reduction atoms default to float32 computation, preserve float64
inputs, and return the input storage dtype. `accumulation_dtype="activation"`
is an explicit alternative. Native reduction does not promise the bitwise
addition order of Reduce@1. `amax` splits gradients between tied maxima as
PyTorch does; `abs` uses its zero subgradient at zero. Softplus uses the stable
PyTorch operation with beta 1 and threshold 20; trainable slopes are separate
Bank operands rather than Python attributes.

Per-atom float32 computation does **not** keep a whole composition in float32:
each instruction normally casts back. Use explicit Cast instructions around
an entire normalization subgraph when intermediate squares/statistics must
remain in float32. A float64 program should retain float64 computation instead
of silently downcasting. The output cast remains explicit as well.

```python
from arti import mechanisms as m

x = m.InputBinding("x", m.TensorType(("B", "D"), ("B", 64), dtype="float16"))
beta = m.BankBinding(
    "beta", "my-package/nonlinear-bank@1", "beta",
    m.TensorType(("D",), (64,), dtype="float16"),
)
work = m.cast(x, dtype="float32")
gate = m.scalar_map_v2(m.scale(work, m.cast(beta, dtype="float32")), mode="sigmoid")
y = m.cast(m.scale(work, gate), dtype="float16")
program = m.FormulaProgram.build(outputs=(y,))
```

`examples/formula_bank_nonlinearity.py` composes PReLU, parameterized Swish,
a smooth saturating nonlinearity, LayerNorm and RMSNorm from ordinary atoms.
Its slope, offset, radius, scale and bias are independent Bank bindings. These
are expression-building examples, not additional opaque neural layers or a
claim that the current search grammar has already learned these structures.

The same example includes independent Bank operands for temperature, visible
logit bias and an optional sink logit. Bias is added only to visible logits;
the biased logits and the sink share the positive temperature. A false
`sink_mask` disables sink participation. The sink joins the normalization
axis before softmax; after removing its
output, the remaining mass is not renormalized. This can express a learned
non-contributing alternative without changing masked-softmax semantics.

Elementary functions do not silently repair their inputs: log requires positive
values, log1p values greater than -1, reciprocal nonzero values, and rsqrt
positive values. Finite forward results do not guarantee bounded gradients
near a pole. Positive temperatures and epsilon floors must be explicit
operands/parameterizations and representable in the chosen dtype. Select is
not lazy evaluation; masking an invalid computed branch later is not a way to
make its gradient valid. Stable library Expm1, Log1p, Softplus and LogSumExp
implementations are used rather than unstable algebraic expansions.

Working-byte admission includes visible computation/storage tensors and the
native LogSumExp temporary. It is not a bound on allocator workspace or all
tensors retained for autograd.

### Index, Mask and Segment Programs

The following alpha atoms use the same typed SSA executor and registry:

| Helper | Reference | Result |
| --- | --- | --- |
| `axis_index(reference, axis=...)` | `arti/formula-atom-axis-index@1` | int64 coordinates from an actual axis length |
| `compare(left, right, mode=...)` | `arti/formula-atom-compare@1` | boolean eq/ne/lt/le/gt/ge |
| `boolean_binary(..., mode=...)` | `arti/formula-atom-boolean-binary@1` | boolean and/or/xor |
| `boolean_not(value)` | `arti/formula-atom-boolean-not@1` | boolean complement |
| `gather_v2(...)` | `arti/formula-atom-gather@2` | indexed float/bool/int64 payload |
| `scatter_add(...)` | `arti/formula-atom-scatter@2` | base plus every indexed update |
| `segment(...)` | `arti/formula-atom-segment@1` | grouped sum/mean/amax/softmax |

AxisIndex reads shape/device, not reference values. Compare requires identical
named shapes, domains and dtypes; broadcast explicitly before comparing.
Boolean inputs support eq/ne only. These discrete outputs have no surrogate
gradient. Select still differentiates its selected value branches, not the
threshold used to construct a comparison mask.

Gather@2 preserves integer payload exactly. The existing `gather()` remains @1.
Scatter@2 permits repeated targets and adds all updates without mutating base;
`scatter()` remains unique replacement @1. Neither operation implicitly edits
persistent Bank state. The existing effect mechanism controls actual writes.

```python
from arti import mechanisms as m

x = m.InputBinding("x", m.TensorType(("B", "N", "D"), ("B", "N", 64)))
ids = m.InputBinding("ids", m.TensorType(("B", "N"), ("B", "N"), dtype="int64"))
mask = m.InputBinding("mask", m.TensorType(("B", "N"), ("B", "N"), dtype="boolean"))
pooled = m.segment(x, ids, mask, axis="N", segment_axis="G", num_segments=8, mode="mean")
program = m.FormulaProgram.build(outputs=(m.axis_index(x, axis="N"), pooled))
```

Segment IDs include the source axis and may include preserved axes. Masks are
boolean and broadcast by named axes, including per-feature masks. Active IDs
must be in `[0, num_segments)`; masked IDs are ignored. Group count is a static
positive bound, not inferred from data. Sum/mean/amax replace the source axis
with the declared segment axis; softmax returns the original source shape and
normalizes independently within each valid group. Empty groups and masked
softmax entries are zero. Physical tensor axes remain nonempty as elsewhere
in Fabric. Mean counts only valid entries; amax uses actual group maxima even
when all values are negative, and ties share their gradient.

Default accumulation is float32, preserving float64 inputs and returning value
storage dtype. IDs and counts stay int64. Grouped softmax keeps max-shift, exp,
sum and division inside one computation dtype. Native parallel indexed sums do
not promise bitwise equality with sequential summation or across devices.

`examples/formula_segment_attention.py` composes Contract, grouped softmax,
Scale and grouped sum with an independently trainable Bank scoring operand.
It is a typed expression example, not an opaque attention layer or a claim
that a network has already discovered this architecture.

Checked plans report invalid active IDs through their device validity result;
plain eager/prepared calls reject them. Internal safe indices prevent an
out-of-bounds access, but do not make an invalid row eligible for use.

### Local Windows and Bank Kernels

`WindowAtom` / `window()` (`arti/formula-atom-window@1`, alpha) expose regular
neighborhoods as ordinary typed tensors. A named axis `N` is replaced in place
by position and tap axes `(P, K)`; for example `[B,N,C] -> [B,P,K,C]`.

```python
patch = window(
    x, axis="N", output_axis="P", window_axis="K",
    kernel_size=3, stride=2, dilation=2, padding=(2, 1), output_size="P",
)
depthwise = contract(patch, kernel_bank, reduce_axes=(("K", "K"),))
```

Here a kernel Bank of shape `[K,C]` gives independent per-channel filters.
For `[K,C,O]` kernels, Contract over `K` followed by `reduce_tensor(...,
axis="C", mode="sum")` gives a shared convolution. `[P,K,C,O]` provides
position-specific local weights. These use the same ordinary Bank bindings;
window positions do not create copies of trainable parameters. Activation
and gating remain separate composable atoms. See
[`formula_local_computation.py`](../examples/formula_local_computation.py).

With left/right padding `L,R`, stride `S`, dilation `D` and kernel size `K`:
`P = floor((N + L + R - (D*(K-1)+1))/S) + 1`. Position `p`, tap `k` reads
`p*S - L + k*D`, or zero outside the input. Taps are not reversed. All sizes
and strides are positive integers, padding is nonnegative, and `P` must be
positive. Static lengths infer `P`; dynamic lengths require an explicit
`output_size` integer or existing TensorSchema symbol. Derived sizes bind
before allocation and cannot override another operand's symbol binding.
Different concrete shapes use the existing preparation/compilation buckets.

Single-axis windows compose across multiple spatial axes. Irregular discrete
neighborhoods use explicit indices with Gather@2. Window may overlap, insert
zeros or omit input values; it is not reversible Fold/UnFold.

The reference lowering uses native padding, `Tensor.unfold`, slicing and axis
movement. Unpadded windows can be overlapping views and must be treated as
read-only; ordinary Fabric operations are out of place. Overlap gradients sum
back to each original position. Logical window size and retained padding
storage are included in admission budgets. This is not a total-autograd-memory
bound: dilated-window backward may allocate over the larger effective window,
and Contract may materialize noncontiguous inputs. Native composition alone
does not establish fused-convolution or GPU speed parity.

### Shared-Body Sequence Scan

`FormulaScan` (`arti/formula-scan@1`, alpha) composes a pure FormulaProgram
over a named sequence axis. Its carry is explicit data, not a hidden Bank.
The body runs once per sequence element and simultaneously returns every
next-carry port and emission head. Bank operands are shared across all steps.

```python
scan = FormulaScan(
    body,
    axis="T",
    sequence_types={"x": TensorType(("B", "T", "D"), ("B", "T", 64))},
    carry_outputs={"h": body.outputs[0]},
    emissions={"hidden": body.outputs[0]},
    max_length=64,
)
result = scan(inputs={"x": sequence, "h": initial}, banks=operands,
              limits=limits)
final = result["carry.h"]
trajectory = result["emit.hidden"]  # [T, B, D]
```

Removing `T` from a sequence type must give the exact corresponding body input
type. Each carry input/output pair must match in shape, dtype and domain.
Emissions may have different types; each gains a leading sequence axis.
All sequence inputs have the same positive length. Non-sequence body inputs
are reused unchanged. The body cannot contain network or topology effects;
state changes are explicit return values. Sequence advancement is distinct
from Refine over the same observation.

`scan.prepare(...)` specializes the actual sequence length and returns the
existing execution plan and prepared bindings. `scan.lower(length, limits=...)`
returns an ordinary FormulaProgram suitable for existing candidates and typed
pools. Slice/Reshape select inputs, renamed SSA instructions connect the carry,
and a balanced Concat tree gathers emissions. Only instruction occurrences are
duplicated: there is one binding for each shared Bank parameter, with gradients
accumulating through the whole recurrence. Inputs and Bank values are not
modified. A new length requires preparation of the corresponding program.

`max_length` does not override FormulaLimits. Expanded instructions, dependency
depth, intermediate values and emission storage remain subject to the ordinary
limits; larger limits must be explicit. This is finite ordered IR lowering,
not a native parallel Scan kernel or constant-size execution graph. Stateful
nonlinear transitions cannot generally use associative reordering. Eager
second derivatives and existing grouped first-derivative paths are distinct
contracts. See [`formula_recurrence.py`](../examples/formula_recurrence.py)
for a gated, two-carry example and ordinary Bank parameters.

### Transformer Programs

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

A parameterized gated MLP can remain fully visible in that program:
`up = Contract(x, W_up)`, `gate = Contract(x, W_gate)`,
`hidden = up * gate * sigmoid(beta * gate)`, followed by an output Contract.
The matrices and feature-wise beta are ordinary Bank operands, not a hidden
activation module.

Lookup@1 now participates in the shared checked/grouped numerical executor.
Ordinary calls still reject out-of-range indices. Checked execution returns
a device validity flag and safe placeholder values for invalid rows; those
rows must be rejected, not consumed as successful lookups. Candidate batching
and typed numerical dispatch enforce that distinction. CPU tests cover full
gated-program values, input/Bank gradients, grouped eager/aot_eager execution,
and heterogeneous int64/bool/float pools. This does not claim GPU kernel fusion
or a speed advantage over optimized attention implementations.

## Bounded Program Query

### Typed Connection Choices

`expand_candidate_bindings()` prepares finite input/output alternatives for a
`FormulaProgramTensorCandidateV4` template. It returns ordinary candidates;
there is no additional grammar interpreter in the execution path.

```python
nodes = expand_candidate_bindings(
    template,
    slot_types=declared_types,
    input_choices={"value": ("original", "earlier", "other_branch")},
    output_choices={template.candidate.program.outputs[0]: ("next",)},
    prefix="activation",
    max_candidates=64,
)
```

Every named input and output must be included. Multi-output templates must
bind their continuation-response outputs as well as their data outputs. The
generator filters by exact TensorType, including symbolic sizes, axes, dtype
and domain. It neither inserts casts nor invents a separate shape unifier.
The ordinary executor still checks concrete tensors at binding time.

Source choices may reuse the same earlier value at multiple input ports.
Outputs must be distinct empty SSA slots and cannot overwrite their inputs.
Template `requires_empty_slots` guards are inherited unless the caller
explicitly supplies a replacement sequence. The limit bounds the type-filtered
Cartesian product before further SSA exclusions; it rejects overflow rather
than silently truncating the candidate catalog.

Each occurrence uses `with_bindings()` to share the same Fabric, operand store
and optional plastic Bank owner. Repeated initialization from the same input
Tensor is not equivalent: the ordinary candidate constructor copies its
initial operands. Sharing across different program templates is not added by
this helper. Unique Parameter ownership is preserved when those occurrences
are mounted in the existing Query. Save/load reconstructs that same catalog;
the helper itself is not a new artifact schema.

Choice order is canonicalized, so reordering the same sets does not change IDs.
Changing those sets can renumber IDs and requires rebuilding the Query,
continuation bindings and prepared dispatch tables. Response-to-successor
connections remain explicit V6/V7 bindings to actual executed outputs; the
generator does not install a separate routing network.

See [`formula_binding_search.py`](../examples/formula_binding_search.py)
for Scale, SiLU/Tanh, multi-parent Add and cross-depth Reduce alternatives.
This example demonstrates an executable candidate space and cooperative replay,
not trained architecture discovery. Native/grouped parameter sharing also does
not imply zero-copy device execution. Prepared numerical groups reuse static
operand rows for a shared store/binding when this saves storage. Fully shared
rows use a broadcast view; partial sharing uses a device index only when its
storage cost is smaller than the avoided copies. Equal but independent Banks
remain distinct. Lane work tensors can still be materialized during execution.

Refresh prepared operands at the search boundary after optimizer/load updates;
refresh preserves table addresses. Changing operand shape/dtype/device or store
sharing requires preparation again, and moving a module invalidates an existing
captured graph. Mutable predecessor Bank values still come from the branch's
live pool, never this static snapshot. Training replay keeps the original
shared Parameters, not detached table copies.

The prepared cooperative search wave also has a fixed `forward_steps()`
composition. It returns every round's original record, supports the existing
independent-event batch path, and performs no host scalar check between rounds.
Reaching this horizon does not mark a still-live branch complete; STOP and
host-imposed limits remain distinct. The caller owns pool capacity and must
consume/decode the returned records before reusing captured storage.

For whole-segment compilation, trace the fixed invocation to ATen once during
preparation, then compile that graph. The current PyTorch version can fail on
direct Dynamo tracing of the Python result wrappers; ATen tracing preserves the
same numerical and scheduling path. This does not compile the host decoder or
dependency replay. A graph is reusable only for its prepared structure,
steps/batch/beam layout and tensor metadata. Keep cold preparation outside the
event loop; a longer horizon also increases graph and record storage.

#### Prepared Numerical Group Compilation

The low-level alpha entry points are
`arti._formula_device_dispatch.FormulaDeviceNumericalDispatch` and
`arti._formula_grouped_training.grouped_formula_training`. The latter context
wraps `query.execute_many(...)` to override automatic selection for diagnostics;
see the [grouped training example](../examples/grouped_formula_training.py).

The internal typed dispatch can prepare ordinary numerical groups with
`prepare_compiled_groups_(sample_run, group_ids=..., backend="inductor")`.
The callback must execute the intended search using caller-owned scratch
frames and pools, covering every required group signature. The implementation
traces the existing checked group to ATen and warms each compiled variant
before an outer CUDA Graph is captured. It does not implement another Formula
executor or change candidate selection, finite checks, or effect ownership.

On CUDA with PyTorch 2.11 or newer, prepared typed dispatch automatically compiles
on the first uncaptured call: whole dispatch for ordinary graphs, ordinary-group
islands for mixed effect graphs. CPU and older PyTorch keep native execution.
Set `execution_backend="native"` when constructing a dispatch for an explicit
uncompiled comparison. Compilation is restricted to ordinary, no-gradient search
groups. Effects retain their existing execution path; gradients continue through
live Formula replay. Explicit Bank pools and handles remain runtime inputs.
Prepared operand buffers retain their addresses and must be refreshed after
optimizer or load updates, just as for uncompiled dispatch.

Reuse requires the same structure, shape, stride, dtype, device, gradient flags
and autocast/inference context. Module conversion invalidates prepared groups;
prepare again and rebuild any outer CUDA Graph before executing it. An already
captured graph does not re-enter Python and cannot enforce this lifecycle check
itself. Do not move or replace its dependencies while retaining that graph.
Cold tracing/compilation is separate from hot search; group compilation can
reduce device kernels whereas CUDA Graph capture reduces host submission work.
Neither optimization alone is evidence of end-to-end training acceleration.

For a dispatch consisting entirely of ordinary typed groups,
`prepare_compiled_dispatch_(sample_run, backend="inductor")` uses the same
preparation contract but compiles the original numerical dispatch as one graph.
It includes output packing and validity aggregation across groups. It does not
compile the surrounding candidate selector or pool writeback. This option and
per-group compilation replace one another; do not stack them. Dispatches with
effect instructions continue to use the original path or ordinary-group islands.

Grouped differentiable execution keeps a separate capture boundary. Supported
CUDA groups use Inductor by default, without a context manager.
`grouped_formula_training(backend="native")` selects the original independent
autograd path for debugging; explicit `eager` still uses grouped differentiation.
Default plans use a bounded cache, and Bank tensors remain live inputs. Its
`inductor` backend compiles forward and PyTorch-computed VJP without automatic
internal CUDA Graphs. The `captured` backend remains the independent native
numeric-capture option. For a fixed training fragment, the caller can instead
capture forward, task loss and VJP together around compile-only execution.
Prepare all variants first and retain the same input/Bank storage, gradient
participation, loss and stream throughout. Different dynamic paths or output
usage patterns require another prepared fragment. Create the training leaves
and perform their first forward/backward on the capture stream as well;
switching an existing gradient graph from the legacy stream can invalidate
capture. Do not replay stale gradient patterns. Captured output and gradient
buffers are reused and must be consumed
or cloned before replay. This is not an automatically captured optimizer step.
Warm the actual forward and backward signatures outside capture first. A cold
signature encountered during capture raises a preparation error instead of
silently claiming acceleration. Compiler failures remain visible.

### Single-Instruction Search

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

With gradients enabled, supported CUDA requests on PyTorch 2.11 or newer use
grouped Inductor forward and differentiation by default. Outputs retain their
individual autograd boundaries, including absent gradients for unused inputs
and parameters; materializing zeros there could change optimizer updates.
CPU, older PyTorch, unsupported requests, and the explicit `native` override
retain separate numerical autograd graphs, aggregating only Boolean validity
flags. `serial=True` selects that native reference path for all requests.
Grouped differentiation does not capture the caller's whole search or optimizer.
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
