# Federal Path Compilation

The Federal compiler turns a selected Federal execution path into an ordinary
neural network.
Federal Bank remains the runtime that owns Query, path selection, local Refine,
and state provenance. Pulse and AdaptivePulse remain usable as operation graphs
inside a Bank path; they are not required to be the outer compiled artifact.

The first compiler is deliberately explicit:

```python
from arti.alpha import FederalPath, FederalPathCompiler

path = FederalPath(
    operations=(frozen_formula, frozen_unfold, frozen_head),
    source_ref="arti/federal-recall@2",
    source_snapshot_fingerprint=bank_snapshot_fingerprint,
    path_ids=("root", "root/formula", "root/terminal"),
    terminal_abi_ref="arti/terminal-output-abi@1",
    refine_steps=3,
)
compiled = FederalPathCompiler.compile(path)
y = compiled(x)
```

`FederalPath` is the evidence boundary. It must name the Federal source,
snapshot, selected path, terminal ABI, operations, and finite Refine depth.
Compilation deep-copies the operations into a `CompiledFederalNetwork`; the
compiled module does not retain FederalRecall, a Bank Query, a Pulse scheduler,
or a runtime callback. Parallel branches may be ordinary PyTorch modules, for
example `FederalParallel`, and remain ordinary branches in the result.

An existing traversal receipt can provide the selected path names without
serializing the live runtime:

```python
path = FederalPath.from_trace(
    federal_trace,
    operations=(frozen_formula, frozen_head),
    source_ref="arti/federal-recall@3",
    source_snapshot_fingerprint=bank_snapshot_fingerprint,
    terminal_abi_ref="arti/terminal-output-abi@1",
)
```

The caller still supplies the frozen operations and Bank snapshot identity;
the trace is evidence, not executable state.

The reference operation wrappers keep the Federal structure visible in the
compiled provenance while executing as ordinary modules:

```python
from arti.alpha import FederalRefine, FederalResidual, FederalTopologyBlock

fold, unfold = fixed_topology.operations()
path = FederalPath(
    operations=(
        FederalTopologyBlock(fold, active_block, unfold),
        FederalRefine(refinement_block, steps=3),
        FederalResidual(residual_branch),
    ),
    source_ref="arti/federal-recall@3",
    source_snapshot_fingerprint=snapshot_fingerprint,
    path_ids=("root", "root/fold", "root/refine", "root/residual"),
    terminal_abi_ref="arti/terminal-output-abi@1",
    input_shape=(None, 128, 64),
    output_shape=(None, 128, 64),
)
```

For a path whose recorded topology changes the visible sequence length, use the
explicit static transport specializations.  They are fixed gather/insert
operations, not learned Query implementations:

```python
from arti.alpha import FederalStaticFold, FederalStaticUnFold

path = FederalPath(
    operations=(
        FederalStaticFold(input_length=128, active_indices=(7, 19, 64, 91)),
        active_block,
        FederalStaticUnFold(
            input_length=4,
            source_indices=(0, -1, 1, 2, 3, -1),
        ),
    ),
    source_ref="arti/federal-recall@3",
    source_snapshot_fingerprint=snapshot_fingerprint,
    path_ids=("root", "root/fold", "root/block", "root/unfold"),
    terminal_abi_ref="arti/terminal-output-abi@1",
    input_shape=(None, 128, 64),
    output_shape=(None, 6, 64),
)
```

`FederalStaticFold` and `FederalStaticUnFold` are ordinary fixed transport
modules.  Their `arti/fold@2` and `arti/unfold@2` dependencies remain visible
in the compile manifest, while dynamic layout, learned queries, data-dependent
lengths, and mutable Bank behavior stay in the Federal runtime and are not
silently compiled.  The older public `arti.nn.Fold` and `arti.nn.UnFold`
implement learned/runtime-dependent behavior; a path containing either must
first provide an explicit fixed transport specialization.

`FederalTopologyBlock` executes the real `arti/fold@2` and
`arti/unfold@2` modules around the active operation. Its exact-static form
requires a fixed topology policy; a learned or input-dependent policy remains
uncompilable. The manifest records those nested Fold/UnFold references as
`dependency_refs`, so the compiled artifact does not erase the topology that
produced it. A finite `FederalRefine` is expanded as a bounded ordinary loop;
it is not an unbounded runtime scheduler.

The compiler distinguishes three cases:

- `exact-static`: the route and Bank snapshot are frozen, shapes are bounded,
  state is immutable, and finite Refine can be expanded.
- `gated-static`: a runtime gate remains, so the path can be inspected but the
  reference compiler does not pretend it is one fixed network.
- `uncompilable`: dynamic route, data-dependent shape, mutable state, an
  unregistered callback, or unbounded Refine remains in the path.

The latter two are explicit boundaries, not hidden fallbacks. A single winner
observed on one input is not enough to compile a dynamic Federation.

## Tensorized dynamic Query

`FederalPathCompiler` specializes an already selected path.  It does not solve
the separate problem of compiling an input-dependent Query.  For that case,
`FederalTensorQueryCompiler` lowers a bounded local Query graph into an ordinary
PyTorch module:

```python
from arti.alpha import FederalTensorQueryCompiler

compiled = FederalTensorQueryCompiler.compile(
    query=query_logits,                 # Tensor[B, action]
    operations=(formula_a, formula_b),  # each Tensor[B, ...] -> Tensor[B, ...]
    example_input=example,
    source_ref="arti/federal-recall@3",
    source_snapshot_fingerprint=bank_snapshot_fingerprint,
    max_steps=8,
    route_mode="hard",
)

output = compiled(value)
exported = FederalTensorQueryCompiler.export(compiled, example)
```

The graph contains no Python Bank-ID lookup, candidate dataclass, Python
`break`, or runtime callback.  Every bounded step re-runs the Query on the
latest state, evaluates the finite operation set, and selects with Tensor
`argmax`/`gather` (or combines it with Tensor soft weights).  Exit is an
identity candidate and previously exited rows are masked to that candidate.
The loop is statically unrolled, so per-row logical exit does not claim a
physical compute saving by itself.

`route_mode="hard"` is the deployment form.  `"soft"` keeps a differentiable
weighted route, while `"straight_through"` has hard forward selection and a
soft backward estimator.  The compiler checks the example ABI before making a
deep-copied graph.  `FederalTensorQueryAdapter` can unwrap a sealed
Bank-owned Query result for implementations whose Query body is exportable;
sealing and runtime validation are pre-compilation concerns, not a promise
that arbitrary Python Query code can be exported.

This is a finite tensorization boundary, not a global shape restriction on
Federal Banks.  Candidate operations inside one block must share one input and
output Tensor ABI.  Heterogeneous Bank shapes are connected by explicit fixed
transport blocks and compiled per ABI family.  The stateful and logical-shape
contracts below cover mutable Bank state, self-modifying effects, open-ended
Refine, and data-dependent sequence lengths without putting Python dispatch in
the graph.

The older finite compiler still exposes `mutable_state` and
`data_dependent_shape` as explicit boundaries.  A set flag raises
`FederalCompileError`; it is not treated as a hint to snapshot and silently
erase the behavior.  Use the explicit stateful or logical-shape contracts when
those semantics are intended.

The component references are `arti/federal-tensor-query-compiler@1`,
`arti/federal-tensor-query-block@1`,
`arti/federal-tensor-query-manifest@1`,
`arti/federal-tensor-federation@1`, and
`arti/federal-tensor-federation-compiler@1`.  `torch.export` is the strict proof
boundary: it requires a full graph and records shape constraints; a successful
export is evidence that this finite Query graph no longer depends on Python
dispatch, not evidence that every dynamic Federation is compilable.

The small scheduler-boundary comparison can be replayed with:

```powershell
uv run python benchmarks/federal_tensor_query_compilation.py `
  --device cpu --backend eager --output .tmp/federal-tensor-query-cpu.json
```

It compares an object-based Python reference, tensorized eager execution, and
`torch.compile(fullgraph=True)`.  Use `--backend inductor` only when the local
PyTorch toolchain can build its backend.  Its tiny two-Bank workload is
deliberately an overhead probe; it must not be quoted as a production
attention/GEMM speed claim.

## Stateful Bank and self-modifying graph

Mutable Bank state and self-modification are compiled as an explicit state
lane, not as an in-place parameter side effect.  A transition receives the
current value, Bank state, and effect state, and returns the next three tensors
plus a per-row stop predicate:

```python
from arti.alpha import FederalStatefulGraphCompiler

graph = FederalStatefulGraphCompiler.compile(
    transition,  # (value, bank, effect) -> (value, bank, effect, stop[B])
    example_value=value,
    example_bank_state=bank_state,
    example_effect_state=effect_state,
    source_ref="arti/federal-federation@1",
    source_snapshot_fingerprint=snapshot_fingerprint,
    max_steps=128,
)

value, bank_state, effect_state, done, steps = graph(
    value, bank_state, effect_state, torch.tensor(128)
)
exported = FederalStatefulGraphCompiler.export(
    graph, value0, bank0, effect0, torch.tensor(128)
)
```

The compiled graph owns no hidden mutable Bank.  Passing `bank_state` from one
call to the next is the persistence boundary; self-modifying nodes change that
returned tensor state while the data lane can remain an identity.  This also
keeps autograd honest: parameters are ordinary trainable parameters of the
transition, while forward-time learning is represented by explicit state
outputs.

`max_steps` is a host-supplied finite safety budget, not a training-time
unroll.  Inside that budget the graph is a real `torch.while_loop`: every
iteration re-queries the current state and can stop per row.  Therefore
“unbounded Refine” means open-ended with respect to the learned stop predicate,
not literal infinite execution.  A finished row is carried without logical
changes; physical packed savings require a separate backend.

## Data-dependent logical shapes

When a Bank changes the logical sequence length from data, use
`FederalRaggedShapeCompiler`.  Its graph ABI is `values[B, capacity, ...]` plus
`lengths[B]`; the transform returns a new padded values tensor and new logical
lengths.  The physical capacity is fixed for export, while the logical shape is
data-dependent and masked in the graph:

```python
from arti.alpha import FederalRaggedShapeCompiler

graph = FederalRaggedShapeCompiler.compile(
    transform,  # (values, lengths) -> (values, lengths)
    example_values=values,
    example_lengths=lengths,
    source_ref="arti/federal-federation@1",
    source_snapshot_fingerprint=snapshot_fingerprint,
)
values, lengths = graph(values, lengths)
exported = FederalRaggedShapeCompiler.export(graph, values0, lengths0)
```

This contract supports data-dependent ragged lengths and preserves a Tensor
graph.  It does not claim that arbitrary data-dependent allocation or rank
change can be exported; those require another explicit carrier ABI.  The
manifest records `shape_semantics="bounded-ragged-logical-shape"`, so a
consumer cannot mistake padded storage for the logical length.

The state-lane control benchmark can be replayed with:

```powershell
uv run python benchmarks/federal_stateful_graph_compilation.py `
  --device cpu --backend eager --max-steps 16 `
  --output .tmp/federal-stateful-graph-cpu.json
```

Its comparison is deliberately narrow: Python reference loop versus the same
explicit-state graph in eager and `torch.compile(fullgraph=True)`. It reports
logical step counts and exact state/output errors, but is not a production
throughput claim. On backends without an installed compiler toolchain,
`compile_error` is recorded instead of being confused with a graph-capture
failure.

The new component references are
`arti/federal-stateful-graph-compiler@1`,
`arti/federal-stateful-refine-graph@1`,
`arti/federal-ragged-shape-compiler@1`, and
`arti/federal-ragged-shape-graph@1`.  These are additional graph contracts;
they do not change the semantics of the older static path compiler.

Compiled artifacts use SafeTensors for module state and JSON for the
`arti/federal-compile-manifest@1` provenance, including operation and nested
dependency references plus optional input/output shape ABI. Reload requires a caller-built
`FederalPath` with the same operation structure and provenance; this is
intentional because Python code and a live Federal runtime are not serialized.

The replay harness records the comparison evidence without becoming a runtime
dependency:

```powershell
uv run python benchmarks/federal_compiler_replay.py --device cpu `
  --output-dir .tmp/federal-replay-cpu
uv run python benchmarks/federal_compiler_replay.py --device cuda `
  --output-dir .tmp/federal-replay-cuda
```

The report contains Federal-to-path and Federal-to-compiled errors, every
intermediate node error, validated terminal ABI fields, a held-out task MSE,
parameter counts, profiler FLOP estimates, latency, and CUDA peak allocation
when CUDA is available. These numbers are replay evidence for one frozen
snapshot, not a claim that a dynamic Federation has become globally static.

The required comparison is Federal versus compiled from the same snapshot and
path: output and intermediate values, terminal ABI, held-out task loss,
parameter/FLOP budget, peak memory, and latency. Compilation is complete only
when the ordinary module can run after the Federal runtime is absent.

An attached `ARTILayer@2` artifact remains an `AdaptivePulse@2` attachment. Its
metadata records that execution surface and the available compiler reference,
but `compiled_artifact_is_separate` is explicit: saving an attachment does not
silently turn a Pulse graph into a Federal specialization. A Federal compiled
artifact must be produced from a frozen Federal snapshot and path evidence.
