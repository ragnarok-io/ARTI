# Program Roles

ARTI separates storage, relations, numerical operations, and execution rather
than using one term for the entire runtime.

| Role | Public type | Responsibility |
| --- | --- | --- |
| Addressable operands | `OperandBank` (Formula-owned) | Stores values or factors addressed by a Formula. A Bank is not an executable graph member. |
| Logical storage | `TensorResource` | Declares tensor shape compatibility, capacity, lifetime, default backing, and optional mounted backing. It does not own a Query or an editor. |
| Relation | `Connection` | Binds source and destination ports and optional resource views. A connection may be direct, continuously trainable, or locally conditional. |
| Learning boundary | `CreditBoundary` on a `Connection` | Preserves the forward TensorView while declaring the credit VJP for that named port relation. It is not a data-transforming node or a new resource payload. |
| Learning topology | `LearningRegion` / `OptimizerDomain` | At an explicit structure window, projects committed credit cuts into parameter ownership and grouped optimizer state. It does not reschedule forward arrival semantics. |
| Composition | `ProgramGraph` | Declares resource and connection composition, including serial dependencies, branch-local execution, explicit parallel frontiers, and named local programs. |
| Local behavior | `RoutedProgram` | Re-queries its latest TensorView while its selected Formula, resource, and NeuralPlasticity proposals remain branch-local until an accepted terminal result. |
| Program composition | `FederatedProgram` | Composes routed programs behind a shared terminal ABI and K-wide, hard-winner policy. It is one executable region that may be mounted by a `ProgramGraph`; it is not a Bank. |
| Retrieval | `Retrieve` | Retrieves Formula operands and advances the current tensor state. It is an operation, not the name of the whole runtime. |
| Iteration | `ExecutionPolicy` | Bounds, stops, traces, and schedules repeated program execution without implying that every iteration repairs a tensor. |
| Numerical operation | Formula Fabric | Performs data, topology, observation, control, and NeuralPlasticity effect operations selected by a program or connection. |

`RoutedProgram`, `FederatedProgram`, `Retrieve`, and `ExecutionPolicy` are the
public construction contracts. Historical `Recall` and `RefinePolicy` remain under
`arti.legacy` only for historical artifacts and inspection.
Historical `TensorViewFormulaProgram` and `FederalRecallV3` names are available
only from `arti.legacy` for inspecting older artifacts; new programs must not
mix those contracts with the role-oriented graph.

## Registering Custom Fabric Regions

An ordinary PyTorch layer can become a named multi-port graph region without
being rewritten as Formula IR.  Declare the port ABI once, construct the layer
normally, then mount the instance onto resources:

```python
import torch
import arti

@arti.fabric_layer(inputs=("source",), outputs={"value": "source"})
class Scale(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(value))

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return source * self.scale

node = arti.as_fabric_node(
    "scale",
    Scale(0.8),
    input_ports={"source": arti.ResourcePort("input")},
    output_ports={"value": arti.ResourcePort("output")},
)
```

For a raw `Tensor` result, the declared output source supplies the TensorView
layout. A layer that changes its logical ABI must return a `TensorView`
explicitly. This makes a shape transition part of the layer contract instead
of a runtime guess.

`@arti.differentiable_fate("name")` marks a registered layer class as an
eligible pure-data node kind. `arti.as_differentiable_fabric_node(...)` then
forms a trainable candidate field; `specialize()` removes the structural
logits and leaves the selected ordinary node. The decorators may be written in
either order. Stateful effects, joins, resource writes, and loop control are
not numerically blended as fates: their arrival and consumption semantics stay
explicit graph operations.

Frozen `FormulaProgramNode` and registered `FabricModuleNode` regions share
the static ProgramGraph lowering path. The result is eligible for
`torch.compile`, `torch.export`, and fixed-bucket CUDA Graph capture. A custom
Python module is still runtime-only at the portable graph-artifact boundary:
callers supply that module again when loading its graph, unless they separately
publish a serializable implementation and lowering.

## Execution Boundaries

A direct `Connection` does not enter a route candidate set automatically. It
can still have learnable continuous transfer parameters. A local conditional
connection only receives its declared context; it does not imply a full
federation scan.

A `ProgramRoute` candidate may name a connection, a node, or a declared local
program of ordinary steps and `ProgramStage` frontiers. The selected program
executes its own ordered path, including serial connection dependencies and
explicit parallel frontiers; the unselected program does not execute. For a
batch route, contained differentiable node fates can be sampled in the same
structure window by passing `route_selections`; unselected fates receive no
credit. For sample-scoped routes, the sampler draws each candidate's node fate
once per structure window. Its structure objective accepts per-row final losses
and the route selections from the actual compiled execution; a fate receives
credit only from rows that executed its path. A batch-mean loss discards that
attribution and is rejected. Both routed and mixed-dataflow compiled plans
return the actual choices with their route score-function term. Route
specialization expands the selected path into ordinary steps
and prunes unreferenced alternatives. Candidate selection is distinct from a
parallel frontier: the latter executes all its members against one snapshot.

`ProgramRoute(..., selection_scope="batch")` selects one hard candidate for the
whole batch. `selection_scope="sample"` instead returns one choice and one
log-probability per row. For a fixed batch bucket, compiled lowering executes
only the selected candidate for each row and preserves row-specific Join
publication and reverse-credit masks. The sample-scoped mode currently requires
`ResourceGraphCompiler.compile_program(...)`; the Python functional executor
does not yet create per-row state receipts. Compiled dispatch executes a
homogeneous route cohort as one batch and recursively splits mixed cohorts.
Routes accept two or more distinct candidates and require a score resource of
shape `[batch, candidate_count]`. Sparse dispatch uses a balanced conditional
tree; only the selected candidate executes. `execution_mode="all_candidates"`
executes every candidate before selecting one result and is limited to
paths without asynchronous Join arrivals.
This fixed-bucket strategy can still cost more than a grouped sparse kernel.
A score-function objective must
pair each row's final task loss with that row's log-probability. Specializing
a sample-scoped route into an ordinary full-batch node is tensor-close, but
floating-point reduction order can differ; low-precision downstream models
must be checked at their own output ABI before claiming exact parity.

`CreditBoundary` belongs to the connection between named ports, so a
multi-port Fabric node can give separate outputs different credit rules without
changing its ordinary tensor ABI. Its data lane remains identity. Bernoulli
credit masks are explicit execution inputs and are replayable under compiled
connection lowering; a `MEAN` permeability or a candidate structure logit is
not by itself a committed region cut.

`TensorResource` state is explicit. Functional execution returns a candidate
`ProgramGraphState`; a federated K-wide search restores only the terminal
winner's resource state. Parallel connections read a common snapshot, and
overlapping writes require a program-supplied merge.

`ResourceGraphCompiler.compile_program(..., iterations=N)` lowers a fixed
local horizon into one functional tensor plan. `N` is internal computation
depth, not a count of external input events. Conditional and shape-dependent
semantics stay explicit; a compiled plan does not claim sparse physical work
merely because its logical graph has a selected path.

`ProgramStage` remains a first-class static frontier after lowering: every
peer reads the same SSA resource snapshot and all of its distinct publications
commit together. The same rule applies inside `compile_loop(...)`, whose
continuation mask provides a finite, per-batch conditional horizon. A
`ProgramJoin` has deliberately different arrival/consume semantics and lowers
through `compile_program(...)` as a tensor-resident dataflow region; it is not
silently treated as a synchronous loop peer.

## Portable Resource Graphs

`arti.save_program_graph(graph, path)` writes a standalone SafeTensors graph
artifact containing the declared resources, their current backings, and native
identity, affine, or Formula-v2 transfer laws. `arti.load_program_graph(path)`
reconstructs fresh resources with the saved state. This is intentionally
separate from `arti.save`, whose model artifact does not own mutable resource
backings. Arbitrary Python callables and conditional activation callables are
runtime-only and fail at the graph artifact boundary.
New graph artifacts retain resource declaration order so a recompiled plan's
`resource_ids` stays aligned with the original positional tensor inputs.
