# Program Roles

ARTI separates storage, relations, numerical operations, and execution rather
than using one term for the entire runtime.

| Role | Public type | Responsibility |
| --- | --- | --- |
| Addressable operands | `OperandBank` (Formula-owned) | Stores values or factors addressed by a Formula. A Bank is not an executable graph member. |
| Logical storage | `TensorResource` | Declares tensor shape compatibility, capacity, lifetime, default backing, and optional mounted backing. It does not own a Query or an editor. |
| Relation | `Connection` | Binds source and destination ports and optional resource views. A connection may be direct, continuously trainable, or locally conditional. |
| Composition | `ProgramGraph` | Declares resource and connection composition, including serial dependencies, branch-local execution, explicit parallel frontiers, and named local programs. |
| Local behavior | `RoutedProgram` | Re-queries its latest TensorView while its selected Formula, resource, and NeuralPlasticity proposals remain branch-local until an accepted terminal result. |
| Program composition | `FederatedProgram` | Composes routed programs behind a shared terminal ABI and K-wide, hard-winner policy. It is one executable region that may be mounted by a `ProgramGraph`; it is not a Bank. |
| Retrieval | `Retrieve` | Retrieves Formula operands and advances the current tensor state. It is an operation, not the name of the whole runtime. |
| Iteration | `ExecutionPolicy` | Bounds, stops, traces, and schedules repeated program execution without implying that every iteration repairs a tensor. |
| Numerical operation | Formula Fabric | Performs data, topology, observation, control, and NeuralPlasticity effect operations selected by a program or connection. |

`RoutedProgram`, `FederatedProgram`, `Retrieve`, and `ExecutionPolicy` are the
public construction contracts. `Recall` and `RefinePolicy` remain under
`arti.legacy` only for historical artifacts and inspection.
Historical `TensorViewFormulaProgram` and `FederalRecallV3` names are available
only from `arti.legacy` for inspecting older artifacts; new programs must not
mix those contracts with the role-oriented graph.

## Execution Boundaries

A direct `Connection` does not enter a route candidate set. It can still have
learnable continuous transfer parameters. A local conditional connection only
receives its declared context; it does not imply a full federation scan.

`TensorResource` state is explicit. Functional execution returns a candidate
`ProgramGraphState`; a federated K-wide search restores only the terminal
winner's resource state. Parallel connections read a common snapshot, and
overlapping writes require a program-supplied merge.

`ResourceGraphCompiler.compile_program(..., iterations=N)` lowers a fixed
local horizon into one functional tensor plan. `N` is internal computation
depth, not a count of external input events. Conditional and shape-dependent
semantics stay explicit; a compiled plan does not claim sparse physical work
merely because its logical graph has a selected path.

## Portable Resource Graphs

`arti.save_program_graph(graph, path)` writes a standalone SafeTensors graph
artifact containing the declared resources, their current backings, and native
identity, affine, or Formula-v2 transfer laws. `arti.load_program_graph(path)`
reconstructs fresh resources with the saved state. This is intentionally
separate from `arti.save`, whose model artifact does not own mutable resource
backings. Arbitrary Python callables and conditional activation callables are
runtime-only and fail at the graph artifact boundary.
