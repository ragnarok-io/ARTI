# Formula Fabric

Formula Fabric is an alpha, fixed-capacity tensor executor. A program declares
a bounded set of Formula cells, while a route source selects their operands.
The executor remains the only implementation of Formula mathematics.

## Objective-Controlled Commits

`ObjectiveFormulaFabricCompute` is an alpha composition adapter that lets an
`ObjectiveExposureBank` provide bounded commit strength to an existing
`FormulaCommitBlend` executor. It does not choose Formula routes, primitives,
fire masks, commit authority, topology, or execution depth.

```python
from arti import alpha

compute = alpha.FormulaFabricCompute(
    alpha.FormulaCommitBlend(alpha.FormulaFabric(program)),
    active_count=8,
)
controlled = alpha.ObjectiveFormulaFabricCompute(
    compute,
    alpha.ObjectiveExposureBank(slots=16, query_dim=64),
)
pulse = alpha.AdaptivePulse(
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
from arti import alpha

routed = alpha.RoutedFormulaFabricCompute(compute, route_source)
refined = alpha.IterativeRoutedFormulaFabricCompute(routed, steps=4)

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
