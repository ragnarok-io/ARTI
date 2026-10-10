# Native JAX Backend

`arti.jax` binds the shared `FormulaProgram` and `ProgramGraph` contracts to
native JAX execution. The graph is static configuration; parameters and forward
state are explicit array PyTrees. A compiled invocation executes JAX operations
without Torch execution or host callbacks.

```bash
uv sync --extra jax
uv sync --extra jax --extra flax  # optional native model integration
```

The root `arti.ARTILayer` is the PyTorch host. Native applications use
`arti.jax.ARTILayer`, whose interface is `init(key, x)` and
`apply(params, state, x, rng=...) -> (value, next_state)`.

## Shared Formula Programs

Create the ordinary typed Formula IR, then bind it to JAX:

```python
import jax
import jax.numpy as jnp
import arti.jax as aj
from arti import mechanisms as m

x = m.InputBinding("x", m.TensorType(("B", "D"), ("B", 8), dtype="float32"))
gain = m.BankBinding(
    "gain", "example/scale@1", "gain",
    m.TensorType((), (), dtype="float32"),
)
program = m.FormulaProgram.build(outputs=(m.scale(x, gain),))
formula = aj.bind_formula(program)
params = formula.init(jax.random.key(0), bank_initializers={"gain": jnp.array(1.0)})
outputs = jax.jit(formula.apply)(params, {"x": jnp.ones((4, 8))})
```

Bank declarations retain their source, partition and member identities in the
shared IR. `init` takes arrays or initializers of the form `(key, TensorType)`.
The parameter dictionary contains the supplied Bank operands; differentiate
the floating operands needed by the application. A graph binding keeps
integer, boolean and frozen floating Bank operands in forward state instead
of optimizer parameters. Bindings of existing graphs preserve the source
Bank's trainability.

Native kernels cover contraction, scale/add, both reduction contracts,
scalar maps, windows, shape operations, gather/scatter, segments, lookup,
masked softmax, index worksets, position encoding and Observation. Exact GELU
and the zero derivative of `abs(0)` match the Torch contract. `Reduce@1`
preserves left-to-right accumulation; `Reduce@2` uses native reductions.

`apply_checked` returns `FormulaResult(outputs, valid)`. The validity scalar
includes finite values, index range and replace-scatter uniqueness. It remains
an array under `jit`; inspect it outside the compiled call before adopting a
result from externally supplied dynamic indices. JAX's default index clipping
is not reported as a successful Formula operation.

Declared `int64` and `float64` contracts require the application's JAX X64
setting. ARTI does not change global JAX precision settings:

```python
jax.config.update("jax_enable_x64", True)
```

`bind_formula_scan(shared_scan)` lowers the existing `FormulaScan` body into
`lax.scan`, retaining simultaneous carry updates, stacked emissions and the
declared horizon. This supports reverse-mode differentiation through the
trajectory. See [JAX control flow](https://docs.jax.dev/en/latest/201/control-flow.html).

## Native Modules And Resources

A native module declares its ports and supplies a pure `init/apply` pair:

```python
@aj.fabric_layer(inputs=("x", "memory"), outputs=("value", "memory"))
def apply(params, state, x, memory, *, rng=None):
    next_memory = 0.9 * memory + 0.1 * x
    return {
        "value": x * params["gain"] + next_memory,
        "memory": next_memory,
    }, state

module = aj.NativeModule(
    init=lambda key, x, memory: ({"gain": jnp.ones((), dtype=x.dtype)}, {}),
    apply=apply,
    implementation_ref="example/memory-layer@1",
)
node = aj.as_fabric_node(
    "memory_layer", module,
    input_ports={"x": m.ResourcePort("x"), "memory": m.ResourcePort("memory")},
    output_ports={"value": m.ResourcePort("out"), "memory": m.ResourcePort("memory")},
)
```

Use `aj.tensor_resource(spec, initial_array, axis_names=..., axis_roles=...)`
to construct a shared resource declaration from a native array. This performs
one host conversion while building the shared graph. Native invocation does
not convert between backends.

```python
def resource(name, lifetime):
    spec = m.TensorResourceSpec(name, m.TensorViewPattern(), lifetime)
    return aj.tensor_resource(spec, jnp.zeros((1, 8)), axis_names=("B", "D"))

graph = m.ProgramGraph(
    (
        resource("x", m.ResourceLifetime.CALL),
        resource("out", m.ResourceLifetime.CALL),
        resource("memory", m.ResourceLifetime.PERSISTENT),
    ),
    (), nodes=(node,), programs={"forward": ("memory_layer",)},
)
bound = aj.bind_resource_graph(graph, "forward")
params, state = bound.init(jax.random.key(1), {"x": jnp.ones((4, 8))})
result = jax.jit(bound.apply)(params, state, {"x": jnp.ones((4, 8))})
state = result.state
```

`ProgramState` contains active and default resource values/masks, backing
selection, epochs, step counters, module state and Join cursors. Resource
lifetime does not determine parameter trainability. Call resources begin each
invocation from their default or mounted backing; persistent resources retain
their returned state. `state.mount(resource_id, value)` and `unmount` select a
compatible backing functionally. Supplied inputs are functional mounts;
unmounting restores their retained default backing. `input_masks` supplies
the new input visibility explicitly. Use a new compiled shape bucket when physical
backing shape or dtype changes.

Forward writes remain differentiable. Carry `result.state` through `lax.scan`
for an episode; use `jax.lax.stop_gradient` explicitly at the application's
chosen cross-episode boundary. Independent instances can be mapped with `vmap`,
with an outer instance axis around each instance's ordinary batch axis.

Shared Torch parameter/Bank objects and reused native module instances bind
to one optimizer leaf. Aliased leaves appear once under `params["shared"]`.
Custom native modules can declare matching `parameter_ids` trees when several
bindings deliberately share a parameter. Identity keys are host metadata and
do not become JAX gradient leaves.

## Graph Execution

Bindings retain the existing graph operations:

- Direct and conditional Connections, including static resource regions.
- Named multi-output nodes and explicit downstream convergence.
- `ProgramStage`: all members read one snapshot, then publish their writes.
- `ProgramRoute`: sparse batch selection uses `lax.switch`; `all_candidates`
  evaluates candidate arrays and selects committed state. Native sample routes
  currently require the explicit `all_candidates` mode.
- `ProgramJoin`: cursor state preserves fresh arrivals across invocations.
- `ProgramLoop`: fixed-horizon `lax.scan`, individual publication masks and
  device-side skipping once every sample stops.

`bind_resource_graph(graph, program_id, native_modules=...)` accepts an existing
Torch graph. Formula nodes and common numeric modules have native lowerings.
Supply `NativeModule` overrides for custom nodes, transfers or conditions.
Conditional edges require example contexts at initialization. Neither a custom
Torch function nor a custom Torch VJP is automatically converted into JAX.

Mutable native module state needs an explicit `state_batch_axis` for sample
routing or independently stopped loops. Global batch-coupled state, such as
BatchNorm statistics, works in ordinary and batch-selected programs; it must
not be silently treated as independent per-sample state.

Credit boundaries are strict identity in forward and use a native custom VJP.
Mean and Bernoulli permeability follow the declared mode independently of
training/evaluation. Bernoulli execution accepts an explicit RNG or replayed
credit masks. Ordinary forward loss does not provide a permeability gradient;
learn it from update consequences using an explicit outer objective.
Repeated uses of an edge share its sample within one program invocation;
bounded loop iterations begin new sampling scopes.

## Flax Integration

`from_flax(module, inputs=..., outputs=..., mutable=..., rng_collections=...)`
separates `variables["params"]` from other variable collections. Mutable
collections are returned in the next native module state. This follows the
[Flax init/apply contract](https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/module.html).

`as_flax_module(native_layer)` exposes an ARTI host as a Linen child module.
`attach` inserts it at specified tensor outputs of a native model:

```python
layer = aj.ARTILayer(bound, input_resource_id="x", output_resource_id="out")
model = aj.attach(flax_model, layer, targets=("encoder.projection",))
variables = model.init(jax.random.key(2), batch)
value, updates = model.apply(variables, batch, mutable=("arti_state",))
variables = {**variables, **updates}
```

Paths are relative to the host model and may use glob patterns. Each matched
call site has its own adapter scope. Host parameters are under `model`, so an
application can use ordinary Optax parameter masks for adapter-only training.
Stochastic programs receive an `arti` RNG collection explicitly.

## Example And Validation

```bash
uv run --extra jax --extra flax python examples/jax_native_training.py
uv run --extra dev --extra jax --extra flax python scripts/quality_gate.py jax --fail-fast
```

The example compiles the model, task gradient, optimizer update and persistent
state progression inside one `lax.scan`. The JAX gate checks Formula forward
and gradient parity, resources, loops, routes, joins, native training and real
Flax attachment.

Current bindings specialize physical shapes and graph structure. Formula
effects with predecessor writeback, automatic lowering of arbitrary federated
Query programs, open-horizon reverse-mode execution and automatic independent
optimizer-region derivation require further native lowerings. A custom native
module can already perform explicit differentiable state updates. Binding
objects are host-bound alpha components; retain the shared static program and
checkpoint the explicit trees with native JAX/Flax checkpoint tooling.

The earlier `init_layer`, `apply_layer`, mask and coordinate helpers remain
available. Their original contract checks run alongside the native program
tests.
