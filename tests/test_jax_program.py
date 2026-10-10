from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import arti.jax as aj
from arti import mechanisms as m


def resource(name, value, *, lifetime=m.ResourceLifetime.CALL):
    value = jnp.asarray(value)
    names = {1: ("B",), 2: ("B", "D"), 3: ("B", "N", "D")}.get(
        value.ndim, ("B",) + tuple(f"D{index}" for index in range(value.ndim - 1))
    )
    spec = m.TensorResourceSpec(
        name,
        m.TensorViewPattern(dtype="boolean" if value.dtype == jnp.bool_ else "floating"),
        lifetime,
    )
    return aj.tensor_resource(spec, value, axis_names=names)


def node(name, function, *, inputs, outputs, initializer=None, state_batch_axis=None):
    module = aj.NativeModule(
        initializer or (lambda key, *values: ({}, {})),
        function,
        tuple(inputs),
        tuple(outputs),
        state_batch_axis=state_batch_axis,
    )
    return aj.as_fabric_node(
        name,
        module,
        input_ports={key: m.ResourcePort(value) for key, value in inputs.items()},
        output_ports={key: m.ResourcePort(value) for key, value in outputs.items()},
    )


def test_identity_host_and_native_registered_module_training():
    identity = aj.ARTILayer()
    x = jnp.ones((3, 2), dtype=jnp.float32)
    p, s = identity.init(jax.random.key(0), x)
    y, next_s = jax.jit(identity.apply)(p, s, x)
    np.testing.assert_array_equal(y, x)
    assert next_s == {}

    @aj.fabric_layer(inputs=("x",), outputs=("y",))
    def apply(params, state, x, *, rng=None):
        return x * params["gain"], state

    module = aj.NativeModule(lambda key, x: ({"gain": jnp.array(0.1, dtype=x.dtype)}, {}), apply)
    transform = aj.as_fabric_node(
        "affine",
        module,
        input_ports={"x": m.ResourcePort("x")},
        output_ports={"y": m.ResourcePort("y")},
    )
    graph = m.ProgramGraph(
        (resource("x", jnp.zeros((1, 2))), resource("y", jnp.zeros((1, 2)))),
        (),
        nodes=(transform,),
        programs={"main": ("affine",)},
    )
    layer = aj.ARTILayer(
        aj.bind_resource_graph(graph, "main"), input_resource_id="x", output_resource_id="y"
    )
    params, state = layer.init(jax.random.key(1), x)

    @jax.jit
    def train(p, s):
        def objective(candidate):
            output, updated = layer.apply(candidate, s, x)
            return jnp.square(output - 2 * x).mean(), updated

        (loss, updated), gradients = jax.value_and_grad(objective, has_aux=True)(p)
        return jax.tree_util.tree_map(lambda a, g: a - 0.1 * g, p, gradients), updated, loss

    first = None
    for _ in range(30):
        params, state, loss = train(params, state)
        first = float(loss) if first is None else first
    assert float(loss) < first * 1e-4
    assert "callback" not in train.lower(params, state).as_text().lower()


def test_shared_formula_node_binds_without_torch_execution():
    x = m.InputBinding("x", m.TensorType(("B", "D"), ("B", 2), dtype="float32"))
    gain = m.BankBinding("gain", "arti/jax-test@1", "gain", m.TensorType((), (), dtype="float32"))
    program = m.FormulaProgram.build(outputs=(m.scale(x, gain),))
    formula = m.FormulaProgramNode(
        "formula",
        m.FormulaFabricV2(program),
        input_ports={"x": m.ResourcePort("x")},
        output_ports={"out": m.ResourcePort("y")},
        output_slots={"out": program.outputs[0]},
        banks={"gain": gain.bind(torch.tensor(3.0, requires_grad=True))},
    )
    graph = m.ProgramGraph(
        (resource("x", jnp.zeros((1, 2))), resource("y", jnp.zeros((1, 2)))),
        (),
        nodes=(formula,),
        programs={"main": ("formula",)},
    )
    bound = aj.bind_resource_graph(graph, "main")
    params, state = bound.init(jax.random.key(2), {"x": jnp.ones((2, 2), dtype=jnp.float32)})
    result = jax.jit(bound.apply)(params, state, {"x": jnp.ones((2, 2), dtype=jnp.float32)})
    np.testing.assert_array_equal(result.outputs["y"], 3)
    gradients = jax.grad(
        lambda p: bound.apply(p, state, {"x": jnp.ones((2, 2))}).outputs["y"].sum()
    )(params)
    np.testing.assert_array_equal(gradients["nodes"]["formula"]["gain"], 4)


def test_parallel_stage_reads_one_snapshot_and_multiplies_named_heads():
    split = node(
        "split",
        lambda p, s, x, rng=None: ({"left": x + 1, "right": x * 2}, s),
        inputs={"x": "x"},
        outputs={"left": "left", "right": "right"},
    )
    join = node(
        "combine",
        lambda p, s, a, b, rng=None: (a * b, s),
        inputs={"a": "left", "b": "right"},
        outputs={"y": "y"},
    )
    graph = m.ProgramGraph(
        tuple(resource(name, jnp.zeros((1, 1))) for name in ("x", "left", "right", "y")),
        (),
        nodes=(split, join),
        programs={"main": ("split", "combine")},
    )
    bound = aj.bind_resource_graph(graph, "main")
    p, s = bound.init(jax.random.key(3))
    result = jax.jit(bound.apply)(p, s, {"x": jnp.array([[3.0]], dtype=jnp.float32)})
    np.testing.assert_array_equal(result.outputs["y"], [[24]])

    first = node(
        "first", lambda p, s, x, rng=None: (x + 10, s), inputs={"x": "x"}, outputs={"left": "left"}
    )
    second = node(
        "second", lambda p, s, x, rng=None: (x * 2, s), inputs={"x": "x"}, outputs={"y": "y"}
    )
    graph = m.ProgramGraph(
        tuple(resource(name, jnp.zeros((1, 1))) for name in ("x", "left", "y")),
        (),
        nodes=(first, second),
        programs={"main": (m.ProgramStage(("first", "second")),)},
    )
    bound = aj.bind_resource_graph(graph, "main")
    p, s = bound.init(jax.random.key(4))
    result = jax.jit(bound.apply)(p, s, {"x": jnp.array([[2.0]], dtype=jnp.float32)})
    np.testing.assert_array_equal(result.outputs["x"], [[2]])
    np.testing.assert_array_equal(result.outputs["left"], [[12]])
    np.testing.assert_array_equal(result.outputs["y"], [[4]])


def test_persistent_resource_scan_carries_learning_state_and_gradients():
    accumulation = node(
        "accumulate",
        lambda p, s, x, memory, rng=None: (memory + p["gain"] * x, s),
        inputs={"x": "x", "memory": "memory"},
        outputs={"memory": "memory"},
        initializer=lambda key, x, memory: ({"gain": jnp.array(0.5, dtype=x.dtype)}, {}),
    )
    graph = m.ProgramGraph(
        (
            resource("x", jnp.zeros((1, 1))),
            resource("memory", jnp.zeros((1, 1)), lifetime=m.ResourceLifetime.PERSISTENT),
        ),
        (),
        nodes=(accumulation,),
        programs={"main": ("accumulate",)},
    )
    bound = aj.bind_resource_graph(graph, "main")
    params, state = bound.init(jax.random.key(5))
    frames = jnp.array([[[1.0]], [[2.0]], [[3.0]]], dtype=jnp.float32)

    def sequence(p, s):
        def step(current, x):
            result = bound.apply(p, current, {"x": x})
            return result.state, result.outputs["memory"]

        return jax.lax.scan(step, s, frames)

    final, outputs = jax.jit(sequence)(params, state)
    np.testing.assert_array_equal(outputs[:, 0, 0], [0.5, 1.5, 3])
    gradient = jax.jit(jax.grad(lambda p: sequence(p, state)[0].values["memory"].sum()))(params)
    np.testing.assert_array_equal(gradient["nodes"]["accumulate"]["gain"], 6)
    mounted = final.mount("memory", jnp.array([[20.0]], dtype=jnp.float32))
    result = jax.jit(bound.apply)(params, mounted, {"x": jnp.ones((1, 1), dtype=jnp.float32)})
    np.testing.assert_array_equal(result.outputs["memory"], [[20.5]])
    np.testing.assert_array_equal(result.state.unmount("memory").values["memory"], [[3]])


def test_fixed_loop_stops_individual_samples_and_retains_module_state():
    def initialize(key, x, limit):
        return {"gain": jnp.array(1.0, dtype=x.dtype)}, {"count": jnp.zeros_like(x)}

    def apply(p, state, x, limit, rng=None):
        value = x + p["gain"]
        return {"x": value, "continue": (value < limit).astype(x.dtype)[:, 0]}, {
            "count": state["count"] + 1
        }

    iteration = node(
        "iteration",
        apply,
        inputs={"x": "x", "limit": "limit"},
        outputs={"x": "x", "continue": "continue"},
        initializer=initialize,
        state_batch_axis=0,
    )
    graph = m.ProgramGraph(
        (
            resource("x", jnp.zeros((1, 1))),
            resource("limit", jnp.ones((1, 1))),
            resource("continue", jnp.ones((1,), dtype=jnp.float32)),
        ),
        (),
        nodes=(iteration,),
        programs={"body": ("iteration",)},
        loops=(m.ProgramLoop("loop", "body", "continue", max_iterations=8),),
    )
    bound = aj.bind_resource_graph(graph, "loop")
    inputs = {
        "x": jnp.zeros((2, 1), dtype=jnp.float32),
        "limit": jnp.array([[1], [3]], dtype=jnp.float32),
    }
    p, s = bound.init(jax.random.key(6), inputs)
    result = jax.jit(bound.apply)(p, s, inputs)
    np.testing.assert_array_equal(result.outputs["x"], [[1], [3]])
    np.testing.assert_array_equal(result.receipt["iterations"], [1, 3])
    np.testing.assert_array_equal(result.state.modules["nodes"]["iteration"]["count"], [[1], [3]])
    assert not np.any(result.receipt["host_limited"])
    gradients = jax.jit(jax.grad(lambda params: bound.apply(params, s, inputs).outputs["x"].sum()))(
        p
    )
    np.testing.assert_array_equal(gradients["nodes"]["iteration"]["gain"], 4)


def test_async_join_keeps_arrivals_across_calls_and_consumes_once():
    left = node(
        "left_node",
        lambda p, s, x, rng=None: (x + 1, s),
        inputs={"x": "x"},
        outputs={"left": "left"},
    )
    right = node(
        "right_node",
        lambda p, s, x, rng=None: (x * 2, s),
        inputs={"x": "x"},
        outputs={"right": "right"},
    )
    merge = node(
        "merge",
        lambda p, s, a, b, rng=None: (a + b, s),
        inputs={"a": "left", "b": "right"},
        outputs={"out": "out"},
    )
    resources = (resource("x", jnp.zeros((1, 1))),) + tuple(
        resource(name, jnp.zeros((1, 1)), lifetime=m.ResourceLifetime.PERSISTENT)
        for name in ("left", "right", "out")
    )
    graph = m.ProgramGraph(
        resources,
        (),
        nodes=(left, right, merge),
        programs={
            "main": (m.ProgramJoin("join", "merge"),),
            "left": ("left_node",),
            "right": ("right_node",),
        },
    )
    waiter, producer_left, producer_right = (
        aj.bind_resource_graph(graph, name) for name in ("main", "left", "right")
    )
    p, s = waiter.init(jax.random.key(7))
    s = producer_left.apply(p, s, {"x": jnp.array([[2.0]], dtype=jnp.float32)}).state
    waiting = jax.jit(waiter.apply)(p, s)
    np.testing.assert_array_equal(waiting.outputs["out"], [[0]])
    s = producer_right.apply(p, waiting.state, {"x": jnp.array([[4.0]], dtype=jnp.float32)}).state
    result = jax.jit(waiter.apply)(p, s)
    np.testing.assert_array_equal(result.outputs["out"], [[11]])
    again = jax.jit(waiter.apply)(p, result.state)
    np.testing.assert_array_equal(again.state.epochs["out"], result.state.epochs["out"])


@pytest.mark.parametrize("scope", ("batch", "sample"))
def test_routes_follow_current_scores_in_jit_and_select_state(scope):
    positive = node(
        "positive", lambda p, s, x, rng=None: (x * 2, s), inputs={"x": "x"}, outputs={"out": "out"}
    )
    negative = node(
        "negative", lambda p, s, x, rng=None: (-x * 3, s), inputs={"x": "x"}, outputs={"out": "out"}
    )
    graph = m.ProgramGraph(
        (
            resource("x", jnp.zeros((1, 1))),
            resource("out", jnp.zeros((1, 1))),
            resource("scores", jnp.zeros((1, 2))),
        ),
        (),
        nodes=(positive, negative),
        programs={
            "main": (
                m.ProgramRoute(
                    "choose",
                    "scores",
                    ("positive", "negative"),
                    selection_scope=scope,
                    execution_mode="all_candidates" if scope == "sample" else "sparse",
                ),
            )
        },
    )
    bound = aj.bind_resource_graph(graph, "main")
    inputs = {
        "x": jnp.array([[1], [2]], dtype=jnp.float32),
        "scores": jnp.array([[5, 0], [0, 4]], dtype=jnp.float32),
    }
    p, s = bound.init(jax.random.key(8), inputs)
    compiled = jax.jit(bound.apply)
    result = compiled(p, s, inputs)
    np.testing.assert_array_equal(
        result.outputs["out"], [[2], [4]] if scope == "batch" else [[2], [-6]]
    )
    other = compiled(p, s, {**inputs, "scores": -inputs["scores"]})
    np.testing.assert_array_equal(
        other.outputs["out"], [[-3], [-6]] if scope == "batch" else [[-3], [4]]
    )
    derivative = jax.grad(lambda x: compiled(p, s, {**inputs, "x": x}).outputs["out"].sum())(
        inputs["x"]
    )
    np.testing.assert_array_equal(derivative, [[2], [2]] if scope == "batch" else [[2], [-3]])


def test_credit_boundary_is_identity_with_independent_backward_rule():
    value = jnp.array([1.0, 2.0], dtype=jnp.float32)
    scale = jnp.array([0.0, 0.25], dtype=jnp.float32)
    output = jax.jit(aj.credit_boundary)(value, scale)
    np.testing.assert_array_equal(output, value)
    dx, ds = jax.grad(lambda x, p: aj.credit_boundary(x, p).sum(), argnums=(0, 1))(value, scale)
    np.testing.assert_array_equal(dx, scale)
    np.testing.assert_array_equal(ds, 0)


def test_existing_affine_connection_and_closed_boundary_bind_natively():
    edge = m.Connection(
        "transfer",
        m.ResourcePort("x"),
        m.ResourcePort("y"),
        transfer=m.LearnableAffineTransfer(gain=2, bias=1),
        credit_boundary=__import__("arti").CreditBoundary(mode="closed"),
    )
    graph = m.ProgramGraph(
        (resource("x", jnp.zeros((1, 2))), resource("y", jnp.zeros((1, 2)))),
        (edge,),
        programs={"main": ("transfer",)},
    )
    bound = aj.bind_resource_graph(graph, "main")
    p, s = bound.init(jax.random.key(9))
    inputs = {"x": jnp.array([[1, 2]], dtype=jnp.float32)}
    result = jax.jit(bound.apply)(p, s, inputs)
    np.testing.assert_array_equal(result.outputs["y"], [[3, 5]])
    gradient = jax.grad(lambda x: bound.apply(p, s, {"x": x}).outputs["y"].sum())(inputs["x"])
    np.testing.assert_array_equal(gradient, 0)


def test_native_graph_vmap_keeps_independent_persistent_instances():
    update = node(
        "accumulator",
        lambda p, s, x, memory, rng=None: (x + memory, s),
        inputs={"x": "x", "memory": "memory"},
        outputs={"memory": "memory"},
    )
    graph = m.ProgramGraph(
        (
            resource("x", jnp.zeros((1, 1))),
            resource("memory", jnp.zeros((1, 1)), lifetime=m.ResourceLifetime.PERSISTENT),
        ),
        (),
        nodes=(update,),
        programs={"main": ("accumulator",)},
    )
    bound = aj.bind_resource_graph(graph, "main")
    p, s = bound.init(jax.random.key(10))
    states = jax.tree_util.tree_map(lambda x: jnp.stack((x, x)), s)
    values = jnp.array([[[1.0]], [[5.0]]], dtype=jnp.float32)
    result = jax.jit(jax.vmap(lambda state, x: bound.apply(p, state, {"x": x})))(states, values)
    np.testing.assert_array_equal(result.outputs["memory"], values)


def test_flax_module_and_native_arti_host_use_standard_variables():
    linen = pytest.importorskip("flax.linen")

    class Dense(linen.Module):
        @linen.compact
        def __call__(self, x):
            return linen.Dense(2)(x)

    module = aj.from_flax(Dense(), inputs=("x",), outputs=("y",))
    transform = aj.as_fabric_node(
        "dense",
        module,
        input_ports={"x": m.ResourcePort("x")},
        output_ports={"y": m.ResourcePort("y")},
    )
    graph = m.ProgramGraph(
        (resource("x", jnp.zeros((1, 2))), resource("y", jnp.zeros((1, 2)))),
        (),
        nodes=(transform,),
        programs={"main": ("dense",)},
    )
    layer = aj.ARTILayer(
        aj.bind_resource_graph(graph, "main"), input_resource_id="x", output_resource_id="y"
    )
    flax_layer = aj.as_flax_module(layer)
    x = jnp.ones((3, 2), dtype=jnp.float32)
    variables = flax_layer.init(jax.random.key(11), x)
    np.testing.assert_array_equal(variables["arti_state"]["program"].steps["y"], 0)
    output, state = jax.jit(lambda v, x: flax_layer.apply(v, x, mutable=("arti_state",)))(
        variables, x
    )
    assert output.shape == (3, 2)
    assert "arti_state" in state
    gradient = jax.jit(
        jax.grad(lambda p: jnp.square(flax_layer.apply({**variables, "params": p}, x)).mean())
    )(variables["params"])
    assert all(np.isfinite(value).all() for value in jax.tree_util.tree_leaves(gradient))


def test_shared_formula_banks_have_one_parameter_owner_and_accumulated_gradient():
    x = m.InputBinding("x", m.TensorType(("B", "D"), ("B", 1), dtype="float32"))
    gain = m.BankBinding("gain", "arti/jax-test@1", "shared", m.TensorType((), (), dtype="float32"))
    program = m.FormulaProgram.build(outputs=(m.scale(x, gain),))
    weight = torch.tensor(2.0, requires_grad=True)
    nodes = tuple(
        m.FormulaProgramNode(
            name,
            m.FormulaFabricV2(program),
            input_ports={"x": m.ResourcePort("x")},
            output_ports={"out": m.ResourcePort(destination)},
            output_slots={"out": program.outputs[0]},
            banks={"gain": gain.bind(weight)},
        )
        for name, destination in (("left_node", "left"), ("right_node", "right"))
    )
    merge = node(
        "merge",
        lambda p, s, a, b, rng=None: (a + b, s),
        inputs={"a": "left", "b": "right"},
        outputs={"out": "out"},
    )
    graph = m.ProgramGraph(
        tuple(resource(name, jnp.zeros((1, 1))) for name in ("x", "left", "right", "out")),
        (),
        nodes=(*nodes, merge),
        programs={"main": (m.ProgramStage(("left_node", "right_node")), "merge")},
    )
    bound = aj.bind_resource_graph(graph, "main")
    inputs = {"x": jnp.ones((2, 1), dtype=jnp.float32)}
    p, s = bound.init(jax.random.key(12), inputs)
    assert len(p["shared"]) == 1
    assert p["nodes"]["left_node"] == p["nodes"]["right_node"] == {}
    gradients = jax.jit(
        jax.grad(lambda params: bound.apply(params, s, inputs).outputs["out"].sum())
    )(p)
    np.testing.assert_array_equal(next(iter(gradients["shared"].values())), 4)


def test_conditional_connection_native_lowering_uses_live_context():
    activation = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False), torch.nn.Sigmoid())
    with torch.no_grad():
        activation[0].weight.fill_(1)
    edge = m.Connection(
        "conditional", m.ResourcePort("x"), m.ResourcePort("out"), activation=activation
    )
    graph = m.ProgramGraph(
        (resource("x", jnp.ones((1, 1))), resource("out", jnp.zeros((1, 1)))),
        (edge,),
        programs={"main": ("conditional",)},
    )
    bound = aj.bind_resource_graph(
        graph, "main", example_contexts={"conditional": jnp.zeros((1, 1))}
    )
    p, s = bound.init(jax.random.key(13))
    apply = jax.jit(
        lambda context: bound.apply(p, s, contexts={"conditional": context}).outputs["out"]
    )
    np.testing.assert_allclose(apply(jnp.array([[0.0]], dtype=jnp.float32)), [[0.5]])
    np.testing.assert_allclose(apply(jnp.array([[1.0]], dtype=jnp.float32)), [[0.7310586]])
    assert float(jax.grad(lambda x: apply(x).sum())(jnp.zeros((1, 1), dtype=jnp.float32))[0, 0]) > 0


def test_bernoulli_boundary_replays_explicit_credit_mask_without_hidden_rng():
    import arti

    edge = m.Connection(
        "membrane",
        m.ResourcePort("x"),
        m.ResourcePort("out"),
        credit_boundary=arti.CreditBoundary(mode="bernoulli"),
    )
    graph = m.ProgramGraph(
        (resource("x", jnp.zeros((1, 2))), resource("out", jnp.zeros((1, 2)))),
        (edge,),
        programs={"main": ("membrane",)},
    )
    bound = aj.bind_resource_graph(graph, "main")
    value = jnp.ones((2, 2), dtype=jnp.float32)
    p, s = bound.init(jax.random.key(14), {"x": value})
    mask = jnp.array([False, True])
    gradient = jax.jit(
        jax.grad(
            lambda x: (
                bound.apply(p, s, {"x": x}, credit_masks={"membrane": mask}).outputs["out"].sum()
            )
        )
    )(value)
    np.testing.assert_array_equal(gradient, [[0, 0], [1, 1]])
    random_gradient = jax.jit(
        jax.grad(
            lambda x, key: bound.apply(p, s, {"x": x}, rng=key).outputs["out"].sum(), argnums=0
        )
    )
    np.testing.assert_array_equal(
        random_gradient(value, jax.random.key(15)), random_gradient(value, jax.random.key(15))
    )


def test_flax_mutable_collections_remain_explicit_forward_state():
    linen = pytest.importorskip("flax.linen")

    class Normalized(linen.Module):
        @linen.compact
        def __call__(self, x):
            return linen.BatchNorm(use_running_average=False, momentum=0.5)(x)

    module = aj.from_flax(Normalized(), inputs=("x",), outputs=("out",), mutable=("batch_stats",))
    transform = aj.as_fabric_node(
        "normalization",
        module,
        input_ports={"x": m.ResourcePort("x")},
        output_ports={"out": m.ResourcePort("out")},
    )
    graph = m.ProgramGraph(
        (resource("x", jnp.zeros((1, 2))), resource("out", jnp.zeros((1, 2)))),
        (),
        nodes=(transform,),
        programs={"main": ("normalization",)},
    )
    bound = aj.bind_resource_graph(graph, "main")
    value = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
    p, s = bound.init(jax.random.key(16), {"x": value})
    result = jax.jit(bound.apply)(p, s, {"x": value})
    mean = result.state.modules["nodes"]["normalization"]["batch_stats"]["BatchNorm_0"]["mean"]
    np.testing.assert_allclose(mean, [1, 1.5])
    assert not np.any(s.modules["nodes"]["normalization"]["batch_stats"]["BatchNorm_0"]["mean"])


def test_formula_validity_propagates_to_graph_receipt_and_integer_banks_are_state():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        x = m.InputBinding("x", m.TensorType(("B", "N", "D"), ("B", 3, 1), dtype="float32"))
        ids = m.BankBinding(
            "ids", "arti/jax-test@1", "indices", m.TensorType(("B", "K"), (1, 1), dtype="int64")
        )
        program = m.FormulaProgram.build(outputs=(m.gather(x, ids, axis="N", index_axis="K"),))
        transform = m.FormulaProgramNode(
            "gather",
            m.FormulaFabricV2(program),
            input_ports={"x": m.ResourcePort("x")},
            output_ports={"out": m.ResourcePort("out")},
            output_slots={"out": program.outputs[0]},
            banks={"ids": ids.bind(torch.tensor([[5]], dtype=torch.int64))},
        )
        graph = m.ProgramGraph(
            (
                resource("x", jnp.zeros((1, 3, 1), dtype=jnp.float32)),
                resource("out", jnp.zeros((1, 1, 1), dtype=jnp.float32)),
            ),
            (),
            nodes=(transform,),
            programs={"main": ("gather",)},
        )
        bound = aj.bind_resource_graph(graph, "main")
        p, s = bound.init(jax.random.key(17))
        assert not jax.tree_util.tree_leaves(p["nodes"]["gather"])
        result = jax.jit(bound.apply)(p, s)
        assert not bool(result.receipt["valid"])
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_flax_attachment_at_named_module_output_is_trainable_and_jitted():
    linen = pytest.importorskip("flax.linen")

    class Host(linen.Module):
        @linen.compact
        def __call__(self, x):
            x = linen.Dense(2, name="projection")(x)
            return linen.Dense(1, name="head")(jax.nn.tanh(x))

    transform = node(
        "adjust",
        lambda p, s, x, rng=None: (x * p["gain"], s),
        inputs={"x": "x"},
        outputs={"out": "out"},
        initializer=lambda key, x: ({"gain": jnp.ones((), dtype=x.dtype)}, {}),
    )
    graph = m.ProgramGraph(
        (resource("x", jnp.zeros((1, 2))), resource("out", jnp.zeros((1, 2)))),
        (),
        nodes=(transform,),
        programs={"main": ("adjust",)},
    )
    layer = aj.ARTILayer(
        aj.bind_resource_graph(graph, "main"), input_resource_id="x", output_resource_id="out"
    )
    model = aj.attach(Host(), layer, targets="projection")
    x = jnp.ones((4, 3), dtype=jnp.float32)
    variables = model.init(jax.random.key(18), x)
    assert "arti_projection" in variables["params"]
    output, updates = jax.jit(lambda v: model.apply(v, x, mutable=("arti_state",)))(variables)
    assert output.shape == (4, 1)
    assert "arti_projection" in updates["arti_state"]
    gradients = jax.jit(
        jax.grad(lambda p: jnp.square(model.apply({**variables, "params": p}, x)).mean())
    )(variables["params"])
    assert abs(float(gradients["arti_projection"]["program"]["nodes"]["adjust"]["gain"])) > 0


def test_inputs_mount_without_overwriting_persistent_default_backing():
    increment = node(
        "increment", lambda p, s, x, rng=None: (x + 1, s), inputs={"x": "x"}, outputs={"x": "x"}
    )
    graph = m.ProgramGraph(
        (resource("x", jnp.zeros((1, 1)), lifetime=m.ResourceLifetime.PERSISTENT),),
        (),
        nodes=(increment,),
        programs={"main": ("increment",)},
    )
    bound = aj.bind_resource_graph(graph, "main")
    p, s = bound.init(jax.random.key(19))
    result = jax.jit(bound.apply)(p, s, {"x": jnp.array([[10.0]], dtype=jnp.float32)})
    np.testing.assert_array_equal(result.outputs["x"], [[11]])
    np.testing.assert_array_equal(result.state.unmount("x").values["x"], [[0]])


def test_batch_route_inside_loop_ignores_stopped_rows():
    first = node(
        "increment_one", lambda p, s, x, rng=None: (x + 1, s), inputs={"x": "x"}, outputs={"x": "x"}
    )
    second = node(
        "increment_two", lambda p, s, x, rng=None: (x + 2, s), inputs={"x": "x"}, outputs={"x": "x"}
    )
    continuation = node(
        "continuation",
        lambda p, s, x, limit, rng=None: ((x < limit).astype(x.dtype)[:, 0], s),
        inputs={"x": "x", "limit": "limit"},
        outputs={"continue": "continue"},
    )
    graph = m.ProgramGraph(
        (
            resource("x", jnp.zeros((1, 1))),
            resource("limit", jnp.ones((1, 1))),
            resource("scores", jnp.zeros((1, 2))),
            resource("continue", jnp.ones((1,))),
        ),
        (),
        nodes=(first, second, continuation),
        programs={
            "body": (
                m.ProgramRoute("route", "scores", ("increment_one", "increment_two")),
                "continuation",
            )
        },
        loops=(m.ProgramLoop("loop", "body", "continue", max_iterations=4),),
    )
    bound = aj.bind_resource_graph(graph, "loop")
    inputs = {
        "x": jnp.zeros((2, 1), dtype=jnp.float32),
        "limit": jnp.array([[1], [2]], dtype=jnp.float32),
        "scores": jnp.array([[100, 0], [0, 1]], dtype=jnp.float32),
    }
    p, s = bound.init(jax.random.key(20), inputs)
    result = jax.jit(bound.apply)(p, s, inputs)
    np.testing.assert_array_equal(result.outputs["x"], [[1], [3]])
    np.testing.assert_array_equal(result.receipt["iterations"], [1, 2])


def test_join_fires_and_consumes_arrivals_per_row():
    merge = node(
        "merge",
        lambda p, s, left, right, rng=None: (left + right, s),
        inputs={"left": "left", "right": "right"},
        outputs={"out": "out"},
    )
    graph = m.ProgramGraph(
        tuple(
            resource(name, jnp.zeros((2, 1)), lifetime=m.ResourceLifetime.PERSISTENT)
            for name in ("left", "right", "out")
        ),
        (),
        nodes=(merge,),
        programs={"main": (m.ProgramJoin("join", "merge"),)},
    )
    bound = aj.bind_resource_graph(graph, "main")
    p, s = bound.init(jax.random.key(21))
    s = s._replace(
        values={
            **s.values,
            "left": jnp.array([[2], [0]], dtype=jnp.float32),
            "right": jnp.array([[20], [0]], dtype=jnp.float32),
        },
        epochs={**s.epochs, "left": jnp.array([1, 0]), "right": jnp.array([1, 0])},
    )
    result = jax.jit(bound.apply)(p, s)
    np.testing.assert_array_equal(result.outputs["out"], [[22], [0]])
    np.testing.assert_array_equal(result.state.joins["join"]["left"], [1, 0])
    np.testing.assert_array_equal(result.state.epochs["out"], [1, 0])


def test_frozen_float_formula_bank_stays_out_of_optimizer_parameters():
    x = m.InputBinding("x", m.TensorType(("B", "D"), ("B", 1), dtype="float32"))
    gain = m.BankBinding("gain", "arti/jax-test@1", "fixed", m.TensorType((), (), dtype="float32"))
    program = m.FormulaProgram.build(outputs=(m.scale(x, gain),))
    transform = m.FormulaProgramNode(
        "fixed",
        m.FormulaFabricV2(program),
        input_ports={"x": m.ResourcePort("x")},
        output_ports={"out": m.ResourcePort("out")},
        output_slots={"out": program.outputs[0]},
        banks={"gain": gain.bind(torch.tensor(2.0))},
    )
    graph = m.ProgramGraph(
        (resource("x", jnp.ones((1, 1))), resource("out", jnp.zeros((1, 1)))),
        (),
        nodes=(transform,),
        programs={"main": ("fixed",)},
    )
    bound = aj.bind_resource_graph(graph, "main")
    p, s = bound.init(jax.random.key(22))
    assert p["nodes"]["fixed"] == {}
    np.testing.assert_array_equal(s.modules["nodes"]["fixed"]["operands"]["gain"], 2)

    def loss(frozen):
        modules = {**s.modules, "nodes": {"fixed": {"operands": {"gain": frozen}}}}
        return bound.apply(p, s._replace(modules=modules)).outputs["out"].sum()

    np.testing.assert_array_equal(jax.grad(loss)(jnp.array(2.0)), 0)


@pytest.mark.parametrize("wrong", ("shape", "dtype"))
def test_native_publication_checks_compiled_resource_abi(wrong):
    operation = (lambda x: x.sum()) if wrong == "shape" else (lambda x: x.astype(jnp.float16))
    transform = node(
        "bad",
        lambda p, s, x, rng=None: (operation(x), s),
        inputs={"x": "x"},
        outputs={"out": "out"},
    )
    graph = m.ProgramGraph(
        (resource("x", jnp.ones((1, 2))), resource("out", jnp.zeros((1, 2)))),
        (),
        nodes=(transform,),
        programs={"main": ("bad",)},
    )
    bound = aj.bind_resource_graph(graph, "main")
    p, s = bound.init(jax.random.key(23))
    with pytest.raises(ValueError, match="compiled"):
        jax.jit(bound.apply)(p, s)


@pytest.mark.parametrize("regional", (False, True))
def test_identity_connection_transports_source_mask_atomically(regional):
    axes = (m.AxisDescriptor("B", "batch", 2), m.AxisDescriptor("D", "feature", 2))
    source = m.TensorResource(
        m.TensorResourceSpec("source", m.TensorViewPattern()),
        m.TensorView(torch.ones(2, 2), axes, mask=torch.tensor([[False, True], [True, False]])),
    )
    target = resource("target", jnp.zeros((2, 2)))
    ranges = (
        {
            "source_view": m.ResourceView("source", (m.AxisRange("D", 0, 1),)),
            "destination_view": m.ResourceView("target", (m.AxisRange("D", 1, 2),)),
        }
        if regional
        else {}
    )
    edge = m.Connection("copy", m.ResourcePort("source"), m.ResourcePort("target"), **ranges)
    graph = m.ProgramGraph((source, target), (edge,), programs={"main": ("copy",)})
    bound = aj.bind_resource_graph(graph, "main")
    p, s = bound.init(jax.random.key(24))
    result = jax.jit(bound.apply)(p, s)
    expected = [[True, False], [True, True]] if regional else [[False, True], [True, False]]
    np.testing.assert_array_equal(result.state.masks["target"], expected)


def test_parallel_shared_immutable_module_has_one_parameter_and_two_vjps():
    import arti

    @arti.fabric_layer(inputs=("x",), outputs={"y": "x"})
    class Linear(torch.nn.Linear):
        pass

    module = Linear(2, 2, bias=False)
    nodes = tuple(
        arti.as_fabric_node(
            name,
            module,
            input_ports={"x": m.ResourcePort("x")},
            output_ports={"y": m.ResourcePort(destination)},
        )
        for name, destination in (("left_node", "left"), ("right_node", "right"))
    )
    graph = m.ProgramGraph(
        tuple(resource(name, jnp.ones((1, 2))) for name in ("x", "left", "right")),
        (),
        nodes=nodes,
        programs={"main": (m.ProgramStage(("left_node", "right_node")),)},
    )
    bound = aj.bind_resource_graph(graph, "main")
    p, s = bound.init(jax.random.key(25))
    assert len(p["nodes"]) == 1
    gradients = jax.jit(
        jax.grad(
            lambda params: sum(
                value.sum() for name, value in bound.apply(params, s).outputs.items() if name != "x"
            )
        )
    )(p)
    np.testing.assert_array_equal(gradients["nodes"]["left_node"]["weight"], 2)


def test_repeated_bernoulli_edge_reuses_one_sample_per_program_invocation():
    import arti

    edge = m.Connection(
        "membrane",
        m.ResourcePort("x"),
        m.ResourcePort("y"),
        credit_boundary=arti.CreditBoundary(mode="bernoulli"),
    )
    accumulator = node(
        "accumulator",
        lambda p, s, y, total, rng=None: (y + total, s),
        inputs={"y": "y", "total": "total"},
        outputs={"total": "total"},
    )
    graph = m.ProgramGraph(
        tuple(resource(name, jnp.zeros((8, 8))) for name in ("x", "y", "total")),
        (edge,),
        nodes=(accumulator,),
        programs={"main": ("membrane", "accumulator", "membrane", "accumulator")},
    )
    bound = aj.bind_resource_graph(graph, "main")
    p, s = bound.init(jax.random.key(26))
    gradient = jax.jit(
        jax.grad(
            lambda x: bound.apply(p, s, {"x": x}, rng=jax.random.key(27)).outputs["total"].sum()
        )
    )(jnp.ones((8, 8)))
    assert set(np.unique(gradient)).issubset({0, 2})


def test_native_transfer_explicit_mask_overrides_source_visibility():
    edge = m.Connection(
        "copy", m.ResourcePort("source"), m.ResourcePort("target"), transfer=torch.nn.Identity()
    )
    graph = m.ProgramGraph(
        (resource("source", jnp.ones((1, 2))), resource("target", jnp.zeros((1, 2)))),
        (edge,),
        programs={"main": ("copy",)},
    )
    module = aj.NativeModule(
        lambda key, x: ({}, {}),
        lambda p, s, x, rng=None: aj.NativeModuleResult(
            x, s, jnp.asarray(True), {"value": jnp.zeros(x.shape, dtype=bool)}
        ),
        mutable_state=False,
    )
    bound = aj.bind_resource_graph(graph, "main", native_transfers={"copy": module})
    p, s = bound.init(jax.random.key(28))
    result = jax.jit(bound.apply)(p, s)
    np.testing.assert_array_equal(result.state.masks["target"], False)


def test_half_layer_norm_accumulates_in_float32_and_matches_weight_vjp():
    import arti

    @arti.fabric_layer(inputs=("x",), outputs={"out": "x"})
    class LayerNorm(torch.nn.LayerNorm):
        pass

    norm = LayerNorm(2).half()
    transform = arti.as_fabric_node(
        "normalization",
        norm,
        input_ports={"x": m.ResourcePort("x")},
        output_ports={"out": m.ResourcePort("out")},
    )
    graph = m.ProgramGraph(
        (
            resource("x", jnp.zeros((1, 2), dtype=jnp.float16)),
            resource("out", jnp.zeros((1, 2), dtype=jnp.float16)),
        ),
        (),
        nodes=(transform,),
        programs={"main": ("normalization",)},
    )
    bound = aj.bind_resource_graph(graph, "main")
    value = jnp.array([[300, -300]], dtype=jnp.float16)
    p, s = bound.init(jax.random.key(29))
    result = jax.jit(bound.apply)(p, s, {"x": value})
    expected = norm(torch.tensor([[300, -300]], dtype=torch.float16))
    np.testing.assert_allclose(result.outputs["out"], expected.detach().numpy(), atol=1e-3)
    cotangent = jnp.array([[1, -1]], dtype=jnp.float16)
    gradients = jax.jit(
        jax.grad(
            lambda params: (bound.apply(params, s, {"x": value}).outputs["out"] * cotangent).sum()
        )
    )(p)
    (expected * torch.tensor([[1, -1]], dtype=torch.float16)).sum().backward()
    np.testing.assert_allclose(
        gradients["nodes"]["normalization"]["weight"], norm.weight.grad.numpy(), atol=1e-3
    )
