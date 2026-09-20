from __future__ import annotations

import torch
import pytest
from torch import Tensor, nn

import arti
from arti import mechanisms


def view(values: list[list[float]]) -> mechanisms.TensorView:
    return mechanisms.TensorView.from_tensor(
        torch.tensor(values, dtype=torch.float32).unsqueeze(-1),
        axis_names=("batch", "token", "feature"),
        axis_roles=("batch", "sequence", "feature"),
    )


def device_view(values: list[list[float]], device: torch.device) -> mechanisms.TensorView:
    source = view(values)
    return mechanisms.TensorView(source.value.to(device), source.axes)


def spec(resource_id: str, *, lifetime: mechanisms.ResourceLifetime) -> mechanisms.TensorResourceSpec:
    return mechanisms.TensorResourceSpec(
        resource_id,
        mechanisms.TensorViewPattern(
            min_rank=3,
            max_rank=3,
            allowed_axis_roles=("batch", "sequence", "feature"),
        ),
        lifetime=lifetime,
        axis_capacity=(("token", 8),),
    )


def test_resource_default_mount_restore_and_fork_keep_storage_separate() -> None:
    resource = mechanisms.TensorResource(spec("memory", lifetime=mechanisms.ResourceLifetime.PERSISTENT), view([[0, 0]]))
    mounted = view([[1, 2]])
    resource.mount(mounted)
    resource.advance(view([[3, 4]]))
    restored = mechanisms.TensorResource.restore(resource.state())
    fork = restored.fork()
    fork.advance(view([[5, 6]]))

    assert restored.resolve().binding.source == "external"
    assert restored.resolve().binding.step_index == 1
    torch.testing.assert_close(restored.resolve().view.value, view([[3, 4]]).value)
    torch.testing.assert_close(fork.resolve().view.value, view([[5, 6]]).value)
    torch.testing.assert_close(restored.resolve().view.value, view([[3, 4]]).value)


def test_graph_state_restores_resource_backings_into_a_fresh_declared_graph() -> None:
    source = mechanisms.TensorResource(
        spec("source", lifetime=mechanisms.ResourceLifetime.CALL), view([[2, 7]])
    )
    memory = mechanisms.TensorResource(
        spec("memory", lifetime=mechanisms.ResourceLifetime.PERSISTENT), view([[0, 0]])
    )
    connection = mechanisms.Connection(
        "copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("memory")
    )
    graph = mechanisms.ProgramGraph((source, memory), (connection,))
    graph.execute(("copy",))
    saved = graph.state()

    fresh = mechanisms.ProgramGraph(
        (
            mechanisms.TensorResource(
                spec("source", lifetime=mechanisms.ResourceLifetime.CALL), view([[9, 9]])
            ),
            mechanisms.TensorResource(
                spec("memory", lifetime=mechanisms.ResourceLifetime.PERSISTENT), view([[0, 0]])
            ),
        ),
        (mechanisms.Connection("copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("memory")),),
    )
    fresh.restore_state(saved)

    assert fresh.contract_fingerprint == saved.contract_fingerprint
    torch.testing.assert_close(fresh.resource("memory").resolve().view.value, view([[2, 7]]).value)


def test_direct_connection_copies_a_whole_resource_without_a_query() -> None:
    source = mechanisms.TensorResource(spec("world", lifetime=mechanisms.ResourceLifetime.CALL), view([[2, 7]]))
    target = mechanisms.TensorResource(spec("workspace", lifetime=mechanisms.ResourceLifetime.PERSISTENT), view([[0, 0]]))
    connection = mechanisms.Connection(
        "copy_world",
        mechanisms.ResourcePort("world"),
        mechanisms.ResourcePort("workspace"),
    )
    graph = mechanisms.ProgramGraph((source, target), (connection,))

    receipt = graph.execute(("copy_world",))

    assert receipt[0].source.resource_id == "world"
    assert receipt[0].destination.resource_id == "workspace"
    assert not connection.is_conditional
    torch.testing.assert_close(target.resolve().view.value, source.resolve().view.value)


def test_named_resource_program_preserves_serial_visibility_and_can_repeat() -> None:
    source = mechanisms.TensorResource(
        spec("source", lifetime=mechanisms.ResourceLifetime.CALL), view([[4, 8]])
    )
    workspace = mechanisms.TensorResource(
        spec("workspace", lifetime=mechanisms.ResourceLifetime.STATE), view([[0, 0]])
    )
    output = mechanisms.TensorResource(
        spec("output", lifetime=mechanisms.ResourceLifetime.CALL), view([[0, 0]])
    )
    graph = mechanisms.ProgramGraph(
        (source, workspace, output),
        (
            mechanisms.Connection("write", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("workspace")),
            mechanisms.Connection("read", mechanisms.ResourcePort("workspace"), mechanisms.ResourcePort("output")),
        ),
        programs={"write_then_read": ("write", "read")},
    )

    first = graph.execute_program("write_then_read")
    source.advance(view([[3, 5]]))
    repeats = graph.execute_repeated("write_then_read", iterations=2)
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "write_then_read")
    compiled = torch.compile(plan, backend="eager", fullgraph=True)
    compiled_values = compiled(
        source.resolve().view.value,
        workspace.resolve().view.value,
        output.resolve().view.value,
    )

    assert tuple(item.connection_id for item in first) == ("write", "read")
    assert len(repeats) == 2
    torch.testing.assert_close(workspace.resolve().view.value, view([[3, 5]]).value)
    torch.testing.assert_close(output.resolve().view.value, view([[3, 5]]).value)
    torch.testing.assert_close(compiled_values[2], view([[3, 5]]).value)


def test_compiled_program_unrolls_a_fixed_local_iteration_horizon() -> None:
    state = mechanisms.TensorResource(
        spec("state", lifetime=mechanisms.ResourceLifetime.STATE), view([[3, 4]])
    )
    graph = mechanisms.ProgramGraph(
        (state,),
        (
            mechanisms.Connection(
                "advance",
                mechanisms.ResourcePort("state"),
                mechanisms.ResourcePort("state"),
                transfer=mechanisms.LearnableAffineTransfer(
                    gain=1.0,
                    bias=1.0,
                    learnable=False,
                ),
            ),
        ),
        programs={"iterate": ("advance",)},
    )

    graph.execute_repeated("iterate", iterations=3)
    plan = mechanisms.ResourceGraphCompiler.compile_program(
        graph,
        "iterate",
        iterations=3,
    )
    torch._dynamo.reset()
    compiled = torch.compile(plan, backend="eager", fullgraph=True)
    initial = view([[3, 4]]).value
    (output,) = compiled(initial)

    torch.testing.assert_close(state.resolve().view.value, view([[6, 7]]).value)
    torch.testing.assert_close(output, view([[6, 7]]).value)
    assert tuple(connection.connection_id for connection in plan.connections) == (
        "advance",
        "advance",
        "advance",
    )


def test_connection_dependencies_define_state_visibility_for_execution_and_compilation() -> None:
    source = mechanisms.TensorResource(
        spec("source", lifetime=mechanisms.ResourceLifetime.CALL), view([[4, 8]])
    )
    workspace = mechanisms.TensorResource(
        spec("workspace", lifetime=mechanisms.ResourceLifetime.STATE), view([[0, 0]])
    )
    output = mechanisms.TensorResource(
        spec("output", lifetime=mechanisms.ResourceLifetime.CALL), view([[0, 0]])
    )
    write = mechanisms.Connection(
        "write", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("workspace")
    )
    read = mechanisms.Connection(
        "read",
        mechanisms.ResourcePort("workspace"),
        mechanisms.ResourcePort("output"),
        depends_on=("write",),
    )
    graph = mechanisms.ProgramGraph((source, workspace, output), (write, read))

    try:
        graph.execute(("read", "write"))
    except mechanisms.ResourceGraphError as error:
        assert "requires prior dependencies" in str(error)
    else:
        raise AssertionError("a dependent read must not observe a missing prior write")
    try:
        mechanisms.ResourceGraphCompiler.compile(graph, ("read", "write"))
    except mechanisms.ResourceGraphError as error:
        assert "requires prior dependencies" in str(error)
    else:
        raise AssertionError("the compiler must preserve declared connection order")

    graph.execute(("write", "read"))
    torch.testing.assert_close(output.resolve().view.value, view([[4, 8]]).value)


def test_functional_graph_execution_keeps_branch_state_out_of_the_live_graph() -> None:
    source = mechanisms.TensorResource(
        spec("source", lifetime=mechanisms.ResourceLifetime.CALL), view([[4, 8]])
    )
    workspace = mechanisms.TensorResource(
        spec("workspace", lifetime=mechanisms.ResourceLifetime.STATE), view([[0, 0]])
    )
    output = mechanisms.TensorResource(
        spec("output", lifetime=mechanisms.ResourceLifetime.CALL), view([[0, 0]])
    )
    graph = mechanisms.ProgramGraph(
        (source, workspace, output),
        (
            mechanisms.Connection(
                "write", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("workspace")
            ),
            mechanisms.Connection(
                "read",
                mechanisms.ResourcePort("workspace"),
                mechanisms.ResourcePort("output"),
                depends_on=("write",),
            ),
        ),
    )

    execution = graph.execute_functional(("write", "read"))

    assert tuple(item.connection_id for item in execution.connections) == ("write", "read")
    torch.testing.assert_close(workspace.resolve().view.value, view([[0, 0]]).value)
    torch.testing.assert_close(output.resolve().view.value, view([[0, 0]]).value)
    graph.restore_state(execution.state)
    torch.testing.assert_close(graph.resource("workspace").resolve().view.value, view([[4, 8]]).value)
    torch.testing.assert_close(graph.resource("output").resolve().view.value, view([[4, 8]]).value)


def test_functional_graph_input_views_are_branch_local_hot_mounts() -> None:
    world = mechanisms.TensorResource(
        spec("world", lifetime=mechanisms.ResourceLifetime.CALL), view([[2, 7]])
    )
    workspace = mechanisms.TensorResource(
        spec("workspace", lifetime=mechanisms.ResourceLifetime.PERSISTENT), view([[0, 0]])
    )
    graph = mechanisms.ProgramGraph(
        (world, workspace),
        (
            mechanisms.Connection(
                "copy",
                mechanisms.ResourcePort("world"),
                mechanisms.ResourcePort("workspace"),
            ),
        ),
    )

    branch = graph.execute_functional(
        ("copy",), input_views={"world": view([[11, 13]])}
    )

    torch.testing.assert_close(world.resolve().view.value, view([[2, 7]]).value)
    branch_workspace = next(
        item for item in branch.state.resources if item.spec.resource_id == "workspace"
    )
    torch.testing.assert_close(branch_workspace.active_view.value, view([[11, 13]]).value)
    graph.restore_state(branch.state)
    torch.testing.assert_close(
        graph.resource("workspace").resolve().view.value,
        view([[11, 13]]).value,
    )


def test_functional_subprogram_invocation_binds_its_declared_input_and_output() -> None:
    world = mechanisms.TensorResource(
        spec("world", lifetime=mechanisms.ResourceLifetime.CALL), view([[0, 0]])
    )
    workspace = mechanisms.TensorResource(
        spec("workspace", lifetime=mechanisms.ResourceLifetime.PERSISTENT), view([[0, 0]])
    )
    output = mechanisms.TensorResource(
        spec("output", lifetime=mechanisms.ResourceLifetime.CALL), view([[0, 0]])
    )
    graph = mechanisms.ProgramGraph(
        (world, workspace, output),
        (
            mechanisms.Connection(
                "write",
                mechanisms.ResourcePort("world"),
                mechanisms.ResourcePort("workspace"),
            ),
            mechanisms.Connection(
                "read",
                mechanisms.ResourcePort("workspace"),
                mechanisms.ResourcePort("output"),
                depends_on=("write",),
            ),
        ),
        programs={"copy_through_workspace": ("write", "read")},
    )

    invocation = graph.invoke_functional(
        graph.program("copy_through_workspace"),
        input_resource_id="world",
        input_view=view([[3, 5]]),
        output_resource_id="output",
    )

    torch.testing.assert_close(invocation.output.value, view([[3, 5]]).value)
    assert tuple(item.connection_id for item in invocation.connections) == ("write", "read")
    torch.testing.assert_close(world.resolve().view.value, view([[0, 0]]).value)


def test_connection_moves_a_multi_element_region_and_a_later_edge_reads_it() -> None:
    world = mechanisms.TensorResource(
        spec("world", lifetime=mechanisms.ResourceLifetime.CALL), view([[1, 2, 3, 4]])
    )
    memory = mechanisms.TensorResource(
        spec("memory", lifetime=mechanisms.ResourceLifetime.PERSISTENT),
        view([[0, 0, 0, 0, 0]]),
    )
    output = mechanisms.TensorResource(
        spec("output", lifetime=mechanisms.ResourceLifetime.CALL), view([[0, 0]])
    )
    world_middle = mechanisms.ResourceView("world", (mechanisms.AxisRange("token", 1, 3),))
    memory_middle = mechanisms.ResourceView("memory", (mechanisms.AxisRange("token", 2, 4),))
    write = mechanisms.Connection(
        "copy_region",
        mechanisms.ResourcePort("world"),
        mechanisms.ResourcePort("memory"),
        source_view=world_middle,
        destination_view=memory_middle,
    )
    read = mechanisms.Connection(
        "read_region",
        mechanisms.ResourcePort("memory"),
        mechanisms.ResourcePort("output"),
        source_view=memory_middle,
    )
    graph = mechanisms.ProgramGraph((world, memory, output), (write, read))

    graph.execute(("copy_region", "read_region"))

    torch.testing.assert_close(memory.resolve().view.value, view([[0, 0, 2, 3, 0]]).value)
    torch.testing.assert_close(output.resolve().view.value, view([[2, 3]]).value)


def test_compiled_connection_preserves_declared_region_write_and_read() -> None:
    world = mechanisms.TensorResource(
        spec("world", lifetime=mechanisms.ResourceLifetime.CALL), view([[1, 2, 3, 4]])
    )
    memory = mechanisms.TensorResource(
        spec("memory", lifetime=mechanisms.ResourceLifetime.PERSISTENT),
        view([[0, 0, 0, 0, 0]]),
    )
    output = mechanisms.TensorResource(
        spec("output", lifetime=mechanisms.ResourceLifetime.CALL), view([[0, 0]])
    )
    world_middle = mechanisms.ResourceView("world", (mechanisms.AxisRange("token", 1, 3),))
    memory_middle = mechanisms.ResourceView("memory", (mechanisms.AxisRange("token", 2, 4),))
    graph = mechanisms.ProgramGraph(
        (world, memory, output),
        (
            mechanisms.Connection(
                "copy_region",
                mechanisms.ResourcePort("world"),
                mechanisms.ResourcePort("memory"),
                source_view=world_middle,
                destination_view=memory_middle,
            ),
            mechanisms.Connection(
                "read_region",
                mechanisms.ResourcePort("memory"),
                mechanisms.ResourcePort("output"),
                source_view=memory_middle,
            ),
        ),
    )

    plan = mechanisms.ResourceGraphCompiler.compile(graph, ("copy_region", "read_region"))
    compiled = torch.compile(plan, backend="eager", fullgraph=True)
    values = compiled(
        world.resolve().view.value,
        memory.resolve().view.value,
        output.resolve().view.value,
    )

    torch.testing.assert_close(values[1], view([[0, 0, 2, 3, 0]]).value)
    torch.testing.assert_close(values[2], view([[2, 3]]).value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_region_connection_executes_on_cuda() -> None:
    device = torch.device("cuda")
    source = mechanisms.TensorResource(
        spec("source", lifetime=mechanisms.ResourceLifetime.CALL),
        device_view([[1, 2, 3, 4]], device),
    )
    target = mechanisms.TensorResource(
        spec("target", lifetime=mechanisms.ResourceLifetime.STATE),
        device_view([[0, 0, 0, 0, 0]], device),
    )
    graph = mechanisms.ProgramGraph(
        (source, target),
        (
            mechanisms.Connection(
                "copy_region",
                mechanisms.ResourcePort("source"),
                mechanisms.ResourcePort("target"),
                source_view=mechanisms.ResourceView(
                    "source", (mechanisms.AxisRange("token", 1, 3),)
                ),
                destination_view=mechanisms.ResourceView(
                    "target", (mechanisms.AxisRange("token", 2, 4),)
                ),
            ),
        ),
    )

    plan = mechanisms.ResourceGraphCompiler.compile(graph, ("copy_region",))
    compiled = torch.compile(plan, backend="eager", fullgraph=True)
    output = compiled(source.resolve().view.value, target.resolve().view.value)[1]

    assert output.device.type == "cuda"
    torch.testing.assert_close(output.cpu(), view([[0, 0, 2, 3, 0]]).value)


class _IdentityView(nn.Module):
    def forward(self, source: mechanisms.TensorView) -> mechanisms.TensorView:
        return source


class _FirstFeatureGate(nn.Module):
    def forward(self, context: Tensor) -> Tensor:
        return context[:, :1].unsqueeze(-1)


class _TrainableGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.5))

    def forward(self, context: Tensor) -> Tensor:
        return self.gain.reshape(1, 1, 1).expand(context.shape[0], 1, 1)


def test_conditional_connection_uses_only_supplied_local_context() -> None:
    source = mechanisms.TensorResource(spec("source", lifetime=mechanisms.ResourceLifetime.CALL), view([[4, 8]]))
    target = mechanisms.TensorResource(spec("target", lifetime=mechanisms.ResourceLifetime.STATE), view([[0, 0]]))
    connection = mechanisms.Connection(
        "local_gate",
        mechanisms.ResourcePort("source"),
        mechanisms.ResourcePort("target"),
        transfer=_IdentityView(),
        activation=_FirstFeatureGate(),
    )
    graph = mechanisms.ProgramGraph((source, target), (connection,))

    graph.execute(("local_gate",), contexts={"local_gate": torch.tensor([[0.25]])})
    torch.testing.assert_close(target.resolve().view.value, view([[1, 2]]).value)
    graph.execute(("local_gate",), contexts={"local_gate": torch.tensor([[0.5]])})
    torch.testing.assert_close(target.resolve().view.value, view([[2, 4]]).value)


def test_continuous_connection_parameters_receive_the_final_output_gradient() -> None:
    source = mechanisms.TensorResource(spec("source", lifetime=mechanisms.ResourceLifetime.CALL), view([[2, 6]]))
    target = mechanisms.TensorResource(spec("target", lifetime=mechanisms.ResourceLifetime.STATE), view([[0, 0]]))
    gate = _TrainableGate()
    graph = mechanisms.ProgramGraph(
        (source, target),
        (
            mechanisms.Connection(
                "learned_gate",
                mechanisms.ResourcePort("source"),
                mechanisms.ResourcePort("target"),
                activation=gate,
            ),
        ),
    )

    graph.execute(("learned_gate",), contexts={"learned_gate": torch.ones(1, 1)})
    target.resolve().view.value.square().mean().backward()

    assert gate.gain.grad is not None
    assert gate.gain.grad.item() > 0


def test_input_independent_learnable_transfer_trains_without_a_query() -> None:
    source = mechanisms.TensorResource(
        spec("source", lifetime=mechanisms.ResourceLifetime.CALL), view([[2, 6]])
    )
    target = mechanisms.TensorResource(
        spec("target", lifetime=mechanisms.ResourceLifetime.STATE), view([[0, 0]])
    )
    transfer = mechanisms.LearnableAffineTransfer(gain=0.25, bias=0.0)
    graph = mechanisms.ProgramGraph(
        (source, target),
        (
            mechanisms.Connection(
                "learned_direct",
                mechanisms.ResourcePort("source"),
                mechanisms.ResourcePort("target"),
                transfer=transfer,
            ),
        ),
    )
    plan = mechanisms.ResourceGraphCompiler.compile(graph, ("learned_direct",))
    optimizer = torch.optim.Adam(plan.parameters(), lr=0.05)
    source_value = source.resolve().view.value
    target_value = target.resolve().view.value
    initial_loss = None
    for _ in range(128):
        optimizer.zero_grad(set_to_none=True)
        predicted = plan(source_value, target_value)[1]
        loss = (predicted - source_value).square().mean()
        if initial_loss is None:
            initial_loss = loss.detach()
        loss.backward()
        optimizer.step()

    predicted = plan(source_value, target_value)[1]
    assert initial_loss is not None
    assert (predicted - source_value).square().mean() < initial_loss * 1e-3
    assert graph.contract_config()["connections"][0]["transfer_kind"] == (
        "module:arti.resource_graph.LearnableAffineTransfer"
    )
    assert arti.component_ref(transfer).startswith("arti/affine-resource-transfer@sha256:")


def test_final_resource_loss_trains_a_connection_without_an_internal_teacher() -> None:
    source = mechanisms.TensorResource(
        spec("source", lifetime=mechanisms.ResourceLifetime.CALL), view([[2, 6]])
    )
    target = mechanisms.TensorResource(
        spec("target", lifetime=mechanisms.ResourceLifetime.STATE), view([[0, 0]])
    )
    gate = _TrainableGate()
    graph = mechanisms.ProgramGraph(
        (source, target),
        (
            mechanisms.Connection(
                "learned_relation",
                mechanisms.ResourcePort("source"),
                mechanisms.ResourcePort("target"),
                activation=gate,
            ),
        ),
    )
    plan = mechanisms.ResourceGraphCompiler.compile(
        graph,
        ("learned_relation",),
        example_contexts={"learned_relation": torch.ones(1, 1)},
    )
    optimizer = torch.optim.SGD(plan.parameters(), lr=0.02)
    source_value = source.resolve().view.value
    target_value = target.resolve().view.value
    context = torch.ones(1, 1)
    for _ in range(16):
        optimizer.zero_grad(set_to_none=True)
        predicted = plan(source_value, target_value, context)[1]
        loss = (predicted - source_value).square().mean()
        loss.backward()
        optimizer.step()

    predicted = plan(source_value, target_value, context)[1]
    torch.testing.assert_close(predicted, source_value, atol=2e-2, rtol=0)


def test_resource_view_preserves_source_coordinates_for_a_range() -> None:
    resource = mechanisms.TensorResource(spec("memory", lifetime=mechanisms.ResourceLifetime.PERSISTENT), view([[1, 2, 3, 4]]))

    region = mechanisms.ResourceView("memory", (mechanisms.AxisRange("token", 1, 4, 2),)).resolve(resource.resolve())

    assert region.value.shape == (1, 2, 1)
    torch.testing.assert_close(region.value, view([[2, 4]]).value)
    assert region.axes[1].origin == 1.0
    assert region.axes[1].scale == 2.0
    assert region.index_map is not None
    assert region.index_map.coordinates.tolist() == [[[1, 0]], [[3, 0]]]


def test_parallel_connections_read_one_common_snapshot_and_need_a_merge() -> None:
    source = mechanisms.TensorResource(spec("source", lifetime=mechanisms.ResourceLifetime.CALL), view([[2, 3]]))
    target = mechanisms.TensorResource(spec("target", lifetime=mechanisms.ResourceLifetime.STATE), view([[0, 0]]))
    first = mechanisms.Connection("first", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("target"))
    second = mechanisms.Connection("second", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("target"))
    graph = mechanisms.ProgramGraph((source, target), (first, second))

    try:
        graph.execute_parallel(("first", "second"))
    except mechanisms.ResourceGraphError as error:
        assert "explicit merge" in str(error)
    else:
        raise AssertionError("parallel writes without a merge must fail")

    graph.execute_parallel(
        ("first", "second"),
        merge={
            "target": lambda _previous, outputs: mechanisms.TensorView(
                outputs[0].value + outputs[1].value,
                outputs[0].axes,
                index_map=outputs[0].index_map,
                mask=outputs[0].mask,
            )
        },
    )
    torch.testing.assert_close(target.resolve().view.value, view([[4, 6]]).value)


def test_functional_parallel_connections_keep_their_common_snapshot_branch_local() -> None:
    source = mechanisms.TensorResource(
        spec("source", lifetime=mechanisms.ResourceLifetime.CALL), view([[2, 3]])
    )
    target = mechanisms.TensorResource(
        spec("target", lifetime=mechanisms.ResourceLifetime.STATE), view([[0, 0]])
    )
    first = mechanisms.Connection("first", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("target"))
    second = mechanisms.Connection("second", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("target"))
    graph = mechanisms.ProgramGraph((source, target), (first, second))

    branch = graph.execute_parallel_functional(
        ("first", "second"),
        merge={
            "target": lambda _previous, outputs: mechanisms.TensorView(
                outputs[0].value + outputs[1].value,
                outputs[0].axes,
                index_map=outputs[0].index_map,
                mask=outputs[0].mask,
            )
        },
    )

    torch.testing.assert_close(target.resolve().view.value, view([[0, 0]]).value)
    branch_target = next(item for item in branch.state.resources if item.spec.resource_id == "target")
    torch.testing.assert_close(branch_target.active_view.value, view([[4, 6]]).value)


def test_connection_reuses_the_existing_typed_formula_fabric() -> None:
    value_type = mechanisms.TensorType.axes(
        ("batch", "token", "feature"), sizes=("B", "S", 1)
    )
    source_binding = mechanisms.InputBinding("source", value_type)
    program = mechanisms.FormulaProgram.build(outputs=(mechanisms.add(source_binding, source_binding),))
    transfer = mechanisms.FormulaTensorViewTransfer(
        mechanisms.FormulaFabricV2(program), source_input="source"
    )
    source = mechanisms.TensorResource(spec("source", lifetime=mechanisms.ResourceLifetime.CALL), view([[3, 5]]))
    target = mechanisms.TensorResource(spec("target", lifetime=mechanisms.ResourceLifetime.STATE), view([[0, 0]]))
    graph = mechanisms.ProgramGraph(
        (source, target),
        (
            mechanisms.Connection(
                "formula_double",
                mechanisms.ResourcePort("source"),
                mechanisms.ResourcePort("target"),
                transfer=transfer,
            ),
        ),
    )

    graph.execute(("formula_double",))

    torch.testing.assert_close(target.resolve().view.value, view([[6, 10]]).value)

    plan = mechanisms.ResourceGraphCompiler.compile(graph, ("formula_double",))
    compiled = torch.compile(plan, backend="eager", fullgraph=True)
    outputs = compiled(source.resolve().view.value, view([[0, 0]]).value)
    torch.testing.assert_close(outputs[1], view([[6, 10]]).value)


def test_static_formula_bindings_are_owned_by_the_lowered_connection_plan() -> None:
    value_type = mechanisms.TensorType.axes(
        ("batch", "token", "feature"), sizes=("B", "S", 1)
    )
    source_binding = mechanisms.InputBinding("source", value_type)
    offset_binding = mechanisms.InputBinding("offset", value_type)
    program = mechanisms.FormulaProgram.build(outputs=(mechanisms.add(source_binding, offset_binding),))
    transfer = mechanisms.FormulaTensorViewTransfer(
        mechanisms.FormulaFabricV2(program),
        source_input="source",
        static_inputs={"offset": view([[10, 20]]).value},
    )
    lowered = transfer.lower_static()

    assert "_static_binding_1" in lowered.state_dict()
    torch.testing.assert_close(lowered(view([[1, 2]])).value, view([[11, 22]]).value)

    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    lowered.to(device)
    result = lowered(device_view([[1, 2]], device))
    assert result.value.device.type == "cuda"
    torch.testing.assert_close(result.value.cpu(), view([[11, 22]]).value)


def test_formula_connection_transforms_one_resource_region_into_another() -> None:
    value_type = mechanisms.TensorType.axes(
        ("batch", "token", "feature"), sizes=("B", "S", 1)
    )
    source_binding = mechanisms.InputBinding("source", value_type)
    program = mechanisms.FormulaProgram.build(outputs=(mechanisms.add(source_binding, source_binding),))
    transfer = mechanisms.FormulaTensorViewTransfer(
        mechanisms.FormulaFabricV2(program), source_input="source"
    )
    source = mechanisms.TensorResource(
        spec("source", lifetime=mechanisms.ResourceLifetime.CALL), view([[1, 2, 3, 4]])
    )
    target = mechanisms.TensorResource(
        spec("target", lifetime=mechanisms.ResourceLifetime.STATE), view([[0, 0, 0, 0, 0]])
    )
    source_region = mechanisms.ResourceView("source", (mechanisms.AxisRange("token", 1, 3),))
    target_region = mechanisms.ResourceView("target", (mechanisms.AxisRange("token", 2, 4),))
    graph = mechanisms.ProgramGraph(
        (source, target),
        (
            mechanisms.Connection(
                "double_region",
                mechanisms.ResourcePort("source"),
                mechanisms.ResourcePort("target"),
                source_view=source_region,
                destination_view=target_region,
                transfer=transfer,
            ),
        ),
    )

    graph.execute(("double_region",))
    plan = mechanisms.ResourceGraphCompiler.compile(graph, ("double_region",))
    compiled = torch.compile(plan, backend="eager", fullgraph=True)
    outputs = compiled(source.resolve().view.value, view([[0, 0, 0, 0, 0]]).value)

    expected = view([[0, 0, 4, 6, 0]]).value
    torch.testing.assert_close(target.resolve().view.value, expected)
    torch.testing.assert_close(outputs[1], expected)


def test_formula_connection_binds_multiple_resource_inputs_through_one_relation() -> None:
    value_type = mechanisms.TensorType.axes(
        ("batch", "token", "feature"), sizes=("B", "S", 1)
    )
    source_binding = mechanisms.InputBinding("source", value_type)
    other_binding = mechanisms.InputBinding("other", value_type)
    program = mechanisms.FormulaProgram.build(outputs=(mechanisms.add(source_binding, other_binding),))
    transfer = mechanisms.FormulaTensorViewTransfer(
        mechanisms.FormulaFabricV2(program),
        source_input="source",
        dynamic_inputs=("other",),
    )
    source = mechanisms.TensorResource(
        spec("source", lifetime=mechanisms.ResourceLifetime.CALL), view([[1, 2]])
    )
    other = mechanisms.TensorResource(
        spec("other", lifetime=mechanisms.ResourceLifetime.PERSISTENT), view([[10, 20]])
    )
    target = mechanisms.TensorResource(
        spec("target", lifetime=mechanisms.ResourceLifetime.STATE), view([[0, 0]])
    )
    graph = mechanisms.ProgramGraph(
        (source, other, target),
        (
            mechanisms.Connection(
                "sum_resources",
                mechanisms.ResourcePort("source"),
                mechanisms.ResourcePort("target"),
                operand_views={"other": mechanisms.ResourceView("other")},
                transfer=transfer,
            ),
        ),
    )

    graph.execute(("sum_resources",))
    plan = mechanisms.ResourceGraphCompiler.compile(graph, ("sum_resources",))
    compiled = torch.compile(plan, backend="eager", fullgraph=True)
    outputs = compiled(
        source.resolve().view.value,
        other.resolve().view.value,
        view([[0, 0]]).value,
    )

    expected = view([[11, 22]]).value
    torch.testing.assert_close(target.resolve().view.value, expected)
    torch.testing.assert_close(outputs[2], expected)


def test_static_direct_connections_lower_to_a_functional_compilable_plan() -> None:
    source = mechanisms.TensorResource(spec("source", lifetime=mechanisms.ResourceLifetime.CALL), view([[3, 7]]))
    target = mechanisms.TensorResource(spec("target", lifetime=mechanisms.ResourceLifetime.STATE), view([[0, 0]]))
    graph = mechanisms.ProgramGraph(
        (source, target),
        (
            mechanisms.Connection(
                "copy",
                mechanisms.ResourcePort("source"),
                mechanisms.ResourcePort("target"),
                transfer=nn.Identity(),
            ),
        ),
    )

    plan = mechanisms.ResourceGraphCompiler.compile(graph, ("copy",))
    compiled = torch.compile(plan, backend="eager", fullgraph=True)
    source_value = source.resolve().view.value.clone().requires_grad_()
    target_value = target.resolve().view.value
    outputs = compiled(source_value, target_value)

    torch.testing.assert_close(outputs[1], source_value)
    torch.testing.assert_close(target.resolve().view.value, target_value)
    outputs[1].square().mean().backward()
    assert source_value.grad is not None


def test_compiler_keeps_conditional_connections_dynamic_on_device() -> None:
    source = mechanisms.TensorResource(spec("source", lifetime=mechanisms.ResourceLifetime.CALL), view([[1, 2]]))
    target = mechanisms.TensorResource(spec("target", lifetime=mechanisms.ResourceLifetime.STATE), view([[0, 0]]))
    graph = mechanisms.ProgramGraph(
        (source, target),
        (
            mechanisms.Connection(
                "dynamic",
                mechanisms.ResourcePort("source"),
                mechanisms.ResourcePort("target"),
                activation=_FirstFeatureGate(),
            ),
        ),
    )

    try:
        mechanisms.ResourceGraphCompiler.compile(graph, ("dynamic",))
    except mechanisms.ResourceGraphCompileError as error:
        assert "example_contexts" in str(error)
    else:
        raise AssertionError("conditional connections require an explicit tensor context contract")

    plan = mechanisms.ResourceGraphCompiler.compile(
        graph,
        ("dynamic",),
        example_contexts={"dynamic": torch.tensor([[0.25]])},
    )
    assert plan.context_connection_ids == ("dynamic",)
    compiled = torch.compile(plan, backend="eager", fullgraph=True)
    first = compiled(source.resolve().view.value, target.resolve().view.value, torch.tensor([[0.25]]))
    second = compiled(source.resolve().view.value, target.resolve().view.value, torch.tensor([[0.5]]))
    torch.testing.assert_close(first[1], view([[0.25, 0.5]]).value)
    torch.testing.assert_close(second[1], view([[0.5, 1.0]]).value)


def test_root_namespace_exposes_the_stable_resource_contract() -> None:
    assert arti.TensorResource is mechanisms.TensorResource
    assert arti.Connection is mechanisms.Connection
    assert arti.ProgramGraph is mechanisms.ProgramGraph


def test_resource_architecture_members_have_independent_content_addresses() -> None:
    source = mechanisms.TensorResource(
        spec("source", lifetime=mechanisms.ResourceLifetime.CALL), view([[1, 2]])
    )
    target = mechanisms.TensorResource(
        spec("target", lifetime=mechanisms.ResourceLifetime.STATE), view([[0, 0]])
    )
    connection = mechanisms.Connection(
        "copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("target")
    )
    graph = mechanisms.ProgramGraph((source, target), (connection,))

    assert arti.component_ref(source).startswith("arti/tensor-resource@sha256:")
    assert arti.component_ref(connection).startswith("arti/connection@sha256:")
    assert arti.component_ref(graph).startswith("arti/program-graph@sha256:")
    root = next(item for item in arti.component_provenance(graph)["components"] if item["path"] == "$")
    assert set(root["dependencies"]) >= {
        arti.component_ref(source),
        arti.component_ref(target),
        arti.component_ref(connection),
    }
