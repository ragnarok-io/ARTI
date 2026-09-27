from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import arti
from arti import mechanisms
from arti.federal_recall import FederalBankStep
from arti.federal_tensor_view import TensorViewFederalCandidate


def _view(values: list[list[float]], *, requires_grad: bool = False) -> mechanisms.TensorView:
    value = torch.tensor(values, dtype=torch.float32).unsqueeze(-1).requires_grad_(requires_grad)
    return mechanisms.TensorView.from_tensor(
        value,
        axis_names=("batch", "token", "feature"),
        axis_roles=("batch", "sequence", "feature"),
    )


def _spec(resource_id: str) -> mechanisms.TensorResourceSpec:
    return mechanisms.TensorResourceSpec(
        resource_id,
        mechanisms.TensorViewPattern(
            min_rank=3,
            max_rank=3,
            allowed_axis_roles=("batch", "sequence", "feature"),
        ),
    )


def _vector_spec(resource_id: str) -> mechanisms.TensorResourceSpec:
    return mechanisms.TensorResourceSpec(
        resource_id,
        mechanisms.TensorViewPattern(
            min_rank=2,
            max_rank=2,
            allowed_axis_roles=("batch", "feature"),
        ),
    )


def _vector_view(value: torch.Tensor) -> mechanisms.TensorView:
    return mechanisms.TensorView.from_tensor(
        value,
        axis_names=("batch", "feature"),
        axis_roles=("batch", "feature"),
    )


def test_output_projection_prunes_only_dead_declared_pure_stage_members() -> None:
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(torch.tensor([[2.0]]))),
        mechanisms.TensorResource(_vector_spec("left"), _vector_view(torch.zeros(1, 1))),
        mechanisms.TensorResource(_vector_spec("right"), _vector_view(torch.zeros(1, 1))),
    )
    left = mechanisms.Connection(
        "left_edge", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("left"),
        transfer=mechanisms.LearnableAffineTransfer(gain=2.0),
    )
    right = mechanisms.Connection(
        "right_edge", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("right"),
        transfer=mechanisms.LearnableAffineTransfer(gain=3.0),
    )
    graph = mechanisms.ProgramGraph(
        resources, (left, right),
        programs={"run": (mechanisms.ProgramStage(("left_edge", "right_edge")),)},
    )
    projected = mechanisms.ResourceGraphCompiler.compile_program_outputs(
        graph, "run", ("left",), discardable_step_ids=("right_edge",),
    )
    observed = mechanisms.ResourceGraphCompiler.compile_program_outputs(
        graph, "run", ("left", "right"), discardable_step_ids=("right_edge",),
    )
    retained = mechanisms.ResourceGraphCompiler.compile_program_outputs(graph, "run", ("left",))
    assert projected.output_resource_ids == ("left",)
    assert projected.executed_step_ids == ("left_edge",)
    assert "right_edge" in observed.executed_step_ids
    assert "right_edge" in retained.executed_step_ids
    assert all(parameter is not right.transfer.gain for parameter in projected.parameters())

    values = tuple(resource.resolve().view.value for resource in resources)
    full = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")(*values)
    actual = torch.compile(projected, backend="eager", fullgraph=True)(*values)
    torch.testing.assert_close(actual[0], full[1], atol=0, rtol=0)
    torch.testing.assert_close(observed(*values), (full[1], full[2]), atol=0, rtol=0)
    expected_gradient = torch.autograd.grad(full[1].sum(), left.transfer.gain, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual[0].sum(), left.transfer.gain)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_output_projection_preserves_previous_value_when_join_is_not_ready() -> None:
    resources = (
        mechanisms.TensorResource(_vector_spec("seed"), _vector_view(torch.tensor([[2.0]]))),
        mechanisms.TensorResource(_vector_spec("left"), _vector_view(torch.tensor([[7.0]]))),
        mechanisms.TensorResource(_vector_spec("right"), _vector_view(torch.tensor([[11.0]]))),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(1, 1))),
    )
    producer = mechanisms.Connection(
        "producer", mechanisms.ResourcePort("seed"), mechanisms.ResourcePort("output"),
        transfer=mechanisms.LearnableAffineTransfer(gain=2.0),
    )
    join = arti.as_fabric_node(
        "join", _DecoratedPairSum(),
        input_ports={"left": mechanisms.ResourcePort("left"),
                     "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (producer,), nodes=(join,),
        programs={"run": ("producer", mechanisms.ProgramJoin("ready", "join"))},
    )
    projected = mechanisms.ResourceGraphCompiler.compile_program_outputs(
        graph, "run", ("output",), discardable_step_ids=("producer",),
    )
    assert projected.executed_step_ids == ("producer", "join")
    full = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    inputs = (*(resource.resolve().view.value for resource in resources),
              torch.zeros((1, full.arrival_width), dtype=torch.bool))
    torch.testing.assert_close(projected(*inputs)[0], full(*inputs)[3], atol=0, rtol=0)
    torch.testing.assert_close(projected(*inputs)[0], torch.tensor([[4.0]]))
    torch.testing.assert_close(
        torch.autograd.grad(projected(*inputs)[0].sum(), producer.transfer.gain)[0],
        torch.tensor(2.0),
    )


def test_output_projection_keeps_live_dynamic_route_and_prunes_dead_diagnostic() -> None:
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(torch.tensor([[2.0]]))),
        mechanisms.TensorResource(_vector_spec("signal"), _vector_view(torch.tensor([[2.0, 1.0]]))),
        mechanisms.TensorResource(_vector_spec("scores"), _vector_view(torch.zeros(1, 2))),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(1, 1))),
        mechanisms.TensorResource(_vector_spec("diagnostic"), _vector_view(torch.zeros(1, 1))),
    )
    score = mechanisms.Connection(
        "score", mechanisms.ResourcePort("signal"), mechanisms.ResourcePort("scores"),
        transfer=mechanisms.LearnableAffineTransfer(gain=1.0),
    )
    first = mechanisms.Connection(
        "first", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
        transfer=mechanisms.LearnableAffineTransfer(gain=2.0),
    )
    second = mechanisms.Connection(
        "second", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
        transfer=mechanisms.LearnableAffineTransfer(gain=-3.0),
    )
    diagnostic = mechanisms.Connection(
        "diagnostic_step", mechanisms.ResourcePort("output"), mechanisms.ResourcePort("diagnostic"),
        transfer=mechanisms.LearnableAffineTransfer(gain=5.0),
    )
    graph = mechanisms.ProgramGraph(
        resources, (score, first, second, diagnostic),
        programs={"run": (
            "score", mechanisms.ProgramRoute("choice", "scores", ("first", "second")),
            "diagnostic_step",
        )},
    )
    projected = mechanisms.ResourceGraphCompiler.compile_program_outputs(
        graph, "run", ("output",), discardable_step_ids=("score", "diagnostic_step"),
    )
    assert projected.executed_step_ids == ("score", "choice")
    assert not any(parameter is diagnostic.transfer.gain for parameter in projected.parameters())
    with pytest.raises(mechanisms.ResourceGraphCompileError, match="must occur"):
        mechanisms.ResourceGraphCompiler.compile_program_outputs(
            graph, "run", ("output",), discardable_step_ids=("choice",),
        )

    values = tuple(resource.resolve().view.value for resource in resources)
    full = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    compiled = torch.compile(projected, backend="eager", fullgraph=True)
    for signal, expected in ((torch.tensor([[2.0, 1.0]]), 4.0), (torch.tensor([[1.0, 2.0]]), -6.0)):
        current = (values[0], signal, *values[2:])
        actual = compiled(*current)[0]
        torch.testing.assert_close(actual, full(*current)[3], atol=0, rtol=0)
        torch.testing.assert_close(actual, torch.tensor([[expected]]))
    first_result = projected(*values)[0]
    gradient = torch.autograd.grad(first_result.sum(), first.transfer.gain)[0]
    torch.testing.assert_close(gradient, torch.tensor(2.0))
    with pytest.raises(mechanisms.ResourceGraphCompileError, match="credit=True"):
        projected.credit_gradient(
            *values, terminal_cotangents={"output": torch.ones_like(first_result)},
        )
    credit_projected = mechanisms.ResourceGraphCompiler.compile_program_outputs(
        graph, "run", ("output",), discardable_step_ids=("score", "diagnostic_step"),
        credit=True,
    )
    credit_full = mechanisms.ResourceGraphCompiler.compile_program(
        graph, "run", _force_dataflow=True,
    )
    credit_inputs = (*values, torch.zeros((1, credit_projected.arrival_width), dtype=torch.bool))
    torch.testing.assert_close(credit_projected(*credit_inputs)[0], first_result)
    torch.testing.assert_close(
        torch.compile(credit_projected, backend="eager", fullgraph=True)(*credit_inputs)[0],
        first_result,
    )
    reverse = credit_projected.credit_gradient(
        *credit_inputs, terminal_cotangents={"output": torch.ones_like(first_result)},
        create_graph=False,
    )
    full_reverse = credit_full.credit_gradient(
        *credit_inputs, terminal_cotangents={"output": torch.ones_like(first_result)},
        create_graph=False,
    )
    projected_parameter_gradient = next(
        value for name, value in reverse.parameter_cotangents.items()
        if reverse.parameters[name] is first.transfer.gain
    )
    full_parameter_gradient = next(
        value for name, value in full_reverse.parameter_cotangents.items()
        if full_reverse.parameters[name] is first.transfer.gain
    )
    torch.testing.assert_close(projected_parameter_gradient, full_parameter_gradient)
    same_name_graph = mechanisms.ProgramGraph(
        resources, (score, first, second, diagnostic),
        programs={"run": (
            "score", mechanisms.ProgramRoute("score", "scores", ("first", "second")),
            "diagnostic_step",
        )},
    )
    same_name_plan = mechanisms.ResourceGraphCompiler.compile_program_outputs(
        same_name_graph, "run", ("output",), discardable_step_ids=("score", "diagnostic_step"),
    )
    torch.testing.assert_close(same_name_plan(*values)[0], first_result)
    if torch.cuda.is_available():
        cuda_resources = tuple(
            mechanisms.TensorResource(
                resource.spec,
                mechanisms.TensorView(resource.resolve().view.value.cuda(), resource.resolve().view.axes),
            )
            for resource in resources
        )
        cuda_graph = mechanisms.ProgramGraph(
            cuda_resources, (score, first, second, diagnostic),
            programs={"run": graph.program("run")},
        ).to("cuda")
        cuda_plan = mechanisms.ResourceGraphCompiler.compile_program_outputs(
            cuda_graph, "run", ("output",), discardable_step_ids=("score", "diagnostic_step"),
        )
        cuda_values = tuple(resource.resolve().view.value for resource in cuda_graph.resources.values())
        cuda_compiled = torch.compile(cuda_plan, backend="inductor", fullgraph=True)
        cuda_output = cuda_compiled(*cuda_values)[0]
        torch.testing.assert_close(cuda_output.cpu(), torch.tensor([[4.0]]))
        cuda_gradient = torch.autograd.grad(cuda_output.sum(), first.transfer.gain)[0]
        torch.testing.assert_close(cuda_gradient.cpu(), torch.tensor(2.0))


def test_loop_output_projection_keeps_cross_iteration_state_and_route_receipt() -> None:
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(torch.tensor([[2.0]]))),
        mechanisms.TensorResource(_vector_spec("carry"), _vector_view(torch.zeros(1, 1))),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(1, 1))),
        mechanisms.TensorResource(_vector_spec("diagnostic"), _vector_view(torch.zeros(1, 1))),
        mechanisms.TensorResource(_vector_spec("scratch"), _vector_view(torch.zeros(1, 1))),
        mechanisms.TensorResource(_vector_spec("scores"), _vector_view(torch.tensor([[2.0, 1.0]]))),
        mechanisms.TensorResource(_continue_spec("continue"), _continue_view(torch.ones(1))),
    )
    read_carry = mechanisms.Connection(
        "read_carry", mechanisms.ResourcePort("carry"), mechanisms.ResourcePort("output"),
    )
    write_carry = mechanisms.Connection(
        "write_carry", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("carry"),
        transfer=mechanisms.LearnableAffineTransfer(gain=1.0),
    )
    side_a = mechanisms.Connection(
        "side_a", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("scratch"),
        transfer=mechanisms.LearnableAffineTransfer(gain=1.0),
    )
    side_b = mechanisms.Connection(
        "side_b", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("scratch"),
        transfer=mechanisms.LearnableAffineTransfer(gain=-1.0),
    )
    diagnostic = mechanisms.Connection(
        "diagnostic_step", mechanisms.ResourcePort("output"), mechanisms.ResourcePort("diagnostic"),
    )
    program = (
        "read_carry", "write_carry",
        mechanisms.ProgramRoute("choose", "scores", ("side_a", "side_b")),
        "diagnostic_step",
    )
    graph = mechanisms.ProgramGraph(
        resources, (read_carry, write_carry, side_a, side_b, diagnostic),
        programs={"tick": program},
        loops=(
            mechanisms.ProgramLoop("bounded", "tick", "continue", max_iterations=2),
            mechanisms.ProgramLoop("open", "tick", "continue", max_iterations=None),
        ),
    )
    bounded = mechanisms.ResourceGraphCompiler.compile_loop_outputs(
        graph, "bounded", ("output",),
        discardable_step_ids=("write_carry", "diagnostic_step"),
    )
    assert bounded.executed_step_ids == ("read_carry", "write_carry", "choose")
    inputs = (*(resource.resolve().view.value for resource in resources),
              torch.zeros((1, bounded.arrival_width), dtype=torch.bool))
    torch.testing.assert_close(bounded(*inputs)[0], torch.tensor([[2.0]]))
    torch.testing.assert_close(
        torch.autograd.grad(bounded(*inputs)[0].sum(), write_carry.transfer.gain)[0],
        torch.tensor(2.0),
    )

    open_plan = mechanisms.ResourceGraphCompiler.compile_loop_outputs(
        graph, "open", ("output",),
        discardable_step_ids=("write_carry", "diagnostic_step"),
    )
    status = open_plan.forward_until_done_with_status(
        *inputs, host_step_limit=torch.tensor(2, dtype=torch.int64),
    )
    torch.testing.assert_close(status[0], torch.tensor([[2.0]]))
    torch.testing.assert_close(status[1], torch.tensor(2))
    torch.testing.assert_close(status[2], torch.tensor([2]))
    torch.testing.assert_close(status[3], torch.tensor([True]))
    compiled_status = torch.compile(
        open_plan.forward_until_done_with_status, backend="eager", fullgraph=True,
    )(*inputs, host_step_limit=torch.tensor(2, dtype=torch.int64))
    for compiled_value, expected_value in zip(compiled_status, status, strict=True):
        torch.testing.assert_close(compiled_value, expected_value)
    receipt = open_plan.forward_until_done_with_route_receipt(
        *inputs, host_step_limit=torch.tensor(2, dtype=torch.int64), receipt_capacity=3,
    )
    torch.testing.assert_close(receipt[0], status[0])
    torch.testing.assert_close(receipt[-1][:2], torch.tensor([0, 0]))
    tensor_receipt = (receipt[1], receipt[4], receipt[5], receipt[6])
    reverse = open_plan.credit_gradient(
        *inputs,
        tensor_receipt=tensor_receipt,
        terminal_cotangents={"output": torch.ones_like(receipt[0])},
        create_graph=False,
    )
    full_reverse = mechanisms.ResourceGraphCompiler.compile_loop(graph, "open").credit_gradient(
        *inputs,
        tensor_receipt=tensor_receipt,
        terminal_cotangents={"output": torch.ones_like(receipt[0])},
        create_graph=False,
    )
    writer_gradient = next(
        value for name, value in reverse.parameter_cotangents.items()
        if reverse.parameters[name] is write_carry.transfer.gain
    )
    torch.testing.assert_close(writer_gradient, torch.tensor(2.0))
    full_writer_gradient = next(
        value for name, value in full_reverse.parameter_cotangents.items()
        if full_reverse.parameters[name] is write_carry.transfer.gain
    )
    torch.testing.assert_close(writer_gradient, full_writer_gradient)
    replayed = open_plan.replay_tensor_receipt(*inputs, tensor_receipt=tensor_receipt)
    torch.testing.assert_close(replayed[0], receipt[0])
    replay_gradient = torch.autograd.grad(replayed[0].sum(), write_carry.transfer.gain)[0]
    torch.testing.assert_close(replay_gradient, writer_gradient)
    compiled_replay = torch.compile(
        open_plan.replay_tensor_receipt, backend="eager", fullgraph=True,
    )(*inputs, tensor_receipt=tensor_receipt)
    torch.testing.assert_close(compiled_replay, replayed)
    for resource_id in open_plan.resource_ids:
        actual = reverse.resource_cotangents[resource_id]
        expected = full_reverse.resource_cotangents[resource_id]
        if actual is None or expected is None:
            assert actual is None and expected is None
        else:
            torch.testing.assert_close(actual, expected)
    with pytest.raises(mechanisms.ResourceGraphCompileError, match="unexposed outputs"):
        open_plan.credit_gradient(
            *inputs,
            terminal_cotangents={"diagnostic": torch.ones_like(receipt[0])},
            tensor_receipt=tensor_receipt,
        )
    if torch.cuda.is_available():
        cuda_resources = tuple(
            mechanisms.TensorResource(
                resource.spec,
                mechanisms.TensorView(resource.resolve().view.value.cuda(), resource.resolve().view.axes),
            )
            for resource in resources
        )
        cuda_graph = mechanisms.ProgramGraph(
            cuda_resources, (read_carry, write_carry, side_a, side_b, diagnostic),
            programs={"tick": program}, loops=(graph.loop("bounded"), graph.loop("open")),
        ).to("cuda")
        cuda_values = tuple(resource.resolve().view.value for resource in cuda_resources)
        cuda_inputs = (*cuda_values, torch.zeros((1, bounded.arrival_width), dtype=torch.bool, device="cuda"))
        cuda_bounded = mechanisms.ResourceGraphCompiler.compile_loop_outputs(
            cuda_graph, "bounded", ("output",),
            discardable_step_ids=("write_carry", "diagnostic_step"),
        ).to("cuda")
        cuda_output = torch.compile(cuda_bounded, backend="inductor", fullgraph=True)(*cuda_inputs)[0]
        torch.testing.assert_close(cuda_output.cpu(), torch.tensor([[2.0]]))
        cuda_gradient = torch.autograd.grad(cuda_output.sum(), write_carry.transfer.gain)[0]
        torch.testing.assert_close(cuda_gradient.cpu(), torch.tensor(2.0))
        cuda_open = mechanisms.ResourceGraphCompiler.compile_loop_outputs(
            cuda_graph, "open", ("output",),
            discardable_step_ids=("write_carry", "diagnostic_step"),
        ).to("cuda")
        cuda_status = torch.compile(
            cuda_open.forward_until_done_with_status, backend="inductor", fullgraph=True,
        )(*cuda_inputs, host_step_limit=torch.tensor(2, dtype=torch.int64, device="cuda"))
        for actual, expected in zip(cuda_status, status, strict=True):
            torch.testing.assert_close(actual.cpu(), expected)
        cuda_receipt = tuple(value.cuda() for value in tensor_receipt)
        cuda_replay = torch.compile(
            cuda_open.replay_tensor_receipt, backend="inductor", fullgraph=True,
        )(*cuda_inputs, tensor_receipt=cuda_receipt)
        torch.testing.assert_close(cuda_replay[0].cpu(), receipt[0])
        cuda_replay_gradient = torch.autograd.grad(
            cuda_replay[0].sum(), write_carry.transfer.gain,
        )[0]
        torch.testing.assert_close(cuda_replay_gradient.cpu(), writer_gradient)


def test_loop_output_projection_supports_plain_program_nodes() -> None:
    resources = (
        mechanisms.TensorResource(_spec("state"), _view([[0.0]])),
        mechanisms.TensorResource(_spec("diagnostic"), _view([[0.0]])),
        mechanisms.TensorResource(_continue_spec("continue"), _continue_view(torch.ones(1))),
    )
    count = arti.as_fabric_node(
        "count", _DecoratedLoopStep(),
        input_ports={"value": mechanisms.ResourcePort("state")},
        output_ports={"value": mechanisms.ResourcePort("state"),
                      "continue": mechanisms.ResourcePort("continue")},
    )
    diagnostic = arti.as_fabric_node(
        "diagnostic_step", _DecoratedScale(2.0),
        input_ports={"source": mechanisms.ResourcePort("state")},
        output_ports={"value": mechanisms.ResourcePort("diagnostic")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (), nodes=(count, diagnostic),
        programs={"tick": ("count", "diagnostic_step")},
        loops=(mechanisms.ProgramLoop("bounded", "tick", "continue", max_iterations=4),),
    )
    projected = mechanisms.ResourceGraphCompiler.compile_loop_outputs(
        graph, "bounded", ("state",), discardable_step_ids=("diagnostic_step",),
    )
    assert projected.executed_step_ids == ("count",)
    inputs = tuple(resource.resolve().view.value for resource in resources)
    torch.testing.assert_close(projected(*inputs)[0], _view([[3.0]]).value)
    inference = projected.forward_until_done(*inputs)
    torch.testing.assert_close(inference[0], _view([[3.0]]).value)
    torch.testing.assert_close(inference[1], torch.tensor(3))
    torch.testing.assert_close(inference[2], torch.tensor([3]))


def test_program_route_dispatches_one_relation_and_credits_sampled_scores(tmp_path) -> None:
    source = mechanisms.TensorResource(_vector_spec("source"), _vector_view(torch.tensor([[2.0]])))
    scores = mechanisms.TensorResource(
        _vector_spec("scores"), _vector_view(torch.tensor([[3.0, -3.0]]))
    )
    output = mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(1, 1)))
    positive = mechanisms.Connection(
        "positive", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
        transfer=mechanisms.LearnableAffineTransfer(gain=2.0, bias=0.0),
    )
    negative = mechanisms.Connection(
        "negative", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
        transfer=mechanisms.LearnableAffineTransfer(gain=-3.0, bias=0.0),
    )
    graph = mechanisms.ProgramGraph(
        (source, scores, output), (positive, negative),
        programs={"run": (mechanisms.ProgramRoute("choose", "scores", ("positive", "negative")),)},
    )
    entry = graph.state()
    execution = graph.execute_program_functional("run", state=entry)
    assert [item.connection_id for item in execution.connections] == ["positive"]
    assert [item.connection_id for item in execution.dispatches[0].members] == ["positive"]
    assert execution.routes[0].candidate_id == "positive"
    assert not execution.routes[0].sampled
    assert {item.spec.resource_id: item for item in execution.state.resources}["output"].active_view.value.item() == 4.0
    assert graph.resource("output").resolve().view.value.item() == 0.0

    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    assert isinstance(plan, mechanisms.StaticRoutedProgramExecutionPlan)
    values = tuple(resource.resolve().view.value for resource in (source, scores, output))
    actual = plan(*values)
    compiled = torch.compile(plan, backend="eager", fullgraph=True)(*values)
    exported = torch.export.export(plan, values).module()(*values)
    torch.testing.assert_close(actual[2], torch.tensor([[4.0]]))
    torch.testing.assert_close(compiled[2], actual[2])
    torch.testing.assert_close(exported[2], actual[2])
    torch.testing.assert_close(
        torch.compile(plan, backend="eager", fullgraph=True)(
            values[0], torch.tensor([[-3.0, 3.0]]), values[2],
        )[2],
        torch.tensor([[-6.0]]),
    )
    loss = compiled[2].sum()
    loss.backward()
    assert positive.transfer.gain.grad is not None
    assert negative.transfer.gain.grad is None or negative.transfer.gain.grad.item() == 0.0

    score_input = torch.tensor([[0.1, -0.1]], requires_grad=True)
    train = torch.compile(plan.forward_with_route_credit, backend="eager", fullgraph=True)
    train_values, (route_choice,), log_probability = train(values[0], score_input, values[2])
    assert route_choice.ndim == 0
    (train_values[2].sum() + train_values[2].detach().square().sum() * log_probability).backward()
    assert score_input.grad is not None
    assert score_input.grad.abs().sum() > 0

    reverse_scores = _vector_view(torch.tensor([[-3.0, 3.0]], requires_grad=True))
    reverse = graph.execute_program_functional(
        "run", state=entry, input_views={"scores": reverse_scores},
    )
    assert reverse.routes[0].candidate_id == "negative"
    assert {item.spec.resource_id: item for item in reverse.state.resources}["output"].active_view.value.item() == -6.0
    with pytest.raises(mechanisms.ResourceGraphError, match="sampled"):
        reverse.routes[0].structure_objective(torch.tensor(1.0))
    torch.manual_seed(4)
    sampled = graph.execute_program_functional(
        "run", state=entry, input_views={"scores": reverse_scores}, sample_routes=True,
    )
    sampled.routes[0].structure_objective(torch.tensor(2.0)).backward()
    assert reverse_scores.value.grad is not None
    assert reverse_scores.value.grad.abs().sum() > 0

    saved = mechanisms.save_program_graph(graph, tmp_path / "routed-program")
    restored = mechanisms.load_program_graph(saved.tensors_path)
    assert restored.contract_fingerprint == graph.contract_fingerprint
    assert restored.execute_program_functional("run").routes[0].candidate_id == "positive"

    if torch.cuda.is_available():
        cuda_plan = plan.to("cuda")
        cuda_values = tuple(value.to("cuda") for value in values)
        cuda_scores = torch.tensor([[0.1, -0.1]], device="cuda", requires_grad=True)
        cuda_forward = torch.compile(cuda_plan, backend="eager", fullgraph=True)
        torch.testing.assert_close(cuda_forward(*cuda_values)[2], torch.tensor([[4.0]], device="cuda"))
        cuda_train = torch.compile(cuda_plan.forward_with_route_credit, backend="eager", fullgraph=True)
        cuda_result, (cuda_route_choice,), cuda_log_probability = cuda_train(
            cuda_values[0], cuda_scores, cuda_values[2]
        )
        assert cuda_route_choice.ndim == 0
        (cuda_result[2].sum() + cuda_result[2].detach().square().sum() * cuda_log_probability).backward()
        assert cuda_scores.grad is not None and cuda_scores.grad.abs().sum() > 0


def test_loop_structure_credit_masks_inactive_route_without_host_sync() -> None:
    resource = mechanisms.TensorResource(
        _vector_spec("scores"), _vector_view(torch.zeros(1, 2)),
    )
    state = mechanisms.ProgramGraph((resource,), ()).state()
    for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
        first = torch.tensor(-0.2, device=device, requires_grad=True)
        inactive = torch.tensor(-0.7, device=device, requires_grad=True)
        execution = mechanisms.ProgramLoopExecution(
            state=state,
            active_masks=(torch.tensor([True], device=device), torch.tensor([False], device=device)),
            connections=(),
            nodes=(),
            routes=(
                mechanisms.ProgramRouteExecution("choose", "left", resource.resolve().binding, first, True, 0),
                mechanisms.ProgramRouteExecution("choose", "right", resource.resolve().binding, inactive, True, 1),
            ),
        )
        credit = torch.compile(
            lambda loss: execution.structure_objective(loss), backend="eager", fullgraph=True,
        )(torch.tensor(2.0, device=device))
        credit.backward()
        torch.testing.assert_close(first.grad, torch.tensor(2.0, device=device))
        torch.testing.assert_close(inactive.grad, torch.tensor(0.0, device=device))


def test_program_route_requeries_new_resource_state_between_wires() -> None:
    source = mechanisms.TensorResource(
        _vector_spec("source"), _vector_view(torch.tensor([[2.0]]))
    )
    scores = mechanisms.TensorResource(
        _vector_spec("scores"), _vector_view(torch.tensor([[3.0, -1.0]]))
    )
    output = mechanisms.TensorResource(
        _vector_spec("output"), _vector_view(torch.zeros(1, 1))
    )
    flip = mechanisms.Connection(
        "flip", mechanisms.ResourcePort("scores"), mechanisms.ResourcePort("scores"),
        transfer=mechanisms.LearnableAffineTransfer(gain=-1.0),
    )
    keep = mechanisms.Connection(
        "keep", mechanisms.ResourcePort("scores"), mechanisms.ResourcePort("scores"),
        transfer=mechanisms.LearnableAffineTransfer(gain=1.0),
    )
    copy = mechanisms.Connection(
        "copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
    )
    graph = mechanisms.ProgramGraph(
        (source, scores, output), (flip, keep, copy),
        programs={"run": (
            mechanisms.ProgramRoute("first", "scores", ("flip", "keep")),
            mechanisms.ProgramRoute("second", "scores", ("flip", "keep")),
            "copy",
        )},
    )
    execution = graph.execute_program_functional("run")
    assert [item.candidate_id for item in execution.routes] == ["flip", "keep"]
    assert [item.connection_id for item in execution.connections] == ["flip", "keep", "copy"]
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    values = tuple(resource.resolve().view.value for resource in (source, scores, output))
    expected = plan(*values)
    compiled = torch.compile(plan, backend="eager", fullgraph=True)(*values)
    torch.testing.assert_close(compiled[1], torch.tensor([[-3.0, 1.0]]))
    for left, right in zip(expected, compiled, strict=True):
        torch.testing.assert_close(left, right)


def test_sample_scoped_route_dispatches_hard_rows_with_join_credit() -> None:
    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[2.0, 4.0], [5.0, 7.0]])),
        mechanisms.TensorResource(
            _vector_spec("scores"), _vector_view(torch.tensor([[3.0, -1.0], [-1.0, 3.0]])),
        ),
        mechanisms.TensorResource(_spec("left"), _view([[0.0, 0.0], [0.0, 0.0]])),
        mechanisms.TensorResource(_spec("right"), _view([[0.0, 0.0], [0.0, 0.0]])),
        mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0], [0.0, 0.0]])),
    )
    left = mechanisms.Connection(
        "left_copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("left"),
        transfer=mechanisms.LearnableAffineTransfer(gain=2.0),
    )
    right = mechanisms.Connection(
        "right_copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("right"),
        transfer=mechanisms.LearnableAffineTransfer(gain=3.0),
    )
    late_left = mechanisms.Connection(
        "late_left", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("left"),
    )
    joined = arti.as_fabric_node(
        "sum", _DecoratedPairSum(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (left, right, late_left), nodes=(joined,),
        programs={"run": (
            mechanisms.ProgramRoute(
                "choose", "scores", ("left_copy", "right_copy"), selection_scope="sample",
            ),
            mechanisms.ProgramJoin("early", "sum"),
            "late_left", mechanisms.ProgramJoin("late", "sum"),
        )},
    )
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    assert isinstance(plan, mechanisms.StaticDataflowProgramExecutionPlan)
    source_value = resources[0].resolve().view.value.detach().clone().requires_grad_(True)
    score_value = resources[1].resolve().view.value.detach().clone().requires_grad_(True)
    inputs = (
        source_value, score_value,
        *(resource.resolve().view.value for resource in resources[2:]),
        torch.zeros((2, plan.arrival_width), dtype=torch.bool),
    )
    forced = plan.forward_with_route_selections(
        *inputs, route_selections=(torch.tensor([0, 1]),),
    )
    torch.testing.assert_close(forced[4], _view([[0.0, 0.0], [20.0, 28.0]]).value)
    compiled = torch.compile(plan, backend="eager", fullgraph=True)(*inputs)
    for reference, actual in zip(forced, compiled, strict=True):
        torch.testing.assert_close(actual, reference)
    source_gradient = torch.autograd.grad(compiled[4].sum(), source_value, retain_graph=True)[0]
    torch.testing.assert_close(source_gradient, _view([[0.0, 0.0], [4.0, 4.0]]).value)
    credit = plan.credit_gradient(
        *inputs, route_selections=(torch.tensor([0, 1]),),
        terminal_cotangents={"output": torch.ones_like(compiled[4])},
    )
    torch.testing.assert_close(credit.resource_values[4], compiled[4])
    torch.testing.assert_close(credit.resource_cotangents["source"], source_gradient)
    sampled, choices, log_probability = plan.forward_with_route_credit(*inputs)
    assert choices[0].shape == (2,)
    assert log_probability.shape == (2,)
    replay = plan.forward_with_route_selections(*inputs, route_selections=choices)
    torch.testing.assert_close(sampled[4], replay[4])
    (log_probability * torch.tensor([1.0, 2.0])).sum().backward()
    assert score_value.grad is not None and score_value.grad.abs().sum() > 0
    if torch.cuda.is_available():
        cuda_plan = plan.to("cuda")
        cuda_inputs = tuple(value.detach().to("cuda") for value in inputs)
        cuda_result = torch.compile(cuda_plan, backend="eager", fullgraph=True)(*cuda_inputs)
        torch.testing.assert_close(cuda_result[4].cpu(), forced[4])
        cuda_inductor = torch.compile(cuda_plan, backend="inductor", fullgraph=True)(*cuda_inputs)
        torch.testing.assert_close(cuda_inductor[4].cpu(), forced[4])
        cuda_credit = cuda_plan.credit_gradient(
            *cuda_inputs,
            route_selections=(torch.tensor([0, 1], device="cuda"),),
            terminal_cotangents={"output": torch.ones_like(cuda_result[4])},
        )
        torch.testing.assert_close(cuda_credit.resource_cotangents["source"].cpu(), source_gradient)


def test_open_loop_sample_routes_wait_for_second_join_arrival_and_replay_credit() -> None:
    @arti.fabric_layer(inputs=("scores",), outputs={"value": "scores"})
    class FlipScores(torch.nn.Module):
        def forward(self, scores: torch.Tensor) -> torch.Tensor:
            return -scores

    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[2.0, 4.0], [5.0, 7.0]])),
        mechanisms.TensorResource(
            _vector_spec("scores"), _vector_view(torch.tensor([[3.0, -1.0], [-1.0, 3.0]])),
        ),
        *(mechanisms.TensorResource(_spec(name), _view([[0.0, 0.0], [0.0, 0.0]]))
          for name in ("left", "right", "output")),
        mechanisms.TensorResource(_continue_spec("continue"), _continue_view(torch.ones(2))),
    )
    left = mechanisms.Connection(
        "left_copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("left"),
        transfer=mechanisms.LearnableAffineTransfer(gain=2.0),
    )
    right = mechanisms.Connection(
        "right_copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("right"),
        transfer=mechanisms.LearnableAffineTransfer(gain=3.0),
    )
    joined = arti.as_fabric_node(
        "sum", _DecoratedPairSum(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    flip = arti.as_fabric_node(
        "flip", FlipScores(),
        input_ports={"scores": mechanisms.ResourcePort("scores")},
        output_ports={"value": mechanisms.ResourcePort("scores")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (left, right), nodes=(joined, flip),
        programs={"iterate": (
            mechanisms.ProgramRoute(
                "choose", "scores", ("left_copy", "right_copy"), selection_scope="sample",
            ),
            mechanisms.ProgramJoin("ready", "sum"), "flip",
        )},
        loops=(mechanisms.ProgramLoop("open", "iterate", "continue", max_iterations=None),),
    )
    plan = mechanisms.ResourceGraphCompiler.compile_loop(graph, "open")
    assert isinstance(plan, mechanisms.StaticDataflowLoopExecutionPlan)
    inputs = (
        *(resource.resolve().view.value for resource in resources),
        torch.zeros((2, plan.body.arrival_width), dtype=torch.bool),
    )
    first = plan.forward_until_done_with_route_receipt(
        *inputs, host_step_limit=torch.tensor(2, dtype=torch.int64), receipt_capacity=1,
    )
    output_index = plan.resource_ids.index("output")
    torch.testing.assert_close(first[output_index], torch.zeros_like(inputs[0]))
    torch.testing.assert_close(first[-7], torch.tensor([[True, False], [False, True]]))
    torch.testing.assert_close(first[-4], torch.tensor([True, True]))
    recorded = plan.forward_until_done_with_route_receipt(
        *inputs, host_step_limit=torch.tensor(2, dtype=torch.int64), receipt_capacity=2,
    )
    compiled = torch.compile(
        plan.forward_until_done_with_route_receipt, backend="eager", fullgraph=True,
    )(*inputs, host_step_limit=torch.tensor(2, dtype=torch.int64), receipt_capacity=2)
    for expected, actual in zip(recorded, compiled, strict=True):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(recorded[output_index], 5 * inputs[0])
    torch.testing.assert_close(recorded[-6], torch.tensor(2))
    torch.testing.assert_close(recorded[-3], torch.ones((2, 2), dtype=torch.bool))
    torch.testing.assert_close(recorded[-1], torch.tensor([[0, 1], [1, 0]]))
    receipt = (recorded[-6], recorded[-3], recorded[-2], recorded[-1])
    replayed = plan.replay_tensor_receipt(*inputs, tensor_receipt=receipt)
    torch.testing.assert_close(replayed[output_index], recorded[output_index])
    direct = torch.autograd.grad(
        replayed[output_index].sum(), (left.transfer.gain, right.transfer.gain),
    )
    reverse = plan.credit_gradient(
        *inputs, tensor_receipt=receipt,
        terminal_cotangents={"output": torch.ones_like(recorded[output_index])},
        create_graph=False,
    )
    for parameter, expected in zip((left.transfer.gain, right.transfer.gain), direct, strict=True):
        observed = next(
            reverse.parameter_cotangents[name] for name, candidate in reverse.parameters.items()
            if candidate is parameter
        )
        torch.testing.assert_close(observed, expected)
    spare_capacity = plan.forward_until_done_with_route_receipt(
        *inputs, host_step_limit=torch.tensor(2, dtype=torch.int64), receipt_capacity=6,
    )
    spare_receipt = (spare_capacity[-6], spare_capacity[-3], spare_capacity[-2], spare_capacity[-1])
    torch.testing.assert_close(spare_receipt[0], torch.tensor(2))
    assert not spare_receipt[1][2:].any()
    replay_calls = 0
    original_forward = plan.body._forward_with_routes

    def count_forward(*args, **kwargs):
        nonlocal replay_calls
        replay_calls += 1
        return original_forward(*args, **kwargs)

    plan.body._forward_with_routes = count_forward
    try:
        spare_replay = plan.replay_tensor_receipt(*inputs, tensor_receipt=spare_receipt)
    finally:
        plan.body._forward_with_routes = original_forward
    assert replay_calls == 2
    torch.testing.assert_close(spare_replay[output_index], replayed[output_index])
    spare_gradients = torch.autograd.grad(
        spare_replay[output_index].sum(), (left.transfer.gain, right.transfer.gain),
    )
    for actual, expected in zip(spare_gradients, direct, strict=True):
        torch.testing.assert_close(actual, expected)
    compiled_spare = torch.compile(
        plan.replay_tensor_receipt, backend="eager", fullgraph=True,
    )(*inputs, tensor_receipt=spare_receipt)
    torch.testing.assert_close(compiled_spare[output_index], replayed[output_index])
    resource_before = {
        resource_id: graph.resources[resource_id].state().active_view.value.clone()
        for resource_id in plan.resource_ids
    }
    checkpointed = checkpoint(
        lambda *values: plan.replay_tensor_receipt(
            *values, tensor_receipt=spare_receipt,
        )[output_index],
        *inputs, use_reentrant=False,
    )
    torch.testing.assert_close(checkpointed, replayed[output_index])
    checkpoint_gradients = torch.autograd.grad(
        checkpointed.sum(), (left.transfer.gain, right.transfer.gain),
    )
    for actual, expected in zip(checkpoint_gradients, direct, strict=True):
        torch.testing.assert_close(actual, expected)
    for resource_id, before in resource_before.items():
        torch.testing.assert_close(graph.resources[resource_id].state().active_view.value, before)
    if torch.cuda.is_available():
        cuda_plan = copy.deepcopy(plan).cuda()
        cuda_inputs = tuple(value.cuda() for value in inputs)
        cuda_recorded = torch.compile(
            cuda_plan.forward_until_done_with_route_receipt, backend="inductor", fullgraph=True,
        )(
            *cuda_inputs, host_step_limit=torch.tensor(2, dtype=torch.int64, device="cuda"),
            receipt_capacity=2,
        )
        for reference, actual in zip(recorded, cuda_recorded, strict=True):
            torch.testing.assert_close(actual.cpu(), reference)
        cuda_reverse = cuda_plan.credit_gradient(
            *cuda_inputs,
            tensor_receipt=(cuda_recorded[-6], cuda_recorded[-3], cuda_recorded[-2], cuda_recorded[-1]),
            terminal_cotangents={"output": torch.ones_like(cuda_recorded[output_index])},
            create_graph=False,
        )
        for parameter, expected in zip((left.transfer.gain, right.transfer.gain), direct, strict=True):
            parameter_name = next(
                name for name, candidate in reverse.parameters.items() if candidate is parameter
            )
            torch.testing.assert_close(
                cuda_reverse.parameter_cotangents[parameter_name].cpu(), expected,
            )
        cuda_spare_receipt = tuple(value.cuda() for value in spare_receipt)
        compiled_cuda_replay = torch.compile(
            cuda_plan.replay_tensor_receipt, backend="inductor", fullgraph=True,
        )
        cuda_replay = compiled_cuda_replay(*cuda_inputs, tensor_receipt=cuda_spare_receipt)
        torch.testing.assert_close(cuda_replay[output_index].cpu(), replayed[output_index])
        cuda_gradients = torch.autograd.grad(
            cuda_replay[output_index].sum(),
            tuple(
                cuda_reverse.parameters[name]
                for parameter in (left.transfer.gain, right.transfer.gain)
                for name, candidate in reverse.parameters.items() if candidate is parameter
            ),
        )
        for actual, expected in zip(cuda_gradients, direct, strict=True):
            torch.testing.assert_close(actual.cpu(), expected)
        cuda_checkpointed = checkpoint(
            lambda *values: compiled_cuda_replay(
                *values, tensor_receipt=cuda_spare_receipt,
            )[output_index],
            *cuda_inputs, use_reentrant=False,
        )
        torch.testing.assert_close(cuda_checkpointed.cpu(), replayed[output_index])
        cuda_checkpoint_gradients = torch.autograd.grad(
            cuda_checkpointed.sum(),
            tuple(
                cuda_reverse.parameters[name]
                for parameter in (left.transfer.gain, right.transfer.gain)
                for name, candidate in reverse.parameters.items() if candidate is parameter
            ),
        )
        for actual, expected in zip(cuda_checkpoint_gradients, direct, strict=True):
            torch.testing.assert_close(actual.cpu(), expected)


def test_sample_scoped_route_without_join_round_trips(tmp_path) -> None:
    resources = (
        mechanisms.TensorResource(
            _vector_spec("source"), _vector_view(torch.tensor([[2.0], [3.0], [4.0], [5.0]])),
        ),
        mechanisms.TensorResource(
            _vector_spec("scores"), _vector_view(torch.tensor([
                [2.0, -1.0], [2.0, -1.0], [-1.0, 2.0], [-1.0, 2.0],
            ])),
        ),
        mechanisms.TensorResource(
            _vector_spec("output"), _vector_view(torch.zeros(4, 1)),
        ),
    )
    positive = mechanisms.Connection(
        "positive", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
        transfer=mechanisms.LearnableAffineTransfer(gain=2.0),
    )
    negative = mechanisms.Connection(
        "negative", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
        transfer=mechanisms.LearnableAffineTransfer(gain=-3.0),
    )
    graph = mechanisms.ProgramGraph(
        resources, (positive, negative),
        programs={"run": (
            mechanisms.ProgramRoute(
                "choose", "scores", ("positive", "negative"), selection_scope="sample",
            ),
        )},
    )
    with pytest.raises(mechanisms.ResourceGraphError, match="requires a compiled"):
        graph.execute_program_functional("run")
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    values = tuple(resource.resolve().view.value for resource in resources)
    assert len(plan.blocks[0]._cohort_branches) == 3
    result = torch.compile(plan, backend="eager", fullgraph=True)(*values)
    torch.testing.assert_close(result[2], torch.tensor([[4.0], [6.0], [-12.0], [-15.0]]))
    exported = torch.export.export(plan, values).module()(*values)
    torch.testing.assert_close(exported[2], result[2])
    sealed = graph.specialize_program_routes({"choose": "positive"}).graph
    sealed_plan = mechanisms.ResourceGraphCompiler.compile_program(sealed, "run")
    forced = plan.blocks[0].forward_with_selection(*values, selection=torch.zeros(4, dtype=torch.long))[0]
    torch.testing.assert_close(sealed_plan(*values)[2], forced[2], atol=0, rtol=0)
    saved = mechanisms.save_program_graph(graph, tmp_path / "sample-route")
    loaded = mechanisms.load_program_graph(saved.tensors_path)
    assert loaded.contract_fingerprint == graph.contract_fingerprint
    loaded_plan = mechanisms.ResourceGraphCompiler.compile_program(loaded, "run")
    torch.testing.assert_close(loaded_plan(*values)[2], result[2])
    if torch.cuda.is_available():
        cuda_plan = plan.to("cuda")
        cuda_values = tuple(value.to("cuda") for value in values)
        cuda_result = torch.compile(cuda_plan, backend="inductor", fullgraph=True)(*cuda_values)
        torch.testing.assert_close(cuda_result[2].cpu(), result[2])


@pytest.mark.parametrize("selection_scope", ("batch", "sample"))
@pytest.mark.parametrize("execution_mode", ("sparse", "all_candidates"))
def test_three_candidate_route_compiles_hard_paths_and_preserves_credit(
    selection_scope: str, execution_mode: str,
) -> None:
    torch._dynamo.reset()

    source = torch.tensor([[2.0], [3.0], [4.0]], requires_grad=True)
    scores = (
        torch.tensor([[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]])
        if selection_scope == "sample" else
        torch.tensor([[0.0, 0.0, 9.0]]).expand(3, -1).clone()
    )
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(source)),
        mechanisms.TensorResource(_vector_spec("scores"), _vector_view(scores)),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(3, 1))),
    )
    connections = tuple(
        mechanisms.Connection(
            f"path_{index}", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
            transfer=mechanisms.LearnableAffineTransfer(gain=gain),
        ) for index, gain in enumerate((2.0, 3.0, 5.0))
    )
    graph = mechanisms.ProgramGraph(
        resources, connections,
        programs={"run": (mechanisms.ProgramRoute(
            "choose", "scores", tuple(connection.connection_id for connection in connections),
            selection_scope=selection_scope, execution_mode=execution_mode,
        ),)},
    )
    plan = mechanisms.ResourceGraphCompiler.compile_program_outputs(
        graph, "run", ("output",), credit=True,
    )
    inputs = (*(resource.resolve().view.value for resource in resources),
              torch.zeros((3, plan.body.arrival_width), dtype=torch.bool))
    expected = (torch.tensor([[4.0], [9.0], [20.0]]) if selection_scope == "sample"
                else torch.tensor([[10.0], [15.0], [20.0]]))
    actual = torch.compile(plan, backend="eager", fullgraph=True)(*inputs)
    torch.testing.assert_close(actual[0], expected, atol=0, rtol=0)
    torch.testing.assert_close(plan(*inputs)[0], expected, atol=0, rtol=0)
    if selection_scope == "sample" and execution_mode == "sparse":
        exported = torch.export.export(plan, inputs).module()(*inputs)
        torch.testing.assert_close(exported[0], expected, atol=0, rtol=0)
    reverse = plan.credit_gradient(
        *inputs, terminal_cotangents={"output": torch.ones_like(expected)},
    )
    expected_source_credit = (
        torch.tensor([[2.0], [3.0], [5.0]]) if selection_scope == "sample"
        else torch.full((3, 1), 5.0)
    )
    torch.testing.assert_close(reverse.resource_cotangents["source"], expected_source_credit)
    gains = tuple(connection.transfer.gain for connection in connections)
    direct_gradients = torch.autograd.grad(plan(*inputs)[0].sum(), gains, allow_unused=True)
    for index, gradient in enumerate(direct_gradients):
        expected_gain_credit = (
            source[index].sum() if selection_scope == "sample" else
            source.sum() if index == 2 else source.new_zeros(())
        )
        if gradient is None:
            torch.testing.assert_close(expected_gain_credit, source.new_zeros(()))
        else:
            torch.testing.assert_close(gradient, expected_gain_credit)
    selected = graph.specialize_program_routes({"choose": "path_2"}).graph
    sealed = mechanisms.ResourceGraphCompiler.compile_program(selected, "run")
    torch.testing.assert_close(sealed(*inputs[:-1])[2], source * 5.0)
    if selection_scope == "sample" and execution_mode == "sparse" and torch.cuda.is_available():
        cuda_plan = plan.cuda()
        cuda_inputs = tuple(value.cuda() for value in inputs)
        cuda_result = torch.compile(cuda_plan, backend="inductor", fullgraph=True)(*cuda_inputs)
        torch.testing.assert_close(cuda_result[0].cpu(), expected, atol=0, rtol=0)


def test_five_candidate_route_compiles_balanced_hard_dispatch(tmp_path) -> None:
    torch._dynamo.reset()

    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(torch.tensor([[2.0]]))),
        mechanisms.TensorResource(_vector_spec("scores"),
                                  _vector_view(torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]]))),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(1, 1))),
    )
    connections = tuple(
        mechanisms.Connection(
            f"path_{index}", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
            transfer=mechanisms.LearnableAffineTransfer(gain=float(index + 1)),
        ) for index in range(5)
    )
    graph = mechanisms.ProgramGraph(
        resources, connections,
        programs={"run": (mechanisms.ProgramRoute(
            "choose", "scores", tuple(connection.connection_id for connection in connections),
        ),)},
    )
    plan = mechanisms.ResourceGraphCompiler.compile_program_outputs(
        graph, "run", ("output",), credit=True,
    )
    inputs = (*(resource.resolve().view.value for resource in resources),
              torch.zeros((1, plan.body.arrival_width), dtype=torch.bool))
    compiled = torch.compile(plan, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(*inputs)[0], torch.tensor([[10.0]]))
    forced = plan.body.forward_with_route_selections(
        *inputs, route_selections=(torch.tensor(0),),
    )
    torch.testing.assert_close(forced[2], torch.tensor([[2.0]]))
    reverse = plan.credit_gradient(
        *inputs, terminal_cotangents={"output": torch.ones(1, 1)},
    )
    torch.testing.assert_close(reverse.resource_cotangents["source"], torch.tensor([[5.0]]))
    saved = mechanisms.save_program_graph(graph, tmp_path / "five-candidate-route")
    restored = mechanisms.load_program_graph(saved.tensors_path)
    assert restored.contract_fingerprint == graph.contract_fingerprint
    restored_plan = mechanisms.ResourceGraphCompiler.compile_program(restored, "run")
    torch.testing.assert_close(restored_plan(*inputs[:-1])[2], torch.tensor([[10.0]]))


def test_sample_route_without_join_accepts_live_candidate_operands() -> None:
    class _ContextGate(torch.nn.Module):
        def forward(self, context: torch.Tensor) -> torch.Tensor:
            return context

    source = torch.tensor([[2.0], [3.0]], requires_grad=True)
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(source)),
        mechanisms.TensorResource(
            _vector_spec("scores"), _vector_view(torch.tensor([[3.0, -3.0], [-3.0, 3.0]])),
        ),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(2, 1))),
    )
    gated = mechanisms.Connection(
        "gated", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
        activation=_ContextGate(),
        credit_boundary=arti.CreditBoundary(mode=arti.CreditBoundaryMode.BERNOULLI),
    )
    direct = mechanisms.Connection(
        "direct", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
        activation=_ContextGate(),
        transfer=mechanisms.LearnableAffineTransfer(gain=-3.0),
        credit_boundary=arti.CreditBoundary(mode=arti.CreditBoundaryMode.BERNOULLI),
    )
    graph = mechanisms.ProgramGraph(
        resources, (gated, direct),
        programs={"run": (mechanisms.ProgramRoute(
            "pick", "scores", ("gated", "direct"), selection_scope="sample",
        ),)},
    )
    context = torch.tensor([[1.0], [2.0]])
    mask = torch.tensor([[False], [True]])
    direct_context = torch.ones_like(context)
    direct_mask = torch.ones_like(mask)
    plan = mechanisms.ResourceGraphCompiler.compile_program(
        graph, "run",
        example_contexts={"gated": context, "direct": direct_context},
        example_credit_masks={"gated": mask, "direct": direct_mask},
    )
    assert isinstance(plan, mechanisms.StaticRoutedProgramExecutionPlan)
    assert plan.context_connection_ids == ("gated", "direct")
    assert plan.credit_mask_connection_ids == ("gated", "direct")
    values = tuple(resource.resolve().view.value for resource in resources)
    inputs = (*values, context, direct_context, mask, direct_mask)
    result = plan(*inputs)
    torch.testing.assert_close(result[2], torch.tensor([[2.0], [-9.0]]))
    changed = plan(*values, torch.tensor([[3.0], [2.0]]), direct_context, mask, direct_mask)
    torch.testing.assert_close(changed[2], torch.tensor([[6.0], [-9.0]]))
    changed_direct = plan(*values, context, torch.full_like(context, 2.0), mask, direct_mask)
    torch.testing.assert_close(changed_direct[2], torch.tensor([[2.0], [-18.0]]))
    torch.testing.assert_close(torch.autograd.grad(result[2].sum(), source)[0], torch.tensor([[0.0], [-3.0]]))
    torch._dynamo.reset()
    compiled = torch.compile(plan, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(*inputs)[2], result[2])


def test_batch_region_preserves_row_specific_view_metadata() -> None:
    value = torch.arange(4, dtype=torch.float32).unsqueeze(-1)
    mask = torch.tensor([[True], [False], [False], [True]])
    coordinates = torch.arange(4, dtype=torch.int64).reshape(4, 1, 1)
    view = mechanisms.TensorView.from_tensor(
        value,
        axis_names=("batch", "feature"),
        axis_roles=("batch", "feature"),
        index_map=mechanisms.TensorIndexMap(
            ("feature",), (4,), (1,), coordinates,
        ),
        mask=mask,
    )
    right = view.slice_batch_range(2, 4)
    torch.testing.assert_close(right.value, value[2:4])
    torch.testing.assert_close(right.mask, mask[2:4])
    torch.testing.assert_close(right.index_map.coordinates, coordinates[2:4])
    assert right.axes[0].extent == 2


def test_sample_route_keeps_offset_sensitive_cohort_plans() -> None:
    mask = torch.tensor([[False], [True]])
    coordinates = torch.arange(2, dtype=torch.int64).reshape(2, 1, 1)
    source = mechanisms.TensorResource(
        _vector_spec("source"),
        mechanisms.TensorView.from_tensor(
            torch.tensor([[2.0], [3.0]]),
            axis_names=("batch", "feature"),
            axis_roles=("batch", "feature"),
            index_map=mechanisms.TensorIndexMap(("feature",), (2,), (1,), coordinates),
            mask=mask,
        ),
    )
    scores = mechanisms.TensorResource(
        _vector_spec("scores"), _vector_view(torch.tensor([[1.0, 0.0], [0.0, 1.0]])),
    )
    output = mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(2, 1)))
    positive = mechanisms.Connection(
        "positive", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
    )
    negative = mechanisms.Connection(
        "negative", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
        transfer=mechanisms.LearnableAffineTransfer(gain=-1.0),
    )
    graph = mechanisms.ProgramGraph(
        (source, scores, output), (positive, negative),
        programs={"run": (
            mechanisms.ProgramRoute(
                "choose", "scores", ("positive", "negative"), selection_scope="sample",
            ),
        )},
    )
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    branch = plan.blocks[0]._cohort_branches["1_1"][0].plan.stages[0][0].plan
    torch.testing.assert_close(branch.templates[0].mask, mask[1:2])
    torch.testing.assert_close(branch.templates[0].index_map.coordinates, coordinates[1:2])


def test_route_selects_and_specializes_compound_program_paths(tmp_path) -> None:
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(torch.tensor([[2.0]]))),
        mechanisms.TensorResource(_vector_spec("scores"), _vector_view(torch.tensor([[2.0, -1.0]]))),
        mechanisms.TensorResource(_vector_spec("left"), _vector_view(torch.zeros(1, 1))),
        mechanisms.TensorResource(_vector_spec("right"), _vector_view(torch.zeros(1, 1))),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(1, 1))),
    )
    left_copy = mechanisms.Connection(
        "left_copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("left"),
        transfer=mechanisms.LearnableAffineTransfer(gain=2.0),
    )
    right_copy = mechanisms.Connection(
        "right_copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("right"),
        transfer=mechanisms.LearnableAffineTransfer(gain=-1.0),
    )
    left_head = arti.as_fabric_node(
        "left_head", _DecoratedScale(3.0),
        input_ports={"source": mechanisms.ResourcePort("left")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    right_head = arti.as_fabric_node(
        "right_head", _DecoratedScale(4.0),
        input_ports={"source": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (left_copy, right_copy), nodes=(left_head, right_head),
        programs={
            "left_path": ("left_copy", "left_head"),
            "right_path": ("right_copy", "right_head"),
            "run": (mechanisms.ProgramRoute(
                "choose", "scores", ("left_path", "right_path"),
            ),),
        },
    )
    execution = graph.execute_program_functional("run")
    assert execution.routes[0].candidate_id == "left_path"
    assert tuple(dispatch.frontier_path for dispatch in execution.dispatches) == (
        (0,), (0, 0), (0, 1),
    )
    assert execution.dispatches[0].route == execution.routes[0]
    assert tuple(len(dispatch.members) for dispatch in execution.dispatches) == (0, 1, 1)
    assert [item.connection_id for item in execution.connections] == ["left_copy"]
    output_state = next(
        item for item in execution.state.resources if item.spec.resource_id == "output"
    )
    torch.testing.assert_close(output_state.active_view.value, torch.tensor([[12.0]]))

    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    values = tuple(resource.resolve().view.value for resource in resources)
    torch.testing.assert_close(plan(*values)[4], torch.tensor([[12.0]]))
    torch.testing.assert_close(
        torch.compile(plan, backend="eager", fullgraph=True)(*values)[4],
        torch.tensor([[12.0]]),
    )
    forced_right = plan.blocks[0].forward_with_selection(
        *values, selection=torch.tensor(1),
    )[0]
    torch.testing.assert_close(forced_right[4], torch.tensor([[-8.0]]))
    sealed = graph.specialize_program_routes({"choose": "left_path"}).graph
    assert sealed.program("run") == ("left_copy", "left_head")
    assert "right_copy" not in sealed.connections
    assert "right_head" not in sealed.nodes
    sealed_plan = mechanisms.ResourceGraphCompiler.compile_program(sealed, "run")
    torch.testing.assert_close(sealed_plan(*values)[4], plan(*values)[4], atol=0, rtol=0)

    saved = mechanisms.save_program_graph(graph, tmp_path / "compound-route")
    restored = mechanisms.load_program_graph(saved.tensors_path, nodes=(left_head, right_head))
    assert restored.contract_fingerprint == graph.contract_fingerprint
    assert restored.execute_program_functional("run").routes[0].candidate_id == "left_path"


def test_nested_route_candidate_preserves_actual_dispatch_and_credit() -> None:
    torch.manual_seed(0)
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(torch.tensor([[2.0]]))),
        mechanisms.TensorResource(_vector_spec("outer_scores"), _vector_view(torch.tensor([[5.0, -5.0]]))),
        mechanisms.TensorResource(_vector_spec("inner_scores"), _vector_view(torch.tensor([[-5.0, 5.0]]))),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(1, 1))),
    )
    connections = tuple(
        mechanisms.Connection(
            name, mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
            transfer=mechanisms.LearnableAffineTransfer(gain=gain),
        )
        for name, gain in (("first", 2.0), ("second", 3.0), ("fallback", 4.0))
    )
    inner = mechanisms.ProgramRoute("inner", "inner_scores", ("first", "second"))
    outer = mechanisms.ProgramRoute("outer", "outer_scores", ("branch", "fallback"))
    graph = mechanisms.ProgramGraph(
        resources, connections,
        programs={"branch": (inner,), "run": (outer,)},
    )
    inner_scores = torch.tensor([[-5.0, 5.0]], requires_grad=True)
    execution = graph.execute_program_functional(
        "run", sample_routes=True,
        input_views={"inner_scores": _vector_view(inner_scores)},
    )
    assert tuple(route.route_id for route in execution.routes) == ("outer", "inner")
    assert tuple(route.candidate_id for route in execution.routes) == ("branch", "second")
    assert all(route.sampled for route in execution.routes)
    inner_credit = execution.routes[1].structure_objective(torch.tensor(1.0))
    assert torch.autograd.grad(inner_credit, inner_scores)[0].abs().sum() > 0
    assert tuple(item.connection_id for item in execution.connections) == ("second",)
    assert tuple(dispatch.frontier_path for dispatch in execution.dispatches) == (
        (0,), (0, 0),
    )
    assert tuple(dispatch.route for dispatch in execution.dispatches) == execution.routes
    assert tuple(item.connection_id for item in execution.dispatches[1].members) == ("second",)
    result = next(item for item in execution.state.resources if item.spec.resource_id == "output")
    torch.testing.assert_close(result.active_view.value, torch.tensor([[6.0]]))
    bypass = graph.execute_program_functional(
        "run", input_views={"outer_scores": _vector_view(torch.tensor([[-5.0, 5.0]]))},
    )
    assert tuple(route.route_id for route in bypass.routes) == ("outer",)
    assert tuple(item.connection_id for item in bypass.connections) == ("fallback",)
    bypass_result = next(item for item in bypass.state.resources if item.spec.resource_id == "output")
    torch.testing.assert_close(bypass_result.active_view.value, torch.tensor([[8.0]]))
    continuation = mechanisms.TensorResource(
        mechanisms.TensorResourceSpec(
            "continue", mechanisms.TensorViewPattern(
                min_rank=1, max_rank=1, allowed_axis_roles=("batch",),
            ),
        ),
        mechanisms.TensorView.from_tensor(
            torch.ones(1), axis_names=("batch",), axis_roles=("batch",),
        ),
    )
    loop_graph = mechanisms.ProgramGraph(
        (*resources, continuation), connections,
        programs={"branch": (inner,), "run": (outer,)},
        loops=(mechanisms.ProgramLoop("repeat", "run", "continue", max_iterations=2),),
    )
    loop_execution = loop_graph.execute_loop_functional("repeat", sample_routes=True)
    assert tuple(route.route_id for route in loop_execution.routes) == (
        "outer", "inner", "outer", "inner",
    )
    assert tuple(route.iteration for route in loop_execution.routes) == (0, 0, 1, 1)
    torch.testing.assert_close(
        loop_execution.structure_objective(torch.tensor(1.0)),
        sum((route.log_probability for route in loop_execution.routes)),
    )
    dynamic_plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    dynamic_inputs = graph.static_program_inputs(dynamic_plan)
    torch.testing.assert_close(dynamic_plan(*dynamic_inputs)[3], torch.tensor([[6.0]]))
    compiled_dynamic = torch.compile(dynamic_plan, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled_dynamic(*dynamic_inputs)[3], torch.tensor([[6.0]]))
    first_scores = (*dynamic_inputs[:2], torch.tensor([[5.0, -5.0]]), *dynamic_inputs[3:])
    torch.testing.assert_close(compiled_dynamic(*first_scores)[3], torch.tensor([[4.0]]))
    bypass_scores = (
        dynamic_inputs[0], torch.tensor([[-5.0, 5.0]]), *dynamic_inputs[2:],
    )
    torch.testing.assert_close(compiled_dynamic(*bypass_scores)[3], torch.tensor([[8.0]]))
    if torch.cuda.is_available():
        cuda_plan = copy.deepcopy(dynamic_plan).cuda()
        cuda_inputs = tuple(value.cuda() for value in dynamic_inputs)
        compiled_cuda = torch.compile(cuda_plan, backend="inductor", fullgraph=True)
        torch.testing.assert_close(compiled_cuda(*cuda_inputs)[3].cpu(), torch.tensor([[6.0]]))
        cuda_bypass = tuple(value.cuda() for value in bypass_scores)
        torch.testing.assert_close(compiled_cuda(*cuda_bypass)[3].cpu(), torch.tensor([[8.0]]))
    projected = mechanisms.ResourceGraphCompiler.compile_program_outputs(
        graph, "run", ("output",),
    )
    torch.testing.assert_close(projected(*dynamic_inputs)[0], torch.tensor([[6.0]]))
    credit_scores = torch.tensor([[-5.0, 5.0]], requires_grad=True)
    credit_inputs = (*dynamic_inputs[:2], credit_scores, *dynamic_inputs[3:])
    routed, choices, log_probability = dynamic_plan.forward_with_route_credit(*credit_inputs)
    torch.testing.assert_close(routed[3], torch.tensor([[6.0]]))
    assert tuple(choice.item() for choice in choices) == (0, 1)
    assert torch.autograd.grad(log_probability, credit_scores)[0].abs().sum() > 0
    replayed, replay_choices, replay_term = dynamic_plan._forward_with_routes(
        *credit_inputs, route_selections=choices,
    )
    torch.testing.assert_close(replayed[3], routed[3], atol=0, rtol=0)
    assert all(torch.equal(left, right) for left, right in zip(replay_choices, choices, strict=True))
    torch.testing.assert_close(replay_term, log_probability, atol=0, rtol=0)
    source_for_credit = dynamic_inputs[0].detach().requires_grad_(True)
    reverse_inputs = (source_for_credit, *dynamic_inputs[1:])
    reverse = dynamic_plan.credit_gradient(
        *reverse_inputs, terminal_cotangents={"output": torch.ones_like(routed[3])},
        route_selections=choices,
    )
    reference_output = dynamic_plan.forward_with_route_selections(
        *reverse_inputs, route_selections=choices,
    )[3]
    reference_parameters = tuple(reverse.parameters.values())
    reference_gradients = torch.autograd.grad(
        reference_output.sum(), (source_for_credit, *reference_parameters),
        allow_unused=True,
    )
    torch.testing.assert_close(reverse.resource_cotangents["source"], reference_gradients[0])
    for name, expected in zip(reverse.parameters, reference_gradients[1:], strict=True):
        actual = reverse.parameter_cotangents[name]
        if expected is None:
            assert actual is None or torch.count_nonzero(actual) == 0
        else:
            torch.testing.assert_close(actual, expected)
    assert tuple(choice.item() for choice in reverse.route_selections) == (0, 1)
    compiled_replay = torch.compile(
        dynamic_plan.forward_with_route_selections, backend="eager", fullgraph=True,
    )
    torch.testing.assert_close(
        compiled_replay(*credit_inputs, route_selections=choices)[3], routed[3], atol=0, rtol=0,
    )
    compiled_credit = torch.compile(
        dynamic_plan.forward_with_route_credit, backend="eager", fullgraph=True,
    )
    compiled_routed, compiled_choices, compiled_term = compiled_credit(*credit_inputs)
    torch.testing.assert_close(compiled_routed[3], routed[3])
    assert tuple(choice.item() for choice in compiled_choices) == (0, 1)
    assert torch.autograd.grad(compiled_term, credit_scores)[0].abs().sum() > 0
    bypassed, bypass_choices, bypass_log_probability = dynamic_plan.forward_with_route_credit(
        *bypass_scores,
    )
    torch.testing.assert_close(bypassed[3], torch.tensor([[8.0]]))
    assert tuple(choice.item() for choice in bypass_choices) == (1, -1)
    assert torch.isfinite(bypass_log_probability)
    replayed_bypass = dynamic_plan.forward_with_route_selections(
        *bypass_scores, route_selections=bypass_choices,
    )
    torch.testing.assert_close(replayed_bypass[3], bypassed[3], atol=0, rtol=0)
    torch.testing.assert_close(
        bypass_log_probability,
        torch.log_softmax(bypass_scores[1].mean(0), dim=0)[1],
    )
    with pytest.raises(mechanisms.ResourceGraphError, match="requires selections"):
        graph.specialize_program_routes({"outer": "branch"})
    inner_only = graph.specialize_program_routes({"inner": "second"}).graph
    assert inner_only.program("branch") == ("second",)
    inner_plan = mechanisms.ResourceGraphCompiler.compile_program(inner_only, "run")
    original_values = tuple(resource.resolve().view.value for resource in resources)
    torch.testing.assert_close(inner_plan(*original_values)[3], torch.tensor([[6.0]]))
    sealed = graph.specialize_program_routes({
        "outer": "branch", "inner": "second",
    }).graph
    assert sealed.program("run") == ("second",)
    assert "first" not in sealed.connections
    assert "fallback" not in sealed.connections
    sealed_plan = mechanisms.ResourceGraphCompiler.compile_program(sealed, "run")
    torch.testing.assert_close(sealed_plan(*original_values)[3], torch.tensor([[6.0]]))
    torch.testing.assert_close(
        torch.compile(sealed_plan, backend="eager", fullgraph=True)(*original_values)[3],
        torch.tensor([[6.0]]),
    )
    bypass_sealed = graph.specialize_program_routes({"outer": "fallback"}).graph
    assert bypass_sealed.program("run") == ("fallback",)

    assert "branch" not in bypass_sealed.contract_config()["programs"]
    assert "first" not in bypass_sealed.connections
    assert "second" not in bypass_sealed.connections

    with pytest.raises(mechanisms.ResourceGraphError, match="must not form a cycle"):
        mechanisms.ProgramGraph(
            resources, connections,
            programs={
                "first_path": (mechanisms.ProgramRoute(
                    "first_route", "outer_scores", ("second_path", "first"),
                ),),
                "second_path": (mechanisms.ProgramRoute(
                    "second_route", "inner_scores", ("first_path", "second"),
                ),),
            },
        )


def test_nested_batch_route_credit_ignores_inactive_rows() -> None:
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(torch.ones(2, 1))),
        mechanisms.TensorResource(_vector_spec("outer_scores"), _vector_view(
            torch.tensor([[8.0, -8.0], [-8.0, 8.0]]),
        )),
        mechanisms.TensorResource(_vector_spec("inner_scores"), _vector_view(
            torch.tensor([[-8.0, 8.0], [8.0, -8.0]]),
        )),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(2, 1))),
    )
    connections = tuple(
        mechanisms.Connection(
            name, mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
            transfer=mechanisms.LearnableAffineTransfer(gain=gain),
        )
        for name, gain in (("first", 2.0), ("second", 3.0), ("fallback", 4.0))
    )
    graph = mechanisms.ProgramGraph(
        resources, connections,
        programs={
            "branch": (mechanisms.ProgramRoute("inner", "inner_scores", ("first", "second")),),
            "run": (mechanisms.ProgramRoute("outer", "outer_scores", ("branch", "fallback")),),
        },
    )
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    inputs = graph.static_program_inputs(plan)
    active = torch.tensor([True, False])
    output, choices, _ = plan.forward_with_route_credit(*inputs, active_rows=active)
    assert tuple(choice.item() for choice in choices) == (0, 1)
    torch.testing.assert_close(output[3], torch.full((2, 1), 3.0))
    compiled_credit = torch.compile(
        plan.forward_with_route_credit, backend="eager", fullgraph=True,
    )
    compiled_output, compiled_choices, _ = compiled_credit(*inputs, active_rows=active)
    torch.testing.assert_close(compiled_output[3], output[3])
    assert tuple(choice.item() for choice in compiled_choices) == (0, 1)
    reverse = plan.credit_gradient(
        *inputs, terminal_cotangents={"output": torch.ones(2, 1)},
        route_selections=choices, active_rows=active,
    )
    assert tuple(choice.item() for choice in reverse.route_selections) == (0, 1)


@pytest.mark.parametrize("outer_score, flip_score, route_steps, expected_gain", (
    (5.0, False, (("outer", "inner"), ("outer", "inner")), 3.0),
    (-5.0, False, (("outer",), ("outer",)), 4.0),
    (5.0, True, (("outer", "inner"), ("outer",)), 4.0),
    (-5.0, True, (("outer",), ("outer", "inner")), 3.0),
))
def test_open_loop_replays_actual_nested_route_credit(
    outer_score: float, flip_score: bool,
    route_steps: tuple[tuple[str, ...], ...], expected_gain: float,
) -> None:
    source = torch.tensor([[2.0]], requires_grad=True)
    credit_mask = torch.ones(1, 1, dtype=torch.bool)
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(source)),
        mechanisms.TensorResource(
            _vector_spec("outer_scores"), _vector_view(torch.tensor([[outer_score, -outer_score]])),
        ),
        mechanisms.TensorResource(
            _vector_spec("inner_scores"), _vector_view(torch.tensor([[-5.0, 5.0]])),
        ),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(1, 1))),
        mechanisms.TensorResource(
            mechanisms.TensorResourceSpec(
                "continue", mechanisms.TensorViewPattern(
                    min_rank=1, max_rank=1, allowed_axis_roles=("batch",),
                ),
            ),
            mechanisms.TensorView.from_tensor(
                torch.ones(1), axis_names=("batch",), axis_roles=("batch",),
            ),
        ),
    )
    writes = tuple(
        mechanisms.Connection(
            name, mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
            transfer=mechanisms.LearnableAffineTransfer(gain=gain),
            credit_boundary=(
                arti.CreditBoundary(mode=arti.CreditBoundaryMode.BERNOULLI)
                if name == "second" else None
            ),
        )
        for name, gain in (("first", 2.0), ("second", 3.0), ("fallback", 4.0))
    )
    flip = mechanisms.Connection(
        "flip_scores", mechanisms.ResourcePort("outer_scores"),
        mechanisms.ResourcePort("outer_scores"),
        transfer=mechanisms.LearnableAffineTransfer(gain=-1.0),
    )
    graph = mechanisms.ProgramGraph(
        resources, (*writes, *((flip,) if flip_score else ())),
        programs={
            "branch": (mechanisms.ProgramRoute("inner", "inner_scores", ("first", "second")),),
            "run": (
                mechanisms.ProgramRoute("outer", "outer_scores", ("branch", "fallback")),
                *(("flip_scores",) if flip_score else ()),
            ),
        },
        loops=(mechanisms.ProgramLoop("repeat", "run", "continue", max_iterations=None),),
    )
    execution = graph.execute_loop_functional(
        "repeat", host_step_limit=2, credit_masks={"second": credit_mask},
    )
    assert execution.actual_iterations == 2
    assert tuple(
        tuple(route.route_id for route in execution.routes if route.iteration == iteration)
        for iteration in range(2)
    ) == route_steps
    plan = mechanisms.ResourceGraphCompiler.compile_loop(
        graph, "repeat", example_credit_masks={"second": credit_mask},
    )
    inputs = tuple(graph.resource(resource_id).resolve().view.value for resource_id in plan.resource_ids)
    inputs = (*inputs, credit_mask, torch.zeros((1, plan.body.arrival_width), dtype=torch.bool))
    reverse = plan.credit_gradient(
        *inputs, execution=execution,
        terminal_cotangents={"output": torch.ones(1, 1)}, create_graph=False,
    )
    eager_output = next(
        item.active_view.value for item in execution.state.resources
        if item.spec.resource_id == "output"
    )
    torch.testing.assert_close(eager_output, torch.tensor([[2.0 * expected_gain]]))
    torch.testing.assert_close(reverse.resource_values[plan.resource_ids.index("output")], eager_output)
    selected_gain = graph.connection(
        "second" if len(route_steps[-1]) == 2 else "fallback"
    ).transfer.gain
    source_gradient, gain_gradient = torch.autograd.grad(
        eager_output.sum(), (source, selected_gain),
    )
    torch.testing.assert_close(
        reverse.resource_cotangents["source"], source_gradient,
    )
    gain_name = next(
        name for name, parameter in reverse.parameters.items() if parameter is selected_gain
    )
    torch.testing.assert_close(reverse.parameter_cotangents[gain_name], gain_gradient)
    assert tuple(
        item.credit_mask for item in execution.connections if item.connection_id == "second"
    ) == (credit_mask,) * sum(len(step) == 2 for step in route_steps)
    assert tuple(tuple(choice.item() for choice in step) for step in reverse.route_selections) == (
        tuple((0, 1) if len(step) == 2 else (1, -1) for step in route_steps)
    )
    recorded = plan.forward_until_done_with_route_receipt(
        *inputs, host_step_limit=torch.tensor(2, dtype=torch.int64), receipt_capacity=2,
        credit_mask_histories=(credit_mask.expand(2, -1, -1),),
    )
    if flip_score and outer_score > 0:
        compiled = torch.compile(
            plan.forward_until_done_with_route_receipt, backend="eager", fullgraph=True,
        )(
            *inputs, host_step_limit=torch.tensor(2, dtype=torch.int64), receipt_capacity=2,
            credit_mask_histories=(credit_mask.expand(2, -1, -1),),
        )
        for actual, reference in zip(compiled, recorded, strict=True):
            torch.testing.assert_close(actual, reference)
    tail = len(plan.resource_ids) + 1
    tensor_receipt = (recorded[tail], recorded[tail + 3], recorded[tail + 4], *recorded[tail + 5:])
    tensor_reverse = plan.credit_gradient(
        *inputs, tensor_receipt=tensor_receipt,
        terminal_cotangents={"output": torch.ones(1, 1)}, create_graph=False,
    )
    torch.testing.assert_close(recorded[plan.resource_ids.index("output")], eager_output)
    torch.testing.assert_close(
        tensor_reverse.resource_cotangents["source"], reverse.resource_cotangents["source"],
    )
    torch.testing.assert_close(
        tensor_reverse.parameter_cotangents[gain_name], gain_gradient,
    )


@pytest.mark.parametrize("second_branch, flip_inner, expected_choices, expected_gain", (
    (True, False, (0, 1, 0, 1), 3.0),
    (False, False, (0, 1, 1, -1), 4.0),
    (True, True, (0, 1, 0, 0), 2.0),
))
def test_open_loop_replays_reused_child_route_slots(
    second_branch: bool, flip_inner: bool,
    expected_choices: tuple[int, ...], expected_gain: float,
) -> None:
    source = torch.tensor([[2.0]], requires_grad=True)
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(source)),
        mechanisms.TensorResource(
            _vector_spec("first_scores"), _vector_view(torch.tensor([[5.0, -5.0]])),
        ),
        mechanisms.TensorResource(
            _vector_spec("second_scores"),
            _vector_view(torch.tensor([[5.0, -5.0]] if second_branch else [[-5.0, 5.0]])),
        ),
        mechanisms.TensorResource(
            _vector_spec("inner_scores"), _vector_view(torch.tensor([[-5.0, 5.0]])),
        ),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(1, 1))),
        mechanisms.TensorResource(
            mechanisms.TensorResourceSpec(
                "continue", mechanisms.TensorViewPattern(
                    min_rank=1, max_rank=1, allowed_axis_roles=("batch",),
                ),
            ),
            mechanisms.TensorView.from_tensor(
                torch.ones(1), axis_names=("batch",), axis_roles=("batch",),
            ),
        ),
    )
    connections = tuple(
        mechanisms.Connection(
            name, mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
            transfer=mechanisms.LearnableAffineTransfer(gain=gain),
            credit_boundary=(
                arti.CreditBoundary(mode=arti.CreditBoundaryMode.BERNOULLI)
                if name == "second" else None
            ),
        )
        for name, gain in (("first", 2.0), ("second", 3.0), ("fallback", 4.0))
    )
    flip = mechanisms.Connection(
        "flip_inner", mechanisms.ResourcePort("inner_scores"),
        mechanisms.ResourcePort("inner_scores"),
        transfer=mechanisms.LearnableAffineTransfer(gain=-1.0),
    )
    graph = mechanisms.ProgramGraph(
        resources, (*connections, *((flip,) if flip_inner else ())),
        programs={
            "child": (mechanisms.ProgramRoute("inner", "inner_scores", ("first", "second")),),
            "run": (
                mechanisms.ProgramRoute("outer_a", "first_scores", ("child", "fallback")),
                *(("flip_inner",) if flip_inner else ()),
                mechanisms.ProgramRoute("outer_b", "second_scores", ("child", "fallback")),
            ),
        },
        loops=(mechanisms.ProgramLoop("repeat", "run", "continue", max_iterations=None),),
    )
    credit_mask = torch.tensor(True)
    execution = graph.execute_loop_functional(
        "repeat", host_step_limit=1, credit_masks={"second": credit_mask},
    )
    plan = mechanisms.ResourceGraphCompiler.compile_loop(
        graph, "repeat", example_credit_masks={"second": credit_mask},
    )
    assert plan.body.route_receipt_ids == ("outer_a", "inner", "outer_b", "inner")
    inputs = tuple(graph.resource(resource_id).resolve().view.value for resource_id in plan.resource_ids)
    inputs = (*inputs, credit_mask, torch.zeros((1, plan.body.arrival_width), dtype=torch.bool))
    reverse = plan.credit_gradient(
        *inputs, execution=execution,
        terminal_cotangents={"output": torch.ones(1, 1)}, create_graph=False,
    )
    assert tuple(tuple(choice.item() for choice in step) for step in reverse.route_selections) == (
        expected_choices,
    )
    eager_output = next(
        item.active_view.value for item in execution.state.resources
        if item.spec.resource_id == "output"
    )
    selected_gain = graph.connection(
        "first" if second_branch and flip_inner else "second" if second_branch else "fallback"
    ).transfer.gain
    source_gradient, gain_gradient = torch.autograd.grad(
        eager_output.sum(), (source, selected_gain),
    )
    torch.testing.assert_close(eager_output, torch.tensor([[2.0 * expected_gain]]))
    torch.testing.assert_close(reverse.resource_cotangents["source"], source_gradient)
    gain_name = next(
        name for name, parameter in reverse.parameters.items() if parameter is selected_gain
    )
    torch.testing.assert_close(reverse.parameter_cotangents[gain_name], gain_gradient)
    recorded = plan.forward_until_done_with_route_receipt(
        *inputs, host_step_limit=torch.tensor(1, dtype=torch.int64), receipt_capacity=1,
    )
    if flip_inner:
        compiled = torch.compile(
            plan.forward_until_done_with_route_receipt, backend="eager", fullgraph=True,
        )(
            *inputs, host_step_limit=torch.tensor(1, dtype=torch.int64), receipt_capacity=1,
        )
        for actual, reference in zip(compiled, recorded, strict=True):
            torch.testing.assert_close(actual, reference)
    tail = len(plan.resource_ids) + 1
    tensor_receipt = (recorded[tail], recorded[tail + 3], recorded[tail + 4], *recorded[tail + 5:])
    tensor_reverse = plan.credit_gradient(
        *inputs, tensor_receipt=tensor_receipt,
        terminal_cotangents={"output": torch.ones(1, 1)}, create_graph=False,
    )
    torch.testing.assert_close(recorded[plan.resource_ids.index("output")], eager_output)
    torch.testing.assert_close(
        tensor_reverse.resource_cotangents["source"], reverse.resource_cotangents["source"],
    )


def test_open_loop_reuses_automatic_credit_sample_within_iteration(monkeypatch) -> None:
    source = torch.tensor([[2.0]], requires_grad=True)
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(source)),
        mechanisms.TensorResource(_vector_spec("first_scores"), _vector_view(torch.tensor([[5.0, -5.0]]))),
        mechanisms.TensorResource(_vector_spec("second_scores"), _vector_view(torch.tensor([[5.0, -5.0]]))),
        mechanisms.TensorResource(_vector_spec("inner_scores"), _vector_view(torch.tensor([[-5.0, 5.0]]))),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(1, 1))),
        mechanisms.TensorResource(
            mechanisms.TensorResourceSpec(
                "continue", mechanisms.TensorViewPattern(
                    min_rank=1, max_rank=1, allowed_axis_roles=("batch",),
                ),
            ),
            mechanisms.TensorView.from_tensor(
                torch.ones(1), axis_names=("batch",), axis_roles=("batch",),
            ),
        ),
    )
    connections = tuple(
        mechanisms.Connection(
            name, mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
            transfer=mechanisms.LearnableAffineTransfer(gain=gain),
            credit_boundary=(
                arti.CreditBoundary(mode=arti.CreditBoundaryMode.BERNOULLI)
                if name == "second" else None
            ),
        )
        for name, gain in (("first", 2.0), ("second", 3.0), ("fallback", 4.0))
    )
    graph = mechanisms.ProgramGraph(
        resources, connections,
        programs={
            "child": (mechanisms.ProgramRoute("inner", "inner_scores", ("first", "second")),),
            "run": (
                mechanisms.ProgramRoute("outer_a", "first_scores", ("child", "fallback")),
                mechanisms.ProgramRoute("outer_b", "second_scores", ("child", "fallback")),
            ),
        },
        loops=(mechanisms.ProgramLoop("repeat", "run", "continue", max_iterations=None),),
    )
    draws = iter((0.1, 0.9))
    sampled_values: list[float] = []

    def fake_rand(*_shape, device=None):
        value = next(draws)
        sampled_values.append(value)
        return torch.tensor(value, device=device)

    monkeypatch.setattr(torch, "rand", fake_rand)
    execution = graph.execute_loop_functional("repeat", host_step_limit=2)
    second_masks = tuple(
        item.credit_mask for item in execution.connections if item.connection_id == "second"
    )
    assert sampled_values == [0.1, 0.9]
    assert tuple(bool(mask) for mask in second_masks) == (True, True, False, False)
    assert second_masks[0] is second_masks[1]
    assert second_masks[2] is second_masks[3]

    monkeypatch.undo()
    plan = mechanisms.ResourceGraphCompiler.compile_loop(
        graph, "repeat", example_credit_masks={"second": torch.tensor(True)},
    )
    inputs = tuple(resource.resolve().view.value for resource in resources)
    inputs = (*inputs, torch.tensor(True), torch.zeros((1, plan.body.arrival_width), dtype=torch.bool))
    eager_reverse = plan.credit_gradient(
        *inputs, execution=execution,
        terminal_cotangents={"output": torch.ones(1, 1)}, create_graph=False,
    )
    eager_output = next(
        item.active_view.value for item in execution.state.resources
        if item.spec.resource_id == "output"
    )
    eager_source_gradient = torch.autograd.grad(eager_output.sum(), source)[0]
    torch.testing.assert_close(eager_reverse.resource_cotangents["source"], eager_source_gradient)

    mask_history = torch.tensor([True, False], dtype=torch.bool)
    recorded = plan.forward_until_done_with_route_receipt(
        *inputs, host_step_limit=torch.tensor(2, dtype=torch.int64), receipt_capacity=2,
        credit_mask_histories=(mask_history,),
    )
    compiled = torch.compile(
        plan.forward_until_done_with_route_receipt, backend="eager", fullgraph=True,
    )(
        *inputs, host_step_limit=torch.tensor(2, dtype=torch.int64), receipt_capacity=2,
        credit_mask_histories=(mask_history,),
    )
    for actual, expected in zip(compiled, recorded, strict=True):
        torch.testing.assert_close(actual, expected)
    tail = len(plan.resource_ids) + 1
    tensor_receipt = (
        recorded[tail], recorded[tail + 3], recorded[tail + 4], *recorded[tail + 5:],
    )
    tensor_reverse = plan.credit_gradient(
        *inputs, tensor_receipt=tensor_receipt,
        terminal_cotangents={"output": torch.ones_like(recorded[plan.resource_ids.index("output")])},
        create_graph=False,
    )
    torch.testing.assert_close(
        tensor_reverse.resource_cotangents["source"], eager_reverse.resource_cotangents["source"],
    )


def test_nested_route_receipt_keeps_candidate_paths_distinct() -> None:
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(torch.ones(1, 1))),
        mechanisms.TensorResource(_vector_spec("outer_scores"), _vector_view(torch.tensor([[8.0, -8.0]]))),
        mechanisms.TensorResource(_vector_spec("a_scores"), _vector_view(torch.tensor([[-8.0, 8.0]]))),
        mechanisms.TensorResource(_vector_spec("b_scores"), _vector_view(torch.tensor([[8.0, -8.0]]))),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(1, 1))),
    )
    connections = tuple(
        mechanisms.Connection(
            name, mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
            transfer=mechanisms.LearnableAffineTransfer(gain=gain),
        )
        for name, gain in (("a0", 2.0), ("a1", 3.0), ("b0", 4.0), ("b1", 5.0))
    )
    graph = mechanisms.ProgramGraph(
        resources, connections,
        programs={
            "path_a": (mechanisms.ProgramRoute("inner_a", "a_scores", ("a0", "a1")),),
            "path_b": (mechanisms.ProgramRoute("inner_b", "b_scores", ("b0", "b1")),),
            "run": (mechanisms.ProgramRoute("outer", "outer_scores", ("path_a", "path_b")),),
        },
    )
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    assert plan.route_receipt_ids == ("outer", "inner_a", "inner_b")
    a_scores = torch.tensor([[-8.0, 8.0]], requires_grad=True)
    b_scores = torch.tensor([[8.0, -8.0]], requires_grad=True)
    inputs = graph.static_program_inputs(plan)
    first_inputs = (*inputs[:2], a_scores, b_scores, *inputs[4:])
    output, choices, term = plan.forward_with_route_credit(*first_inputs)
    torch.testing.assert_close(output[4], torch.tensor([[3.0]]))
    assert tuple(choice.item() for choice in choices) == (0, 1, -1)
    gradients = torch.autograd.grad(term, (a_scores, b_scores), allow_unused=True)
    assert gradients[0] is not None and gradients[0].abs().sum() > 0
    assert gradients[1] is None or torch.count_nonzero(gradients[1]) == 0
    torch.testing.assert_close(
        plan.forward_with_route_selections(*first_inputs, route_selections=choices)[4],
        output[4], atol=0, rtol=0,
    )
    second_inputs = (inputs[0], torch.tensor([[-8.0, 8.0]]), *first_inputs[2:])
    other, other_choices, _ = plan.forward_with_route_credit(*second_inputs)
    torch.testing.assert_close(other[4], torch.tensor([[4.0]]))
    assert tuple(choice.item() for choice in other_choices) == (1, -1, 0)
    torch.testing.assert_close(
        plan.forward_with_route_selections(*second_inputs, route_selections=other_choices)[4],
        other[4], atol=0, rtol=0,
    )
    continuation = mechanisms.TensorResource(
        mechanisms.TensorResourceSpec(
            "continue", mechanisms.TensorViewPattern(
                min_rank=1, max_rank=1, allowed_axis_roles=("batch",),
            ),
        ),
        mechanisms.TensorView.from_tensor(
            torch.ones(1), axis_names=("batch",), axis_roles=("batch",),
        ),
    )
    loop_graph = mechanisms.ProgramGraph(
        (*resources, continuation), connections,
        programs={name: graph.program(name) for name in ("path_a", "path_b", "run")},
        loops=(mechanisms.ProgramLoop("open", "run", "continue", max_iterations=None),),
    )
    loop_plan = mechanisms.ResourceGraphCompiler.compile_loop(loop_graph, "open")
    loop_inputs = (
        *(resource.resolve().view.value for resource in (*resources, continuation)),
        torch.zeros((1, loop_plan.body.arrival_width), dtype=torch.bool),
    )
    recorded = loop_plan.forward_until_done_with_route_receipt(
        *loop_inputs, host_step_limit=torch.tensor(2, dtype=torch.int64), receipt_capacity=3,
    )
    compiled_recorded = torch.compile(
        loop_plan.forward_until_done_with_route_receipt, backend="eager", fullgraph=True,
    )(
        *loop_inputs, host_step_limit=torch.tensor(2, dtype=torch.int64), receipt_capacity=3,
    )
    for actual, expected in zip(compiled_recorded, recorded, strict=True):
        torch.testing.assert_close(actual, expected)
    output_index = loop_plan.resource_ids.index("output")
    torch.testing.assert_close(recorded[output_index], torch.tensor([[3.0]]))
    tail = len(loop_plan.resource_ids) + 1
    receipt = (recorded[tail], recorded[tail + 3], recorded[tail + 4], *recorded[tail + 5:])
    assert len(receipt) == 6
    torch.testing.assert_close(receipt[0], torch.tensor(2))
    torch.testing.assert_close(receipt[3][:2], torch.tensor([0, 0]))
    torch.testing.assert_close(receipt[4][:2], torch.tensor([1, 1]))
    torch.testing.assert_close(receipt[5][:2], torch.tensor([-1, -1]))
    replayed = loop_plan.replay_tensor_receipt(*loop_inputs, tensor_receipt=receipt)
    torch.testing.assert_close(replayed[output_index], recorded[output_index], atol=0, rtol=0)
    compiled_replay = torch.compile(
        loop_plan.replay_tensor_receipt, backend="eager", fullgraph=True,
    )(*loop_inputs, tensor_receipt=receipt)
    torch.testing.assert_close(compiled_replay[output_index], replayed[output_index])
    reverse = loop_plan.credit_gradient(
        *loop_inputs, tensor_receipt=receipt,
        terminal_cotangents={"output": torch.ones_like(recorded[output_index])},
        create_graph=False,
    )
    expected_gain = torch.autograd.grad(replayed[output_index].sum(), connections[1].transfer.gain)[0]
    gain_name = next(
        name for name, parameter in reverse.parameters.items()
        if parameter is connections[1].transfer.gain
    )
    torch.testing.assert_close(reverse.parameter_cotangents[gain_name], expected_gain)
    assert reverse.route_ids == ("outer", "inner_a", "inner_b")
    sampled_a_scores = torch.tensor([[-8.0, 8.0]], requires_grad=True)
    sampled_b_scores = torch.tensor([[8.0, -8.0]], requires_grad=True)
    sampled_inputs = (*loop_inputs[:2], sampled_a_scores, sampled_b_scores, *loop_inputs[4:])
    sampled = loop_plan.forward_until_done_with_route_receipt(
        *sampled_inputs, host_step_limit=torch.tensor(2, dtype=torch.int64),
        receipt_capacity=2, sample_routes=True,
    )
    sampled_receipt = (
        sampled[tail], sampled[tail + 3], sampled[tail + 4], *sampled[tail + 5:],
    )
    sampled_reverse = loop_plan.credit_gradient(
        *sampled_inputs, tensor_receipt=sampled_receipt,
        terminal_cotangents={"output": torch.ones_like(sampled[output_index])},
        create_graph=False,
    )
    assert sampled_reverse.route_log_probability is not None
    score_gradients = torch.autograd.grad(
        sampled_reverse.route_log_probability,
        (sampled_a_scores, sampled_b_scores), allow_unused=True,
    )
    assert score_gradients[0] is not None and score_gradients[0].abs().sum() > 0
    assert score_gradients[1] is None or torch.count_nonzero(score_gradients[1]) == 0
    if torch.cuda.is_available():
        cuda_plan = copy.deepcopy(loop_plan).cuda()
        cuda_inputs = tuple(value.cuda() for value in loop_inputs)
        cuda_recorded = torch.compile(
            cuda_plan.forward_until_done_with_route_receipt,
            backend="inductor", fullgraph=True,
        )(
            *cuda_inputs, host_step_limit=torch.tensor(2, dtype=torch.int64, device="cuda"),
            receipt_capacity=3,
        )
        for actual, expected in zip(cuda_recorded, recorded, strict=True):
            torch.testing.assert_close(actual.cpu(), expected)
        cuda_receipt = (
            cuda_recorded[tail], cuda_recorded[tail + 3], cuda_recorded[tail + 4],
            *cuda_recorded[tail + 5:],
        )
        cuda_replay = torch.compile(
            cuda_plan.replay_tensor_receipt, backend="inductor", fullgraph=True,
        )(*cuda_inputs, tensor_receipt=cuda_receipt)
        torch.testing.assert_close(cuda_replay[output_index].cpu(), replayed[output_index])


def test_trajectory_fate_credit_handles_reused_nested_route_slots() -> None:
    fate = mechanisms.ProgramFateSample(
        selections=(), log_probability=torch.tensor(-0.7),
        fate_log_probabilities=(torch.tensor(-0.7),),
        fate_route_paths=(((('shared', 1),),),),
    )
    activity = (torch.tensor([True]),)
    for choices in (
        (torch.tensor(0), torch.tensor(1), torch.tensor(-1)),
        (torch.tensor(1), torch.tensor(-1), torch.tensor(1)),
    ):
        torch.testing.assert_close(
            fate.trajectory_structure_objective(
                torch.tensor(2.0), route_ids=("outer", "shared", "shared"),
                route_selections=(choices,), iteration_active=activity,
            ),
            torch.tensor(-1.4),
        )
    torch.testing.assert_close(
        fate.trajectory_structure_objective(
            torch.tensor(2.0), route_ids=("outer", "shared", "shared"),
            route_selections=((torch.tensor(0), torch.tensor(-1), torch.tensor(-1)),),
            iteration_active=activity,
        ),
        torch.tensor(0.0),
    )


def test_trajectory_fate_credit_uses_only_rows_that_reached_the_node() -> None:
    logits = torch.nn.Parameter(torch.tensor([0.3, -0.4]))
    fate = mechanisms.ProgramFateSample(
        selections=(("reader", "stored"),),
        log_probability=torch.log_softmax(logits, dim=0)[0],
        fate_log_probabilities=(torch.log_softmax(logits, dim=0)[0],),
        fate_route_paths=(((('outer', 0), ('inner', 1)),),),
        route_selection_scopes=(("outer", "sample"), ("inner", "sample")),
    )
    loss = torch.tensor([2.0, 10.0, 100.0, 1000.0])
    credit = fate.trajectory_structure_objective(
        loss, baseline=1.0, route_ids=("outer", "inner"),
        route_selections=(
            (torch.tensor([0, 0, 1, 0]), torch.tensor([1, 0, -1, 1])),
            (torch.tensor([1, 0, 0, 0]), torch.tensor([-1, 1, 1, 1])),
        ),
        iteration_active=(
            torch.tensor([True, True, True, False]),
            torch.tensor([True, True, False, False]),
        ),
    )
    torch.testing.assert_close(credit, fate.log_probability * 2.5)
    expected = 2.5 * (torch.tensor([1.0, 0.0]) - torch.softmax(logits.detach(), dim=0))
    torch.testing.assert_close(torch.autograd.grad(credit, logits)[0], expected)


def test_loop_route_credit_uses_actual_rows_and_global_route_mean() -> None:
    row_logits = torch.nn.Parameter(torch.tensor([[0.3, -0.4], [-0.1, 0.2], [0.5, 0.0]]))
    global_logits = torch.nn.Parameter(torch.tensor([0.2, -0.3]))
    row_terms = torch.log_softmax(row_logits, dim=-1)[:, 0]
    global_term = torch.log_softmax(global_logits, dim=-1)[1]
    result = arti.StaticDataflowLoopCreditResult(
        resource_values=(), resource_cotangents={}, parameter_cotangents={},
        parameters={}, context_cotangents={}, arrivals=torch.zeros(3, 1, dtype=torch.bool),
        iteration_active=(
            torch.tensor([True, False, True]),
            torch.tensor([False, True, True]),
        ),
        join_ready=(), route_log_probability=row_terms[[0, 2]].sum() + global_term,
        route_log_terms=(row_terms, global_term),
    )
    losses = torch.tensor([2.0, 7.0, 20.0])
    objective = result.structure_objective(losses, baseline=1.0)
    expected = row_terms[0] + 19.0 * row_terms[2] + 12.5 * global_term
    torch.testing.assert_close(objective, expected)
    actual_gradients = torch.autograd.grad(
        objective, (row_logits, global_logits), retain_graph=True,
    )
    expected_gradients = torch.autograd.grad(expected, (row_logits, global_logits))
    for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, reference)
    assert torch.count_nonzero(actual_gradients[0][1]) == 0


def test_compound_route_honors_connection_dependencies_across_stages() -> None:
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(torch.tensor([[2.0]]))),
        mechanisms.TensorResource(_vector_spec("scores"), _vector_view(torch.tensor([[2.0, -1.0]]))),
        mechanisms.TensorResource(_vector_spec("middle"), _vector_view(torch.zeros(1, 1))),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(1, 1))),
    )
    first = mechanisms.Connection(
        "first", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("middle"),
        transfer=mechanisms.LearnableAffineTransfer(gain=2.0),
    )
    second = mechanisms.Connection(
        "second", mechanisms.ResourcePort("middle"), mechanisms.ResourcePort("output"),
        transfer=mechanisms.LearnableAffineTransfer(gain=3.0), depends_on=("first",),
    )
    fallback = mechanisms.Connection(
        "fallback", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
    )
    graph = mechanisms.ProgramGraph(
        resources, (first, second, fallback),
        programs={
            "dependent": ("first", "second"),
            "alternative": ("fallback",),
            "run": (mechanisms.ProgramRoute(
                "choose", "scores", ("dependent", "alternative"),
            ),),
        },
    )
    output = next(
        item.active_view.value for item in graph.execute_program_functional("run").state.resources
        if item.spec.resource_id == "output"
    )
    torch.testing.assert_close(output, torch.tensor([[12.0]]))
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    values = tuple(resource.resolve().view.value for resource in resources)
    torch.testing.assert_close(plan(*values)[3], output)


def test_sample_route_selects_distinct_compound_paths_per_row() -> None:
    resources = (
        mechanisms.TensorResource(
            _vector_spec("source"), _vector_view(torch.tensor([[2.0], [3.0]])),
        ),
        mechanisms.TensorResource(
            _vector_spec("scores"), _vector_view(torch.tensor([[2.0, -1.0], [-1.0, 2.0]])),
        ),
        mechanisms.TensorResource(_vector_spec("middle"), _vector_view(torch.zeros(2, 1))),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(2, 1))),
    )
    first = mechanisms.Connection(
        "first", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("middle"),
        transfer=mechanisms.LearnableAffineTransfer(gain=2.0),
    )
    second = mechanisms.Connection(
        "second", mechanisms.ResourcePort("middle"), mechanisms.ResourcePort("output"),
        transfer=mechanisms.LearnableAffineTransfer(gain=3.0), depends_on=("first",),
    )
    fallback = mechanisms.Connection(
        "fallback", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
        transfer=mechanisms.LearnableAffineTransfer(gain=-1.0),
    )
    graph = mechanisms.ProgramGraph(
        resources, (first, second, fallback),
        programs={
            "dependent": ("first", "second"),
            "alternative": ("fallback",),
            "run": (mechanisms.ProgramRoute(
                "choose", "scores", ("dependent", "alternative"), selection_scope="sample",
            ),),
        },
    )
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    values = tuple(resource.resolve().view.value for resource in resources)
    expected = torch.tensor([[12.0], [-3.0]])
    torch.testing.assert_close(plan(*values)[3], expected)
    torch.testing.assert_close(
        torch.compile(plan, backend="eager", fullgraph=True)(*values)[3], expected,
    )


def test_compound_route_samples_nested_differentiable_fates() -> None:
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(torch.tensor([[2.0]]))),
        mechanisms.TensorResource(_vector_spec("scores"), _vector_view(torch.tensor([[2.0, -1.0]]))),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.tensor([[0.0]]))),
    )
    choice = arti.as_differentiable_fabric_node(
        "choice", {"double": _DoubleFate(), "quadruple": _QuadrupleFate()},
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    fallback = mechanisms.Connection(
        "fallback", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
    )
    graph = mechanisms.ProgramGraph(
        resources, (fallback,), nodes=(choice,),
        programs={
            "learned": ("choice",),
            "ordinary": ("fallback",),
            "run": (mechanisms.ProgramRoute("choose", "scores", ("learned", "ordinary")),),
        },
    )
    joint = graph.sample_program_fates("run", generator=torch.Generator().manual_seed(7))
    assert joint.fate_route_paths == (((('choose', 0),),),)
    sample = graph.sample_program_fates(
        "run", generator=torch.Generator().manual_seed(7),
        route_selections={"choose": "learned"},
    )
    assert tuple(node_id for node_id, _ in sample.selections) == ("choice",)
    execution = graph.execute_program_functional("run", candidate_selections=dict(sample.selections))
    output = next(item.active_view.value for item in execution.state.resources if item.spec.resource_id == "output")
    assert output.item() in (4.0, 8.0)


def test_nested_route_fate_credit_requires_every_ancestor_selection() -> None:
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(torch.tensor([[2.0]]))),
        mechanisms.TensorResource(_vector_spec("outer_scores"), _vector_view(torch.tensor([[1.0, 0.0]]))),
        mechanisms.TensorResource(_vector_spec("inner_scores"), _vector_view(torch.tensor([[1.0, 0.0]]))),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(1, 1))),
    )
    fates = tuple(
        arti.as_differentiable_fabric_node(
            name, {"double": _DoubleFate(), "quadruple": _QuadrupleFate()},
            input_ports={"source": mechanisms.ResourcePort("source")},
            output_ports={"value": mechanisms.ResourcePort("output")},
        )
        for name in ("inner_a", "inner_b")
    )
    fallback = mechanisms.Connection(
        "fallback", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
    )
    graph = mechanisms.ProgramGraph(
        resources, (fallback,), nodes=fates,
        programs={
            "branch": (mechanisms.ProgramRoute(
                "inner", "inner_scores", ("inner_a", "inner_b"),
            ),),
            "run": (mechanisms.ProgramRoute(
                "outer", "outer_scores", ("branch", "fallback"),
            ),),
        },
    )
    sampled = graph.sample_program_fates("run", generator=torch.Generator().manual_seed(17))
    assert sampled.fate_route_paths == (
        ((("outer", 0), ("inner", 0)),),
        ((("outer", 0), ("inner", 1)),),
    )
    bypass = sampled.structure_objective(
        torch.tensor(1.0), route_selections={"outer": torch.tensor(1), "inner": torch.tensor(0)},
    )
    for node in fates:
        torch.testing.assert_close(
            torch.autograd.grad(bypass, node.logits, retain_graph=True)[0],
            torch.zeros_like(node.logits),
        )
    selected = sampled.structure_objective(
        torch.tensor(1.0), route_selections={"outer": torch.tensor(0), "inner": torch.tensor(0)},
    )
    assert torch.autograd.grad(selected, fates[0].logits, retain_graph=True)[0].abs().sum() > 0
    torch.testing.assert_close(
        torch.autograd.grad(selected, fates[1].logits)[0], torch.zeros_like(fates[1].logits),
    )


def test_compound_route_does_not_credit_unselected_fate() -> None:
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(torch.tensor([[2.0]]))),
        mechanisms.TensorResource(_vector_spec("scores"), _vector_view(torch.tensor([[2.0, -1.0]]))),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.tensor([[0.0]]))),
    )
    choices = tuple(
        arti.as_differentiable_fabric_node(
            name, {"double": _DoubleFate(), "quadruple": _QuadrupleFate()},
            input_ports={"source": mechanisms.ResourcePort("source")},
            output_ports={"value": mechanisms.ResourcePort("output")},
        )
        for name in ("chosen", "unused")
    )
    graph = mechanisms.ProgramGraph(
        resources, (), nodes=choices,
        programs={
            "chosen_path": ("chosen",),
            "unused_path": ("unused",),
            "run": (mechanisms.ProgramRoute(
                "choose", "scores", ("chosen_path", "unused_path"),
            ),),
        },
    )
    sample = graph.sample_program_fates(
        "run", generator=torch.Generator().manual_seed(11),
        route_selections={"choose": "chosen_path"},
    )
    assert tuple(node_id for node_id, _ in sample.selections) == ("chosen",)
    execution = graph.execute_program_functional("run", candidate_selections=dict(sample.selections))
    output = next(item.active_view.value for item in execution.state.resources if item.spec.resource_id == "output")
    objective = sample.structure_objective(output.square().mean())
    assert torch.autograd.grad(objective, choices[0].logits, retain_graph=True)[0] is not None
    assert torch.autograd.grad(objective, choices[1].logits, allow_unused=True)[0] is None


def test_batch_route_jointly_samples_fates_and_credits_only_executed_path() -> None:
    resources = (
        mechanisms.TensorResource(
            _vector_spec("source"), _vector_view(torch.tensor([[2.0], [3.0]])),
        ),
        mechanisms.TensorResource(
            _vector_spec("scores"), _vector_view(torch.tensor([[2.0, -1.0], [2.0, -1.0]])),
        ),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(2, 1))),
    )
    choices = tuple(
        arti.as_differentiable_fabric_node(
            name, {"double": _DoubleFate(), "quadruple": _QuadrupleFate()},
            input_ports={"source": mechanisms.ResourcePort("source")},
            output_ports={"value": mechanisms.ResourcePort("output")},
        )
        for name in ("first", "second")
    )
    graph = mechanisms.ProgramGraph(
        resources, (), nodes=choices,
        programs={
            "first_path": ("first",),
            "second_path": ("second",),
            "run": (mechanisms.ProgramRoute(
                "choose", "scores", ("first_path", "second_path"),
            ),),
        },
    )
    sample = graph.sample_program_fates("run", generator=torch.Generator().manual_seed(19))
    assert tuple(node_id for node_id, _ in sample.selections) == ("first", "second")
    assert sample.fate_route_paths == (((("choose", 0),),), ((("choose", 1),),))
    sealed = graph.specialize_differentiable_nodes(dict(sample.selections)).graph
    plan = mechanisms.ResourceGraphCompiler.compile_program(sealed, "run")
    values = tuple(resource.resolve().view.value for resource in resources)
    outputs, (actual_route,), _ = plan.forward_with_route_credit(*values)
    assert actual_route.ndim == 0
    per_row_loss = outputs[2].square().squeeze(-1)
    with pytest.raises(mechanisms.ResourceGraphError, match="actual route selections"):
        sample.structure_objective(per_row_loss)
    with pytest.raises(mechanisms.ResourceGraphError, match="batch route selection must be scalar"):
        sample.structure_objective(
            per_row_loss, route_selections={"choose": torch.zeros(2, dtype=torch.long)},
        )
    for loss in (per_row_loss, per_row_loss.mean()):
        objective = sample.structure_objective(loss, route_selections={"choose": actual_route})
        for index, node in enumerate(choices):
            gradient = torch.autograd.grad(objective, node.logits, retain_graph=True)[0]
            if index == actual_route.item():
                candidate_id = dict(sample.selections)[node.node_id]
                indicator = torch.zeros_like(node.logits)
                indicator[node.candidate_ids.index(candidate_id)] = 1.0
                expected = loss.detach().mean() * (indicator - node.probabilities().detach())
                torch.testing.assert_close(gradient, expected)
            else:
                torch.testing.assert_close(gradient, torch.zeros_like(gradient))


def test_sample_route_credits_only_rows_that_executed_each_fate() -> None:
    resources = (
        mechanisms.TensorResource(
            _vector_spec("source"), _vector_view(torch.tensor([[2.0], [3.0]])),
        ),
        mechanisms.TensorResource(
            _vector_spec("scores"), _vector_view(torch.tensor([[2.0, -1.0], [-1.0, 2.0]])),
        ),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(2, 1))),
    )
    choices = tuple(
        arti.as_differentiable_fabric_node(
            name, {"double": _DoubleFate(), "quadruple": _QuadrupleFate()},
            input_ports={"source": mechanisms.ResourcePort("source")},
            output_ports={"value": mechanisms.ResourcePort("output")},
        )
        for name in ("first", "second")
    )
    graph = mechanisms.ProgramGraph(
        resources, (), nodes=choices,
        programs={
            "first_path": ("first",),
            "second_path": ("second",),
            "run": (mechanisms.ProgramRoute(
                "choose", "scores", ("first_path", "second_path"),
                selection_scope="sample",
            ),),
        },
    )
    sample = graph.sample_program_fates("run", generator=torch.Generator().manual_seed(19))
    assert tuple(node_id for node_id, _ in sample.selections) == ("first", "second")
    selected = dict(sample.selections)
    sealed = graph.specialize_differentiable_nodes(selected).graph
    plan = mechanisms.ResourceGraphCompiler.compile_program(sealed, "run")
    values = tuple(resource.resolve().view.value for resource in resources)
    output = torch.compile(plan, backend="eager", fullgraph=True)(*values)[2]
    per_row_loss = output.square().squeeze(-1)
    routes = torch.tensor([0, 1])
    with pytest.raises(mechanisms.ResourceGraphError, match="per-row final losses"):
        sample.structure_objective(per_row_loss.mean())
    with pytest.raises(mechanisms.ResourceGraphError, match="actual route selections"):
        sample.structure_objective(per_row_loss)
    with pytest.raises(mechanisms.ResourceGraphError, match="baseline must"):
        sample.structure_objective(
            per_row_loss, baseline=torch.zeros(2, 1),
            route_selections={"choose": torch.tensor([0, 1])},
        )
    objective = sample.structure_objective(
        per_row_loss, route_selections={"choose": routes},
    )
    assert sample.structure_objective(
        per_row_loss, baseline=torch.zeros(1, 1), route_selections={"choose": routes},
    ).ndim == 0
    for row, node in enumerate(choices):
        candidate_id = selected[node.node_id]
        indicator = torch.zeros_like(node.logits)
        indicator[node.candidate_ids.index(candidate_id)] = 1.0
        expected = per_row_loss[row].detach() / 2 * (indicator - node.probabilities().detach())
        torch.testing.assert_close(
            torch.autograd.grad(objective, node.logits, retain_graph=True)[0], expected,
        )
    wrong_routes = sample.structure_objective(
        per_row_loss, route_selections={"choose": torch.tensor([1, 0])},
    )
    correct_gradient = torch.autograd.grad(objective, choices[0].logits, retain_graph=True)[0]
    wrong_gradient = torch.autograd.grad(wrong_routes, choices[0].logits, retain_graph=True)[0]
    assert not torch.equal(correct_gradient, wrong_gradient)
    first_only = sample.structure_objective(
        per_row_loss, route_selections={"choose": torch.zeros(2, dtype=torch.long)},
    )
    torch.testing.assert_close(
        torch.autograd.grad(first_only, choices[1].logits, retain_graph=True)[0],
        torch.zeros_like(choices[1].logits),
    )
    sampled_values, (actual_routes,), route_log_probability = plan.forward_with_route_credit(*values)
    assert actual_routes.shape == (2,)
    assert route_log_probability.shape == (2,)
    sampled_losses = sampled_values[2].square().squeeze(-1)
    sampled_objective = sample.structure_objective(
        sampled_losses, route_selections={"choose": actual_routes},
    )
    for candidate_index, node in enumerate(choices):
        candidate_id = selected[node.node_id]
        indicator = torch.zeros_like(node.logits)
        indicator[node.candidate_ids.index(candidate_id)] = 1.0
        coefficient = torch.where(
            actual_routes == candidate_index, sampled_losses.detach(),
            torch.zeros_like(sampled_losses),
        ).mean()
        expected = coefficient * (indicator - node.probabilities().detach())
        torch.testing.assert_close(
            torch.autograd.grad(sampled_objective, node.logits, retain_graph=True)[0],
            expected,
        )


def test_batch_selected_fate_keeps_full_batch_credit_when_reused_by_sample_route() -> None:
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(torch.ones(2, 1))),
        mechanisms.TensorResource(
            _vector_spec("scores"), _vector_view(torch.tensor([[2.0, -1.0], [2.0, -1.0]])),
        ),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(2, 1))),
    )
    choices = tuple(
        arti.as_differentiable_fabric_node(
            name, {"double": _DoubleFate(), "quadruple": _QuadrupleFate()},
            input_ports={"source": mechanisms.ResourcePort("source")},
            output_ports={"value": mechanisms.ResourcePort("output")},
        )
        for name in ("common", "other")
    )
    graph = mechanisms.ProgramGraph(
        resources, (), nodes=choices,
        programs={
            "common_path": ("common",),
            "other_path": ("other",),
            "run": (
                mechanisms.ProgramRoute("batch", "scores", ("common_path", "other_path")),
                mechanisms.ProgramRoute(
                    "sample", "scores", ("common_path", "other_path"),
                    selection_scope="sample",
                ),
            ),
        },
    )
    sampled = graph.sample_program_fates(
        "run", route_selections={"batch": "common_path"},
        generator=torch.Generator().manual_seed(13),
    )
    assert sampled.fate_route_paths == (((), (("sample", 0),)), ((("sample", 1),),))
    objective = sampled.structure_objective(
        torch.tensor([2.0, 6.0]), route_selections={"sample": torch.zeros(2, dtype=torch.long)},
    )
    assert torch.autograd.grad(objective, choices[0].logits, retain_graph=True)[0].abs().sum() > 0
    torch.testing.assert_close(
        torch.autograd.grad(objective, choices[1].logits)[0],
        torch.zeros_like(choices[1].logits),
    )


def test_compound_route_parallel_stage_and_async_join_share_reverse_tape() -> None:
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(torch.tensor([[1.0]]))),
        mechanisms.TensorResource(_vector_spec("scores"), _vector_view(torch.tensor([[2.0, -1.0]]))),
        *(mechanisms.TensorResource(_vector_spec(name), _vector_view(torch.zeros(1, 1)))
          for name in ("left", "right", "selected", "reference", "final")),
    )
    left_copy = mechanisms.Connection(
        "left_copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("left"),
        transfer=mechanisms.LearnableAffineTransfer(gain=2.0),
    )
    right_copy = mechanisms.Connection(
        "right_copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("right"),
        transfer=mechanisms.LearnableAffineTransfer(gain=3.0),
    )
    fallback = mechanisms.Connection(
        "fallback", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("selected"),
        transfer=mechanisms.LearnableAffineTransfer(gain=-1.0),
    )
    reference_copy = mechanisms.Connection(
        "reference_copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("reference"),
    )
    combine = arti.as_fabric_node(
        "combine", _DecoratedPairSum(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("selected")},
    )
    joined = arti.as_fabric_node(
        "joined", _DecoratedPairSum(),
        input_ports={"left": mechanisms.ResourcePort("selected"), "right": mechanisms.ResourcePort("reference")},
        output_ports={"value": mechanisms.ResourcePort("final")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (left_copy, right_copy, fallback, reference_copy), nodes=(combine, joined),
        programs={
            "wide_path": (
                mechanisms.ProgramStage(("left_copy", "right_copy")),
                mechanisms.ProgramJoin("sub_ready", "combine"),
            ),
            "short_path": ("fallback",),
            "run": (
                mechanisms.ProgramRoute("choose", "scores", ("wide_path", "short_path")),
                "reference_copy", mechanisms.ProgramJoin("ready", "joined"),
            ),
        },
    )
    execution = graph.execute_program_functional("run")
    assert execution.routes[0].candidate_id == "wide_path"
    assert tuple(dispatch.frontier_path for dispatch in execution.dispatches) == (
        (0,), (0, 0), (0, 1), (1,), (2,),
    )
    assert execution.dispatches[0].route == execution.routes[0]
    assert execution.dispatches[0].members == ()
    assert tuple(item.connection_id for item in execution.dispatches[1].members) == (
        "left_copy", "right_copy",
    )
    assert execution.dispatches[2].members[0].node_id == "combine"
    assert execution.dispatches[2].join == execution.joins[0]
    assert execution.dispatches[-1].join == execution.joins[-1]
    assert execution.joins[0].fired
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    source = resources[0].resolve().view.value.detach().clone().requires_grad_(True)
    inputs = (
        source, *(resource.resolve().view.value for resource in resources[1:]),
        torch.zeros((1, plan.arrival_width), dtype=torch.bool),
    )
    forced = plan.forward_with_route_selections(
        *inputs, route_selections=(torch.tensor(0),),
    )
    torch.testing.assert_close(forced[6], torch.tensor([[6.0]]))
    compiled = torch.compile(plan, backend="eager", fullgraph=True)(*inputs)
    torch.testing.assert_close(compiled[6], forced[6])
    ordinary_gradient = torch.autograd.grad(compiled[6].sum(), source, retain_graph=True)[0]
    credit = plan.credit_gradient(
        *inputs, route_selections=(torch.tensor(0),),
        terminal_cotangents={"final": torch.ones_like(compiled[6])},
    )
    torch.testing.assert_close(credit.resource_cotangents["source"], ordinary_gradient)
    torch.testing.assert_close(ordinary_gradient, torch.tensor([[6.0]]))


def test_routed_relation_publishes_only_selected_port_to_async_join() -> None:
    class _ContextGate(torch.nn.Module):
        def forward(self, context: torch.Tensor) -> torch.Tensor:
            return context[:, :1].unsqueeze(-1)

    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[2.0, 4.0]])),
        mechanisms.TensorResource(_vector_spec("scores"), _vector_view(torch.tensor([[3.0, -1.0, -2.0]]))),
        mechanisms.TensorResource(_spec("left"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("right"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]])),
    )
    left = mechanisms.Connection(
        "left_copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("left"),
        transfer=mechanisms.LearnableAffineTransfer(gain=2.0),
    )
    right = mechanisms.Connection(
        "right_copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("right"),
        transfer=mechanisms.LearnableAffineTransfer(gain=3.0),
    )
    alternate_left = mechanisms.Connection(
        "alternate_left", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("left"),
        transfer=mechanisms.LearnableAffineTransfer(gain=4.0),
    )
    gated_right = mechanisms.Connection(
        "gated_right", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("right"),
        activation=_ContextGate(),
        credit_boundary=arti.CreditBoundary(mode=arti.CreditBoundaryMode.BERNOULLI),
    )
    pair_sum = arti.as_fabric_node(
        "sum", _DecoratedPairSum(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (left, right, alternate_left, gated_right), nodes=(pair_sum,),
        programs={"run": (
            mechanisms.ProgramRoute("choose", "scores", ("left_copy", "right_copy", "alternate_left")),
            mechanisms.ProgramJoin("early", "sum"),
            "gated_right", mechanisms.ProgramJoin("late", "sum"),
        )},
    )
    context = torch.ones(1, 1)
    credit_mask = torch.ones(1, 2, 1, dtype=torch.bool)
    plan = mechanisms.ResourceGraphCompiler.compile_program(
        graph, "run",
        example_contexts={"gated_right": context},
        example_credit_masks={"gated_right": credit_mask},
    )
    assert isinstance(plan, mechanisms.StaticDataflowProgramExecutionPlan)
    values = tuple(resource.resolve().view.value for resource in resources)
    arrivals = torch.zeros((1, plan.arrival_width), dtype=torch.bool)
    inputs = (*values, context, credit_mask, arrivals)
    result = plan(*inputs)
    torch.testing.assert_close(result[4], _view([[6.0, 12.0]]).value)
    eager = graph.execute_program_functional(
        "run", contexts={"gated_right": context},
        credit_masks={"gated_right": credit_mask},
    )
    assert [join.fired for join in eager.joins] == [False, True]
    gated_receipt = next(
        item for item in eager.connections if item.connection_id == "gated_right"
    )
    assert gated_receipt.context is context
    assert gated_receipt.destination_before is not None
    assert gated_receipt.destination_before.epoch < gated_receipt.destination.epoch
    assert gated_receipt.credit_mask is credit_mask
    compiled = torch.compile(plan, backend="eager", fullgraph=True)(*inputs)
    exported = torch.export.export(plan, inputs).module()(*inputs)
    for reference, candidate, exported_value in zip(result, compiled, exported, strict=True):
        torch.testing.assert_close(reference, candidate)
        torch.testing.assert_close(reference, exported_value)
    credit = plan.credit_gradient(
        *inputs, terminal_cotangents={"output": torch.ones_like(result[4])},
    )
    torch.testing.assert_close(credit.resource_cotangents["source"], torch.full_like(values[0], 3.0))
    assert any(
        name.startswith("choose.") and name.endswith("transfer.gain") and value is not None
        for name, value in credit.parameter_cotangents.items()
    )
    torch.testing.assert_close(credit.context_cotangents["gated_right"], torch.tensor([[6.0]]))
    blocked_mask = credit_mask.clone()
    blocked_mask[:, 1] = False
    blocked = plan.credit_gradient(
        *(*values, context, blocked_mask, arrivals),
        terminal_cotangents={"output": torch.ones_like(result[4])},
    )
    torch.testing.assert_close(blocked.resource_cotangents["source"], _view([[3.0, 2.0]]).value)
    torch.testing.assert_close(blocked.context_cotangents["gated_right"], torch.tensor([[2.0]]))

    source_variable = values[0].detach().clone().requires_grad_(True)
    score_variable = torch.tensor([[0.1, -0.1, -0.2]], requires_grad=True)
    sampled_inputs = (source_variable, score_variable, *values[2:], context, credit_mask, arrivals)
    sampled_forward = torch.compile(plan.forward_with_route_credit, backend="eager", fullgraph=True)
    sampled_values, choices, joint_log_probability = sampled_forward(*sampled_inputs)
    assert len(choices) == 1 and choices[0].dtype == torch.long
    replay = plan.credit_gradient(
        *sampled_inputs, route_selections=choices,
        terminal_cotangents={"output": torch.ones_like(sampled_values[4])},
    )
    torch.testing.assert_close(replay.resource_values[4], sampled_values[4])
    torch.testing.assert_close(replay.arrivals, sampled_values[-1])
    ordinary = torch.autograd.grad(
        sampled_values[4].sum(), source_variable, allow_unused=True, retain_graph=True,
    )[0]
    torch.testing.assert_close(
        replay.resource_cotangents["source"],
        torch.zeros_like(source_variable) if ordinary is None else ordinary,
    )
    ((sampled_values[4].detach().square().sum() + 1.0) * joint_log_probability).backward()
    assert score_variable.grad is not None and score_variable.grad.abs().sum() > 0

    for chosen_index, expected_output in (
        (0, _view([[6.0, 12.0]]).value),
        (1, torch.zeros_like(values[4])),
        (2, _view([[10.0, 20.0]]).value),
    ):
        chosen = (torch.tensor(chosen_index),)
        forced_source = values[0].detach().clone().requires_grad_(True)
        forced_inputs = (forced_source, torch.tensor([[0.1, -0.1, -0.2]]), *values[2:], context, credit_mask, arrivals)
        forced = plan.forward_with_route_selections(*forced_inputs, route_selections=chosen)
        forced_credit = plan.credit_gradient(
            *forced_inputs, route_selections=chosen,
            terminal_cotangents={"output": torch.ones_like(forced[4])},
        )
        torch.testing.assert_close(forced[4], expected_output)
        torch.testing.assert_close(forced_credit.resource_values[4], forced[4])
        torch.testing.assert_close(forced_credit.arrivals, forced[-1])
        torch.testing.assert_close(forced_credit.route_selections[0], chosen[0])
        forced_gradient = torch.autograd.grad(
            forced[4].sum(), forced_source, allow_unused=True,
        )[0]
        torch.testing.assert_close(
            forced_credit.resource_cotangents["source"],
            torch.zeros_like(forced_source) if forced_gradient is None else forced_gradient,
        )

    other_scores = torch.tensor([[-1.0, 3.0, -2.0]])
    other = (*values[:1], other_scores, *values[2:], context, credit_mask, arrivals)
    result_other = plan(*other)
    eager_other = graph.execute_program_functional(
        "run", input_views={"scores": _vector_view(other_scores)},
        contexts={"gated_right": context}, credit_masks={"gated_right": credit_mask},
    )
    assert [join.fired for join in eager_other.joins] == [False, False]
    torch.testing.assert_close(result_other[4], torch.zeros_like(result_other[4]))
    credit_other = plan.credit_gradient(
        *other, terminal_cotangents={"output": torch.ones_like(result_other[4])},
    )
    torch.testing.assert_close(credit_other.resource_cotangents["source"], torch.zeros_like(values[0]))
    if torch.cuda.is_available():
        cuda_plan = plan.to("cuda")
        cuda_inputs = tuple(value.to("cuda") for value in inputs)
        cuda_compiled = torch.compile(cuda_plan, backend="eager", fullgraph=True)(*cuda_inputs)
        torch.testing.assert_close(cuda_compiled[4], result[4].to("cuda"))


def _continue_spec(resource_id: str) -> mechanisms.TensorResourceSpec:
    return mechanisms.TensorResourceSpec(
        resource_id,
        mechanisms.TensorViewPattern(
            min_rank=1,
            max_rank=1,
            allowed_axis_roles=("batch",),
        ),
    )


def _continue_view(value: torch.Tensor) -> mechanisms.TensorView:
    return mechanisms.TensorView.from_tensor(
        value,
        axis_names=("batch",),
        axis_roles=("batch",),
    )


def _federated_program() -> mechanisms.FederatedProgram:
    pattern = mechanisms.TensorViewPattern(
        min_rank=2, max_rank=2, allowed_axis_roles=("batch", "feature")
    )
    schema = mechanisms.TensorSchema(
        dtype="float32", device_class="any", dimensions=("B", 2),
        semantic_axes=("batch", "feature"), mask_semantics="none",
    )
    abi = mechanisms.TerminalOutputABI(
        fields=(
            mechanisms.TerminalField("value", schema, "terminal-value"),
            mechanisms.TerminalField(
                "validity",
                mechanisms.TensorSchema(
                    dtype="boolean", device_class="any", dimensions=("B",),
                    semantic_axes=("batch",), mask_semantics="boolean-validity",
                ),
                "terminal-validity",
            ),
            mechanisms.TerminalField(
                "score",
                mechanisms.TensorSchema(
                    dtype="float32", device_class="any", dimensions=("B",),
                    semantic_axes=("batch",), mask_semantics="none",
                ),
                "terminal-score",
            ),
        ),
        factor_order=(),
        validity_contract="one validity bit per row",
        packing_contract="named terminal tensors",
        score_contract="one score per row",
        consumer_contract="one winner",
        gradient_contract=mechanisms.GradientContract.autograd(),
    )
    observer = mechanisms.CoordinateTensorViewObserver(max_rank=2, query_dim=2)
    with torch.no_grad():
        observer.encoder[0].weight.zero_()
        observer.encoder[0].bias.zero_()
        observer.encoder[2].weight.zero_()
        observer.encoder[2].bias.copy_(torch.tensor([0.0, 1.0]))
    query = mechanisms.seal_tensor_view_bank_query(
        mechanisms.TensorViewBankQuery(
            pattern=pattern,
            observer=observer,
            matcher=mechanisms.BankMemberMatcher(
                torch.eye(2), member_ids=("double", "exit")
            ),
        )
    )
    value = mechanisms.InputBinding(
        "value", mechanisms.TensorType(
            ("B", "D"), ("B", 2), dtype="float32", domain="activation"
        ),
    )
    action = mechanisms.TensorViewFormulaAction(
        mechanisms.BankLocalFormulaAction(
            "double", mechanisms.FormulaProgram.build(outputs=(mechanisms.add(value, value),)),
            input_schema=schema, output_schema=schema, operands={},
        ),
        layout=mechanisms.TensorViewLayoutTransition(
            ("batch", "feature"), ("batch", "feature"), index_transition="identity"
        ),
    )
    program = mechanisms.RoutedProgram(
        program_id="terminal-local", query=query, actions=(action,),
        terminal_action=mechanisms.ProgramTerminalAction("exit", input_schema=schema),
        local_iteration=mechanisms.LocalIterationPolicy(min_steps=1, max_steps=1),
        input_pattern=pattern, exit_pattern=pattern, terminal_abi=abi,
    )
    return mechanisms.FederatedProgram(
        {program.program_id: program}, terminal_abi=abi,
        root_program_ids=(program.program_id,), max_levels=1, max_k=1,
    )


class _ScaleNode(mechanisms.ProgramNode):
    def __init__(self) -> None:
        super().__init__("scale", input_resource_id="workspace", output_resource_id="output")
        self.scale = torch.nn.Parameter(torch.tensor(2.0))

    def contract_config(self) -> dict[str, object]:
        return {**super().contract_config(), "operation": "scale"}

    def invoke(self, view: mechanisms.TensorView) -> mechanisms.ProgramNodeInvocation:
        return mechanisms.ProgramNodeInvocation(
            mechanisms.TensorView(view.value * self.scale, view.axes, index_map=view.index_map, mask=view.mask),
            {"operation": "scale"},
        )


class _OffsetNode(mechanisms.ProgramNode):
    def __init__(self, node_id: str, input_resource_id: str, output_resource_id: str, offset: float) -> None:
        super().__init__(node_id, input_resource_id=input_resource_id, output_resource_id=output_resource_id)
        self.offset = float(offset)

    def contract_config(self) -> dict[str, object]:
        return {**super().contract_config(), "operation": "offset", "offset": self.offset}

    def invoke(self, view: mechanisms.TensorView) -> mechanisms.ProgramNodeInvocation:
        return mechanisms.ProgramNodeInvocation(
            mechanisms.TensorView(view.value + self.offset, view.axes, index_map=view.index_map, mask=view.mask)
        )


class _PairSumNode(mechanisms.MultiPortProgramNode):
    def __init__(self) -> None:
        super().__init__(
            "pair_sum",
            input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
            output_ports={"value": mechanisms.ResourcePort("output")},
        )

    def invoke_ports(self, inputs: dict[str, mechanisms.TensorView]) -> mechanisms.MultiPortProgramNodeInvocation:
        left, right = inputs["left"], inputs["right"]
        return mechanisms.MultiPortProgramNodeInvocation(
            {"value": mechanisms.TensorView(left.value + right.value, left.axes)}
        )


@arti.fabric_layer(inputs=("source",), outputs={"value": "source"})
class _DecoratedScale(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return source * self.scale


def _scale_local_vjp(
    module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    outputs: tuple[torch.Tensor, ...],
    output_cotangents: tuple[torch.Tensor, ...],
    parameters: tuple[torch.nn.Parameter, ...],
    create_graph: bool,
) -> mechanisms.LocalVJPResult:
    del outputs, create_graph
    source = inputs[0]
    cotangent = output_cotangents[0]
    scale = parameters[0]
    return mechanisms.LocalVJPResult(
        (cotangent * scale,),
        ((cotangent * source).sum().reshape_as(scale),),
    )


@arti.fabric_layer(
    inputs=("source",),
    outputs={"value": "source"},
    local_vjp=_scale_local_vjp,
)
class _ExplicitVJPScale(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return source * self.scale


@arti.fabric_layer(inputs=("source",), outputs={"value": "source"})
class _BoundaryScale(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))
        self.boundary = arti.CreditBoundary(mode=arti.CreditBoundaryMode.MEAN)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return self.boundary(source) * self.scale


@arti.differentiable_fate("double")
@arti.fabric_layer(inputs=("source",), outputs={"value": "source"})
class _DoubleFate(torch.nn.Module):
    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return source * 2.0


@arti.differentiable_fate("quadruple")
@arti.fabric_layer(inputs=("source",), outputs={"value": "source"})
class _QuadrupleFate(torch.nn.Module):
    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return source * 4.0


@arti.fabric_layer(inputs=("source",), outputs={"value": "source"})
@arti.differentiable_fate("identity")
class _IdentityFate(torch.nn.Module):
    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return source


@arti.fabric_layer(
    inputs=("left", "right"), outputs={"value": "left"}
)
class _DecoratedPairSum(torch.nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left + right


@arti.fabric_layer(
    inputs=("left", "right"), outputs={"value": "left"}
)
class _DecoratedPairProduct(torch.nn.Module):
    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left * right


@arti.fabric_layer(
    inputs=("value",), outputs={"value": "value", "continue": "value"}
)
class _DecoratedLoopStep(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> dict[str, torch.Tensor | mechanisms.TensorView]:
        next_value = value + 1.0
        continuation = (next_value[:, 0, 0] < 3.0).to(dtype=value.dtype)
        return {
            "value": next_value,
            "continue": mechanisms.TensorView.from_tensor(
                continuation,
                axis_names=("batch",),
                axis_roles=("batch",),
            ),
        }


@arti.fabric_layer(
    inputs=("left", "right"), outputs={"value": "left", "continue": "left"}
)
class _DecoratedPairSumStop(torch.nn.Module):
    """A join body that terminates its enclosing loop after one publication."""

    def forward(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> dict[str, torch.Tensor | mechanisms.TensorView]:
        return {
            "value": left + right,
            "continue": mechanisms.TensorView.from_tensor(
                torch.zeros(left.shape[0], dtype=left.dtype, device=left.device),
                axis_names=("batch",),
                axis_roles=("batch",),
            ),
        }


class _SplitNode(mechanisms.MultiPortProgramNode):
    """A two-head region used to exercise explicit graph fan-out."""

    def __init__(self) -> None:
        super().__init__(
            "split",
            input_ports={"value": mechanisms.ResourcePort("workspace")},
            output_ports={
                "left": mechanisms.ResourcePort("left"),
                "right": mechanisms.ResourcePort("right"),
            },
        )

    def invoke_ports(
        self, inputs: dict[str, mechanisms.TensorView]
    ) -> mechanisms.MultiPortProgramNodeInvocation:
        value = inputs["value"]
        return mechanisms.MultiPortProgramNodeInvocation(
            {
                "left": mechanisms.TensorView(
                    value.value + 1.0, value.axes, index_map=value.index_map, mask=value.mask
                ),
                "right": mechanisms.TensorView(
                    value.value * 2.0, value.axes, index_map=value.index_map, mask=value.mask
                ),
            },
            {"operation": "split"},
        )


class _CountingLoopNode(mechanisms.MultiPortProgramNode):
    def __init__(self) -> None:
        super().__init__(
            "count",
            input_ports={"state": mechanisms.ResourcePort("state")},
            output_ports={
                "state": mechanisms.ResourcePort("state"),
                "continue": mechanisms.ResourcePort("continue"),
            },
        )

    def invoke_ports(
        self, inputs: dict[str, mechanisms.TensorView]
    ) -> mechanisms.MultiPortProgramNodeInvocation:
        state = inputs["state"]
        next_state = mechanisms.TensorView(
            state.value + 1.0, state.axes, index_map=state.index_map, mask=state.mask
        )
        return mechanisms.MultiPortProgramNodeInvocation(
            {
                "state": next_state,
                "continue": _continue_view(
                    (next_state.value[:, 0, 0] < 2.0).to(dtype=next_state.value.dtype)
                ),
            }
        )


class _TerminalRoutedProgram(mechanisms.RoutedProgram):
    """Minimal terminal local region for the graph-node contract test."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.program_id = "terminal-local"

    @property
    def effect_actions(self) -> tuple[object, ...]:
        return ()

    @property
    def all_resource_actions(self) -> tuple[object, ...]:
        return ()

    def contract_config(self) -> dict[str, object]:
        return {"program_id": self.program_id, "mode": "test-terminal"}

    def forward(self, view: mechanisms.TensorView, *, max_candidates: int) -> FederalBankStep:
        assert max_candidates == 1
        batch = view.value.shape[view.batch_axis]
        return FederalBankStep(
            (
                TensorViewFederalCandidate.terminal_view(
                    "terminal",
                    local_log_score=view.value.new_zeros(()),
                    outputs={
                        "value": view.value * 3.0,
                        "validity": torch.ones(batch, dtype=torch.bool, device=view.value.device),
                        "score": torch.zeros(batch, dtype=view.value.dtype, device=view.value.device),
                    },
                ),
            )
        )


def test_mixed_program_executes_direct_edges_then_mounted_node_functionally() -> None:
    source_view = _view([[3.0, 5.0]], requires_grad=True)
    source = mechanisms.TensorResource(_spec("source"), source_view)
    workspace = mechanisms.TensorResource(_spec("workspace"), _view([[0.0, 0.0]]))
    output = mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]]))
    copy = mechanisms.Connection(
        "copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("workspace")
    )
    node = _ScaleNode()
    graph = mechanisms.ProgramGraph(
        (source, workspace, output),
        (copy,),
        nodes=(node,),
        programs={"mixed": ("copy", "scale")},
    )

    result = graph.execute_program_functional("mixed")
    output_state = next(item for item in result.state.resources if item.spec.resource_id == "output")

    assert tuple(item.connection_id for item in result.connections) == ("copy",)
    assert tuple(item.node_id for item in result.nodes) == ("scale",)
    assert result.nodes[0].receipt == {"operation": "scale"}
    assert tuple(
        tuple(item.connection_id if isinstance(item, mechanisms.ConnectionExecution) else item.node_id for item in dispatch.members)
        for dispatch in result.dispatches
    ) == (("copy",), ("scale",))
    assert tuple((dispatch.iteration, dispatch.frontier) for dispatch in result.dispatches) == (
        (0, 0), (0, 1)
    )
    torch.testing.assert_close(output_state.active_view.value, _view([[6.0, 10.0]]).value)
    torch.testing.assert_close(output.resolve().view.value, _view([[0.0, 0.0]]).value)

    output_state.active_view.value.sum().backward()
    assert node.scale.grad is not None
    assert source_view.value.grad is not None


def test_mixed_parallel_frontier_retains_connection_receipt_and_shared_snapshot() -> None:
    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[3.0, 5.0]])),
        mechanisms.TensorResource(_spec("workspace"), _view([[1.0, 2.0]])),
        mechanisms.TensorResource(_spec("scratch"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]])),
    )
    graph = mechanisms.ProgramGraph(
        resources,
        (mechanisms.Connection("copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("scratch")),),
        nodes=(_ScaleNode(),),
        programs={"mixed_stage": (mechanisms.ProgramStage(("scale", "copy")),)},
    )

    result = graph.execute_program_functional("mixed_stage", iterations=2)
    assert tuple(
        tuple(item.connection_id if isinstance(item, mechanisms.ConnectionExecution) else item.node_id for item in dispatch.members)
        for dispatch in result.dispatches
    ) == (("scale", "copy"), ("scale", "copy"))
    assert tuple((dispatch.iteration, dispatch.frontier) for dispatch in result.dispatches) == (
        (0, 0), (1, 0)
    )
    assert len(result.connections) == 2
    assert result.connections[0].source.resource_id == "source"
    assert result.connections[0].destination.resource_id == "scratch"
    final = {item.spec.resource_id: item.active_view.value for item in result.state.resources}
    torch.testing.assert_close(final["scratch"], _view([[3.0, 5.0]]).value)
    torch.testing.assert_close(final["output"], _view([[2.0, 4.0]]).value)


def test_mixed_program_gradient_matches_its_direct_tensor_reference() -> None:
    graph_input = torch.tensor([[[1.0], [3.0]]], requires_grad=True)
    source = mechanisms.TensorResource(
        _spec("source"),
        mechanisms.TensorView.from_tensor(
            graph_input,
            axis_names=("batch", "token", "feature"),
            axis_roles=("batch", "sequence", "feature"),
        ),
    )
    workspace = mechanisms.TensorResource(_spec("workspace"), _view([[0.0, 0.0]]))
    output = mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]]))
    node = _ScaleNode()
    with torch.no_grad():
        node.scale.fill_(1.75)
    graph = mechanisms.ProgramGraph(
        (source, workspace, output),
        (mechanisms.Connection("copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("workspace")),),
        nodes=(node,),
        programs={"mixed": ("copy", "scale")},
    )

    execution = graph.execute_program_functional("mixed")
    graph_output = next(
        item.active_view.value for item in execution.state.resources if item.spec.resource_id == "output"
    )
    graph_loss = graph_output.square().mean()
    graph_loss.backward()
    assert graph_input.grad is not None
    assert node.scale.grad is not None

    reference_input = graph_input.detach().clone().requires_grad_(True)
    reference_scale = node.scale.detach().clone().requires_grad_(True)
    reference_loss = (reference_input * reference_scale).square().mean()
    reference_loss.backward()

    torch.testing.assert_close(graph_output, reference_input.detach() * reference_scale.detach())
    torch.testing.assert_close(graph_input.grad, reference_input.grad)
    torch.testing.assert_close(node.scale.grad, reference_scale.grad)


def test_multi_port_node_fans_out_before_an_explicit_formula_join() -> None:
    value_type = mechanisms.TensorType.axes(
        ("batch", "token", "feature"), sizes=("B", "S", 1)
    )
    left_binding = mechanisms.InputBinding("left", value_type)
    right_binding = mechanisms.InputBinding("right", value_type)
    add_program = mechanisms.FormulaProgram.build(
        outputs=(mechanisms.add(left_binding, right_binding),)
    )
    join = mechanisms.FormulaProgramNode(
        "join",
        mechanisms.FormulaFabricV2(add_program),
        input_ports={
            "left": mechanisms.ResourcePort("left"),
            "right": mechanisms.ResourcePort("right"),
        },
        output_ports={"sum": mechanisms.ResourcePort("joined")},
        output_slots={"sum": "%0"},
    )
    source = mechanisms.TensorResource(_spec("source"), _view([[2.0, 4.0]]))
    workspace = mechanisms.TensorResource(_spec("workspace"), _view([[0.0, 0.0]]))
    left = mechanisms.TensorResource(_spec("left"), _view([[0.0, 0.0]]))
    right = mechanisms.TensorResource(_spec("right"), _view([[0.0, 0.0]]))
    joined = mechanisms.TensorResource(_spec("joined"), _view([[0.0, 0.0]]))
    copy = mechanisms.Connection(
        "copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("workspace")
    )
    graph = mechanisms.ProgramGraph(
        (source, workspace, left, right, joined),
        (copy,),
        nodes=(_SplitNode(), join),
        programs={"fanout_then_join": ("copy", "split", "join")},
    )

    result = graph.execute_program_functional("fanout_then_join")
    outputs = {item.spec.resource_id: item.active_view for item in result.state.resources}

    assert isinstance(result.nodes[0], mechanisms.MultiPortProgramNodeExecution)
    assert result.nodes[0].receipt == {"operation": "split"}
    assert tuple(name for name, _ in result.nodes[0].outputs) == ("left", "right")
    torch.testing.assert_close(outputs["left"].value, _view([[3.0, 5.0]]).value)
    torch.testing.assert_close(outputs["right"].value, _view([[4.0, 8.0]]).value)
    torch.testing.assert_close(outputs["joined"].value, _view([[7.0, 13.0]]).value)


def test_multi_port_node_rejects_implicit_output_convergence() -> None:
    with pytest.raises(mechanisms.ResourceGraphError, match="explicit Formula Fabric join"):
        mechanisms.MultiPortProgramNode(
            "invalid",
            input_ports={"value": mechanisms.ResourcePort("source")},
            output_ports={
                "first": mechanisms.ResourcePort("target"),
                "second": mechanisms.ResourcePort("target"),
            },
        )


def test_frozen_formula_multi_port_subgraph_lowers_to_static_tensor_ssa() -> None:
    value_type = mechanisms.TensorType.axes(
        ("batch", "token", "feature"), sizes=("B", "S", 1)
    )
    left_binding = mechanisms.InputBinding("left", value_type)
    right_binding = mechanisms.InputBinding("right", value_type)
    program = mechanisms.FormulaProgram.build(
        outputs=(
            mechanisms.add(left_binding, right_binding),
            mechanisms.add(left_binding, left_binding),
        )
    )
    left = mechanisms.TensorResource(_spec("left"), _view([[1.0, 3.0]]))
    right = mechanisms.TensorResource(_spec("right"), _view([[2.0, 5.0]]))
    summed = mechanisms.TensorResource(_spec("summed"), _view([[0.0, 0.0]]))
    doubled = mechanisms.TensorResource(_spec("doubled"), _view([[0.0, 0.0]]))
    node = mechanisms.FormulaProgramNode(
        "fanout",
        mechanisms.FormulaFabricV2(program),
        input_ports={
            "left": mechanisms.ResourcePort("left"),
            "right": mechanisms.ResourcePort("right"),
        },
        output_ports={
            "summed": mechanisms.ResourcePort("summed"),
            "doubled": mechanisms.ResourcePort("doubled"),
        },
        output_slots={"summed": "%0", "doubled": "%1"},
    )
    graph = mechanisms.ProgramGraph(
        (left, right, summed, doubled), (), nodes=(node,), programs={"fanout": ("fanout",)}
    )

    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "fanout")
    assert isinstance(plan, mechanisms.StaticProgramGraphExecutionPlan)
    eager = plan(
        left.resolve().view.value,
        right.resolve().view.value,
        summed.resolve().view.value,
        doubled.resolve().view.value,
    )
    compiled = torch.compile(plan, backend="eager", fullgraph=True)(
        left.resolve().view.value,
        right.resolve().view.value,
        summed.resolve().view.value,
        doubled.resolve().view.value,
    )
    exported = torch.export.export(
        plan,
        (
            left.resolve().view.value,
            right.resolve().view.value,
            summed.resolve().view.value,
            doubled.resolve().view.value,
        ),
    ).module()(
        left.resolve().view.value,
        right.resolve().view.value,
        summed.resolve().view.value,
        doubled.resolve().view.value,
    )

    torch.testing.assert_close(eager[2], _view([[3.0, 8.0]]).value)
    torch.testing.assert_close(eager[3], _view([[2.0, 6.0]]).value)
    for eager_value, compiled_value in zip(eager, compiled, strict=True):
        torch.testing.assert_close(eager_value, compiled_value)
    for eager_value, exported_value in zip(eager, exported, strict=True):
        torch.testing.assert_close(eager_value, exported_value)
def test_static_program_credit_lowering_composes_declared_local_vjps() -> None:
    source = mechanisms.TensorResource(_spec("source"), _view([[1.0, 2.0]]))
    hidden = mechanisms.TensorResource(_spec("hidden"), _view([[0.0, 0.0]]))
    output = mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]]))
    first = arti.as_fabric_node(
        "first",
        _DecoratedScale(2.0),
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("hidden")},
    )
    second = arti.as_fabric_node(
        "second",
        _ExplicitVJPScale(3.0),
        input_ports={"source": mechanisms.ResourcePort("hidden")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        (source, hidden, output),
        (),
        nodes=(first, second),
        programs={"chain": ("first", "second")},
    )
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "chain")
    assert isinstance(plan, mechanisms.StaticProgramGraphExecutionPlan)
    values = (
        source.resolve().view.value,
        hidden.resolve().view.value,
        output.resolve().view.value,
    )
    ordinary = plan(*values)
    ordinary_gradients = torch.autograd.grad(
        ordinary[-1].sum(),
        (first.module.scale, second.module.scale),
    )
    result = plan.credit_gradient(
        *values,
        terminal_cotangents={"output": torch.ones_like(ordinary[-1])},
    )

    torch.testing.assert_close(result.resource_values[-1], ordinary[-1].detach())
    torch.testing.assert_close(result.resource_cotangents["source"], torch.full_like(values[0], 6.0))
    torch.testing.assert_close(result.parameter_cotangents["first.scale"], ordinary_gradients[0])
    torch.testing.assert_close(result.parameter_cotangents["second.scale"], ordinary_gradients[1])


def test_static_mixed_program_composes_connection_boundary_and_node_credit() -> None:
    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[1.0, 2.0]])),
        mechanisms.TensorResource(_spec("hidden"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("middle"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]])),
    )
    first_transfer = mechanisms.LearnableAffineTransfer()
    final_transfer = mechanisms.LearnableAffineTransfer()
    with torch.no_grad():
        first_transfer.gain.fill_(2.0)
        final_transfer.gain.fill_(4.0)
    first = mechanisms.Connection(
        "first", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("hidden"),
        transfer=first_transfer,
        credit_boundary=arti.CreditBoundary(mode=arti.CreditBoundaryMode.MEAN),
    )
    node = arti.as_fabric_node(
        "multiply", _DecoratedScale(3.0),
        input_ports={"source": mechanisms.ResourcePort("hidden")},
        output_ports={"value": mechanisms.ResourcePort("middle")},
    )
    final = mechanisms.Connection(
        "final", mechanisms.ResourcePort("middle"), mechanisms.ResourcePort("output"),
        transfer=final_transfer,
    )
    graph = mechanisms.ProgramGraph(
        resources, (first, final), nodes=(node,),
        programs={"mixed": ("first", "multiply", "final")},
    )
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "mixed")
    assert isinstance(plan, mechanisms.StaticProgramGraphExecutionPlan)
    inputs = tuple(resource.resolve().view.value for resource in resources)
    runtime = graph.execute_program_functional("mixed")
    eager = plan(*inputs)
    compiled = torch.compile(plan, backend="eager", fullgraph=True)(*inputs)
    torch.testing.assert_close(eager[-1], _view([[24.0, 48.0]]).value)
    runtime_output = next(
        item.active_view.value for item in runtime.state.resources if item.spec.resource_id == "output"
    )
    torch.testing.assert_close(eager[-1], runtime_output)
    for eager_value, compiled_value in zip(eager, compiled, strict=True):
        torch.testing.assert_close(eager_value, compiled_value)

    result = plan.credit_gradient(
        *inputs, terminal_cotangents={"output": torch.ones_like(eager[-1])}
    )
    torch.testing.assert_close(result.resource_values[-1], eager[-1])
    torch.testing.assert_close(result.resource_cotangents["source"], torch.full_like(inputs[0], 12.0))
    assert result.parameter_cotangents["first.transfer.gain"] is not None
    assert result.parameter_cotangents["multiply.scale"] is not None
    assert result.parameter_cotangents["final.transfer.gain"] is not None
    first_step = plan.stages[0][0]
    assert isinstance(first_step, torch.nn.Module)
    boundary = first_step.plan.connections[0].credit_boundary
    assert boundary is not None
    trial_gain = first_step.plan.connections[0].transfer.gain - 0.1 * result.parameter_cotangents["first.transfer.gain"]
    assert torch.autograd.grad((trial_gain - 1.0).square(), boundary.alpha)[0].abs() > 0.0


def test_static_mixed_program_returns_conditional_context_credit_with_bernoulli_mask() -> None:
    class _ContextGate(torch.nn.Module):
        def forward(self, context: torch.Tensor) -> torch.Tensor:
            return context[:, :1].unsqueeze(-1)

    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[1.0, 2.0]])),
        mechanisms.TensorResource(_spec("hidden"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]])),
    )
    connection = mechanisms.Connection(
        "dynamic", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("hidden"),
        activation=_ContextGate(),
        credit_boundary=arti.CreditBoundary(mode=arti.CreditBoundaryMode.BERNOULLI),
    )
    node = arti.as_fabric_node(
        "multiply", _DecoratedScale(3.0),
        input_ports={"source": mechanisms.ResourcePort("hidden")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (connection,), nodes=(node,),
        programs={"mixed": ("dynamic", "multiply")},
    )
    context = torch.tensor([[0.25]])
    mask = torch.tensor([[[True], [False]]])
    plan = mechanisms.ResourceGraphCompiler.compile_program(
        graph, "mixed",
        example_contexts={"dynamic": context},
        example_credit_masks={"dynamic": mask},
    )
    assert plan.context_connection_ids == ("dynamic",)
    assert plan.credit_mask_connection_ids == ("dynamic",)
    inputs = (*tuple(resource.resolve().view.value for resource in resources), context, mask)
    output = plan(*inputs)[-1]
    torch.testing.assert_close(output, _view([[0.75, 1.5]]).value)
    result = plan.credit_gradient(
        *inputs, terminal_cotangents={"output": torch.ones_like(output)}
    )
    torch.testing.assert_close(result.resource_cotangents["source"], _view([[0.75, 0.0]]).value)
    torch.testing.assert_close(result.context_cotangents["dynamic"], torch.tensor([[3.0]]))


def test_static_mixed_parallel_stage_uses_one_resource_snapshot() -> None:
    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[2.0, 4.0]])),
        mechanisms.TensorResource(_spec("hidden"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]])),
    )
    connection = mechanisms.Connection(
        "copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("hidden")
    )
    node = arti.as_fabric_node(
        "multiply", _DecoratedScale(3.0),
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (connection,), nodes=(node,),
        programs={"parallel": (mechanisms.ProgramStage(("copy", "multiply")),)},
    )
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "parallel")
    inputs = tuple(resource.resolve().view.value for resource in resources)
    runtime = graph.execute_program_functional("parallel")
    values = plan(*inputs)
    by_id = {item.spec.resource_id: item.active_view.value for item in runtime.state.resources}
    for resource_id, value in zip(plan.resource_ids, values, strict=True):
        torch.testing.assert_close(value, by_id[resource_id])
    result = plan.credit_gradient(
        *inputs, terminal_cotangents={
            "hidden": torch.ones_like(values[1]),
            "output": torch.ones_like(values[2]),
        },
    )
    torch.testing.assert_close(result.resource_cotangents["source"], torch.full_like(inputs[0], 4.0))


def test_static_mixed_connection_and_program_wait_for_async_join() -> None:
    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[2.0, 4.0]])),
        mechanisms.TensorResource(_spec("left"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("right"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]])),
    )
    transfer = mechanisms.LearnableAffineTransfer()
    with torch.no_grad():
        transfer.gain.fill_(2.0)
    connection = mechanisms.Connection(
        "copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("left"),
        transfer=transfer,
    )
    right_head = arti.as_fabric_node(
        "right_head", _DecoratedScale(3.0),
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("right")},
    )
    pair_sum = arti.as_fabric_node(
        "sum", _DecoratedPairSum(),
        input_ports={
            "left": mechanisms.ResourcePort("left"),
            "right": mechanisms.ResourcePort("right"),
        },
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (connection,), nodes=(right_head, pair_sum),
        programs={"mixed": (
            "copy", mechanisms.ProgramJoin("early", "sum"),
            "right_head", mechanisms.ProgramJoin("late", "sum"),
        )},
    )
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "mixed")
    assert isinstance(plan, mechanisms.StaticDataflowProgramExecutionPlan)
    inputs = (*tuple(resource.resolve().view.value for resource in resources),
              torch.zeros((1, plan.arrival_width), dtype=torch.bool))
    values = plan(*inputs)
    compiled = torch.compile(plan, backend="eager", fullgraph=True)(*inputs)
    exported = torch.export.export(plan, inputs).module()(*inputs)
    for value, compiled_value, exported_value in zip(values, compiled, exported, strict=True):
        torch.testing.assert_close(value, compiled_value)
        torch.testing.assert_close(value, exported_value)
    runtime = graph.execute_program_functional("mixed")
    runtime_output = next(
        item.active_view.value for item in runtime.state.resources if item.spec.resource_id == "output"
    )
    torch.testing.assert_close(values[3], runtime_output)
    torch.testing.assert_close(values[3], _view([[10.0, 20.0]]).value)
    assert tuple(item.fired for item in runtime.joins) == (False, True)
    result = plan.credit_gradient(
        *inputs, terminal_cotangents={"output": torch.ones_like(values[3])}
    )
    assert result.join_ready[1] is not None and not result.join_ready[1].any()
    assert result.join_ready[3] is not None and result.join_ready[3].all()
    torch.testing.assert_close(result.resource_cotangents["source"], torch.full_like(inputs[0], 5.0))
    assert result.parameter_cotangents["copy.transfer.gain"] is not None
    assert result.parameter_cotangents["right_head.scale"] is not None


def test_static_mixed_async_join_preserves_conditional_credit_boundary() -> None:
    class _ContextGate(torch.nn.Module):
        def forward(self, context: torch.Tensor) -> torch.Tensor:
            return context[:, :1].unsqueeze(-1)

    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[2.0, 4.0]])),
        mechanisms.TensorResource(_spec("left"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("right"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]])),
    )
    connection = mechanisms.Connection(
        "dynamic", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("left"),
        activation=_ContextGate(),
        credit_boundary=arti.CreditBoundary(mode=arti.CreditBoundaryMode.BERNOULLI),
    )
    right_head = arti.as_fabric_node(
        "right_head", _DecoratedScale(3.0),
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("right")},
    )
    pair_sum = arti.as_fabric_node(
        "sum", _DecoratedPairSum(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (connection,), nodes=(right_head, pair_sum),
        programs={"mixed": ("dynamic", "right_head", mechanisms.ProgramJoin("ready", "sum"))},
    )
    context = torch.tensor([[0.25]])
    mask = torch.tensor([[[True], [False]]])
    plan = mechanisms.ResourceGraphCompiler.compile_program(
        graph, "mixed",
        example_contexts={"dynamic": context},
        example_credit_masks={"dynamic": mask},
    )
    assert isinstance(plan, mechanisms.StaticDataflowProgramExecutionPlan)
    inputs = graph.static_program_inputs(
        plan, contexts={"dynamic": context}, credit_masks={"dynamic": mask},
    )
    torch.testing.assert_close(inputs[-1], torch.zeros((1, plan.arrival_width), dtype=torch.bool))
    output = plan(*inputs)[3]
    torch.testing.assert_close(output, _view([[6.5, 13.0]]).value)
    result = plan.credit_gradient(
        *inputs, terminal_cotangents={"output": torch.ones_like(output)}
    )
    torch.testing.assert_close(result.resource_cotangents["source"], _view([[3.25, 3.0]]).value)
    torch.testing.assert_close(result.context_cotangents["dynamic"], torch.tensor([[2.0]]))


def test_static_mixed_async_join_keeps_parallel_stage_snapshot() -> None:
    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[2.0, 4.0]])),
        mechanisms.TensorResource(_spec("left"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("right"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]])),
    )
    connection = mechanisms.Connection(
        "copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("left")
    )
    right_head = arti.as_fabric_node(
        "right_head", _DecoratedScale(3.0),
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("right")},
    )
    pair_sum = arti.as_fabric_node(
        "sum", _DecoratedPairSum(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (connection,), nodes=(right_head, pair_sum),
        programs={"mixed": (
            mechanisms.ProgramStage(("copy", "right_head")),
            mechanisms.ProgramJoin("ready", "sum"),
        )},
    )
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "mixed")
    inputs = (*tuple(resource.resolve().view.value for resource in resources),
              torch.zeros((1, plan.arrival_width), dtype=torch.bool))
    runtime = graph.execute_program_functional("mixed")
    values = plan(*inputs)
    by_id = {item.spec.resource_id: item.active_view.value for item in runtime.state.resources}
    for resource_id, value in zip(plan.resource_ids, values[:-1], strict=True):
        torch.testing.assert_close(value, by_id[resource_id])
    torch.testing.assert_close(values[3], 4.0 * inputs[0])
    result = plan.credit_gradient(
        *inputs, terminal_cotangents={"output": torch.ones_like(values[3])}
    )
    torch.testing.assert_close(result.resource_cotangents["source"], torch.full_like(inputs[0], 4.0))


def test_static_mixed_async_join_rejects_intra_stage_read_after_write() -> None:
    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[2.0, 4.0]])),
        mechanisms.TensorResource(_spec("left"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("right"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]])),
    )
    connection = mechanisms.Connection(
        "copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("left")
    )
    right_head = arti.as_fabric_node(
        "right_head", _DecoratedScale(3.0),
        input_ports={"source": mechanisms.ResourcePort("left")},
        output_ports={"value": mechanisms.ResourcePort("right")},
    )
    pair_sum = arti.as_fabric_node(
        "sum", _DecoratedPairSum(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (connection,), nodes=(right_head, pair_sum),
        programs={"mixed": (
            mechanisms.ProgramStage(("copy", "right_head")),
            mechanisms.ProgramJoin("ready", "sum"),
        )},
    )
    with pytest.raises(mechanisms.ResourceGraphError, match="peers cannot read"):
        mechanisms.ResourceGraphCompiler.compile_program(graph, "mixed")


def test_static_mixed_async_join_preserves_per_sample_wait_and_credit() -> None:
    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[1.0, 2.0], [3.0, 4.0]])),
        mechanisms.TensorResource(_spec("left"), _view([[0.0, 0.0], [0.0, 0.0]])),
        mechanisms.TensorResource(_spec("right"), _view([[10.0, 20.0], [30.0, 40.0]])),
        mechanisms.TensorResource(_spec("output"), _view([[7.0, 7.0], [7.0, 7.0]])),
    )
    connection = mechanisms.Connection(
        "copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("left")
    )
    pair_sum = arti.as_fabric_node(
        "sum", _DecoratedPairSum(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (connection,), nodes=(pair_sum,),
        programs={"mixed": ("copy", mechanisms.ProgramJoin("ready", "sum"))},
    )
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "mixed")
    arrivals = torch.tensor([[False, True], [False, False]])
    inputs = (*tuple(resource.resolve().view.value for resource in resources), arrivals)
    output = plan(*inputs)[3]
    torch.testing.assert_close(output, _view([[11.0, 22.0], [7.0, 7.0]]).value)
    result = plan.credit_gradient(
        *inputs, terminal_cotangents={"output": torch.ones_like(output)}
    )
    torch.testing.assert_close(result.resource_cotangents["source"], _view([[1.0, 1.0], [0.0, 0.0]]).value)
    torch.testing.assert_close(result.resource_cotangents["right"], _view([[1.0, 1.0], [0.0, 0.0]]).value)
    torch.testing.assert_close(result.resource_cotangents["output"], _view([[0.0, 0.0], [1.0, 1.0]]).value)


def test_program_loop_preserves_pending_join_arrivals_across_iterations() -> None:
    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[2.0, 4.0]])),
        mechanisms.TensorResource(_spec("left"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("right"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_continue_spec("continue"), _continue_view(torch.ones(1))),
    )
    left_copy = mechanisms.Connection(
        "left_copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("left")
    )
    right_copy = mechanisms.Connection(
        "right_copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("right")
    )
    pair_sum = arti.as_fabric_node(
        "sum", _DecoratedPairSum(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (left_copy, right_copy), nodes=(pair_sum,),
        programs={
            "iterate": (mechanisms.ProgramJoin("ready", "sum"), "right_copy"),
            "idle": (mechanisms.ProgramJoin("idle_ready", "sum"),),
        },
        loops=(
            mechanisms.ProgramLoop("bounded", "iterate", "continue", max_iterations=2),
            mechanisms.ProgramLoop("idle_loop", "idle", "continue", max_iterations=2),
        ),
    )
    prepared = graph.execute_functional(("left_copy",))
    idle = graph.execute_loop_functional("idle_loop", state=prepared.state)
    assert tuple(item.fired for item in idle.joins) == (False, False)
    prior_right = next(item for item in prepared.state.resources if item.spec.resource_id == "right")
    idle_right = next(item for item in idle.state.resources if item.spec.resource_id == "right")
    assert idle_right.epoch == prior_right.epoch
    result = graph.execute_loop_functional("bounded", state=prepared.state)
    assert tuple(item.fired for item in result.joins) == (False, True)
    output = next(item.active_view.value for item in result.state.resources if item.spec.resource_id == "output")
    torch.testing.assert_close(output, _view([[4.0, 8.0]]).value)


def test_static_mixed_dataflow_loop_composes_join_credit_and_stop() -> None:
    class _ContextGate(torch.nn.Module):
        def forward(self, context: torch.Tensor) -> torch.Tensor:
            return context[:, :1].unsqueeze(-1)

    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[2.0, 4.0]])),
        mechanisms.TensorResource(_spec("left"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("right"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_continue_spec("continue"), _continue_view(torch.ones(1))),
    )
    connection = mechanisms.Connection(
        "copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("left"),
        activation=_ContextGate(),
        credit_boundary=arti.CreditBoundary(mode=arti.CreditBoundaryMode.BERNOULLI),
    )
    right_head = arti.as_fabric_node(
        "right_head", _DecoratedScale(3.0),
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("right")},
    )
    pair_sum = arti.as_fabric_node(
        "sum", _DecoratedPairSumStop(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={
            "value": mechanisms.ResourcePort("output"),
            "continue": mechanisms.ResourcePort("continue"),
        },
    )
    graph = mechanisms.ProgramGraph(
        resources, (connection,), nodes=(right_head, pair_sum),
        programs={"iterate": (
            mechanisms.ProgramStage(("copy", "right_head")),
            mechanisms.ProgramJoin("ready", "sum"),
        )},
        loops=(mechanisms.ProgramLoop("bounded", "iterate", "continue", max_iterations=3),),
    )
    context = torch.tensor([[0.5]])
    mask = torch.tensor([[[True], [False]]])
    plan = mechanisms.ResourceGraphCompiler.compile_loop(
        graph, "bounded",
        example_contexts={"copy": context},
        example_credit_masks={"copy": mask},
    )
    assert isinstance(plan, mechanisms.StaticDataflowLoopExecutionPlan)
    inputs = (*tuple(resource.resolve().view.value for resource in resources),
              context, mask, torch.zeros((1, plan.body.arrival_width), dtype=torch.bool))
    values = plan(*inputs)
    dynamic = plan.forward_until_done(*inputs)
    for expected, actual in zip(values, dynamic[:-2], strict=True):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(dynamic[-2], torch.tensor(1))
    torch.testing.assert_close(dynamic[-1], torch.tensor([1]))
    with torch.no_grad():
        inference_values = plan(*inputs)
        compiled_inference = torch.compile(plan, backend="eager", fullgraph=True)(*inputs)
    for expected, actual, compiled_value in zip(values, inference_values, compiled_inference, strict=True):
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(compiled_value, expected)
    compiled_dynamic = torch.compile(plan.forward_until_done, backend="eager", fullgraph=True)(*inputs)
    for expected, actual in zip(dynamic, compiled_dynamic, strict=True):
        torch.testing.assert_close(actual, expected)
    class _DynamicPlan(torch.nn.Module):
        def __init__(self, body: torch.nn.Module) -> None:
            super().__init__()
            self.body = body

        def forward(self, *values: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return self.body.forward_until_done(*values)

    exported_dynamic = torch.export.export(_DynamicPlan(plan), inputs)
    assert "while_loop" in str(exported_dynamic.graph_module.graph)
    for expected, actual in zip(dynamic, exported_dynamic.module()(*inputs), strict=True):
        torch.testing.assert_close(actual, expected)
    compiled = torch.compile(plan, backend="eager", fullgraph=True)(*inputs)
    exported = torch.export.export(plan, inputs).module()(*inputs)
    for value, compiled_value, exported_value in zip(values, compiled, exported, strict=True):
        torch.testing.assert_close(value, compiled_value)
        torch.testing.assert_close(value, exported_value)
    torch.testing.assert_close(values[3], _view([[7.0, 14.0]]).value)
    runtime = graph.execute_loop_functional(
        "bounded", contexts={"copy": context}, credit_masks={"copy": mask}
    )
    runtime_output = next(
        item.active_view.value for item in runtime.state.resources if item.spec.resource_id == "output"
    )
    torch.testing.assert_close(values[3], runtime_output)
    result = plan.credit_gradient(
        *inputs, terminal_cotangents={"output": torch.ones_like(values[3])}
    )
    assert tuple(bool(activity.all()) for activity in result.iteration_active) == (True, False, False)
    torch.testing.assert_close(result.resource_cotangents["source"], _view([[3.5, 3.0]]).value)
    torch.testing.assert_close(result.context_cotangents["copy"], torch.tensor([[2.0]]))
    assert result.parameter_cotangents["right_head.scale"] is not None
    open_graph = mechanisms.ProgramGraph(
        tuple(graph.resources.values()), tuple(graph.connections.values()),
        nodes=tuple(graph.nodes.values()), programs={"iterate": graph.program("iterate")},
        loops=(mechanisms.ProgramLoop("open", "iterate", "continue", max_iterations=None),),
    )
    open_execution = open_graph.execute_loop_functional(
        "open", host_step_limit=8, contexts={"copy": context}, credit_masks={"copy": mask},
    )
    assert open_execution.actual_iterations == 1
    open_plan = mechanisms.ResourceGraphCompiler.compile_loop(
        open_graph, "open",
        example_contexts={"copy": context}, example_credit_masks={"copy": mask},
    )
    open_credit = open_plan.credit_gradient(
        *inputs, execution=open_execution,
        terminal_cotangents={"output": torch.ones_like(values[3])},
    )
    torch.testing.assert_close(open_credit.resource_values[3], values[3])
    torch.testing.assert_close(
        open_credit.resource_cotangents["source"], result.resource_cotangents["source"]
    )
    torch.testing.assert_close(
        open_credit.context_cotangents["copy"], result.context_cotangents["copy"]
    )
    torch.testing.assert_close(
        open_credit.parameter_cotangents["right_head.scale"],
        result.parameter_cotangents["right_head.scale"],
    )
    if torch.cuda.is_available():
        cuda_plan = plan.cuda()
        cuda_inputs = tuple(value.cuda() for value in inputs)
        cuda_values = cuda_plan(*cuda_inputs)
        for cpu_value, cuda_value in zip(values, cuda_values, strict=True):
            torch.testing.assert_close(cuda_value.cpu(), cpu_value)
        cuda_dynamic = cuda_plan.forward_until_done(*cuda_inputs)
        for cpu_value, cuda_value in zip(dynamic, cuda_dynamic, strict=True):
            torch.testing.assert_close(cuda_value.cpu(), cpu_value)
        captured = cuda_plan.capture(*cuda_inputs)
        for cpu_value, replayed in zip(values, captured.replay(*cuda_inputs), strict=True):
            torch.testing.assert_close(replayed.cpu(), cpu_value)
        cuda_result = cuda_plan.credit_gradient(
            *cuda_inputs, terminal_cotangents={"output": torch.ones_like(cuda_values[3])}
        )
        torch.testing.assert_close(
            cuda_result.resource_cotangents["source"].cpu(), result.resource_cotangents["source"]
        )
        torch.testing.assert_close(
            cuda_result.context_cotangents["copy"].cpu(), result.context_cotangents["copy"]
        )


def test_program_route_specialization_keeps_shared_parameters_and_join_credit() -> None:
    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[2.0, 4.0]])),
        mechanisms.TensorResource(_vector_spec("scores"), _vector_view(torch.tensor([[3.0, -1.0]]))),
        mechanisms.TensorResource(_spec("left"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("right"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]])),
    )
    shared = mechanisms.LearnableAffineTransfer(gain=2.0)
    left = mechanisms.Connection(
        "left_candidate", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("left"),
        transfer=shared,
    )
    alternative = mechanisms.Connection(
        "alternative", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("left"),
        transfer=mechanisms.LearnableAffineTransfer(gain=3.0),
    )
    right = mechanisms.Connection(
        "right_fixed", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("right"),
        transfer=shared,
    )
    joined = arti.as_fabric_node(
        "sum", _DecoratedPairSum(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (left, alternative, right), nodes=(joined,),
        programs={"run": (
            mechanisms.ProgramRoute("pick", "scores", ("left_candidate", "alternative")),
            "right_fixed", mechanisms.ProgramJoin("ready", "sum"),
        )},
    )
    sealed = graph.specialize_program_routes({"pick": "left_candidate"})
    assert sealed.selections == (("pick", "left_candidate"),)
    assert sealed.graph.program("run")[0] == "left_candidate"
    assert "alternative" not in sealed.graph.connections
    sealed_shared = sealed.graph.connections["left_candidate"].transfer.gain
    assert sealed_shared is sealed.graph.connections["right_fixed"].transfer.gain
    assert sealed_shared is not shared.gain
    original_plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    sealed_plan = mechanisms.ResourceGraphCompiler.compile_program(sealed.graph, "run")
    source = resources[0].resolve().view.value.detach().clone().requires_grad_(True)
    values = (source, *(resource.resolve().view.value for resource in resources[1:]))
    arrivals = torch.zeros((1, original_plan.arrival_width), dtype=torch.bool)
    inputs = (*values, arrivals)
    forced = original_plan.forward_with_route_selections(
        *inputs, route_selections=(torch.tensor(0),),
    )
    specialized = torch.compile(sealed_plan, backend="eager", fullgraph=True)(*inputs)
    for reference, actual in zip(forced, specialized, strict=True):
        torch.testing.assert_close(actual, reference)
    forced_gradient = torch.autograd.grad(forced[4].sum(), (source, shared.gain), retain_graph=True)
    specialized_gradient = torch.autograd.grad(specialized[4].sum(), (source, sealed_shared))
    for reference, actual in zip(forced_gradient, specialized_gradient, strict=True):
        torch.testing.assert_close(actual, reference)
    sealed_credit = sealed_plan.credit_gradient(
        *inputs, terminal_cotangents={"output": torch.ones_like(specialized[4])},
    )
    torch.testing.assert_close(sealed_credit.resource_cotangents["source"], forced_gradient[0])
    with torch.no_grad():
        shared.gain.sub_(0.01 * forced_gradient[1])
        sealed_shared.sub_(0.01 * specialized_gradient[1])
    next_forced = original_plan.forward_with_route_selections(
        *inputs, route_selections=(torch.tensor(0),),
    )
    next_specialized = sealed_plan(*inputs)
    torch.testing.assert_close(next_specialized[4], next_forced[4])


def test_joint_fate_and_route_live_specialization_keeps_training_parameters() -> None:
    names = ("source", "scores", "left", "right", "joined", "selected", "output")
    values = (
        torch.tensor([[2.0]]), torch.tensor([[3.0, -1.0]]),
        *(torch.zeros(1, 1) for _ in range(5)),
    )
    resources = tuple(
        mechanisms.TensorResource(_vector_spec(name), _vector_view(value))
        for name, value in zip(names, values, strict=True)
    )
    choices = tuple(
        arti.as_differentiable_fabric_node(
            node_id, {"double": _DoubleFate(), "quadruple": _QuadrupleFate()},
            input_ports={"source": mechanisms.ResourcePort("source")},
            output_ports={"value": mechanisms.ResourcePort(destination)},
        )
        for node_id, destination in (("left_fate", "left"), ("right_fate", "right"))
    )
    join = arti.as_fabric_node(
        "sum", _DecoratedPairSum(),
        input_ports={
            "left": mechanisms.ResourcePort("left"),
            "right": mechanisms.ResourcePort("right"),
        },
        output_ports={"value": mechanisms.ResourcePort("joined")},
    )
    transfer = mechanisms.LearnableAffineTransfer(gain=1.5)
    graph = mechanisms.ProgramGraph(
        resources,
        (
            mechanisms.Connection(
                "joined_copy", mechanisms.ResourcePort("joined"),
                mechanisms.ResourcePort("selected"), transfer=transfer,
            ),
            mechanisms.Connection(
                "fallback", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("selected"),
            ),
            mechanisms.Connection(
                "final", mechanisms.ResourcePort("selected"), mechanisms.ResourcePort("output"),
            ),
        ),
        nodes=(*choices, join),
        programs={
            "joined_path": (
                mechanisms.ProgramStage(("left_fate", "right_fate")),
                mechanisms.ProgramJoin("ready", "sum"), "joined_copy",
            ),
            "direct_path": ("fallback",),
            "run": (
                mechanisms.ProgramRoute("pick", "scores", ("joined_path", "direct_path")),
                "final",
            ),
        },
    )
    fates = graph.specialize_differentiable_nodes(
        {"left_fate": "double", "right_fate": "quadruple"},
        share_module_state=True,
    ).graph
    selected = fates.specialize_program_routes(
        {"pick": "joined_path"}, share_module_state=True,
    ).graph
    assert selected.connections["joined_copy"].transfer.gain is transfer.gain
    assert "fallback" not in selected.connections
    independent = fates.specialize_program_routes({"pick": "joined_path"}).graph
    assert independent.connections["joined_copy"].transfer.gain is not transfer.gain

    routed_plan = mechanisms.ResourceGraphCompiler.compile_program(fates, "run")
    selected_plan = mechanisms.ResourceGraphCompiler.compile_program(selected, "run")
    inputs = (*values, torch.zeros((1, routed_plan.arrival_width), dtype=torch.bool))
    routed = routed_plan.forward_with_route_selections(
        *inputs, route_selections=(torch.tensor(0),),
    )[selected_plan.resource_ids.index("output")]
    specialized = selected_plan(*inputs)[selected_plan.resource_ids.index("output")]
    torch.testing.assert_close(specialized, routed)
    torch.testing.assert_close(specialized, torch.tensor([[18.0]]))
    compiled = torch.compile(selected_plan, backend="eager", fullgraph=True)
    torch.testing.assert_close(
        compiled(*inputs)[selected_plan.resource_ids.index("output")], specialized,
    )

    optimizer = torch.optim.AdamW((transfer.gain,), lr=0.1, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    specialized.square().sum().backward()
    assert transfer.gain.grad is not None
    optimizer.step()
    next_output = selected_plan(*inputs)[selected_plan.resource_ids.index("output")]
    assert not torch.equal(next_output, specialized)
    torch.testing.assert_close(
        next_output,
        routed_plan.forward_with_route_selections(
            *inputs, route_selections=(torch.tensor(0),),
        )[selected_plan.resource_ids.index("output")],
    )


def test_route_can_select_a_joining_subprogram_then_specialize_its_topology() -> None:
    source_view = _view([[2.0, 4.0]], requires_grad=True)
    source = source_view.value
    scores = torch.tensor([[3.0, -3.0]], requires_grad=True)
    resources = (
        mechanisms.TensorResource(_spec("source"), source_view),
        mechanisms.TensorResource(_vector_spec("scores"), _vector_view(scores)),
        *(mechanisms.TensorResource(_spec(name), _view([[0.0, 0.0]]))
          for name in ("left", "right", "output")),
    )
    left = mechanisms.Connection(
        "left_edge", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("left"),
        transfer=mechanisms.LearnableAffineTransfer(gain=2.0),
    )
    right = mechanisms.Connection(
        "right_edge", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("right"),
        transfer=mechanisms.LearnableAffineTransfer(gain=3.0),
    )
    direct = mechanisms.Connection(
        "direct_edge", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
        transfer=mechanisms.LearnableAffineTransfer(gain=5.0),
    )
    joined = arti.as_fabric_node(
        "sum", _DecoratedPairSum(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (left, right, direct), nodes=(joined,),
        programs={
            "join_path": (
                mechanisms.ProgramStage(("left_edge", "right_edge")),
                mechanisms.ProgramJoin("ready", "sum"),
            ),
            "direct_path": ("direct_edge",),
            "run": (mechanisms.ProgramRoute("pick", "scores", ("join_path", "direct_path")),),
        },
    )
    executed = graph.execute_program_functional("run")
    assert executed.routes[0].candidate_id == "join_path"
    assert [join.fired for join in executed.joins] == [True]
    assert len(executed.connections) == 2
    output = next(item.active_view.value for item in executed.state.resources
                  if item.spec.resource_id == "output")
    torch.testing.assert_close(output, source * 5)
    dynamic_plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    assert isinstance(dynamic_plan, mechanisms.StaticDataflowProgramExecutionPlan)
    values = tuple(resource.resolve().view.value for resource in resources)
    arrivals = torch.zeros((1, dynamic_plan.arrival_width), dtype=torch.bool)
    dynamic_result = dynamic_plan(*values, arrivals)
    torch.testing.assert_close(dynamic_result[4], output)
    dynamic_gradient = torch.autograd.grad(
        dynamic_result[4].sum(),
        (source, left.transfer.gain, right.transfer.gain),
    )
    reference_gradient = torch.autograd.grad(
        output.sum(), (source, left.transfer.gain, right.transfer.gain),
        retain_graph=True,
    )
    for reference, actual in zip(reference_gradient, dynamic_gradient, strict=True):
        torch.testing.assert_close(reference, actual)
    dynamic_credit = dynamic_plan.credit_gradient(
        *values, arrivals, terminal_cotangents={"output": torch.ones_like(output)},
    )
    torch.testing.assert_close(
        dynamic_credit.resource_cotangents["source"], reference_gradient[0],
    )
    torch._dynamo.reset()
    compiled_dynamic = torch.compile(dynamic_plan, backend="eager", fullgraph=True)
    compiled_result = compiled_dynamic(*values, arrivals)
    for expected, actual in zip(dynamic_result, compiled_result, strict=True):
        torch.testing.assert_close(actual, expected)
    direct_result = dynamic_plan.forward_with_route_selections(
        *values, arrivals, route_selections=(torch.tensor(1),),
    )
    torch.testing.assert_close(direct_result[4], source * 5)
    torch.testing.assert_close(direct_result[-1], arrivals)

    selected = graph.specialize_program_routes({"pick": "join_path"}).graph
    assert "join_path" not in selected._programs
    assert "direct_path" not in selected._programs
    assert "direct_edge" not in selected.connections
    assert tuple(type(step) for step in selected.program("run")) == (
        mechanisms.ProgramStage, mechanisms.ProgramJoin,
    )
    lowered = mechanisms.ResourceGraphCompiler.compile_program(selected, "run")
    arrivals = torch.zeros((1, lowered.arrival_width), dtype=torch.bool)
    compiled = lowered(*values, arrivals)
    torch.testing.assert_close(compiled[4], output)
    eager_gradient = torch.autograd.grad(output.sum(), (source, left.transfer.gain, right.transfer.gain))
    selected_gradient = torch.autograd.grad(
        compiled[4].sum(),
        (source, selected.connections["left_edge"].transfer.gain,
         selected.connections["right_edge"].transfer.gain),
    )
    for reference, actual in zip(eager_gradient, selected_gradient, strict=True):
        torch.testing.assert_close(reference, actual)

    sampled = graph.execute_program_functional("run", sample_routes=True)
    sampled_output = next(item.active_view.value for item in sampled.state.resources
                          if item.spec.resource_id == "output")
    sampled.routes[0].structure_objective(sampled_output.square().mean()).backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()

    direct_only = graph.specialize_program_routes({"pick": "direct_path"}).graph
    assert "sum" not in direct_only.nodes
    assert "left_edge" not in direct_only.connections
    assert "right_edge" not in direct_only.connections
    assert "join_path" not in direct_only._programs
    assert "direct_path" not in direct_only._programs


def test_nested_dynamic_route_preserves_join_arrivals_and_output_gradient() -> None:
    source = torch.tensor([[2.0]], requires_grad=True)
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(source)),
        mechanisms.TensorResource(_vector_spec("outer_scores"), _vector_view(torch.tensor([[5.0, -5.0]]))),
        mechanisms.TensorResource(_vector_spec("inner_scores"), _vector_view(torch.tensor([[5.0, -5.0]]))),
        *(mechanisms.TensorResource(_vector_spec(name), _vector_view(torch.zeros(1, 1)))
          for name in ("left", "right", "output")),
    )
    connections = tuple(
        mechanisms.Connection(
            name, mechanisms.ResourcePort("source"), mechanisms.ResourcePort(destination),
            transfer=mechanisms.LearnableAffineTransfer(gain=gain),
        )
        for name, destination, gain in (
            ("left_edge", "left", 2.0), ("right_edge", "right", 3.0),
            ("inner_direct", "output", 4.0), ("outer_direct", "output", 7.0),
        )
    )
    join = arti.as_fabric_node(
        "sum", _DecoratedPairSum(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        resources, connections, nodes=(join,),
        programs={
            "joined": (mechanisms.ProgramStage(("left_edge", "right_edge")),
                       mechanisms.ProgramJoin("ready", "sum")),
            "branch": (mechanisms.ProgramRoute(
                "inner", "inner_scores", ("joined", "inner_direct"),
            ),),
            "run": (mechanisms.ProgramRoute(
                "outer", "outer_scores", ("branch", "outer_direct"),
            ),),
        },
    )
    reference = graph.execute_program_functional("run")
    assert tuple(route.route_id for route in reference.routes) == ("outer", "inner")
    assert tuple(join_receipt.fired for join_receipt in reference.joins) == (True,)
    reference_output = next(
        item.active_view.value for item in reference.state.resources
        if item.spec.resource_id == "output"
    )
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    inputs = graph.static_program_inputs(plan)
    output_index = plan.resource_ids.index("output")
    observed = plan(*inputs)
    torch.testing.assert_close(observed[output_index], reference_output)
    torch.testing.assert_close(observed[-1], torch.zeros_like(inputs[-1]))
    compiled = torch.compile(plan, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(*inputs)[output_index], reference_output)
    observed_gradient = torch.autograd.grad(
        observed[output_index].sum(),
        (source, connections[0].transfer.gain, connections[1].transfer.gain),
        retain_graph=True,
    )
    reference_gradient = torch.autograd.grad(
        reference_output.sum(),
        (source, connections[0].transfer.gain, connections[1].transfer.gain),
    )
    for actual, expected in zip(observed_gradient, reference_gradient, strict=True):
        torch.testing.assert_close(actual, expected)
    _, nested_choices, _ = plan.forward_with_route_credit(*inputs)
    assert tuple(choice.item() for choice in nested_choices) == (0, 0)
    reverse = plan.credit_gradient(
        *inputs, terminal_cotangents={"output": torch.ones_like(observed[output_index])},
        route_selections=nested_choices,
    )
    torch.testing.assert_close(reverse.resource_cotangents["source"], observed_gradient[0])
    for gain, expected in zip(
        (connections[0].transfer.gain, connections[1].transfer.gain),
        observed_gradient[1:], strict=True,
    ):
        name = next(name for name, parameter in reverse.parameters.items() if parameter is gain)
        torch.testing.assert_close(reverse.parameter_cotangents[name], expected)
    direct_inputs = (*inputs[:2], torch.tensor([[-5.0, 5.0]]), *inputs[3:])
    torch.testing.assert_close(compiled(*direct_inputs)[output_index], torch.tensor([[8.0]]))
    bypass_inputs = (inputs[0], torch.tensor([[-5.0, 5.0]]), *inputs[2:])
    torch.testing.assert_close(compiled(*bypass_inputs)[output_index], torch.tensor([[14.0]]))


def test_sample_scoped_outer_route_compiles_nested_candidate_per_cohort() -> None:
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(torch.tensor([[2.0], [3.0]]))),
        mechanisms.TensorResource(
            _vector_spec("outer_scores"), _vector_view(torch.tensor([[5.0, -5.0], [-5.0, 5.0]])),
        ),
        mechanisms.TensorResource(
            _vector_spec("inner_scores"), _vector_view(torch.tensor([[5.0, -5.0], [-5.0, 5.0]])),
        ),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(2, 1))),
    )
    connections = tuple(
        mechanisms.Connection(
            name, mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
            transfer=mechanisms.LearnableAffineTransfer(gain=gain),
        )
        for name, gain in (("first", 2.0), ("second", 3.0), ("fallback", 4.0))
    )
    graph = mechanisms.ProgramGraph(
        resources, connections,
        programs={
            "branch": (mechanisms.ProgramRoute(
                "inner", "inner_scores", ("first", "second"),
            ),),
            "run": (mechanisms.ProgramRoute(
                "outer", "outer_scores", ("branch", "fallback"), selection_scope="sample",
            ),),
        },
    )
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    inputs = graph.static_program_inputs(plan)
    expected = torch.tensor([[4.0], [12.0]])
    torch.testing.assert_close(plan(*inputs)[3], expected)
    torch.testing.assert_close(
        torch.compile(plan, backend="eager", fullgraph=True)(*inputs)[3], expected,
    )
    inner_scores = inputs[2].detach().requires_grad_(True)
    routed_inputs = (*inputs[:2], inner_scores, *inputs[3:])
    routed, choices, log_probability = plan.forward_with_route_credit(*routed_inputs)
    torch.testing.assert_close(routed[3], expected)
    torch.testing.assert_close(choices[0], torch.tensor([0, 1]))
    torch.testing.assert_close(choices[1], torch.tensor([0, -1]))
    assert log_probability.shape == (2,)
    inner_gradient = torch.autograd.grad(log_probability.sum(), inner_scores)[0]
    assert inner_gradient[0].abs().sum() > 0
    torch.testing.assert_close(inner_gradient[1], torch.zeros_like(inner_gradient[1]))
    replayed = plan.forward_with_route_selections(*routed_inputs, route_selections=choices)
    torch.testing.assert_close(replayed[3], routed[3], atol=0, rtol=0)
    reverse = plan.credit_gradient(
        *routed_inputs, terminal_cotangents={"output": torch.ones_like(expected)},
        route_selections=choices,
    )
    direct_gradients = torch.autograd.grad(
        replayed[3].sum(), (connections[0].transfer.gain, connections[2].transfer.gain),
    )
    for gain, direct in zip(
        (connections[0].transfer.gain, connections[2].transfer.gain), direct_gradients, strict=True,
    ):
        name = next(name for name, parameter in reverse.parameters.items() if parameter is gain)
        torch.testing.assert_close(reverse.parameter_cotangents[name], direct)
    compiled_credit = torch.compile(
        plan.forward_with_route_credit, backend="eager", fullgraph=True,
    )(*routed_inputs)
    torch.testing.assert_close(compiled_credit[0][3], expected)
    for actual, recorded in zip(compiled_credit[1], choices, strict=True):
        torch.testing.assert_close(actual, recorded)


@pytest.mark.parametrize("inner_scope", ("sample", "batch"))
def test_nested_cohort_route_credit_counts_each_decision_once(inner_scope: str) -> None:
    outer_scores = torch.tensor([[8.0, -8.0], [8.0, -8.0], [-8.0, 8.0], [-8.0, 8.0]])
    inner_scores = torch.tensor([[8.0, -8.0], [-4.0, 4.0], [8.0, -8.0], [8.0, -8.0]])
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(torch.ones(4, 1))),
        mechanisms.TensorResource(_vector_spec("outer_scores"), _vector_view(outer_scores)),
        mechanisms.TensorResource(_vector_spec("inner_scores"), _vector_view(inner_scores)),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(4, 1))),
    )
    connections = tuple(
        mechanisms.Connection(
            name, mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
            transfer=mechanisms.LearnableAffineTransfer(gain=gain),
        )
        for name, gain in (("first", 2.0), ("second", 3.0), ("fallback", 4.0))
    )
    graph = mechanisms.ProgramGraph(
        resources, connections,
        programs={
            "branch": (mechanisms.ProgramRoute(
                "inner", "inner_scores", ("first", "second"), selection_scope=inner_scope,
            ),),
            "run": (mechanisms.ProgramRoute(
                "outer", "outer_scores", ("branch", "fallback"), selection_scope="sample",
            ),),
        },
    )
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    assert plan.route_receipt_scopes == ("sample", "sample")
    inputs = graph.static_program_inputs(plan)
    output, choices, joint_term = plan.forward_with_route_credit(*inputs)
    torch.testing.assert_close(choices[0], torch.tensor([0, 0, 1, 1]))
    expected_inner = torch.tensor([0, 1, -1, -1]) if inner_scope == "sample" else torch.tensor([0, 0, -1, -1])
    torch.testing.assert_close(choices[1], expected_inner)
    expected_output = torch.tensor([[2.0], [3.0], [4.0], [4.0]]) if inner_scope == "sample" else torch.tensor([[2.0], [2.0], [4.0], [4.0]])
    torch.testing.assert_close(output[3], expected_output)
    outer_term = torch.log_softmax(outer_scores, dim=-1)[torch.arange(4), choices[0]].sum()
    if inner_scope == "sample":
        inner_term = torch.log_softmax(inner_scores[:2], dim=-1)[
            torch.arange(2), expected_inner[:2]
        ].sum()
    else:
        inner_term = torch.log_softmax(inner_scores[:2].mean(0), dim=-1)[0]
    torch.testing.assert_close(joint_term.sum(), outer_term + inner_term)
    replayed = plan.forward_with_route_selections(*inputs, route_selections=choices)
    torch.testing.assert_close(replayed[3], output[3], atol=0, rtol=0)
    compiled = torch.compile(
        plan.forward_with_route_credit, backend="eager", fullgraph=True,
    )(*inputs)
    torch.testing.assert_close(compiled[0][3], output[3])
    torch.testing.assert_close(compiled[2], joint_term)
    if torch.cuda.is_available():
        plan = plan.to("cuda")
        cuda_inputs = tuple(
            value.to("cuda").detach().requires_grad_(index == 2)
            for index, value in enumerate(inputs)
        )
        cuda_output, cuda_choices, cuda_term = torch.compile(
            plan.forward_with_route_credit, backend="inductor", fullgraph=True,
        )(*cuda_inputs)
        torch.testing.assert_close(cuda_output[3].cpu(), output[3])
        torch.testing.assert_close(cuda_term.cpu(), joint_term)
        for actual, expected in zip(cuda_choices, choices, strict=True):
            torch.testing.assert_close(actual.cpu(), expected)
        reference_scores = inner_scores.detach().requires_grad_(True)
        if inner_scope == "sample":
            reference_term = torch.log_softmax(reference_scores[:2], dim=-1)[
                torch.arange(2), expected_inner[:2]
            ].sum()
        else:
            reference_term = torch.log_softmax(reference_scores[:2].mean(0), dim=-1)[0]
        expected_gradient = torch.autograd.grad(reference_term, reference_scores)[0]
        actual_gradient = torch.autograd.grad(cuda_term.sum(), cuda_inputs[2])[0]
        torch.testing.assert_close(actual_gradient.cpu(), expected_gradient)


def test_open_loop_nested_sample_receipt_fails_explicitly() -> None:
    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(torch.tensor([[2.0], [3.0]]))),
        mechanisms.TensorResource(
            _vector_spec("outer_scores"),
            _vector_view(torch.tensor([[8.0, -8.0], [-8.0, 8.0]])),
        ),
        mechanisms.TensorResource(
            _vector_spec("inner_scores"),
            _vector_view(torch.tensor([[8.0, -8.0], [-8.0, 8.0]])),
        ),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(2, 1))),
        mechanisms.TensorResource(_continue_spec("continue"), _continue_view(torch.ones(2))),
    )
    connections = tuple(
        mechanisms.Connection(
            name, mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
            transfer=mechanisms.LearnableAffineTransfer(gain=gain),
        )
        for name, gain in (("first", 2.0), ("second", 3.0), ("fallback", 4.0))
    )
    graph = mechanisms.ProgramGraph(
        resources, connections,
        programs={
            "branch": (mechanisms.ProgramRoute(
                "inner", "inner_scores", ("first", "second"), selection_scope="sample",
            ),),
            "run": (mechanisms.ProgramRoute(
                "outer", "outer_scores", ("branch", "fallback"), selection_scope="sample",
            ),),
        },
        loops=(mechanisms.ProgramLoop("open", "run", "continue", max_iterations=None),),
    )
    plan = mechanisms.ResourceGraphCompiler.compile_loop(graph, "open")
    assert plan.body.route_receipt_ids == ("outer", "inner")
    inputs = tuple(
        graph.resource(resource_id).resolve().view.value for resource_id in plan.resource_ids
    )
    inputs = (*inputs, torch.zeros((2, plan.body.arrival_width), dtype=torch.bool))

    with pytest.raises(
        mechanisms.ResourceGraphCompileError,
        match="sample-scoped nested routes require cohort route receipt lowering",
    ):
        plan.forward_until_done_with_route_receipt(
            *inputs, host_step_limit=torch.tensor(2, dtype=torch.int64), receipt_capacity=2,
        )


def test_compiled_loop_requeries_nested_route_after_state_change() -> None:
    @arti.fabric_layer(inputs=("scores",), outputs={"value": "scores"})
    class FlipScores(torch.nn.Module):
        def forward(self, scores: torch.Tensor) -> torch.Tensor:
            return -scores

    resources = (
        mechanisms.TensorResource(_vector_spec("source"), _vector_view(torch.tensor([[2.0]]))),
        mechanisms.TensorResource(_vector_spec("outer_scores"), _vector_view(torch.tensor([[5.0, -5.0]]))),
        mechanisms.TensorResource(_vector_spec("inner_scores"), _vector_view(torch.tensor([[5.0, -5.0]]))),
        mechanisms.TensorResource(_vector_spec("output"), _vector_view(torch.zeros(1, 1))),
        mechanisms.TensorResource(_continue_spec("continue"), _continue_view(torch.ones(1))),
    )
    connections = tuple(
        mechanisms.Connection(
            name, mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
            transfer=mechanisms.LearnableAffineTransfer(gain=gain),
        )
        for name, gain in (("first", 2.0), ("second", 3.0), ("fallback", 4.0))
    )
    flip = arti.as_fabric_node(
        "flip", FlipScores(),
        input_ports={"scores": mechanisms.ResourcePort("inner_scores")},
        output_ports={"value": mechanisms.ResourcePort("inner_scores")},
    )
    graph = mechanisms.ProgramGraph(
        resources, connections, nodes=(flip,),
        programs={
            "branch": (mechanisms.ProgramRoute(
                "inner", "inner_scores", ("first", "second"),
            ),),
            "iterate": (
                mechanisms.ProgramRoute("outer", "outer_scores", ("branch", "fallback")),
                "flip",
            ),
        },
        loops=(mechanisms.ProgramLoop("bounded", "iterate", "continue", max_iterations=2),),
    )
    eager = graph.execute_loop_functional("bounded")
    assert tuple((route.route_id, route.candidate_id) for route in eager.routes) == (
        ("outer", "branch"), ("inner", "first"),
        ("outer", "branch"), ("inner", "second"),
    )
    plan = mechanisms.ResourceGraphCompiler.compile_loop(graph, "bounded")
    inputs = (
        *(resource.resolve().view.value for resource in resources),
        torch.zeros((1, plan.body.arrival_width), dtype=torch.bool),
    )
    output_index = plan.resource_ids.index("output")
    torch.testing.assert_close(plan.forward_bounded(*inputs)[output_index], torch.tensor([[6.0]]))
    torch.testing.assert_close(
        torch.compile(plan.forward_bounded, backend="eager", fullgraph=True)(*inputs)[output_index],
        torch.tensor([[6.0]]),
    )


def test_routed_two_head_program_writes_persistent_resource_for_later_read() -> None:
    @arti.fabric_layer(inputs=("source",), outputs={"left": "source", "right": "source"})
    class TwoHeads(torch.nn.Module):
        def forward(self, source: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"left": source + 1.0, "right": source * 2.0}

    scores = torch.tensor([[2.0, -2.0]], requires_grad=True)
    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[2.0]], requires_grad=True)),
        mechanisms.TensorResource(_vector_spec("scores"), _vector_view(scores)),
        *(
            mechanisms.TensorResource(_spec(name), _view([[0.0]]))
            for name in ("left", "right", "joined")
        ),
        mechanisms.TensorResource(
            mechanisms.TensorResourceSpec(
                "memory", _spec("memory").view_pattern,
                lifetime=mechanisms.ResourceLifetime.PERSISTENT,
            ),
            _view([[0.0]]),
        ),
        mechanisms.TensorResource(_spec("unused"), _view([[0.0]])),
        mechanisms.TensorResource(_spec("output"), _view([[0.0]])),
    )
    split = arti.as_fabric_node(
        "split", TwoHeads(),
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={
            "left": mechanisms.ResourcePort("left"),
            "right": mechanisms.ResourcePort("right"),
        },
    )
    joined = arti.as_fabric_node(
        "sum", _DecoratedPairSum(),
        input_ports={
            "left": mechanisms.ResourcePort("left"),
            "right": mechanisms.ResourcePort("right"),
        },
        output_ports={"value": mechanisms.ResourcePort("joined")},
    )
    graph = mechanisms.ProgramGraph(
        resources,
        (
            mechanisms.Connection(
                "write", mechanisms.ResourcePort("joined"), mechanisms.ResourcePort("memory"),
                transfer=mechanisms.LearnableAffineTransfer(gain=1.5),
            ),
            mechanisms.Connection(
                "skip", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("unused"),
            ),
            mechanisms.Connection(
                "read", mechanisms.ResourcePort("memory"), mechanisms.ResourcePort("output"),
                transfer=mechanisms.LearnableAffineTransfer(gain=2.0),
            ),
        ),
        nodes=(split, joined),
        programs={
            "write_path": ("split", mechanisms.ProgramJoin("ready", "sum"), "write"),
            "skip_path": ("skip",),
            "run": (
                mechanisms.ProgramRoute("choose", "scores", ("write_path", "skip_path")),
                "read",
            ),
        },
    )
    first = graph.execute_program_functional("run")
    first_state = {item.spec.resource_id: item for item in first.state.resources}
    torch.testing.assert_close(first_state["memory"].active_view.value, _view([[10.5]]).value)
    torch.testing.assert_close(first_state["output"].active_view.value, _view([[21.0]]).value)
    second = graph.execute_program_functional(
        "run", state=first.state,
        input_views={"scores": _vector_view(torch.tensor([[-2.0, 2.0]]))},
    )
    second_state = {item.spec.resource_id: item for item in second.state.resources}
    assert second_state["memory"].epoch == first_state["memory"].epoch
    torch.testing.assert_close(second_state["output"].active_view.value, _view([[21.0]]).value)

    continued = graph.specialize_program_routes(
        {"choose": "skip_path"}, state=first.state,
    ).graph
    continued_state = continued.state()
    carried_memory = next(
        item for item in continued_state.resources if item.spec.resource_id == "memory"
    )
    torch.testing.assert_close(carried_memory.active_view.value, first_state["memory"].active_view.value)
    assert carried_memory.epoch == first_state["memory"].epoch
    assert continued.resource("memory").resolve().view.value.grad_fn is None
    continued_execution = continued.execute_program_functional(
        "run", input_views={"scores": _vector_view(torch.tensor([[-2.0, 2.0]]))},
    )
    continued_output = next(
        item.active_view.value for item in continued_execution.state.resources
        if item.spec.resource_id == "output"
    )
    torch.testing.assert_close(continued_output, second_state["output"].active_view.value)
    torch.testing.assert_close(graph.resource("memory").resolve().view.value, _view([[0.0]]).value)

    kept_join = graph.specialize_program_routes(
        {"choose": "write_path"}, state=first.state,
    ).graph
    assert kept_join.state().join_cursors == first.state.join_cursors

    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    assert isinstance(plan, mechanisms.StaticDataflowProgramExecutionPlan)
    inputs = (
        *(resource.resolve().view.value for resource in resources),
        torch.zeros((1, plan.arrival_width), dtype=torch.bool),
    )
    written = plan.forward_with_route_selections(
        *inputs, route_selections=(torch.tensor(0),),
    )
    output_index = plan.resource_ids.index("output")
    memory_index = plan.resource_ids.index("memory")
    torch.testing.assert_close(written[output_index], _view([[21.0]]).value)
    torch.testing.assert_close(written[memory_index], _view([[10.5]]).value)
    credit = plan.credit_gradient(
        *inputs, route_selections=(torch.tensor(0),),
        terminal_cotangents={"output": torch.ones_like(written[output_index])},
    )
    torch.testing.assert_close(credit.resource_cotangents["source"], torch.full_like(inputs[0], 9.0))
    write_credit = tuple(
        credit.parameter_cotangents[name]
        for name, parameter in credit.parameters.items()
        if parameter is graph.connections["write"].transfer.gain
    )
    assert len(write_credit) == 1
    torch.testing.assert_close(write_credit[0], torch.tensor(14.0))
    compiled = torch.compile(plan, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(*inputs)[output_index], written[output_index])

    skipped_inputs = list(written)
    skipped_inputs[plan.resource_ids.index("scores")] = torch.tensor([[-2.0, 2.0]])
    skipped = plan.forward_with_route_selections(
        *skipped_inputs, route_selections=(torch.tensor(1),),
    )
    torch.testing.assert_close(skipped[memory_index], written[memory_index])
    torch.testing.assert_close(skipped[output_index], written[output_index])
    skipped_credit = plan.credit_gradient(
        *skipped_inputs, route_selections=(torch.tensor(1),),
        terminal_cotangents={"output": torch.ones_like(skipped[output_index])},
    )
    source_credit = skipped_credit.resource_cotangents["source"]
    assert source_credit is None or torch.count_nonzero(source_credit) == 0
    torch.testing.assert_close(
        skipped_credit.resource_cotangents["memory"],
        torch.full_like(skipped_inputs[memory_index], 2.0),
    )
    skipped_write_credit = tuple(
        skipped_credit.parameter_cotangents[name]
        for name, parameter in skipped_credit.parameters.items()
        if parameter is graph.connections["write"].transfer.gain
    )
    assert len(skipped_write_credit) == 1
    assert skipped_write_credit[0] is None or torch.count_nonzero(skipped_write_credit[0]) == 0
    selected = graph.specialize_program_routes({"choose": "write_path"}).graph
    assert "skip" not in selected.connections
    selected_plan = mechanisms.ResourceGraphCompiler.compile_program(selected, "run")
    torch.testing.assert_close(selected_plan(*inputs)[output_index], written[output_index])
    sampled_values, (sampled_route,), log_probability = plan.forward_with_route_credit(*inputs)
    final_loss = (sampled_values[output_index] - 5.0).square().mean()
    route_objective = final_loss.detach() * log_probability
    score_gradient = torch.autograd.grad(route_objective, scores)[0]
    expected_score_gradient = final_loss.detach() * (
        torch.nn.functional.one_hot(sampled_route, num_classes=2).to(scores.dtype)
        - torch.softmax(scores, dim=-1)
    )
    torch.testing.assert_close(score_gradient, expected_score_gradient)
    if torch.cuda.is_available():
        cuda_plan = plan.cuda()
        cuda_inputs = tuple(value.cuda() for value in inputs)
        cuda_written = cuda_plan.forward_with_route_selections(
            *cuda_inputs, route_selections=(torch.tensor(0, device="cuda"),),
        )
        torch.testing.assert_close(cuda_written[output_index].cpu(), written[output_index])
        cuda_credit = cuda_plan.credit_gradient(
            *cuda_inputs, route_selections=(torch.tensor(0, device="cuda"),),
            terminal_cotangents={"output": torch.ones_like(cuda_written[output_index])},
        )
        torch.testing.assert_close(
            cuda_credit.resource_cotangents["source"].cpu(),
            credit.resource_cotangents["source"],
        )


def test_dynamic_route_preserves_join_arrivals_across_calls() -> None:
    resources = tuple(
        mechanisms.TensorResource(_spec(name), _view([[value]]))
        for name, value in (
            ("seed", 2.0), ("left", 0.0), ("right", 0.0), ("output", 1.0),
        )
    ) + (
        mechanisms.TensorResource(
            _vector_spec("scores"), _vector_view(torch.tensor([[3.0, -3.0]])),
        ),
    )
    right_edge = mechanisms.Connection(
        "right_edge", mechanisms.ResourcePort("seed"), mechanisms.ResourcePort("right"),
    )
    left_edge = mechanisms.Connection(
        "left_edge", mechanisms.ResourcePort("output"), mechanisms.ResourcePort("left"),
    )
    direct_edge = mechanisms.Connection(
        "direct_edge", mechanisms.ResourcePort("seed"), mechanisms.ResourcePort("output"),
    )
    joined = arti.as_fabric_node(
        "sum", _DecoratedPairSum(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (right_edge, left_edge, direct_edge), nodes=(joined,),
        programs={
            "prime": ("right_edge",),
            "wait_path": ("left_edge", mechanisms.ProgramJoin("ready", "sum")),
            "direct_path": ("direct_edge",),
            "run": (mechanisms.ProgramRoute("pick", "scores", ("wait_path", "direct_path")),),
        },
    )
    primed = graph.execute_program_functional("prime")
    first_reference = graph.execute_program_functional("run", state=primed.state)
    second_reference = graph.execute_program_functional("run", state=first_reference.state)
    assert [item.fired for item in first_reference.joins] == [True]
    assert [item.fired for item in second_reference.joins] == [False]

    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    assert isinstance(plan, mechanisms.StaticDataflowProgramExecutionPlan)
    assert plan.arrival_width == 2
    inputs = graph.static_program_inputs(plan, state=primed.state)
    torch.testing.assert_close(inputs[-1], torch.tensor([[False, True]]))
    first = plan(*inputs)
    second = plan(*first)
    output_index = plan.resource_ids.index("output")
    for result, reference in ((first, first_reference), (second, second_reference)):
        expected = next(item.active_view.value for item in reference.state.resources
                        if item.spec.resource_id == "output")
        torch.testing.assert_close(result[output_index], expected)
    torch.testing.assert_close(first[-1], torch.tensor([[False, False]]))
    torch.testing.assert_close(second[-1], torch.tensor([[True, False]]))
    compiled = torch.compile(plan, backend="eager", fullgraph=True)(*inputs)
    torch.testing.assert_close(compiled[output_index], first[output_index])
    torch.testing.assert_close(compiled[-1], first[-1])

    sealed = graph.specialize_program_routes(
        {"pick": "wait_path"}, state=primed.state,
    ).graph
    sealed_plan = mechanisms.ResourceGraphCompiler.compile_program(sealed, "run")
    assert isinstance(sealed_plan, mechanisms.StaticDataflowProgramExecutionPlan)
    sealed_inputs = sealed.static_program_inputs(sealed_plan)
    torch.testing.assert_close(sealed_inputs[-1], inputs[-1])
    sealed_first = sealed_plan(*sealed_inputs)
    torch.testing.assert_close(
        sealed_first[sealed_plan.resource_ids.index("output")], first[output_index],
    )
    torch.testing.assert_close(sealed_first[-1], first[-1])


def test_sample_route_with_join_keeps_row_specific_arrivals() -> None:
    seed = _view([[2.0], [3.0]], requires_grad=True)
    resources = tuple(
        mechanisms.TensorResource(_spec(name), view)
        for name, view in (
            ("seed", seed), ("left", _view([[0.0], [0.0]])),
            ("right", _view([[2.0], [3.0]])),
            ("output", _view([[1.0], [1.0]])),
        )
    ) + (
        mechanisms.TensorResource(
            _vector_spec("scores"), _vector_view(torch.tensor([[3.0, -3.0], [-3.0, 3.0]])),
        ),
    )
    left_edge = mechanisms.Connection(
        "left_edge", mechanisms.ResourcePort("seed"), mechanisms.ResourcePort("left"),
        credit_boundary=arti.CreditBoundary(mode=arti.CreditBoundaryMode.MEAN),
    )
    direct_edge = mechanisms.Connection(
        "direct_edge", mechanisms.ResourcePort("seed"), mechanisms.ResourcePort("output"),
    )
    joined = arti.as_fabric_node(
        "sum", _DecoratedPairSum(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (left_edge, direct_edge), nodes=(joined,),
        programs={
            "wait_path": ("left_edge", mechanisms.ProgramJoin("ready", "sum")),
            "direct_path": ("direct_edge",),
            "run": (mechanisms.ProgramRoute(
                "pick", "scores", ("wait_path", "direct_path"), selection_scope="sample",
            ),),
        },
    )
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    assert isinstance(plan, mechanisms.StaticDataflowProgramExecutionPlan)
    values = tuple(resource.resolve().view.value for resource in resources)
    arrivals = torch.tensor([[False, True], [False, True]])
    result = plan(*values, arrivals)
    torch.testing.assert_close(result[3], _view([[4.0], [3.0]]).value)
    torch.testing.assert_close(result[-1], torch.tensor([[False, False], [False, True]]))
    source_gradient = torch.autograd.grad(result[3].sum(), seed.value)[0]
    torch.testing.assert_close(source_gradient, _view([[0.5], [1.0]]).value)
    credit = plan.credit_gradient(
        *values, arrivals, terminal_cotangents={"output": torch.ones_like(result[3])},
    )
    torch.testing.assert_close(credit.resource_cotangents["seed"], source_gradient)
    second = plan(*result)
    torch.testing.assert_close(second[3], result[3])
    torch.testing.assert_close(second[-1], torch.tensor([[True, False], [False, True]]))
    torch._dynamo.reset()
    compiled = torch.compile(plan, backend="eager", fullgraph=True)
    for expected, actual in zip(result, compiled(*values, arrivals), strict=True):
        torch.testing.assert_close(actual, expected)


def test_sample_route_with_join_uses_live_candidate_context_and_credit_mask() -> None:
    class _ContextGate(torch.nn.Module):
        def forward(self, context: torch.Tensor) -> torch.Tensor:
            return context[:, :1].unsqueeze(-1)

    seed = _view([[2.0], [3.0]], requires_grad=True)
    resources = tuple(
        mechanisms.TensorResource(_spec(name), view)
        for name, view in (
            ("seed", seed), ("left", _view([[0.0], [0.0]])),
            ("right", _view([[2.0], [3.0]])),
            ("output", _view([[1.0], [1.0]])),
        )
    ) + (
        mechanisms.TensorResource(
            _vector_spec("scores"), _vector_view(torch.tensor([[3.0, -3.0], [-3.0, 3.0]])),
        ),
    )
    gated = mechanisms.Connection(
        "gated", mechanisms.ResourcePort("seed"), mechanisms.ResourcePort("left"),
        activation=_ContextGate(),
        credit_boundary=arti.CreditBoundary(mode=arti.CreditBoundaryMode.BERNOULLI),
    )
    direct = mechanisms.Connection(
        "direct", mechanisms.ResourcePort("seed"), mechanisms.ResourcePort("output"),
    )
    joined = arti.as_fabric_node(
        "sum", _DecoratedPairSum(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (gated, direct), nodes=(joined,),
        programs={
            "gated_path": ("gated", mechanisms.ProgramJoin("ready", "sum")),
            "direct_path": ("direct",),
            "run": (mechanisms.ProgramRoute(
                "pick", "scores", ("gated_path", "direct_path"), selection_scope="sample",
            ),),
        },
    )
    context = torch.tensor([[1.0], [2.0]])
    mask = torch.tensor([[[False]], [[True]]])
    plan = mechanisms.ResourceGraphCompiler.compile_program(
        graph, "run", example_contexts={"gated": context}, example_credit_masks={"gated": mask},
    )
    assert isinstance(plan, mechanisms.StaticDataflowProgramExecutionPlan)
    assert plan.context_connection_ids == ("gated",)
    assert plan.credit_mask_connection_ids == ("gated",)
    values = tuple(resource.resolve().view.value for resource in resources)
    arrivals = torch.tensor([[False, True], [False, True]])
    result = plan(*values, context, mask, arrivals)
    torch.testing.assert_close(result[3], _view([[4.0], [3.0]]).value)
    torch.testing.assert_close(result[-1], torch.tensor([[False, False], [False, True]]))
    changed_context = torch.tensor([[2.0], [2.0]])
    changed = plan(*values, changed_context, mask, arrivals)
    torch.testing.assert_close(changed[3], _view([[6.0], [3.0]]).value)
    credit = plan.credit_gradient(
        *values, context, mask, arrivals,
        terminal_cotangents={"output": torch.ones_like(result[3])},
    )
    torch.testing.assert_close(credit.resource_cotangents["seed"], _view([[0.0], [1.0]]).value)
    ordinary = torch.autograd.grad(result[3].sum(), seed.value)[0]
    torch.testing.assert_close(ordinary, credit.resource_cotangents["seed"])
    torch._dynamo.reset()
    compiled = torch.compile(plan, backend="eager", fullgraph=True)
    for expected, actual in zip(changed, compiled(*values, changed_context, mask, arrivals), strict=True):
        torch.testing.assert_close(actual, expected)


def test_bounded_route_join_loop_tracks_hard_path_and_stop() -> None:
    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[2.0, 4.0]])),
        mechanisms.TensorResource(_vector_spec("scores"), _vector_view(torch.tensor([[3.0, -1.0]]))),
        mechanisms.TensorResource(_spec("left"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("right"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]])),
        mechanisms.TensorResource(_continue_spec("continue"), _continue_view(torch.ones(1))),
    )
    left = mechanisms.Connection(
        "left_copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("left"),
        transfer=mechanisms.LearnableAffineTransfer(gain=2.0),
    )
    right = mechanisms.Connection(
        "right_copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("right"),
        transfer=mechanisms.LearnableAffineTransfer(gain=4.0),
    )
    right_head = arti.as_fabric_node(
        "right_head", _DecoratedScale(3.0),
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("right")},
    )
    pair_sum = arti.as_fabric_node(
        "sum", _DecoratedPairSumStop(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={
            "value": mechanisms.ResourcePort("output"),
            "continue": mechanisms.ResourcePort("continue"),
        },
    )
    graph = mechanisms.ProgramGraph(
        resources, (left, right), nodes=(right_head, pair_sum),
        programs={"iterate": (
            mechanisms.ProgramRoute("choose", "scores", ("left_copy", "right_copy")),
            "right_head", mechanisms.ProgramJoin("ready", "sum"),
        )},
        loops=(mechanisms.ProgramLoop("bounded", "iterate", "continue", max_iterations=3),),
    )
    plan = mechanisms.ResourceGraphCompiler.compile_loop(graph, "bounded")
    assert isinstance(plan, mechanisms.StaticDataflowLoopExecutionPlan)
    values = tuple(resource.resolve().view.value for resource in resources)
    arrivals = torch.zeros((1, plan.body.arrival_width), dtype=torch.bool)
    compiled_bounded = torch.compile(plan.forward_bounded, backend="eager", fullgraph=True)
    compiled_dynamic = torch.compile(plan.forward_until_done, backend="eager", fullgraph=True)
    for scores, expected_output, expected_steps in (
        (torch.tensor([[3.0, -1.0]]), _view([[10.0, 20.0]]).value, 1),
        (torch.tensor([[-1.0, 3.0]]), torch.zeros_like(values[4]), 3),
    ):
        inputs = (values[0], scores, *values[2:], arrivals)
        result = plan.forward_bounded(*inputs)
        dynamic = plan.forward_until_done(*inputs)
        compiled = compiled_bounded(*inputs)
        compiled_until_done = compiled_dynamic(*inputs)
        for expected, actual, compiled_value in zip(result, dynamic[:-2], compiled, strict=True):
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(compiled_value, expected)
        for expected, compiled_value in zip(dynamic, compiled_until_done, strict=True):
            torch.testing.assert_close(compiled_value, expected)
        torch.testing.assert_close(result[4], expected_output)
        torch.testing.assert_close(dynamic[-2], torch.tensor(expected_steps))
        credit = plan.credit_gradient(
            *inputs, terminal_cotangents={"output": torch.ones_like(result[4])},
        )
        torch.testing.assert_close(credit.resource_values[4], result[4])
        torch.testing.assert_close(
            credit.resource_cotangents["source"],
            torch.full_like(values[0], 5.0) if expected_steps == 1 else torch.zeros_like(values[0]),
        )
        assert tuple(bool(mask.all()) for mask in credit.iteration_active) == (
            (True, False, False) if expected_steps == 1 else (True, True, True)
        )
        if torch.cuda.is_available():
            cuda_plan = copy.deepcopy(plan).to("cuda")
            cuda_inputs = tuple(value.to("cuda") for value in inputs)
            cuda_result = torch.compile(
                cuda_plan.forward_bounded, backend="eager", fullgraph=True,
            )(*cuda_inputs)
            cuda_dynamic = torch.compile(
                cuda_plan.forward_until_done, backend="eager", fullgraph=True,
            )(*cuda_inputs)
            for expected, actual in zip(result, cuda_result, strict=True):
                torch.testing.assert_close(actual.cpu(), expected)
            for expected, actual in zip(dynamic, cuda_dynamic, strict=True):
                torch.testing.assert_close(actual.cpu(), expected)
            cuda_credit = cuda_plan.credit_gradient(
                *cuda_inputs, terminal_cotangents={"output": torch.ones_like(cuda_result[4])},
            )
            torch.testing.assert_close(
                cuda_credit.resource_cotangents["source"].cpu(),
                credit.resource_cotangents["source"],
            )

    sampled_source = values[0].detach().clone().requires_grad_(True)
    sampled_scores = torch.tensor([[0.1, -0.1]], requires_grad=True)
    sampled_inputs = (sampled_source, sampled_scores, *values[2:], arrivals)
    sampled_forward = torch.compile(plan.forward_with_route_credit, backend="eager", fullgraph=True)
    sampled_values, choices, joint_log_probability, activity = sampled_forward(*sampled_inputs)
    assert len(choices) == plan.max_iterations
    assert all(len(iteration_choices) == 1 for iteration_choices in choices)
    replayed = plan.forward_with_route_selections(*sampled_inputs, route_selections=choices)
    for sampled, replay in zip(sampled_values, replayed, strict=True):
        torch.testing.assert_close(sampled, replay)
    expected_log_probability = torch.stack([
        torch.log_softmax(sampled_scores.mean(dim=0), dim=0)[iteration_choices[0]]
        * active.any().to(sampled_scores.dtype)
        for iteration_choices, active in zip(choices, activity, strict=True)
    ]).sum()
    torch.testing.assert_close(joint_log_probability, expected_log_probability)
    replay_credit = plan.credit_gradient(
        *sampled_inputs, route_selections=choices,
        terminal_cotangents={"output": torch.ones_like(sampled_values[4])},
    )
    assert len(replay_credit.route_selections) == plan.max_iterations
    for chosen, recorded in zip(choices, replay_credit.route_selections, strict=True):
        torch.testing.assert_close(chosen[0], recorded[0])
    for sampled, replay in zip(sampled_values[:-1], replay_credit.resource_values, strict=True):
        torch.testing.assert_close(sampled, replay)
    for expected, actual in zip(activity, replay_credit.iteration_active, strict=True):
        torch.testing.assert_close(actual, expected)
    ordinary_gradient = torch.autograd.grad(
        sampled_values[4].sum(), sampled_source, allow_unused=True, retain_graph=True,
    )[0]
    torch.testing.assert_close(
        replay_credit.resource_cotangents["source"],
        torch.zeros_like(sampled_source) if ordinary_gradient is None else ordinary_gradient,
    )
    ((sampled_values[4].detach().square().sum() + 1.0) * joint_log_probability).backward()
    assert sampled_scores.grad is not None and sampled_scores.grad.abs().sum() > 0
    if torch.cuda.is_available():
        cuda_plan = copy.deepcopy(plan).cuda()
        cuda_inputs = tuple(value.detach().to("cuda") for value in sampled_inputs)
        cuda_values, cuda_choices, _, cuda_activity = torch.compile(
            cuda_plan.forward_with_route_credit, backend="eager", fullgraph=True,
        )(*cuda_inputs)
        cuda_credit = cuda_plan.credit_gradient(
            *cuda_inputs, route_selections=cuda_choices,
            terminal_cotangents={"output": torch.ones_like(cuda_values[4])},
        )
        torch.testing.assert_close(cuda_credit.resource_values[4], cuda_values[4])
        for expected, actual in zip(cuda_activity, cuda_credit.iteration_active, strict=True):
            torch.testing.assert_close(actual, expected)

    open_graph = mechanisms.ProgramGraph(
        tuple(graph.resources.values()), tuple(graph.connections.values()),
        nodes=tuple(graph.nodes.values()),
        programs={"iterate": graph.program("iterate")},
        loops=(mechanisms.ProgramLoop("open", "iterate", "continue", max_iterations=None),),
    )
    open_plan = mechanisms.ResourceGraphCompiler.compile_loop(open_graph, "open")
    assert isinstance(open_plan, mechanisms.StaticDataflowLoopExecutionPlan)
    compiled_open = torch.compile(open_plan.forward_until_done, backend="eager", fullgraph=True)
    compiled_status = torch.compile(open_plan.forward_until_done_with_status, backend="eager", fullgraph=True)
    for scores, limit, expected_steps, remaining_active in (
        (torch.tensor([[3.0, -1.0]]), 4, 1, False),
        (torch.tensor([[3.0, -1.0]]), 1, 1, False),
        (torch.tensor([[-1.0, 3.0]]), 4, 4, True),
        (torch.tensor([[-1.0, 3.0]]), 7, 7, True),
    ):
        inputs = (values[0], scores, *values[2:], arrivals)
        budget = torch.tensor(limit, dtype=torch.int64)
        expected = open_plan.forward_until_done(*inputs, host_step_limit=budget)
        actual = compiled_open(*inputs, host_step_limit=budget)
        for reference, compiled_value in zip(expected, actual, strict=True):
            torch.testing.assert_close(compiled_value, reference)
        torch.testing.assert_close(actual[-2], torch.tensor(expected_steps))
        status = compiled_status(*inputs, host_step_limit=budget)
        for reference, compiled_value in zip(actual, status[:-1], strict=True):
            torch.testing.assert_close(compiled_value, reference)
        torch.testing.assert_close(status[-1], torch.tensor([remaining_active]))
    with pytest.raises(mechanisms.ResourceGraphCompileError, match="fixed training horizon"):
        open_plan.forward_bounded(*inputs)
    with pytest.raises(mechanisms.ResourceGraphCompileError, match="open-horizon reverse tape"):
        open_plan.credit_gradient(
            *inputs, terminal_cotangents={"output": torch.ones_like(values[4])},
        )

    batch_resources = (
        mechanisms.TensorResource(_spec("source"), _view([[2.0, 4.0], [2.0, 4.0]])),
        mechanisms.TensorResource(
            _vector_spec("scores"), _vector_view(torch.tensor([[2.0, -2.0], [-5.0, 5.0]]))
        ),
        mechanisms.TensorResource(_spec("left"), _view([[0.0, 0.0], [0.0, 0.0]])),
        mechanisms.TensorResource(_spec("right"), _view([[0.0, 0.0], [0.0, 0.0]])),
        mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0], [0.0, 0.0]])),
        mechanisms.TensorResource(_continue_spec("continue"), _continue_view(torch.ones(2))),
    )
    batch_graph = mechanisms.ProgramGraph(
        batch_resources, tuple(graph.connections.values()), nodes=tuple(graph.nodes.values()),
        programs={"iterate": graph.program("iterate")},
    )
    batch_body = mechanisms.ResourceGraphCompiler.compile_program(batch_graph, "iterate")
    assert isinstance(batch_body, mechanisms.StaticDataflowProgramExecutionPlan)
    batch_inputs = (
        *(resource.resolve().view.value for resource in batch_resources),
        torch.zeros((2, batch_body.arrival_width), dtype=torch.bool),
    )
    active = torch.tensor([True, False])
    compiled_batch = torch.compile(batch_body.forward_with_active_rows, backend="eager", fullgraph=True)
    active_result = compiled_batch(*batch_inputs, active_rows=active)
    unmasked_result = batch_body(*batch_inputs)
    torch.testing.assert_close(active_result[4][0], _view([[10.0, 20.0]]).value[0])
    torch.testing.assert_close(unmasked_result[4], torch.zeros_like(unmasked_result[4]))
    if torch.cuda.is_available():
        cuda_plan = copy.deepcopy(open_plan).cuda()
        cuda_forward = torch.compile(
            cuda_plan.forward_until_done, backend="inductor", fullgraph=True,
        )
        cuda_inputs = tuple(value.cuda() for value in inputs)
        cuda_budget = torch.tensor(7, dtype=torch.int64, device="cuda")
        cuda_result = cuda_forward(*cuda_inputs, host_step_limit=cuda_budget)
        for reference, actual in zip(
            open_plan.forward_until_done(*inputs, host_step_limit=torch.tensor(7)),
            cuda_result, strict=True,
        ):
            torch.testing.assert_close(actual.cpu(), reference)


def test_bounded_routed_join_loop_uses_live_context_and_credit_mask() -> None:
    class _ContextGate(torch.nn.Module):
        def forward(self, context: torch.Tensor) -> torch.Tensor:
            return context[:, :1].unsqueeze(-1)

    source = _view([[2.0]], requires_grad=True)
    resources = (
        mechanisms.TensorResource(_spec("source"), source),
        mechanisms.TensorResource(_vector_spec("scores"), _vector_view(torch.tensor([[3.0, -3.0]]))),
        mechanisms.TensorResource(_spec("left"), _view([[0.0]])),
        mechanisms.TensorResource(_spec("right"), _view([[2.0]])),
        mechanisms.TensorResource(_spec("output"), _view([[0.0]])),
        mechanisms.TensorResource(_continue_spec("continue"), _continue_view(torch.ones(1))),
    )
    gated = mechanisms.Connection(
        "gated", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("left"),
        activation=_ContextGate(),
        credit_boundary=arti.CreditBoundary(mode=arti.CreditBoundaryMode.BERNOULLI),
    )
    direct = mechanisms.Connection(
        "direct", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
    )
    joined = arti.as_fabric_node(
        "sum", _DecoratedPairSumStop(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={
            "value": mechanisms.ResourcePort("output"),
            "continue": mechanisms.ResourcePort("continue"),
        },
    )
    graph = mechanisms.ProgramGraph(
        resources, (gated, direct), nodes=(joined,),
        programs={
            "gated_path": ("gated", mechanisms.ProgramJoin("ready", "sum")),
            "direct_path": ("direct",),
            "iterate": (mechanisms.ProgramRoute("pick", "scores", ("gated_path", "direct_path")),),
        },
        loops=(mechanisms.ProgramLoop("bounded", "iterate", "continue", max_iterations=2),),
    )
    context = torch.tensor([[1.0]])
    mask = torch.tensor([[[False]]])
    plan = mechanisms.ResourceGraphCompiler.compile_loop(
        graph, "bounded", example_contexts={"gated": context},
        example_credit_masks={"gated": mask},
    )
    assert isinstance(plan, mechanisms.StaticDataflowLoopExecutionPlan)
    values = tuple(resource.resolve().view.value for resource in resources)
    arrivals = torch.tensor([[False, True]])
    inputs = (*values, context, mask, arrivals)
    result = plan.forward_bounded(*inputs)
    torch.testing.assert_close(result[4], _view([[4.0]]).value)
    torch.testing.assert_close(plan.forward_until_done(*inputs)[-2], torch.tensor(1))
    changed_context = torch.tensor([[2.0]])
    changed = plan.forward_bounded(*values, changed_context, mask, arrivals)
    torch.testing.assert_close(changed[4], _view([[6.0]]).value)
    credit = plan.credit_gradient(
        *inputs, terminal_cotangents={"output": torch.ones_like(result[4])},
    )
    torch.testing.assert_close(credit.resource_cotangents["source"], torch.zeros_like(source.value))
    torch.testing.assert_close(torch.autograd.grad(result[4].sum(), source.value)[0], torch.zeros_like(source.value))
    torch._dynamo.reset()
    compiled = torch.compile(plan.forward_bounded, backend="eager", fullgraph=True)
    for expected, actual in zip(changed, compiled(*values, changed_context, mask, arrivals), strict=True):
        torch.testing.assert_close(actual, expected)

    open_graph = mechanisms.ProgramGraph(
        tuple(graph.resources.values()), tuple(graph.connections.values()),
        nodes=tuple(graph.nodes.values()),
        programs={name: graph.program(name) for name in ("gated_path", "direct_path", "iterate")},
        loops=(mechanisms.ProgramLoop("open", "iterate", "continue", max_iterations=None),),
    )
    open_plan = mechanisms.ResourceGraphCompiler.compile_loop(
        open_graph, "open", example_contexts={"gated": context},
        example_credit_masks={"gated": mask},
    )
    open_execution = open_graph.execute_loop_functional(
        "open", input_views={"right": _view([[2.0]])},
        contexts={"gated": context}, credit_masks={"gated": mask},
        host_step_limit=2,
    )
    open_result = open_plan.forward_until_done(
        *inputs, host_step_limit=torch.tensor(2, dtype=torch.int64),
    )
    compiled_open = torch.compile(open_plan.forward_until_done, backend="eager", fullgraph=True)
    compiled_result = compiled_open(
        *inputs, host_step_limit=torch.tensor(2, dtype=torch.int64),
    )
    assert open_execution.actual_iterations == 1
    assert open_execution.termination_reason == "endogenous"
    torch.testing.assert_close(open_result[4], result[4])
    torch.testing.assert_close(open_result[-2], torch.tensor(1))
    for expected, actual in zip(open_result, compiled_result, strict=True):
        torch.testing.assert_close(actual, expected)
    open_credit = open_plan.credit_gradient(
        *inputs, execution=open_execution,
        terminal_cotangents={"output": torch.ones_like(open_result[4])},
    )
    torch.testing.assert_close(open_credit.resource_cotangents["source"], credit.resource_cotangents["source"])
    torch.testing.assert_close(open_credit.context_cotangents["gated"], credit.context_cotangents["gated"])
    live_context = context.clone().requires_grad_()
    live_mask = torch.ones_like(mask)
    live_inputs = (*values, live_context, live_mask, arrivals)
    live_execution = open_graph.execute_loop_functional(
        "open", input_views={"right": _view([[2.0]])},
        contexts={"gated": live_context}, credit_masks={"gated": live_mask},
        host_step_limit=2,
    )
    live_values = open_plan.forward_until_done(
        *live_inputs, host_step_limit=torch.tensor(2, dtype=torch.int64),
    )
    live_credit = open_plan.credit_gradient(
        *live_inputs, execution=live_execution,
        terminal_cotangents={"output": torch.ones_like(live_values[4])},
    )
    torch.testing.assert_close(live_credit.resource_cotangents["source"], torch.ones_like(source.value))
    torch.testing.assert_close(live_credit.context_cotangents["gated"], source.value[:, 0, :])


def test_bounded_routed_connection_loop_compiles_without_join() -> None:
    source = _view([[2.0]], requires_grad=True)
    resources = (
        mechanisms.TensorResource(_spec("source"), source),
        mechanisms.TensorResource(_vector_spec("scores"), _vector_view(torch.tensor([[3.0, -3.0]]))),
        mechanisms.TensorResource(_spec("output"), _view([[0.0]])),
        mechanisms.TensorResource(_continue_spec("continue"), _continue_view(torch.ones(1))),
    )
    first = mechanisms.Connection(
        "first", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
        transfer=mechanisms.LearnableAffineTransfer(gain=2.0),
    )
    second = mechanisms.Connection(
        "second", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
        transfer=mechanisms.LearnableAffineTransfer(gain=3.0),
    )
    graph = mechanisms.ProgramGraph(
        resources, (first, second),
        programs={"iterate": (mechanisms.ProgramRoute(
            "pick", "scores", ("first", "second"),
        ),)},
        loops=(mechanisms.ProgramLoop("bounded", "iterate", "continue", max_iterations=2),),
    )
    plan = mechanisms.ResourceGraphCompiler.compile_loop(graph, "bounded")
    assert isinstance(plan, mechanisms.StaticDataflowLoopExecutionPlan)
    assert plan.body.arrival_width == 0
    values = tuple(resource.resolve().view.value for resource in resources)
    arrivals = torch.zeros((1, 0), dtype=torch.bool)
    result = plan.forward_bounded(*values, arrivals)
    torch.testing.assert_close(result[2], _view([[4.0]]).value)
    credit = plan.credit_gradient(
        *values, arrivals, terminal_cotangents={"output": torch.ones_like(result[2])},
    )
    torch.testing.assert_close(credit.resource_cotangents["source"], torch.full_like(source.value, 2.0))
    torch._dynamo.reset()
    compiled = torch.compile(plan.forward_bounded, backend="eager", fullgraph=True)
    for expected, actual in zip(result, compiled(*values, arrivals), strict=True):
        torch.testing.assert_close(actual, expected)

    open_graph = mechanisms.ProgramGraph(
        tuple(graph.resources.values()), tuple(graph.connections.values()),
        programs={"iterate": graph.program("iterate")},
        loops=(mechanisms.ProgramLoop("open", "iterate", "continue", max_iterations=None),),
    )
    open_plan = mechanisms.ResourceGraphCompiler.compile_loop(open_graph, "open")
    assert isinstance(open_plan, mechanisms.StaticDataflowLoopExecutionPlan)
    budget = torch.tensor(2, dtype=torch.int64)
    open_result = open_plan.forward_until_done(*values, arrivals, host_step_limit=budget)
    torch.testing.assert_close(open_result[2], result[2])
    torch.testing.assert_close(open_result[-2], budget)
    execution = open_graph.execute_loop_functional("open", host_step_limit=2)
    assert execution.actual_iterations == 2
    assert tuple(route.candidate_id for route in execution.routes) == ("first", "first")
    open_credit = open_plan.credit_gradient(
        *values, arrivals, execution=execution,
        terminal_cotangents={"output": torch.ones_like(open_result[2])},
    )
    torch.testing.assert_close(open_credit.resource_cotangents["source"], credit.resource_cotangents["source"])
    torch._dynamo.reset()
    compiled_open = torch.compile(open_plan.forward_until_done, backend="eager", fullgraph=True)
    for expected, actual in zip(open_result, compiled_open(*values, arrivals, host_step_limit=budget), strict=True):
        torch.testing.assert_close(actual, expected)


def test_bounded_sample_route_loop_compiles_with_aliased_resource_inputs() -> None:
    source = _view([[2.0], [3.0]], requires_grad=True)
    resources = (
        mechanisms.TensorResource(_spec("source"), source),
        mechanisms.TensorResource(
            _vector_spec("scores"),
            _vector_view(torch.tensor([[4.0, -4.0], [-4.0, 4.0]])),
        ),
        mechanisms.TensorResource(_spec("output"), _view([[0.0], [0.0]])),
        mechanisms.TensorResource(
            _continue_spec("continue"), _continue_view(torch.ones(2)),
        ),
        mechanisms.TensorResource(_spec("alias"), source),
    )
    graph = mechanisms.ProgramGraph(
        resources,
        (
            mechanisms.Connection(
                "double_copy", mechanisms.ResourcePort("source"),
                mechanisms.ResourcePort("output"),
                transfer=mechanisms.LearnableAffineTransfer(gain=2.0),
            ),
            mechanisms.Connection(
                "triple_copy", mechanisms.ResourcePort("source"),
                mechanisms.ResourcePort("output"),
                transfer=mechanisms.LearnableAffineTransfer(gain=3.0),
            ),
        ),
        programs={"iterate": (mechanisms.ProgramRoute(
            "choose", "scores", ("double_copy", "triple_copy"),
            selection_scope="sample", execution_mode="sparse",
        ),)},
        loops=(mechanisms.ProgramLoop("bounded", "iterate", "continue", max_iterations=2),),
    )
    plan = mechanisms.ResourceGraphCompiler.compile_loop(graph, "bounded")
    values = tuple(resource.resolve().view.value for resource in resources)
    arrivals = torch.zeros((2, plan.body.arrival_width), dtype=torch.bool)
    choices = ((torch.tensor([0, 1]),), (torch.tensor([0, 1]),))
    inputs = (*values, arrivals)
    expected = plan.forward_with_route_selections(*inputs, route_selections=choices)
    torch.testing.assert_close(expected[2], torch.tensor([[[4.0]], [[9.0]]]))
    torch.testing.assert_close(expected[4], source.value)
    compiled = torch.compile(
        plan.forward_with_route_selections, backend="eager", fullgraph=True,
    )
    actual = compiled(*inputs, route_selections=choices)
    for reference, result in zip(expected, actual, strict=True):
        torch.testing.assert_close(result, reference)
    gradient = torch.autograd.grad(actual[2].sum(), source.value)[0]
    torch.testing.assert_close(gradient, torch.tensor([[[2.0]], [[3.0]]]))
    if torch.cuda.is_available():
        cuda_plan = plan.to("cuda")
        cuda_source = source.value.detach().to("cuda").requires_grad_(True)
        cuda_scores = values[1].to("cuda").requires_grad_(True)
        cuda_inputs = (
            cuda_source, cuda_scores, values[2].to("cuda"),
            values[3].to("cuda"), cuda_source, arrivals.to("cuda"),
        )
        cuda_choices = tuple(
            (choice.to("cuda"),) for (choice,) in choices
        )
        cuda_reference = cuda_plan.forward_with_route_selections(
            *cuda_inputs, route_selections=cuda_choices,
        )
        cuda_reference_gradient = torch.autograd.grad(
            cuda_reference[2].sum(), cuda_source,
        )[0]
        cuda_compiled = torch.compile(
            cuda_plan.forward_with_route_selections, backend="inductor", fullgraph=True,
        )
        cuda_output = cuda_compiled(*cuda_inputs, route_selections=cuda_choices)
        for reference, result in zip(cuda_reference, cuda_output, strict=True):
            torch.testing.assert_close(result, reference)
        cuda_gradient = torch.autograd.grad(cuda_output[2].sum(), cuda_source)[0]
        torch.testing.assert_close(cuda_gradient, cuda_reference_gradient)
        sampled_forward = torch.compile(
            cuda_plan.forward_with_route_credit, backend="inductor", fullgraph=True,
        )
        sampled_values, sampled_choices, log_probability, _activity = sampled_forward(
            *cuda_inputs,
        )
        replayed = cuda_plan.forward_with_route_selections(
            *cuda_inputs, route_selections=sampled_choices,
        )
        for sampled, replay in zip(sampled_values, replayed, strict=True):
            torch.testing.assert_close(sampled, replay)
        expected_log_probability = sum(
            torch.log_softmax(cuda_inputs[1], dim=-1).gather(
                1, iteration_choices[0].unsqueeze(-1),
            ).squeeze(-1)
            for iteration_choices in sampled_choices
        )
        torch.testing.assert_close(log_probability, expected_log_probability)
        expected_score_gradient = sum(
            torch.nn.functional.one_hot(
                iteration_choices[0], num_classes=2,
            ).to(cuda_scores.dtype) - torch.softmax(cuda_scores, dim=-1)
            for iteration_choices in sampled_choices
        )
        score_gradient = torch.autograd.grad(log_probability.sum(), cuda_scores)[0]
        torch.testing.assert_close(score_gradient, expected_score_gradient)


def test_static_mixed_dataflow_loop_keeps_per_row_depth_and_recurrent_credit() -> None:
    @arti.fabric_layer(inputs=("left", "right"), outputs={"value": "left", "continue": "left"})
    class _RowStopJoin(torch.nn.Module):
        def forward(self, left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor | mechanisms.TensorView]:
            continuation = (right[:, 0, 0] < 2.0).to(dtype=right.dtype)
            return {
                "value": left + right,
                "continue": mechanisms.TensorView.from_tensor(
                    continuation, axis_names=("batch",), axis_roles=("batch",),
                ),
            }

    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[1.0, 1.0], [3.0, 3.0]])),
        mechanisms.TensorResource(_spec("left"), _view([[0.0, 0.0], [0.0, 0.0]])),
        mechanisms.TensorResource(_spec("right"), _view([[0.0, 0.0], [0.0, 0.0]])),
        mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0], [0.0, 0.0]])),
        mechanisms.TensorResource(_continue_spec("continue"), _continue_view(torch.ones(2))),
    )
    connection = mechanisms.Connection(
        "copy", mechanisms.ResourcePort("output"), mechanisms.ResourcePort("left")
    )
    right_head = arti.as_fabric_node(
        "right_head", _DecoratedScale(1.0),
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("right")},
    )
    pair_sum = arti.as_fabric_node(
        "sum", _RowStopJoin(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={
            "value": mechanisms.ResourcePort("output"),
            "continue": mechanisms.ResourcePort("continue"),
        },
    )
    graph = mechanisms.ProgramGraph(
        resources, (connection,), nodes=(right_head, pair_sum),
        programs={"iterate": (
            mechanisms.ProgramStage(("copy", "right_head")),
            mechanisms.ProgramJoin("ready", "sum"),
        )},
        loops=(mechanisms.ProgramLoop("bounded", "iterate", "continue", max_iterations=3),),
    )
    plan = mechanisms.ResourceGraphCompiler.compile_loop(graph, "bounded")
    assert isinstance(plan, mechanisms.StaticDataflowLoopExecutionPlan)
    inputs = (*tuple(resource.resolve().view.value for resource in resources),
              torch.zeros((2, plan.body.arrival_width), dtype=torch.bool))
    values = plan(*inputs)
    dynamic = plan.forward_until_done(*inputs)
    for expected, actual in zip(values, dynamic[:-2], strict=True):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(dynamic[-2], torch.tensor(3))
    torch.testing.assert_close(dynamic[-1], torch.tensor([3, 1]))
    compiled = torch.compile(plan, backend="eager", fullgraph=True)(*inputs)
    for value, compiled_value in zip(values, compiled, strict=True):
        torch.testing.assert_close(value, compiled_value)
    torch.testing.assert_close(values[3], _view([[3.0, 3.0], [3.0, 3.0]]).value)
    result = plan.credit_gradient(
        *inputs, terminal_cotangents={"output": torch.ones_like(values[3])}
    )
    assert tuple(mask.tolist() for mask in result.iteration_active) == (
        [True, True], [True, False], [True, False],
    )
    torch.testing.assert_close(
        result.resource_cotangents["source"], _view([[3.0, 3.0], [1.0, 1.0]]).value
    )


def test_static_connection_credit_lowering_applies_boundary_at_the_port() -> None:
    source = mechanisms.TensorResource(_spec("source"), _view([[1.0, 2.0]]))
    hidden = mechanisms.TensorResource(_spec("hidden"), _view([[0.0, 0.0]]))
    output = mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]]))
    first_transfer = mechanisms.LearnableAffineTransfer()
    second_transfer = mechanisms.LearnableAffineTransfer()
    with torch.no_grad():
        first_transfer.gain.fill_(2.0)
        second_transfer.gain.fill_(3.0)
    first = mechanisms.Connection(
        "first",
        mechanisms.ResourcePort("source"),
        mechanisms.ResourcePort("hidden"),
        transfer=first_transfer,
        credit_boundary=arti.CreditBoundary(mode=arti.CreditBoundaryMode.MEAN),
    )
    second = mechanisms.Connection(
        "second",
        mechanisms.ResourcePort("hidden"),
        mechanisms.ResourcePort("output"),
        transfer=second_transfer,
    )
    graph = mechanisms.ProgramGraph(
        (source, hidden, output),
        (first, second),
        programs={"chain": ("first", "second")},
    )
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "chain")
    assert isinstance(plan, mechanisms.ResourceGraphExecutionPlan)
    compiled_first = plan.connections[0].transfer
    compiled_second = plan.connections[1].transfer
    assert isinstance(compiled_first, mechanisms.LearnableAffineTransfer)
    assert isinstance(compiled_second, mechanisms.LearnableAffineTransfer)
    values = (
        source.resolve().view.value,
        hidden.resolve().view.value,
        output.resolve().view.value,
    )
    ordinary = plan(*values)
    ordinary_gradients = torch.autograd.grad(
        ordinary[-1].sum(),
        (compiled_first.gain, compiled_second.gain),
    )
    result = plan.credit_gradient(
        *values,
        terminal_cotangents={"output": torch.ones_like(ordinary[-1])},
    )

    torch.testing.assert_close(result.resource_values[-1], ordinary[-1].detach())
    torch.testing.assert_close(result.resource_cotangents["source"], torch.full_like(values[0], 3.0))
    torch.testing.assert_close(result.parameter_cotangents["first.transfer.gain"], ordinary_gradients[0])
    torch.testing.assert_close(result.parameter_cotangents["second.transfer.gain"], ordinary_gradients[1])
    trial_gain = compiled_first.gain - 0.1 * result.parameter_cotangents["first.transfer.gain"]
    meta_gradient = torch.autograd.grad((trial_gain - 1.0).square(), plan.connections[0].credit_boundary.alpha)[0]
    assert meta_gradient.abs() > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for compiled credit coverage")
def test_static_program_credit_boundary_compiles_through_inductor_backward() -> None:
    torch._dynamo.reset()
    try:
        source = mechanisms.TensorResource(_spec("source"), _view([[1.0, 2.0]]))
        output = mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]]))
        node = arti.as_fabric_node(
            "boundary",
            _BoundaryScale(2.0),
            input_ports={"source": mechanisms.ResourcePort("source")},
            output_ports={"value": mechanisms.ResourcePort("output")},
        )
        graph = mechanisms.ProgramGraph(
            (source, output), (), nodes=(node,), programs={"run": ("boundary",)}
        )
        plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run").to("cuda")
        assert isinstance(plan, mechanisms.StaticProgramGraphExecutionPlan)
        compiled = torch.compile(plan, backend="inductor", fullgraph=True)
        source_value = source.resolve().view.value.detach().clone().to("cuda").requires_grad_(True)
        output_value = output.resolve().view.value.to("cuda")
        result = compiled(source_value, output_value)[-1]
        result.square().mean().backward()

        expected = source_value.detach() * 2.0
        torch.testing.assert_close(node.module.scale.grad, (2.0 * expected * source_value.detach()).mean())
        torch.testing.assert_close(source_value.grad, expected)
    finally:
        torch._dynamo.reset()


def test_formula_multi_port_program_has_explicit_fixed_horizon_loop_semantics() -> None:
    value_type = mechanisms.TensorType.axes(
        ("batch", "token", "feature"), sizes=("B", "S", 1)
    )
    state_binding = mechanisms.InputBinding("state", value_type)
    program = mechanisms.FormulaProgram.build(
        outputs=(mechanisms.add(state_binding, state_binding),)
    )
    state = mechanisms.TensorResource(_spec("state"), _view([[1.0, 3.0]]))
    node = mechanisms.FormulaProgramNode(
        "twice",
        mechanisms.FormulaFabricV2(program),
        input_ports={"state": mechanisms.ResourcePort("state")},
        output_ports={"next": mechanisms.ResourcePort("state")},
        output_slots={"next": "%0"},
    )
    graph = mechanisms.ProgramGraph(
        (state,), (), nodes=(node,), programs={"iterate": ("twice",)}
    )

    runtime = graph.execute_program_functional("iterate", iterations=3)
    runtime_state = runtime.state.resources[0].active_view.value
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "iterate", iterations=3)
    static = plan(state.resolve().view.value)

    torch.testing.assert_close(runtime_state, _view([[8.0, 24.0]]).value)
    torch.testing.assert_close(static[0], runtime_state)


def test_repeated_compiled_join_consumes_one_arrival_only_once() -> None:
    resources = tuple(
        mechanisms.TensorResource(_spec(name), _view([[value]], requires_grad=name == "output"))
        for name, value in (
            ("seed", 2.0), ("left", 0.0), ("right", 0.0), ("output", 1.0),
        )
    )
    right_edge = mechanisms.Connection(
        "right_edge", mechanisms.ResourcePort("seed"), mechanisms.ResourcePort("right"),
        transfer=mechanisms.LearnableAffineTransfer(gain=1.0, bias=0.0),
    )
    left_edge = mechanisms.Connection(
        "left_edge", mechanisms.ResourcePort("output"), mechanisms.ResourcePort("left"),
        transfer=mechanisms.LearnableAffineTransfer(gain=1.0, bias=0.0),
    )
    joined = arti.as_fabric_node(
        "sum", _DecoratedPairSum(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        resources, (right_edge, left_edge), nodes=(joined,),
        programs={
            "prime": ("right_edge",),
            "repeat": ("left_edge", mechanisms.ProgramJoin("ready", "sum")),
        },
    )
    primed = graph.execute_program_functional("prime")
    reference = graph.execute_program_functional("repeat", state=primed.state, iterations=2)
    assert [join.fired for join in reference.joins] == [True, False]
    reference_output = next(
        item.active_view.value for item in reference.state.resources if item.spec.resource_id == "output"
    )
    torch.testing.assert_close(reference_output, _view([[3.0]]).value)

    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "repeat", iterations=2)
    assert plan.arrival_width == 2
    inputs = graph.static_program_inputs(plan, state=primed.state)
    arrivals = inputs[-1]
    torch.testing.assert_close(arrivals, torch.tensor([[False, True]]))
    result = plan(*inputs)
    torch.testing.assert_close(result[plan.resource_ids.index("output")], reference_output)
    torch.testing.assert_close(result[-1], torch.tensor([[True, False]]))
    reference_input_grad, reference_gain_grad = torch.autograd.grad(
        result[plan.resource_ids.index("output")].sum(),
        (inputs[plan.resource_ids.index("output")], left_edge.transfer.gain),
        retain_graph=True,
    )
    credit = plan.credit_gradient(
        *inputs,
        terminal_cotangents={"output": torch.ones_like(reference_output)},
    )
    torch.testing.assert_close(credit.resource_cotangents["output"], reference_input_grad)
    torch.testing.assert_close(
        credit.parameter_cotangents["left_edge.transfer.gain"], reference_gain_grad,
    )
    torch._dynamo.reset()
    try:
        compiled = torch.compile(plan, backend="eager", fullgraph=True)(
            *inputs,
        )
        torch.testing.assert_close(compiled[plan.resource_ids.index("output")], reference_output)
        torch.testing.assert_close(compiled[-1], result[-1])
    finally:
        torch._dynamo.reset()


def test_program_loop_keeps_stopped_batch_rows_at_their_prior_state() -> None:
    state = mechanisms.TensorResource(_spec("state"), _view([[0.0], [2.0]]))
    continuation = mechanisms.TensorResource(
        _continue_spec("continue"), _continue_view(torch.tensor([1.0, 1.0]))
    )
    loop = mechanisms.ProgramLoop("adaptive", "count", "continue", max_iterations=3)
    graph = mechanisms.ProgramGraph(
        (state, continuation),
        (),
        nodes=(_CountingLoopNode(),),
        programs={"count": ("count",)},
        loops=(loop,),
    )

    result = graph.execute_loop_functional("adaptive")
    outputs = {item.spec.resource_id: item.active_view for item in result.state.resources}

    torch.testing.assert_close(outputs["state"].value, _view([[2.0], [3.0]]).value)
    torch.testing.assert_close(
        result.active_masks[0], torch.tensor([True, True], dtype=torch.bool)
    )
    torch.testing.assert_close(
        result.active_masks[1], torch.tensor([True, False], dtype=torch.bool)
    )
    torch.testing.assert_close(
        result.active_masks[2], torch.tensor([False, False], dtype=torch.bool)
    )
    assert tuple((dispatch.iteration, dispatch.frontier) for dispatch in result.dispatches) == (
        (0, 0), (1, 0), (2, 0)
    )
    assert tuple(dispatch.members[0].node_id for dispatch in result.dispatches) == (
        "count", "count", "count"
    )


def test_open_horizon_loop_records_dynamic_routes_and_termination(tmp_path) -> None:
    class ScoreCurrentState(mechanisms.MultiPortProgramNode):
        def __init__(self) -> None:
            super().__init__(
                "score",
                input_ports={"state": mechanisms.ResourcePort("state")},
                output_ports={"scores": mechanisms.ResourcePort("scores")},
            )
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def invoke_ports(
            self, inputs: dict[str, mechanisms.TensorView]
        ) -> mechanisms.MultiPortProgramNodeInvocation:
            state = inputs["state"].value[:, 0, 0:1]
            scores = torch.cat(((0.5 - state) * self.scale, (state - 0.5) * self.scale), dim=1)
            return mechanisms.MultiPortProgramNodeInvocation({"scores": _vector_view(scores)})

    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[2.0]])),
        mechanisms.TensorResource(_spec("state"), _view([[0.0]])),
        mechanisms.TensorResource(_vector_spec("scores"), _vector_view(torch.zeros(1, 2))),
        mechanisms.TensorResource(_spec("output"), _view([[0.0]])),
        mechanisms.TensorResource(_continue_spec("continue"), _continue_view(torch.ones(1))),
    )
    positive = mechanisms.Connection(
        "positive", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
        transfer=mechanisms.LearnableAffineTransfer(gain=2.0, bias=0.0),
    )
    negative = mechanisms.Connection(
        "negative", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
        transfer=mechanisms.LearnableAffineTransfer(gain=-3.0, bias=0.0),
    )
    score = ScoreCurrentState()
    graph = mechanisms.ProgramGraph(
        resources, (positive, negative), nodes=(score, _CountingLoopNode()),
        programs={"iterate": (
            "score", mechanisms.ProgramRoute("choose", "scores", ("positive", "negative")),
            "count",
        )},
        loops=(mechanisms.ProgramLoop("dynamic", "iterate", "continue", max_iterations=None),),
    )

    with pytest.raises(mechanisms.ResourceGraphError, match="host_step_limit"):
        graph.execute_loop_functional("dynamic")
    with pytest.raises(mechanisms.ResourceGraphCompileError, match="open-horizon"):
        mechanisms.ResourceGraphCompiler.compile_loop(graph, "dynamic")

    completed = graph.execute_loop_functional("dynamic", host_step_limit=10)
    assert completed.termination_reason == "endogenous"
    assert completed.actual_iterations == 2
    assert tuple(route.candidate_id for route in completed.routes) == ("positive", "negative")
    assert tuple(route.iteration for route in completed.routes) == (0, 1)
    assert len(completed.dispatches) == 6
    assert tuple(dispatch.route for dispatch in completed.dispatches if dispatch.route is not None) == (
        *completed.routes,
    )
    final = {item.spec.resource_id: item.active_view.value for item in completed.state.resources}
    torch.testing.assert_close(final["state"], _view([[2.0]]).value)
    torch.testing.assert_close(final["output"], _view([[-6.0]]).value)

    truncated = graph.execute_loop_functional("dynamic", host_step_limit=1)
    assert truncated.termination_reason == "host_limit"
    assert truncated.actual_iterations == 1
    assert tuple(route.candidate_id for route in truncated.routes) == ("positive",)

    torch.manual_seed(7)
    sampled = graph.execute_loop_functional("dynamic", host_step_limit=10, sample_routes=True)
    sampled_output = next(
        item.active_view.value for item in sampled.state.resources
        if item.spec.resource_id == "output"
    )
    final_loss = sampled_output.square().mean()
    (final_loss + sampled.structure_objective(final_loss)).backward()
    assert score.scale.grad is not None and score.scale.grad.abs().item() > 0
    assert positive.transfer.gain.grad is not None or negative.transfer.gain.grad is not None
    assert all(route.sampled for route in sampled.routes)

    saved = mechanisms.save_program_graph(graph, tmp_path / "open-loop")
    restored = mechanisms.load_program_graph(
        saved.tensors_path, nodes=(ScoreCurrentState(), _CountingLoopNode())
    )
    assert restored.loop("dynamic").max_iterations is None

    batch_resources = (
        mechanisms.TensorResource(_spec("source"), _view([[2.0], [2.0]])),
        mechanisms.TensorResource(_spec("state"), _view([[-1.0], [10.0]])),
        mechanisms.TensorResource(_vector_spec("scores"), _vector_view(torch.zeros(2, 2))),
        mechanisms.TensorResource(_spec("output"), _view([[0.0], [0.0]])),
        mechanisms.TensorResource(_continue_spec("continue"), _continue_view(torch.ones(2))),
    )
    batch_graph = mechanisms.ProgramGraph(
        batch_resources, (positive, negative), nodes=(score, _CountingLoopNode()),
        programs={"iterate": graph.program("iterate")},
        loops=(graph.loop("dynamic"),),
    )
    batch_result = batch_graph.execute_loop_functional("dynamic", host_step_limit=10)
    assert batch_result.termination_reason == "endogenous"
    assert batch_result.actual_iterations == 3
    assert tuple(route.candidate_id for route in batch_result.routes) == (
        "negative", "positive", "negative"
    )
    torch.testing.assert_close(batch_result.active_masks[1], torch.tensor([True, False]))
    batch_final = {
        item.spec.resource_id: item.active_view.value for item in batch_result.state.resources
    }
    torch.testing.assert_close(batch_final["state"], _view([[2.0], [11.0]]).value)
    torch.testing.assert_close(batch_final["output"], _view([[-6.0], [-6.0]]).value)


@pytest.mark.parametrize("stochastic_boundary", (False, True))
def test_compiled_open_loop_requeries_routes_after_each_state_change(stochastic_boundary: bool) -> None:
    torch._dynamo.reset()

    @arti.fabric_layer(inputs=("state",), outputs={"scores": "state"})
    class StateScores(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, state: torch.Tensor) -> dict[str, mechanisms.TensorView]:
            value = state[:, 0, 0:1]
            scores = torch.cat((0.5 - value, value - 0.5), dim=1) * self.scale
            return {"scores": _vector_view(scores)}

    @arti.fabric_layer(inputs=("output",), outputs={"value": "output"})
    class PublishResult(torch.nn.Module):
        def forward(self, output: torch.Tensor) -> torch.Tensor:
            return output

    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[2.0]])),
        mechanisms.TensorResource(_spec("state"), _view([[0.0]])),
        mechanisms.TensorResource(_vector_spec("scores"), _vector_view(torch.zeros(1, 2))),
        mechanisms.TensorResource(_spec("output"), _view([[0.0]])),
        mechanisms.TensorResource(_spec("result"), _view([[0.0]])),
        mechanisms.TensorResource(_continue_spec("continue"), _continue_view(torch.ones(1))),
    )
    connections = (
        mechanisms.Connection(
            "positive", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
            transfer=mechanisms.LearnableAffineTransfer(gain=2.0),
            credit_boundary=(
                arti.CreditBoundary(initial_logit=-1.0, mode=arti.CreditBoundaryMode.BERNOULLI)
                if stochastic_boundary else None
            ),
        ),
        mechanisms.Connection(
            "negative", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
            transfer=mechanisms.LearnableAffineTransfer(gain=-3.0),
        ),
    )
    nodes = (
        arti.as_fabric_node(
            "score", StateScores(),
            input_ports={"state": mechanisms.ResourcePort("state")},
            output_ports={"scores": mechanisms.ResourcePort("scores")},
        ),
        arti.as_fabric_node(
            "count", _DecoratedLoopStep(),
            input_ports={"value": mechanisms.ResourcePort("state")},
            output_ports={
                "value": mechanisms.ResourcePort("state"),
                "continue": mechanisms.ResourcePort("continue"),
            },
        ),
        arti.as_fabric_node(
            "publish", PublishResult(),
            input_ports={"output": mechanisms.ResourcePort("output")},
            output_ports={"value": mechanisms.ResourcePort("result")},
        ),
    )
    graph = mechanisms.ProgramGraph(
        resources, connections, nodes=nodes,
        programs={"iterate": (
            "score", mechanisms.ProgramRoute("choose", "scores", ("positive", "negative")),
            "count", mechanisms.ProgramJoin("ready", "publish"),
        )},
        loops=(mechanisms.ProgramLoop("open", "iterate", "continue", max_iterations=None),),
    )
    eager = graph.execute_loop_functional("open", host_step_limit=10)
    assert tuple(route.candidate_id for route in eager.routes) == (
        "positive", "negative", "negative"
    )
    plan = mechanisms.ResourceGraphCompiler.compile_loop(
        graph, "open",
        example_credit_masks={"positive": torch.tensor(True)} if stochastic_boundary else None,
    )
    assert isinstance(plan, mechanisms.StaticDataflowLoopExecutionPlan)
    inputs = (
        *(resource.resolve().view.value for resource in resources),
        *((torch.tensor(True),) if stochastic_boundary else ()),
        torch.zeros((1, plan.body.arrival_width), dtype=torch.bool),
    )
    compiled = torch.compile(plan.forward_until_done_with_status, backend="eager", fullgraph=True)
    actual = compiled(*inputs, host_step_limit=torch.tensor(10, dtype=torch.int64))
    recorded = torch.compile(
        plan.forward_until_done_with_route_receipt, backend="eager", fullgraph=True,
    )(*inputs, host_step_limit=torch.tensor(10, dtype=torch.int64), receipt_capacity=6)
    resource_count = len(plan.resource_ids)
    first_chunk = plan.forward_until_done_with_route_receipt(
        *inputs, host_step_limit=torch.tensor(10, dtype=torch.int64), receipt_capacity=2,
    )
    next_inputs = (
        *first_chunk[:resource_count], *inputs[resource_count:-1], first_chunk[resource_count],
    )
    second_chunk = torch.compile(
        plan.forward_until_done_with_route_receipt, backend="eager", fullgraph=True,
    )(
        *next_inputs, host_step_limit=torch.tensor(10, dtype=torch.int64), receipt_capacity=2,
        initial_active=first_chunk[-4], completed_steps=first_chunk[-6],
    )
    torch.testing.assert_close(first_chunk[-6] + second_chunk[-6], recorded[-6])
    torch.testing.assert_close(second_chunk[:resource_count + 1], recorded[:resource_count + 1])
    torch.testing.assert_close(second_chunk[-4], recorded[-4])
    torch.testing.assert_close(
        torch.cat((first_chunk[-1], second_chunk[-1][:1])), recorded[-1][:3],
    )
    first_receipt = (first_chunk[-6], first_chunk[-3], first_chunk[-2], first_chunk[-1])
    second_receipt = (second_chunk[-6], second_chunk[-3], second_chunk[-2], second_chunk[-1])
    first_replay = plan.replay_tensor_receipt(*inputs, tensor_receipt=first_receipt)
    next_replay_inputs = (
        *first_replay[:resource_count], *inputs[resource_count:-1], first_replay[resource_count],
    )
    second_replay = plan.replay_tensor_receipt(*next_replay_inputs, tensor_receipt=second_receipt)
    whole_replay = plan.replay_tensor_receipt(
        *inputs, tensor_receipt=(recorded[-6], recorded[-3], recorded[-2], recorded[-1]),
    )
    torch.testing.assert_close(second_replay[4], whole_replay[4])
    torch.testing.assert_close(first_replay[-1] + second_replay[-1], whole_replay[-1])
    chunk_gradients = torch.autograd.grad(
        second_replay[4].sum(),
        (connections[0].transfer.gain, connections[1].transfer.gain),
    )
    whole_gradients = torch.autograd.grad(
        whole_replay[4].sum(),
        (connections[0].transfer.gain, connections[1].transfer.gain),
    )
    for chunk_gradient, whole_gradient in zip(chunk_gradients, whole_gradients, strict=True):
        torch.testing.assert_close(chunk_gradient, whole_gradient)
    exhausted_chunk = plan.forward_until_done_with_route_receipt(
        *next_inputs, host_step_limit=torch.tensor(10, dtype=torch.int64), receipt_capacity=2,
        initial_active=torch.zeros_like(first_chunk[-4]), completed_steps=first_chunk[-6],
    )
    torch.testing.assert_close(exhausted_chunk[-6], torch.zeros_like(exhausted_chunk[-6]))
    deeper_graph = mechanisms.ProgramGraph(
        resources, connections, nodes=nodes,
        programs={"iterate": graph.program("iterate")},
        loops=(mechanisms.ProgramLoop(
            "deeper", "iterate", "continue", max_iterations=None, min_iterations=4,
        ),),
    )
    deeper_plan = mechanisms.ResourceGraphCompiler.compile_loop(
        deeper_graph, "deeper",
        example_credit_masks={"positive": torch.tensor(True)} if stochastic_boundary else None,
    )
    deeper_inputs = (*inputs[:-1], torch.zeros((1, deeper_plan.body.arrival_width), dtype=torch.bool))
    deeper_whole = deeper_plan.forward_until_done_with_route_receipt(
        *deeper_inputs, host_step_limit=torch.tensor(10, dtype=torch.int64), receipt_capacity=6,
    )
    deeper_first = deeper_plan.forward_until_done_with_route_receipt(
        *deeper_inputs, host_step_limit=torch.tensor(10, dtype=torch.int64), receipt_capacity=2,
    )
    deeper_second_inputs = (
        *deeper_first[:resource_count], *deeper_inputs[resource_count:-1],
        deeper_first[resource_count],
    )
    deeper_second = deeper_plan.forward_until_done_with_route_receipt(
        *deeper_second_inputs, host_step_limit=torch.tensor(10, dtype=torch.int64),
        receipt_capacity=2, initial_active=deeper_first[-4], completed_steps=deeper_first[-6],
    )
    torch.testing.assert_close(deeper_whole[-6], torch.tensor(4))
    torch.testing.assert_close(deeper_first[-6] + deeper_second[-6], deeper_whole[-6])
    torch.testing.assert_close(deeper_second[-4], deeper_whole[-4])
    torch.testing.assert_close(deeper_second[:resource_count + 1], deeper_whole[:resource_count + 1])
    for expected, observed in zip(actual, recorded[:len(actual)], strict=True):
        torch.testing.assert_close(observed, expected)
    torch.testing.assert_close(recorded[-3][:3], torch.ones((3, 1), dtype=torch.bool))
    torch.testing.assert_close(recorded[-2], torch.tensor(False))
    torch.testing.assert_close(recorded[-1][:3], torch.tensor([0, 1, 1]))
    capacity_exit = plan.forward_until_done_with_route_receipt(
        *inputs, host_step_limit=torch.tensor(10, dtype=torch.int64), receipt_capacity=1,
    )
    torch.testing.assert_close(capacity_exit[-6], torch.tensor(1))
    torch.testing.assert_close(capacity_exit[-4], torch.tensor([True]))
    torch.testing.assert_close(capacity_exit[-1], torch.tensor([0]))
    torch.testing.assert_close(actual[1], _view([[3.0]]).value)
    torch.testing.assert_close(actual[4], _view([[-6.0]]).value)
    torch.testing.assert_close(actual[-3], torch.tensor(3))
    torch.testing.assert_close(actual[-1], torch.tensor([False]))
    target = torch.tensor([[[-2.0]]])
    eager_result = next(
        item.active_view.value for item in eager.state.resources
        if item.spec.resource_id == "result"
    )
    final_loss = (eager_result - target).square().mean()
    expected_gain = torch.autograd.grad(final_loss, connections[1].transfer.gain)[0]
    cotangent = 2 * (actual[4].detach() - target) / actual[4].numel()
    reverse = plan.credit_gradient(
        *inputs, execution=eager,
        terminal_cotangents={"result": cotangent}, create_graph=False,
    )
    recorded_reverse = plan.credit_gradient(
        *inputs, tensor_receipt=(recorded[-6], recorded[-3], recorded[-2], recorded[-1]),
        terminal_cotangents={"result": cotangent}, create_graph=False,
    )
    replayed = torch.compile(plan.replay_tensor_receipt, backend="eager", fullgraph=True)(
        *inputs, tensor_receipt=(recorded[-6], recorded[-3], recorded[-2], recorded[-1]),
    )
    torch.testing.assert_close(replayed[4], recorded[4])
    replayed_gain = torch.autograd.grad(
        replayed[4], connections[1].transfer.gain, grad_outputs=cotangent,
    )[0]
    torch.testing.assert_close(
        replayed_gain,
        recorded_reverse.parameter_cotangents[
            next(name for name, parameter in recorded_reverse.parameters.items()
                 if parameter is connections[1].transfer.gain)
        ],
    )
    torch.testing.assert_close(recorded_reverse.resource_values[4], actual[4])
    assert tuple(int(choices[0]) for choices in recorded_reverse.route_selections) == (0, 1, 1)
    torch.testing.assert_close(reverse.resource_values[4], actual[4])
    assert len(reverse.iteration_active) == 3
    assert tuple(int(choices[0]) for choices in reverse.route_selections) == (0, 1, 1)
    gain_cotangents = {
        id(parameter): reverse.parameter_cotangents[name]
        for name, parameter in reverse.parameters.items()
        if name in reverse.parameter_cotangents
    }
    torch.testing.assert_close(gain_cotangents[id(connections[1].transfer.gain)], expected_gain)
    recorded_gains = {
        id(parameter): recorded_reverse.parameter_cotangents[name]
        for name, parameter in recorded_reverse.parameters.items()
        if name in recorded_reverse.parameter_cotangents
    }
    torch.testing.assert_close(recorded_gains[id(connections[1].transfer.gain)], expected_gain)
    if stochastic_boundary:
        masks = (torch.tensor([False, True, True, True, True, True]),)
        sampled = plan.forward_until_done_with_route_receipt(
            *inputs, host_step_limit=torch.tensor(10, dtype=torch.int64),
            receipt_capacity=6, sample_routes=True, credit_mask_histories=masks,
        )
        sampled_reverse = plan.credit_gradient(
            *inputs, tensor_receipt=(sampled[-7], sampled[-4], sampled[-3], sampled[-2], sampled[-1]),
            terminal_cotangents={"result": torch.ones_like(sampled[4])}, create_graph=False,
        )
        assert sampled_reverse.route_log_probability is not None
        sampled_replay = plan.replay_tensor_receipt(
            *inputs,
            tensor_receipt=(sampled[-7], sampled[-4], sampled[-3], sampled[-2], sampled[-1]),
        )
        torch.testing.assert_close(sampled_replay[4], sampled[4])
        torch.testing.assert_close(sampled_replay[-1], sampled_reverse.route_log_probability)
        row_advantage = torch.tensor([2.0])
        weighted_replay = plan.replay_tensor_receipt(
            *inputs,
            tensor_receipt=(sampled[-7], sampled[-4], sampled[-3], sampled[-2], sampled[-1]),
            row_advantage=row_advantage,
        )
        manual_objective = sampled_reverse.structure_objective(row_advantage)
        torch.testing.assert_close(weighted_replay[-1], manual_objective)
        torch.testing.assert_close(
            torch.autograd.grad(weighted_replay[-1], nodes[0].module.scale, retain_graph=True)[0],
            torch.autograd.grad(manual_objective, nodes[0].module.scale, retain_graph=True)[0],
        )
        route_objective = sampled_reverse.structure_objective(torch.tensor(1.0))
        assert route_objective.ndim == 0
        route_gradient = torch.autograd.grad(route_objective, nodes[0].module.scale)[0]
        assert torch.isfinite(route_gradient) and route_gradient.abs() > 0
        torch.testing.assert_close(sampled_reverse.iteration_active[0], torch.tensor([True]))
        torch.testing.assert_close(sampled[-1][0], torch.tensor(False))
        masked_first = plan.forward_until_done_with_route_receipt(
            *inputs, host_step_limit=torch.tensor(10, dtype=torch.int64),
            receipt_capacity=1, credit_mask_histories=(masks[0][:1],),
        )
        first_reverse = plan.credit_gradient(
            *inputs,
            tensor_receipt=(masked_first[-7], masked_first[-4], masked_first[-3],
                            masked_first[-2], masked_first[-1]),
            terminal_cotangents={"result": torch.ones_like(masked_first[4])},
            create_graph=False,
        )
        first_gains = {
            id(parameter): first_reverse.parameter_cotangents[name]
            for name, parameter in first_reverse.parameters.items()
            if name in first_reverse.parameter_cotangents
        }
        torch.testing.assert_close(first_gains[id(connections[0].transfer.gain)], torch.tensor(0.0))
    positive_credit = gain_cotangents.get(id(connections[0].transfer.gain))
    assert positive_credit is None or torch.count_nonzero(positive_credit) == 0
    if stochastic_boundary:
        torch.manual_seed(0)
    truncated = graph.execute_loop_functional("open", host_step_limit=1)
    if stochastic_boundary:
        assert truncated.connections[0].credit_mask is not None
        torch.testing.assert_close(truncated.connections[0].credit_mask, torch.tensor(False))
    assert truncated.termination_reason == "host_limit"
    truncated_reverse = plan.credit_gradient(
        *inputs, execution=truncated,
        terminal_cotangents={"result": torch.ones_like(actual[4])}, create_graph=False,
    )
    torch.testing.assert_close(truncated_reverse.resource_values[4], _view([[4.0]]).value)
    truncated_gains = {
        id(parameter): truncated_reverse.parameter_cotangents[name]
        for name, parameter in truncated_reverse.parameters.items()
        if name in truncated_reverse.parameter_cotangents
    }
    torch.testing.assert_close(
        truncated_gains[id(connections[0].transfer.gain)],
        torch.tensor(0.0 if stochastic_boundary else 2.0),
    )
    negative_credit = truncated_gains.get(id(connections[1].transfer.gain))
    assert negative_credit is None or torch.count_nonzero(negative_credit) == 0
    if torch.cuda.is_available():
        cuda_plan = copy.deepcopy(plan).cuda()
        cuda_inputs = tuple(value.cuda() for value in inputs)
        cuda_result = torch.compile(
            cuda_plan.forward_until_done_with_status, backend="inductor", fullgraph=True,
        )(
            *cuda_inputs,
            host_step_limit=torch.tensor(10, dtype=torch.int64, device="cuda"),
        )
        for reference, compiled_value in zip(actual, cuda_result, strict=True):
            torch.testing.assert_close(compiled_value.cpu(), reference)
        compiled_cuda_receipt = torch.compile(
            cuda_plan.forward_until_done_with_route_receipt, backend="inductor", fullgraph=True,
        )
        cuda_recorded = compiled_cuda_receipt(
            *cuda_inputs,
            host_step_limit=torch.tensor(10, dtype=torch.int64, device="cuda"),
            receipt_capacity=6,
        )
        for reference, compiled_value in zip(recorded, cuda_recorded, strict=True):
            torch.testing.assert_close(compiled_value.cpu(), reference)
        cuda_first = compiled_cuda_receipt(
            *cuda_inputs, host_step_limit=torch.tensor(10, dtype=torch.int64, device="cuda"),
            receipt_capacity=2,
        )
        cuda_next_inputs = (
            *cuda_first[:resource_count], *cuda_inputs[resource_count:-1],
            cuda_first[resource_count],
        )
        cuda_second = compiled_cuda_receipt(
            *cuda_next_inputs, host_step_limit=torch.tensor(10, dtype=torch.int64, device="cuda"),
            receipt_capacity=2, initial_active=cuda_first[-4], completed_steps=cuda_first[-6],
        )
        for reference, compiled_value in zip(
            recorded[:resource_count + 1], cuda_second[:resource_count + 1], strict=True,
        ):
            torch.testing.assert_close(compiled_value.cpu(), reference)
        torch.testing.assert_close(
            cuda_first[-6] + cuda_second[-6], cuda_recorded[-6],
        )


def test_open_loop_three_candidate_route_replays_actual_transition_path() -> None:
    @arti.fabric_layer(inputs=("state",), outputs={"scores": "state"})
    class StateScores(torch.nn.Module):
        def forward(self, state: torch.Tensor) -> dict[str, mechanisms.TensorView]:
            value = state[:, 0, 0:1]
            scores = torch.cat(tuple(-(value - index).abs() for index in range(3)), dim=1)
            return {"scores": _vector_view(scores)}

    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[2.0]])),
        mechanisms.TensorResource(_spec("state"), _view([[0.0]])),
        mechanisms.TensorResource(_vector_spec("scores"), _vector_view(torch.zeros(1, 3))),
        mechanisms.TensorResource(_spec("output"), _view([[0.0]])),
        mechanisms.TensorResource(_continue_spec("continue"), _continue_view(torch.ones(1))),
    )
    connections = tuple(
        mechanisms.Connection(
            f"path_{index}", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("output"),
            transfer=mechanisms.LearnableAffineTransfer(gain=gain),
        ) for index, gain in enumerate((2.0, 3.0, 5.0))
    )
    nodes = (
        arti.as_fabric_node(
            "score", StateScores(),
            input_ports={"state": mechanisms.ResourcePort("state")},
            output_ports={"scores": mechanisms.ResourcePort("scores")},
        ),
        arti.as_fabric_node(
            "count", _DecoratedLoopStep(),
            input_ports={"value": mechanisms.ResourcePort("state")},
            output_ports={"value": mechanisms.ResourcePort("state"),
                          "continue": mechanisms.ResourcePort("continue")},
        ),
    )
    graph = mechanisms.ProgramGraph(
        resources, connections, nodes=nodes,
        programs={"iterate": (
            "score", mechanisms.ProgramRoute(
                "choose", "scores", tuple(connection.connection_id for connection in connections),
            ), "count",
        )},
        loops=(mechanisms.ProgramLoop("open", "iterate", "continue", max_iterations=None),),
    )
    eager = graph.execute_loop_functional("open", host_step_limit=6)
    assert tuple(route.candidate_id for route in eager.routes) == ("path_0", "path_1", "path_2")
    plan = mechanisms.ResourceGraphCompiler.compile_loop(graph, "open")
    inputs = (*(resource.resolve().view.value for resource in resources),
              torch.zeros((1, plan.body.arrival_width), dtype=torch.bool))
    recorded = torch.compile(
        plan.forward_until_done_with_route_receipt, backend="eager", fullgraph=True,
    )(*inputs, host_step_limit=torch.tensor(6, dtype=torch.int64), receipt_capacity=6)
    torch.testing.assert_close(recorded[3], _view([[10.0]]).value)
    torch.testing.assert_close(recorded[-1][:3], torch.tensor([0, 1, 2]))
    receipt = (recorded[-6], recorded[-3], recorded[-2], recorded[-1])
    replay = plan.replay_tensor_receipt(*inputs, tensor_receipt=receipt)
    torch.testing.assert_close(replay[3], recorded[3])
    reverse = plan.credit_gradient(
        *inputs, tensor_receipt=receipt,
        terminal_cotangents={"output": torch.ones_like(recorded[3])},
    )
    torch.testing.assert_close(reverse.resource_cotangents["source"], _view([[5.0]]).value)
    assert tuple(int(choices[0]) for choices in reverse.route_selections) == (0, 1, 2)
    final_gain = connections[2].transfer.gain
    torch.testing.assert_close(
        next(value for name, value in reverse.parameter_cotangents.items()
             if reverse.parameters[name] is final_gain),
        torch.tensor(2.0),
    )


def test_open_loop_replays_each_implicit_bernoulli_credit_mask() -> None:
    state = mechanisms.TensorResource(_spec("state"), _view([[1.0]]))
    continuation = mechanisms.TensorResource(
        _continue_spec("continue"), _continue_view(torch.ones(1)),
    )
    scores = mechanisms.TensorResource(
        _vector_spec("scores"), _vector_view(torch.tensor([[2.0, -2.0]])),
    )
    connection = mechanisms.Connection(
        "scale", mechanisms.ResourcePort("state"), mechanisms.ResourcePort("state"),
        transfer=mechanisms.LearnableAffineTransfer(gain=2.0),
        credit_boundary=arti.CreditBoundary(mode=arti.CreditBoundaryMode.BERNOULLI),
    )
    alternate = mechanisms.Connection(
        "alternate", mechanisms.ResourcePort("state"), mechanisms.ResourcePort("state"),
    )
    graph = mechanisms.ProgramGraph(
        (state, continuation, scores), (connection, alternate),
        programs={"iterate": (mechanisms.ProgramRoute(
            "choice", "scores", ("scale", "alternate"),
        ),)},
        loops=(mechanisms.ProgramLoop("open", "iterate", "continue", max_iterations=None),),
    )
    plan = mechanisms.ResourceGraphCompiler.compile_loop(
        graph, "open", example_credit_masks={"scale": torch.tensor(True)},
    )
    inputs = (
        state.resolve().view.value, continuation.resolve().view.value, scores.resolve().view.value,
        torch.tensor(True), torch.zeros((1, plan.body.arrival_width), dtype=torch.bool),
    )
    torch.manual_seed(1)
    execution = graph.execute_loop_functional("open", host_step_limit=2)
    assert execution.termination_reason == "host_limit"
    assert [bool(item.credit_mask) for item in execution.connections] == [False, True]
    final_state = next(
        item.active_view.value for item in execution.state.resources
        if item.spec.resource_id == "state"
    )
    direct = torch.autograd.grad(final_state.sum(), connection.transfer.gain)[0]
    reverse = plan.credit_gradient(
        *inputs, execution=execution,
        terminal_cotangents={"state": torch.ones_like(final_state)}, create_graph=False,
    )
    recorded = next(
        cotangent for name, parameter in reverse.parameters.items()
        if parameter is connection.transfer.gain
        for cotangent in (reverse.parameter_cotangents[name],)
    )
    torch.testing.assert_close(recorded, direct)
    torch.testing.assert_close(recorded, torch.tensor(2.0))


def test_program_stage_runs_distinct_formula_heads_from_one_snapshot_then_joins() -> None:
    value_type = mechanisms.TensorType.axes(
        ("batch", "token", "feature"), sizes=("B", "S", 1)
    )
    source_binding = mechanisms.InputBinding("source", value_type)
    double_program = mechanisms.FormulaProgram.build(
        outputs=(mechanisms.add(source_binding, source_binding),)
    )
    left_binding = mechanisms.InputBinding("left", value_type)
    right_binding = mechanisms.InputBinding("right", value_type)
    join_program = mechanisms.FormulaProgram.build(
        outputs=(mechanisms.add(left_binding, right_binding),)
    )
    source = mechanisms.TensorResource(_spec("source"), _view([[2.0, 4.0]]))
    left = mechanisms.TensorResource(_spec("left"), _view([[0.0, 0.0]]))
    right = mechanisms.TensorResource(_spec("right"), _view([[0.0, 0.0]]))
    output = mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]]))
    left_head = mechanisms.FormulaProgramNode(
        "left_head",
        mechanisms.FormulaFabricV2(double_program),
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("left")},
        output_slots={"value": "%0"},
    )
    right_head = mechanisms.FormulaProgramNode(
        "right_head",
        mechanisms.FormulaFabricV2(double_program),
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("right")},
        output_slots={"value": "%0"},
    )
    join = mechanisms.FormulaProgramNode(
        "join_heads",
        mechanisms.FormulaFabricV2(join_program),
        input_ports={
            "left": mechanisms.ResourcePort("left"),
            "right": mechanisms.ResourcePort("right"),
        },
        output_ports={"value": mechanisms.ResourcePort("output")},
        output_slots={"value": "%0"},
    )
    graph = mechanisms.ProgramGraph(
        (source, left, right, output),
        (),
        nodes=(left_head, right_head, join),
        programs={
            "parallel_then_join": (
                mechanisms.ProgramStage(("left_head", "right_head")),
                "join_heads",
            )
        },
    )

    result = graph.execute_program_functional("parallel_then_join")
    outputs = {item.spec.resource_id: item.active_view.value for item in result.state.resources}

    assert tuple(node.node_id for node in result.nodes) == (
        "left_head",
        "right_head",
        "join_heads",
    )
    torch.testing.assert_close(outputs["left"], _view([[4.0, 8.0]]).value)
    torch.testing.assert_close(outputs["right"], _view([[4.0, 8.0]]).value)
    torch.testing.assert_close(outputs["output"], _view([[8.0, 16.0]]).value)
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "parallel_then_join")
    compiled = torch.compile(plan, backend="eager", fullgraph=True)(
        source.resolve().view.value,
        left.resolve().view.value,
        right.resolve().view.value,
        output.resolve().view.value,
    )
    torch.testing.assert_close(compiled[-1], _view([[8.0, 16.0]]).value)


def test_differentiable_fabric_node_specializes_to_one_compilable_formula_region(tmp_path) -> None:
    value_type = mechanisms.TensorType.axes(
        ("batch", "token", "feature"), sizes=("B", "S", 1)
    )
    source_binding = mechanisms.InputBinding("source", value_type)
    double_program = mechanisms.FormulaProgram.build(
        outputs=(mechanisms.add(source_binding, source_binding),)
    )
    doubled = mechanisms.add(source_binding, source_binding)
    quadruple_program = mechanisms.FormulaProgram.build(
        outputs=(mechanisms.add(doubled, doubled),)
    )
    source = mechanisms.TensorResource(_spec("source"), _view([[2.0, 4.0]], requires_grad=True))
    output = mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]]))
    ports = {"source": mechanisms.ResourcePort("source")}
    outputs = {"value": mechanisms.ResourcePort("output")}
    node = mechanisms.DifferentiableFabricNode(
        "fate",
        {
            "double": mechanisms.FormulaProgramNode(
                "double_candidate", mechanisms.FormulaFabricV2(double_program),
                input_ports=ports, output_ports=outputs,
                output_slots={"value": double_program.outputs[0]},
            ),
            "quadruple": mechanisms.FormulaProgramNode(
                "quadruple_candidate", mechanisms.FormulaFabricV2(quadruple_program),
                input_ports=ports, output_ports=outputs,
                output_slots={"value": quadruple_program.outputs[0]},
            ),
        },
    )
    graph = mechanisms.ProgramGraph((source, output), (), nodes=(node,), programs={"run": ("fate",)})
    assert arti.component_ref(node).startswith("arti/differentiable-fabric-node@sha256:")
    saved = mechanisms.save_program_graph(graph, tmp_path / "differentiable-fate")
    restored = mechanisms.load_program_graph(saved.tensors_path, nodes=(node,))
    assert arti.component_ref(restored) == arti.component_ref(graph)
    trainable = graph.execute_program_functional("run")
    value = next(item.active_view.value for item in trainable.state.resources if item.spec.resource_id == "output")
    torch.testing.assert_close(value, _view([[6.0, 12.0]]).value)
    value.square().sum().backward()
    assert node.logits.grad is not None and torch.count_nonzero(node.logits.grad) > 0

    with torch.no_grad():
        node.logits.copy_(torch.tensor([-5.0, 5.0]))
    specialization = node.specialize()
    assert specialization.candidate_id == "quadruple"
    sealed_graph = mechanisms.ProgramGraph(
        (source, output), (), nodes=(specialization.node,), programs={"run": ("fate",)}
    )
    plan = mechanisms.ResourceGraphCompiler.compile_program(sealed_graph, "run")
    static = torch.compile(plan, backend="eager", fullgraph=True)(
        source.resolve().view.value, output.resolve().view.value
    )
    torch.testing.assert_close(static[-1], _view([[8.0, 16.0]]).value)


def test_decorated_pytorch_layers_mount_as_fabric_nodes_and_differentiate_fates() -> None:
    source = mechanisms.TensorResource(_spec("source"), _view([[2.0, 4.0]], requires_grad=True))
    output = mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]]))
    input_ports = {"source": mechanisms.ResourcePort("source")}
    output_ports = {"value": mechanisms.ResourcePort("output")}

    node = arti.as_fabric_node(
        "scale",
        _DecoratedScale(3.0),
        input_ports=input_ports,
        output_ports=output_ports,
    )
    graph = mechanisms.ProgramGraph((source, output), (), nodes=(node,), programs={"run": ("scale",)})
    result = graph.execute_program_functional("run")
    value = next(item.active_view.value for item in result.state.resources if item.spec.resource_id == "output")
    torch.testing.assert_close(value, _view([[6.0, 12.0]]).value)
    value.sum().backward()
    assert node.module.scale.grad is not None
    assert arti.component_ref(node).startswith("arti/fabric-module-node@sha256:")
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "run")
    inputs = (source.resolve().view.value, output.resolve().view.value)
    eager = plan(*inputs)
    compiled = torch.compile(plan, backend="eager", fullgraph=True)(*inputs)
    exported = torch.export.export(plan, inputs).module()(*inputs)
    for eager_value, compiled_value, exported_value in zip(eager, compiled, exported, strict=True):
        torch.testing.assert_close(eager_value, compiled_value)
        torch.testing.assert_close(eager_value, exported_value)
    torch.testing.assert_close(eager[-1], _view([[6.0, 12.0]]).value)

    fates = arti.as_differentiable_fabric_node(
        "fate",
        {"double": _DoubleFate(), "quadruple": _QuadrupleFate()},
        input_ports=input_ports,
        output_ports=output_ports,
    )
    fate_graph = mechanisms.ProgramGraph((source, output), (), nodes=(fates,), programs={"run": ("fate",)})
    mixed = fate_graph.execute_program_functional("run")
    mixed_value = next(
        item.active_view.value for item in mixed.state.resources if item.spec.resource_id == "output"
    )
    torch.testing.assert_close(mixed_value, _view([[6.0, 12.0]]).value)
    mixed_value.square().sum().backward()
    assert fates.logits.grad is not None and torch.count_nonzero(fates.logits.grad) > 0

    with torch.no_grad():
        fates.logits.copy_(torch.tensor([-4.0, 4.0]))
    specialization = fates.specialize()
    assert specialization.candidate_id == "quadruple"
    assert isinstance(specialization.node, arti.FabricModuleNode)
    sealed = mechanisms.ProgramGraph(
        (source, output), (), nodes=(specialization.node,), programs={"run": ("fate",)}
    )
    final = sealed.execute_program_functional("run")
    final_value = next(item.active_view.value for item in final.state.resources if item.spec.resource_id == "output")
    torch.testing.assert_close(final_value, _view([[8.0, 16.0]]).value)
    assert arti.fabric_registration(_IdentityFate()).input_names == ("source",)


def test_differentiable_fate_uses_hard_candidate_losses_not_mixed_output_loss() -> None:
    source = mechanisms.TensorResource(_spec("source"), _view([[1.0]]))
    node = arti.as_differentiable_fabric_node(
        "choice",
        {"double": _DoubleFate(), "quadruple": _QuadrupleFate()},
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    inputs = {"source": source.resolve().view}
    mixed = node.invoke_ports(inputs).outputs["value"].value
    torch.testing.assert_close(mixed, _view([[3.0]]).value)
    assert mixed.sub(3.0).square().mean().item() == pytest.approx(0.0)

    hard_losses = {
        candidate_id: node.invoke_candidate_ports(candidate_id, inputs).outputs["value"].value.sub(3.0).square().mean()
        for candidate_id in node.candidate_ids
    }
    assert all(loss.item() == pytest.approx(1.0) for loss in hard_losses.values())
    structure = node.discrete_structure_objective(hard_losses)
    assert structure.item() == pytest.approx(1.0)
    structure.backward()
    assert node.logits.grad is not None
    torch.testing.assert_close(node.logits.grad, torch.zeros_like(node.logits))


def test_differentiable_fate_structure_gradient_and_candidate_exposure_are_separate() -> None:
    node = arti.as_differentiable_fabric_node(
        "choice",
        {"double": _DoubleFate(), "quadruple": _QuadrupleFate()},
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    losses = {"double": torch.tensor(3.0, requires_grad=True), "quadruple": torch.tensor(1.0, requires_grad=True)}
    structure = node.discrete_structure_objective(losses)
    structure.backward(retain_graph=True)
    assert node.logits.grad is not None
    torch.testing.assert_close(node.logits.grad, torch.tensor([0.5, -0.5]))
    assert losses["double"].grad is None and losses["quadruple"].grad is None

    node.logits.grad = None
    candidate_training = node.candidate_training_objective(losses, exposure={"double": 0.25, "quadruple": 0.75})
    candidate_training.backward()
    assert node.logits.grad is None
    assert losses["double"].grad is not None and losses["double"].grad.item() == pytest.approx(0.25)
    assert losses["quadruple"].grad is not None and losses["quadruple"].grad.item() == pytest.approx(0.75)


def test_program_graph_executes_an_explicit_hard_fate_without_mutating_structure_logits() -> None:
    source = mechanisms.TensorResource(_spec("source"), _view([[2.0]]))
    output = mechanisms.TensorResource(_spec("output"), _view([[0.0]]))
    choice = arti.as_differentiable_fabric_node(
        "choice",
        {"double": _DoubleFate(), "quadruple": _QuadrupleFate()},
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph((source, output), (), nodes=(choice,), programs={"run": ("choice",)})
    original_logits = choice.logits.detach().clone()

    result = graph.execute_program_functional("run", candidate_selections={"choice": "quadruple"})
    final = next(item.active_view.value for item in result.state.resources if item.spec.resource_id == "output")
    torch.testing.assert_close(final, _view([[8.0]]).value)
    assert result.nodes[0].receipt["candidate_id"] == "quadruple"
    torch.testing.assert_close(choice.logits, original_logits)


def test_program_graph_joint_hard_fate_sample_credits_the_real_final_loss() -> None:
    source = mechanisms.TensorResource(_spec("source"), _view([[2.0]]))
    middle = mechanisms.TensorResource(_spec("middle"), _view([[0.0]]))
    chosen = mechanisms.TensorResource(_spec("chosen"), _view([[0.0]]))
    output = mechanisms.TensorResource(_spec("output"), _view([[0.0]]))
    first = arti.as_differentiable_fabric_node(
        "first",
        {"double": _DoubleFate(), "quadruple": _QuadrupleFate()},
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("middle")},
    )
    second = arti.as_differentiable_fabric_node(
        "second",
        {"double": _DoubleFate(), "quadruple": _QuadrupleFate()},
        input_ports={"source": mechanisms.ResourcePort("middle")},
        output_ports={"value": mechanisms.ResourcePort("chosen")},
    )
    transfer = mechanisms.LearnableAffineTransfer()
    graph = mechanisms.ProgramGraph(
        (source, middle, chosen, output),
        (mechanisms.Connection(
            "final", mechanisms.ResourcePort("chosen"), mechanisms.ResourcePort("output"),
            transfer=transfer,
        ),),
        nodes=(first, second),
        programs={"run": ("first", "second", "final")},
    )
    with torch.no_grad():
        first.logits.copy_(torch.tensor([0.4, -0.2]))
        second.logits.copy_(torch.tensor([-0.3, 0.6]))
    sample = graph.sample_program_fates("run", generator=torch.Generator().manual_seed(17))
    assert tuple(node_id for node_id, _candidate in sample.selections) == ("first", "second")
    execution = graph.execute_program_functional("run", candidate_selections=dict(sample.selections))
    result = next(item.active_view.value for item in execution.state.resources if item.spec.resource_id == "output")
    final_loss = (result - 11.0).square().mean()
    objective = sample.structure_objective(final_loss, baseline=1.0)
    for node, (node_id, candidate_id) in zip((first, second), sample.selections, strict=True):
        assert node.node_id == node_id
        indicator = torch.zeros_like(node.logits)
        indicator[node.candidate_ids.index(candidate_id)] = 1.0
        expected = (final_loss.detach() - 1.0) * (indicator - node.probabilities().detach())
        observed = torch.autograd.grad(objective, node.logits, retain_graph=True)[0]
        torch.testing.assert_close(observed, expected)
    assert torch.autograd.grad(objective, transfer.gain, allow_unused=True)[0] is None
    assert torch.autograd.grad(final_loss, transfer.gain)[0].abs().item() > 0.0


@pytest.mark.parametrize("initial_scores", ((0.2, -0.2), (-0.2, 0.2)))
def test_open_loop_jointly_credits_routes_and_executed_fates(
    initial_scores: tuple[float, float],
) -> None:
    torch._dynamo.reset()

    @arti.fabric_layer(inputs=("source",), outputs={"scores": "source"})
    class _TrainableScores(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor(initial_scores))

        def forward(self, source: torch.Tensor) -> dict[str, mechanisms.TensorView]:
            return {"scores": _vector_view(self.logits.unsqueeze(0).expand(source.shape[0], -1))}

    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[2.0]])),
        *(mechanisms.TensorResource(_spec(name), _view([[0.0]]))
          for name in ("left", "right", "selected", "output")),
        mechanisms.TensorResource(_vector_spec("scores"), _vector_view(torch.zeros(1, 2))),
        mechanisms.TensorResource(_continue_spec("continue"), _continue_view(torch.ones(1))),
    )
    scoring = arti.as_fabric_node(
        "score", _TrainableScores(),
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"scores": mechanisms.ResourcePort("scores")},
    )
    prepare = arti.as_differentiable_fabric_node(
        "prepare", {"double": _DoubleFate(), "quadruple": _QuadrupleFate()},
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("left")},
    )
    branch_fate = arti.as_differentiable_fabric_node(
        "branch_fate", {"double": _DoubleFate(), "quadruple": _QuadrupleFate()},
        input_ports={"source": mechanisms.ResourcePort("left")},
        output_ports={"value": mechanisms.ResourcePort("selected")},
    )
    join = arti.as_fabric_node(
        "join", _DecoratedPairSum(),
        input_ports={
            "left": mechanisms.ResourcePort("selected"),
            "right": mechanisms.ResourcePort("right"),
        },
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    connections = (
        mechanisms.Connection("reference", mechanisms.ResourcePort("left"), mechanisms.ResourcePort("right")),
        mechanisms.Connection("alternate", mechanisms.ResourcePort("left"), mechanisms.ResourcePort("selected")),
        mechanisms.Connection("feedback", mechanisms.ResourcePort("output"), mechanisms.ResourcePort("source")),
    )
    graph = mechanisms.ProgramGraph(
        resources, connections, nodes=(scoring, prepare, branch_fate, join),
        programs={"iterate": (
            "score", "prepare", "reference",
            mechanisms.ProgramRoute("choice", "scores", ("branch_fate", "alternate")),
            mechanisms.ProgramJoin("ready", "join"), "feedback",
        )},
        loops=(mechanisms.ProgramLoop("open", "iterate", "continue", max_iterations=None),),
    )
    sample = graph.sample_program_fates("iterate", generator=torch.Generator().manual_seed(13))
    torch.manual_seed(19)
    execution = graph.execute_loop_functional(
        "open", candidate_selections=dict(sample.selections),
        sample_routes=True, host_step_limit=2,
    )
    assert execution.actual_iterations == 2
    assert len(execution.routes) == 2
    assert len(execution.joins) == 2 and all(join_receipt.fired for join_receipt in execution.joins)
    output = next(
        resource.active_view.value for resource in execution.state.resources
        if resource.spec.resource_id == "output"
    )
    loss = (output - 11.0).square().mean()
    objective = execution.structure_objective(loss, baseline=1.0, fate_sample=sample)
    trajectory_fate_credit = sample.trajectory_structure_objective(
        loss, baseline=1.0, route_ids=("choice",),
        route_selections=tuple(
            (torch.tensor(("branch_fate", "alternate").index(route.candidate_id)),)
            for route in execution.routes
        ),
        iteration_active=execution.active_masks,
    )
    route_credit = sum(
        route.structure_objective(loss, baseline=1.0) for route in execution.routes
    )
    torch.testing.assert_close(trajectory_fate_credit + route_credit, objective)
    advantage = loss.detach() - 1.0
    for fate in (prepare, branch_fate):
        observed = torch.autograd.grad(objective, fate.logits, retain_graph=True, allow_unused=True)[0]
        reached = any(node.node_id == fate.node_id for node in execution.nodes)
        if not reached:
            assert observed is None or torch.count_nonzero(observed) == 0
            continue
        choice = dict(sample.selections)[fate.node_id]
        indicator = torch.zeros_like(fate.logits)
        indicator[fate.candidate_ids.index(choice)] = 1.0
        torch.testing.assert_close(observed, advantage * (indicator - fate.probabilities().detach()))
    score_parameter = scoring.module.logits
    route_gradient = torch.autograd.grad(objective, score_parameter)[0]
    expected_route_gradient = sum(
        (
            torch.nn.functional.one_hot(
                torch.tensor(("branch_fate", "alternate").index(route.candidate_id)), 2,
            ).to(score_parameter.dtype)
            - torch.softmax(score_parameter.detach(), dim=0)
    ) for route in execution.routes
    ) * advantage
    torch.testing.assert_close(route_gradient, expected_route_gradient)

    selections = dict(sample.selections)
    forced = graph.execute_loop_functional(
        "open", candidate_selections=selections, host_step_limit=2,
    )
    route_id = forced.routes[0].candidate_id
    assert all(route.candidate_id == route_id for route in forced.routes)
    specialized = graph.specialize_differentiable_nodes(selections).graph
    specialized = specialized.specialize_program_routes({"choice": route_id}).graph
    assert not any(isinstance(node, arti.DifferentiableFabricNode) for node in specialized.nodes.values())
    assert not any(isinstance(step, mechanisms.ProgramRoute) for step in specialized.program("iterate"))
    plan = mechanisms.ResourceGraphCompiler.compile_loop(specialized, "open")
    assert isinstance(plan, mechanisms.StaticDataflowLoopExecutionPlan)
    values = tuple(specialized.resources[resource_id].resolve().view.value for resource_id in plan.resource_ids)
    arrivals = torch.zeros((1, plan.body.arrival_width), dtype=torch.bool)
    budget = torch.tensor(2, dtype=torch.int64)
    expected = plan.forward_until_done_with_status(*values, arrivals, host_step_limit=budget)
    compiled = torch.compile(plan.forward_until_done_with_status, backend="eager", fullgraph=True)
    actual = compiled(*values, arrivals, host_step_limit=budget)
    for reference, compiled_value in zip(expected, actual, strict=True):
        torch.testing.assert_close(compiled_value, reference)
    forced_output = next(
        resource.active_view.value for resource in forced.state.resources
        if resource.spec.resource_id == "output"
    )
    torch.testing.assert_close(actual[plan.resource_ids.index("output")], forced_output)


def test_program_graph_specialization_preserves_shared_candidate_parameter_identity() -> None:
    @arti.fabric_layer(inputs=("source",), outputs={"value": "source"})
    class SharedScale(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, source: torch.Tensor) -> torch.Tensor:
            return source * self.scale

    @arti.differentiable_fate("shared")
    @arti.fabric_layer(inputs=("source",), outputs={"value": "source"})
    class SharedFate(torch.nn.Module):
        def __init__(self, shared: SharedScale) -> None:
            super().__init__()
            self.shared = shared

        def forward(self, source: torch.Tensor) -> torch.Tensor:
            return self.shared(source)

    @arti.differentiable_fate("scaled_twice")
    @arti.fabric_layer(inputs=("source",), outputs={"value": "source"})
    class TwiceFate(torch.nn.Module):
        def __init__(self, shared: SharedScale) -> None:
            super().__init__()
            self.shared = shared

        def forward(self, source: torch.Tensor) -> torch.Tensor:
            return self.shared(source) * 2.0

    shared = SharedScale()
    source = mechanisms.TensorResource(_spec("source"), _view([[2.0]], requires_grad=True))
    middle = mechanisms.TensorResource(_spec("middle"), _view([[0.0]]))
    output = mechanisms.TensorResource(_spec("output"), _view([[0.0]]))
    choice = arti.as_differentiable_fabric_node(
        "choice",
        {"shared": SharedFate(shared), "scaled_twice": TwiceFate(shared)},
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("middle")},
    )
    observer = arti.as_fabric_node(
        "observer",
        shared,
        input_ports={"source": mechanisms.ResourcePort("middle")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph(
        (source, middle, output), (), nodes=(choice, observer), programs={"run": ("choice", "observer")}
    )
    forced = choice.invoke_candidate_ports("shared", {"source": source.resolve().view}).outputs["value"].value
    torch.testing.assert_close(forced, _view([[4.0]]).value)
    sealed = graph.specialize_differentiable_nodes({"choice": "shared"})
    assert sealed.selections == (("choice", "shared"),)
    selected = sealed.graph.nodes["choice"]
    retained_observer = sealed.graph.nodes["observer"]
    assert isinstance(selected, arti.FabricModuleNode)
    assert isinstance(retained_observer, arti.FabricModuleNode)
    assert selected.module.shared.scale is retained_observer.module.scale
    result = sealed.graph.execute_program_functional("run")
    final = next(item.active_view.value for item in result.state.resources if item.spec.resource_id == "output")
    torch.testing.assert_close(final, _view([[8.0]]).value)

    live = graph.specialize_differentiable_nodes(
        {"choice": "shared"}, share_module_state=True,
    ).graph
    assert live.nodes["choice"].module.shared.scale is shared.scale
    assert live.nodes["observer"].module.scale is shared.scale
    assert sealed.graph.nodes["choice"].module.shared.scale is not shared.scale
    plan = mechanisms.ResourceGraphCompiler.compile_program(live, "run")
    assert sum(parameter is shared.scale for parameter in plan.parameters()) == 1
    inputs = tuple(resource.resolve().view.value for resource in (source, middle, output))
    before = plan(*inputs)[plan.resource_ids.index("output")]
    optimizer = torch.optim.AdamW((shared.scale,), lr=0.1, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    before.square().sum().backward()
    assert shared.scale.grad is not None
    optimizer.step()
    after = plan(*inputs)[plan.resource_ids.index("output")]
    assert not torch.equal(before, after)


def test_two_hard_fates_specialize_with_shared_credit_through_join_and_connection() -> None:
    @arti.differentiable_fate("scale")
    @arti.fabric_layer(inputs=("source",), outputs={"value": "source"})
    class _ScaleFate(torch.nn.Module):
        def __init__(self, shared: _DecoratedScale) -> None:
            super().__init__()
            self.shared = shared

        def forward(self, source: torch.Tensor) -> torch.Tensor:
            return self.shared(source)

    @arti.differentiable_fate("identity")
    @arti.fabric_layer(inputs=("source",), outputs={"value": "source"})
    class _IdentityFate(torch.nn.Module):
        def forward(self, source: torch.Tensor) -> torch.Tensor:
            return source

    shared = _DecoratedScale(2.0)
    resources = (
        mechanisms.TensorResource(_spec("source"), _view([[2.0]], requires_grad=True)),
        mechanisms.TensorResource(_spec("left"), _view([[0.0]])),
        mechanisms.TensorResource(_spec("right"), _view([[0.0]])),
        mechanisms.TensorResource(_spec("joined"), _view([[0.0]])),
        mechanisms.TensorResource(_spec("output"), _view([[0.0]])),
    )
    choices = tuple(
        arti.as_differentiable_fabric_node(
            node_id,
            {"scale": _ScaleFate(shared), "identity": _IdentityFate()},
            input_ports={"source": mechanisms.ResourcePort("source")},
            output_ports={"value": mechanisms.ResourcePort(destination)},
        )
        for node_id, destination in (("left_choice", "left"), ("right_choice", "right"))
    )
    join = arti.as_fabric_node(
        "sum", _DecoratedPairSum(),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("joined")},
    )
    transfer = mechanisms.LearnableAffineTransfer()
    with torch.no_grad():
        transfer.gain.fill_(1.5)
    final = mechanisms.Connection(
        "final", mechanisms.ResourcePort("joined"), mechanisms.ResourcePort("output"),
        transfer=transfer,
    )
    graph = mechanisms.ProgramGraph(
        resources, (final,), nodes=(*choices, join),
        programs={"run": (
            mechanisms.ProgramStage(("left_choice", "right_choice")),
            mechanisms.ProgramJoin("ready", "sum"), "final",
        )},
    )
    sampled = graph.sample_program_fates("run", generator=torch.Generator().manual_seed(29))
    assert tuple(node_id for node_id, _candidate in sampled.selections) == (
        "left_choice", "right_choice"
    )
    forced = graph.execute_program_functional(
        "run", candidate_selections={"left_choice": "scale", "right_choice": "scale"}
    )
    forced_output = next(
        item.active_view.value for item in forced.state.resources if item.spec.resource_id == "output"
    )
    sealed = graph.specialize_differentiable_nodes(
        {"left_choice": "scale", "right_choice": "scale"}
    ).graph
    left_scale = sealed.nodes["left_choice"].module.shared.scale
    right_scale = sealed.nodes["right_choice"].module.shared.scale
    assert left_scale is right_scale
    assert left_scale is not shared.scale
    sealed_transfer = sealed.connections["final"].transfer.gain
    plan = mechanisms.ResourceGraphCompiler.compile_program(sealed, "run")
    assert isinstance(plan, mechanisms.StaticDataflowProgramExecutionPlan)
    compiled_connection = next(
        step.plan.connections[0] for step in plan.nodes
        if getattr(step, "connection_id", None) == "final"
    )
    assert compiled_connection.transfer.gain is sealed_transfer
    inputs = (
        *(resource.resolve().view.value for resource in sealed.resources.values()),
        torch.zeros((1, plan.arrival_width), dtype=torch.bool),
    )
    compiled_output = plan(*inputs)[plan.resource_ids.index("output")]
    torch.testing.assert_close(forced_output, _view([[12.0]]).value)
    torch.testing.assert_close(compiled_output, forced_output)
    forced_grads = torch.autograd.grad(forced_output.sum(), (shared.scale, transfer.gain))
    compiled_grads = torch.autograd.grad(compiled_output.sum(), (left_scale, sealed_transfer))
    for forced_grad, compiled_grad in zip(forced_grads, compiled_grads, strict=True):
        torch.testing.assert_close(compiled_grad, forced_grad)
    credit = plan.credit_gradient(
        *inputs, terminal_cotangents={"output": torch.ones_like(compiled_output)}
    )
    torch.testing.assert_close(
        credit.resource_cotangents["source"],
        torch.full_like(inputs[plan.resource_ids.index("source")], 6.0),
    )
    torch.testing.assert_close(credit.parameter_cotangents["left_choice.shared.scale"], forced_grads[0])
    torch.testing.assert_close(credit.parameter_cotangents["final.transfer.gain"], forced_grads[1])
    with torch.no_grad():
        shared.scale.sub_(0.1 * forced_grads[0])
        transfer.gain.sub_(0.1 * forced_grads[1])
        left_scale.sub_(0.1 * compiled_grads[0])
        sealed_transfer.sub_(0.1 * compiled_grads[1])
    torch.testing.assert_close(left_scale, shared.scale)
    torch.testing.assert_close(sealed_transfer, transfer.gain)


def test_program_graph_specialization_preserves_resource_values_without_prior_autograd_history() -> None:
    source_leaf = torch.tensor([[3.0]], requires_grad=True)
    source = mechanisms.TensorResource(
        _spec("source"),
        mechanisms.TensorView.from_tensor(
            (source_leaf * 2.0).unsqueeze(-1),
            axis_names=("batch", "token", "feature"),
            axis_roles=("batch", "sequence", "feature"),
        ),
    )
    output = mechanisms.TensorResource(_spec("output"), _view([[0.0]]))
    choice = arti.as_differentiable_fabric_node(
        "choice",
        {"double": _DoubleFate(), "quadruple": _QuadrupleFate()},
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph((source, output), (), nodes=(choice,), programs={"run": ("choice",)})

    sealed = graph.specialize_differentiable_nodes({"choice": "double"}).graph
    copied = sealed.resource("source").resolve().view.value
    torch.testing.assert_close(copied, source.resolve().view.value)
    assert copied.data_ptr() != source.resolve().view.value.data_ptr()
    assert copied.requires_grad
    assert copied.grad_fn is None


def test_differentiable_fate_specialization_can_continue_from_functional_resource_state() -> None:
    source = mechanisms.TensorResource(_spec("source"), _view([[2.0]]))
    memory = mechanisms.TensorResource(
        mechanisms.TensorResourceSpec(
            "memory", _spec("memory").view_pattern,
            lifetime=mechanisms.ResourceLifetime.PERSISTENT,
        ),
        _view([[0.0]]),
    )
    output = mechanisms.TensorResource(_spec("output"), _view([[0.0]]))
    choice = arti.as_differentiable_fabric_node(
        "choice",
        {"double": _DoubleFate(), "quadruple": _QuadrupleFate()},
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("memory")},
    )
    graph = mechanisms.ProgramGraph(
        (source, memory, output),
        (mechanisms.Connection(
            "read", mechanisms.ResourcePort("memory"), mechanisms.ResourcePort("output"),
        ),),
        nodes=(choice,),
        programs={"write": ("choice",), "read_only": ("read",)},
    )
    written = graph.execute_program_functional(
        "write", candidate_selections={"choice": "quadruple"},
    )
    sealed = graph.specialize_differentiable_nodes(
        {"choice": "quadruple"}, state=written.state,
    ).graph
    carried = sealed.resource("memory").resolve()
    torch.testing.assert_close(carried.view.value, _view([[8.0]]).value)
    assert sealed.resource("memory").state().epoch == 1
    assert carried.view.value.grad_fn is None
    assert graph.resource("memory").state().epoch == 0

    resumed = sealed.execute_program_functional("read_only")
    final = next(
        item.active_view.value for item in resumed.state.resources
        if item.spec.resource_id == "output"
    )
    torch.testing.assert_close(final, _view([[8.0]]).value)
    plan = mechanisms.ResourceGraphCompiler.compile_program(sealed, "read_only")
    inputs = tuple(sealed.resource(resource_id).resolve().view.value for resource_id in plan.resource_ids)
    compiled = torch.compile(plan, backend="eager", fullgraph=True)(*inputs)
    torch.testing.assert_close(compiled[plan.resource_ids.index("output")], final)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_static_decorated_fabric_node_matches_inductor_gradients_and_cuda_graph() -> None:
    device = torch.device("cuda")
    source = mechanisms.TensorResource(
        _spec("source"),
        mechanisms.TensorView.from_tensor(
            torch.zeros(3, 2, 1, device=device),
            axis_names=("batch", "token", "feature"),
            axis_roles=("batch", "sequence", "feature"),
        ),
    )
    output = mechanisms.TensorResource(
        _spec("output"),
        mechanisms.TensorView.from_tensor(
            torch.zeros(3, 2, 1, device=device),
            axis_names=("batch", "token", "feature"),
            axis_roles=("batch", "sequence", "feature"),
        ),
    )
    node = arti.as_fabric_node(
        "scale",
        _DecoratedScale(1.75).to(device),
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    graph = mechanisms.ProgramGraph((source, output), (), nodes=(node,), programs={"run": ("scale",)})
    prototype = mechanisms.ResourceGraphCompiler.compile_program(graph, "run").to(device)
    eager_plan = copy.deepcopy(prototype).to(device)
    compiled_plan = copy.deepcopy(prototype).to(device)
    eager_source = torch.randn(3, 2, 1, device=device, requires_grad=True)
    compiled_source = eager_source.detach().clone().requires_grad_(True)
    eager_target = torch.zeros_like(eager_source)
    compiled_target = torch.zeros_like(compiled_source)
    eager_output = eager_plan(eager_source, eager_target)[1]
    eager_gradients = torch.autograd.grad(
        eager_output.square().mean(), (eager_source, *eager_plan.parameters())
    )
    compiled_output = torch.compile(compiled_plan, backend="inductor", fullgraph=True)(
        compiled_source, compiled_target
    )[1]
    compiled_gradients = torch.autograd.grad(
        compiled_output.square().mean(), (compiled_source, *compiled_plan.parameters())
    )
    torch.testing.assert_close(compiled_output, eager_output)
    for actual, expected in zip(compiled_gradients, eager_gradients, strict=True):
        torch.testing.assert_close(actual, expected)

    inference_plan = copy.deepcopy(prototype).to(device)
    capture_source = torch.randn(3, 2, 1, device=device)
    capture_target = torch.zeros_like(capture_source)
    captured = inference_plan.capture(capture_source, capture_target)
    expected = inference_plan(capture_source, capture_target)[1]
    replayed = captured.replay(capture_source, capture_target)[1]
    torch.testing.assert_close(replayed, expected)


def test_decorated_fabric_nodes_lower_through_join_and_bounded_loop() -> None:
    torch._dynamo.reset()
    left = mechanisms.TensorResource(_spec("left"), _view([[2.0, 4.0]]))
    right = mechanisms.TensorResource(_spec("right"), _view([[3.0, 5.0]]))
    output = mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]]))
    sum_node = arti.as_fabric_node(
        "sum",
        _DecoratedPairSum(),
        input_ports={
            "left": mechanisms.ResourcePort("left"),
            "right": mechanisms.ResourcePort("right"),
        },
        output_ports={"value": mechanisms.ResourcePort("output")},
    )
    join_graph = mechanisms.ProgramGraph(
        (left, right, output),
        (),
        nodes=(sum_node,),
        programs={"join": (mechanisms.ProgramJoin("ready", "sum"),)},
    )
    join_plan = mechanisms.ResourceGraphCompiler.compile_join(join_graph, "ready")
    program_plan = mechanisms.ResourceGraphCompiler.compile_program(join_graph, "join")
    join_inputs = (
        left.resolve().view.value,
        right.resolve().view.value,
        output.resolve().view.value,
        torch.tensor([[True, True]]),
    )
    expected_join = join_plan(*join_inputs)
    compiled_join = torch.compile(join_plan, backend="eager", fullgraph=True)(*join_inputs)
    exported_join = torch.export.export(join_plan, join_inputs).module()(*join_inputs)
    whole_program = program_plan(*join_inputs)
    for expected, compiled, exported in zip(expected_join, compiled_join, exported_join, strict=True):
        torch.testing.assert_close(compiled, expected)
        torch.testing.assert_close(exported, expected)
    torch.testing.assert_close(whole_program[2], expected_join[2])
    torch.testing.assert_close(expected_join[2], _view([[5.0, 9.0]]).value)

    value = mechanisms.TensorResource(_spec("value"), _view([[0.0, 0.0]]))
    continuation = mechanisms.TensorResource(
        _continue_spec("continue"), _continue_view(torch.ones(1))
    )
    loop_node = arti.as_fabric_node(
        "step",
        _DecoratedLoopStep(),
        input_ports={"value": mechanisms.ResourcePort("value")},
        output_ports={
            "value": mechanisms.ResourcePort("value"),
            "continue": mechanisms.ResourcePort("continue"),
        },
    )
    loop_graph = mechanisms.ProgramGraph(
        (value, continuation),
        (),
        nodes=(loop_node,),
        programs={"step": ("step",)},
        loops=(mechanisms.ProgramLoop("bounded", "step", "continue", max_iterations=5),),
    )
    loop_plan = mechanisms.ResourceGraphCompiler.compile_loop(loop_graph, "bounded")
    loop_inputs = (value.resolve().view.value, continuation.resolve().view.value)
    expected_loop = loop_plan(*loop_inputs)
    dynamic_loop = loop_plan.forward_until_done(*loop_inputs)
    for expected, actual in zip(expected_loop, dynamic_loop[:-2], strict=True):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(dynamic_loop[-2], torch.tensor(3))
    torch.testing.assert_close(dynamic_loop[-1], torch.tensor([3]))
    compiled_dynamic_loop = torch.compile(loop_plan.forward_until_done, backend="eager", fullgraph=True)(
        *loop_inputs
    )
    for expected, actual in zip(dynamic_loop, compiled_dynamic_loop, strict=True):
        torch.testing.assert_close(actual, expected)
    with torch.no_grad():
        inference_loop = loop_plan(*loop_inputs)
    for expected, actual in zip(expected_loop, inference_loop, strict=True):
        torch.testing.assert_close(actual, expected)
    compiled_loop = torch.compile(loop_plan, backend="eager", fullgraph=True)(*loop_inputs)
    exported_loop = torch.export.export(loop_plan, loop_inputs).module()(*loop_inputs)
    for expected, compiled, exported in zip(expected_loop, compiled_loop, exported_loop, strict=True):
        torch.testing.assert_close(compiled, expected)
        torch.testing.assert_close(exported, expected)
    torch.testing.assert_close(expected_loop[0], _view([[3.0, 3.0]]).value)

    minimum_graph = mechanisms.ProgramGraph(
        (value, continuation), (), nodes=(loop_node,), programs={"step": ("step",)},
        loops=(mechanisms.ProgramLoop(
            "minimum", "step", "continue", max_iterations=5, min_iterations=4,
        ),),
    )
    minimum_plan = mechanisms.ResourceGraphCompiler.compile_loop(minimum_graph, "minimum")
    minimum = minimum_plan.forward_until_done(*loop_inputs)
    torch.testing.assert_close(minimum[-2], torch.tensor(4))
    torch.testing.assert_close(minimum[-1], torch.tensor([4]))
    torch.testing.assert_close(minimum[0], _view([[4.0, 4.0]]).value)


def test_static_loop_preserves_parallel_frontiers_before_conditional_exit() -> None:
    """A compiled loop may contain a same-snapshot fan-out before its join."""

    source = mechanisms.TensorResource(_spec("source"), _view([[1.0, 2.0]]))
    left = mechanisms.TensorResource(_spec("left"), _view([[0.0, 0.0]]))
    right = mechanisms.TensorResource(_spec("right"), _view([[0.0, 0.0]]))
    state = mechanisms.TensorResource(_spec("state"), _view([[0.0, 0.0]]))
    continuation = mechanisms.TensorResource(
        _continue_spec("continue"), _continue_view(torch.ones(1))
    )
    left_head = arti.as_fabric_node(
        "left_head",
        _DecoratedScale(2.0),
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("left")},
    )
    right_head = arti.as_fabric_node(
        "right_head",
        _DecoratedScale(3.0),
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("right")},
    )
    join = arti.as_fabric_node(
        "join",
        _DecoratedPairSumStop(),
        input_ports={
            "left": mechanisms.ResourcePort("left"),
            "right": mechanisms.ResourcePort("right"),
        },
        output_ports={
            "value": mechanisms.ResourcePort("state"),
            "continue": mechanisms.ResourcePort("continue"),
        },
    )
    loop = mechanisms.ProgramLoop("bounded", "iterate", "continue", max_iterations=3)
    graph = mechanisms.ProgramGraph(
        (source, left, right, state, continuation),
        (),
        nodes=(left_head, right_head, join),
        programs={
            "iterate": (mechanisms.ProgramStage(("left_head", "right_head")), "join")
        },
        loops=(loop,),
    )
    runtime = graph.execute_loop_functional("bounded")
    plan = mechanisms.ResourceGraphCompiler.compile_loop(graph, "bounded")
    inputs = tuple(resource.resolve().view.value for resource in graph.resources.values())
    eager = plan(*inputs)
    compiled = torch.compile(plan, backend="eager", fullgraph=True)(*inputs)
    exported = torch.export.export(plan, inputs).module()(*inputs)
    expected = _view([[5.0, 10.0]]).value
    runtime_state = next(
        item.active_view.value for item in runtime.state.resources if item.spec.resource_id == "state"
    )
    torch.testing.assert_close(runtime_state, expected)
    for result in (eager, compiled, exported):
        torch.testing.assert_close(result[3], expected)
        torch.testing.assert_close(result[4], torch.zeros(1))


def test_static_program_compiles_parallel_two_by_two_cross_layer_dataflow() -> None:
    """Two independent heads may feed two distinct concurrent consumers."""

    source = mechanisms.TensorResource(_spec("source"), _view([[2.0, 3.0]]))
    left = mechanisms.TensorResource(_spec("left"), _view([[0.0, 0.0]]))
    right = mechanisms.TensorResource(_spec("right"), _view([[0.0, 0.0]]))
    sum_output = mechanisms.TensorResource(_spec("sum_output"), _view([[0.0, 0.0]]))
    product_output = mechanisms.TensorResource(
        _spec("product_output"), _view([[0.0, 0.0]])
    )
    left_head = arti.as_fabric_node(
        "left_head",
        _DecoratedScale(2.0),
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("left")},
    )
    right_head = arti.as_fabric_node(
        "right_head",
        _DecoratedScale(3.0),
        input_ports={"source": mechanisms.ResourcePort("source")},
        output_ports={"value": mechanisms.ResourcePort("right")},
    )
    sum_node = arti.as_fabric_node(
        "sum",
        _DecoratedPairSum(),
        input_ports={
            "left": mechanisms.ResourcePort("left"),
            "right": mechanisms.ResourcePort("right"),
        },
        output_ports={"value": mechanisms.ResourcePort("sum_output")},
    )
    product_node = arti.as_fabric_node(
        "product",
        _DecoratedPairProduct(),
        input_ports={
            "left": mechanisms.ResourcePort("left"),
            "right": mechanisms.ResourcePort("right"),
        },
        output_ports={"value": mechanisms.ResourcePort("product_output")},
    )
    graph = mechanisms.ProgramGraph(
        (source, left, right, sum_output, product_output),
        (),
        nodes=(left_head, right_head, sum_node, product_node),
        programs={
            "two_by_two": (
                mechanisms.ProgramStage(("left_head", "right_head")),
                mechanisms.ProgramStage(("sum", "product")),
            )
        },
    )
    runtime = graph.execute_program_functional("two_by_two")
    plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "two_by_two")
    inputs = tuple(resource.resolve().view.value for resource in graph.resources.values())
    eager = plan(*inputs)
    compiled = torch.compile(plan, backend="eager", fullgraph=True)(*inputs)
    exported = torch.export.export(plan, inputs).module()(*inputs)
    expected_sum = _view([[10.0, 15.0]]).value
    expected_product = _view([[24.0, 54.0]]).value
    runtime_values = {item.spec.resource_id: item.active_view.value for item in runtime.state.resources}
    torch.testing.assert_close(runtime_values["sum_output"], expected_sum)
    torch.testing.assert_close(runtime_values["product_output"], expected_product)
    for result in (eager, compiled, exported):
        torch.testing.assert_close(result[3], expected_sum)
        torch.testing.assert_close(result[4], expected_product)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_static_parallel_loop_matches_inductor_gradients_and_cuda_graph() -> None:
    """The parallel loop lowering remains tensor-only on its CUDA fast path."""

    device = torch.device("cuda")

    def resource(resource_id: str, value: torch.Tensor) -> mechanisms.TensorResource:
        return mechanisms.TensorResource(
            _spec(resource_id),
            mechanisms.TensorView.from_tensor(
                value,
                axis_names=("batch", "token", "feature"),
                axis_roles=("batch", "sequence", "feature"),
            ),
        )

    def graph_with_parallel_loop() -> mechanisms.ProgramGraph:
        source = resource("source", torch.zeros(2, 1, 1, device=device))
        left = resource("left", torch.zeros_like(source.resolve().view.value))
        right = resource("right", torch.zeros_like(source.resolve().view.value))
        state = resource("state", torch.zeros_like(source.resolve().view.value))
        continuation = mechanisms.TensorResource(
            _continue_spec("continue"), _continue_view(torch.ones(2, device=device))
        )
        left_head = arti.as_fabric_node(
            "left_head",
            _DecoratedScale(2.0).to(device),
            input_ports={"source": mechanisms.ResourcePort("source")},
            output_ports={"value": mechanisms.ResourcePort("left")},
        )
        right_head = arti.as_fabric_node(
            "right_head",
            _DecoratedScale(3.0).to(device),
            input_ports={"source": mechanisms.ResourcePort("source")},
            output_ports={"value": mechanisms.ResourcePort("right")},
        )
        join = arti.as_fabric_node(
            "join",
            _DecoratedPairSumStop().to(device),
            input_ports={
                "left": mechanisms.ResourcePort("left"),
                "right": mechanisms.ResourcePort("right"),
            },
            output_ports={
                "value": mechanisms.ResourcePort("state"),
                "continue": mechanisms.ResourcePort("continue"),
            },
        )
        return mechanisms.ProgramGraph(
            (source, left, right, state, continuation),
            (),
            nodes=(left_head, right_head, join),
            programs={
                "iterate": (mechanisms.ProgramStage(("left_head", "right_head")), "join")
            },
            loops=(mechanisms.ProgramLoop("bounded", "iterate", "continue", max_iterations=3),),
        )

    prototype = mechanisms.ResourceGraphCompiler.compile_loop(
        graph_with_parallel_loop(), "bounded"
    ).to(device)
    eager_plan = copy.deepcopy(prototype).to(device)
    compiled_plan = copy.deepcopy(prototype).to(device)
    source = torch.randn(2, 1, 1, device=device, requires_grad=True)
    compiled_source = source.detach().clone().requires_grad_(True)
    tail = (
        torch.zeros_like(source),
        torch.zeros_like(source),
        torch.zeros_like(source),
        torch.ones(2, device=device),
    )
    compiled_tail = tuple(value.detach().clone() for value in tail)
    eager_output = eager_plan(source, *tail)[3]
    eager_gradients = torch.autograd.grad(
        eager_output.square().mean(), (source, *eager_plan.parameters())
    )
    compiled_output = torch.compile(compiled_plan, backend="inductor", fullgraph=True)(
        compiled_source, *compiled_tail
    )[3]
    compiled_gradients = torch.autograd.grad(
        compiled_output.square().mean(), (compiled_source, *compiled_plan.parameters())
    )
    torch.testing.assert_close(compiled_output, eager_output)
    for actual, expected in zip(compiled_gradients, eager_gradients, strict=True):
        torch.testing.assert_close(actual, expected)

    inference_plan = copy.deepcopy(prototype).to(device)
    capture_inputs = (
        torch.randn(2, 1, 1, device=device),
        torch.zeros(2, 1, 1, device=device),
        torch.zeros(2, 1, 1, device=device),
        torch.zeros(2, 1, 1, device=device),
        torch.ones(2, device=device),
    )
    captured = inference_plan.capture(*capture_inputs)
    torch.testing.assert_close(captured.replay(*capture_inputs), inference_plan(*capture_inputs))


def test_program_join_waits_for_independent_arrivals_then_consumes_each_once() -> None:
    left_input = mechanisms.TensorResource(_spec("left_input"), _view([[0.0, 0.0]]))
    right_input = mechanisms.TensorResource(_spec("right_input"), _view([[0.0, 0.0]]))
    left = mechanisms.TensorResource(_spec("left"), _view([[0.0, 0.0]]))
    right = mechanisms.TensorResource(_spec("right"), _view([[0.0, 0.0]]))
    output = mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]]))
    graph = mechanisms.ProgramGraph(
        (left_input, right_input, left, right, output),
        (),
        nodes=(
            _OffsetNode("left_write", "left_input", "left", 1.0),
            _OffsetNode("right_write", "right_input", "right", 2.0),
            _PairSumNode(),
        ),
        programs={
            "left": ("left_write",),
            "right": ("right_write",),
            "join": (mechanisms.ProgramJoin("pair_ready", "pair_sum"),),
        },
    )

    left_state = graph.execute_program_functional("left", input_views={"left_input": _view([[3.0, 5.0]])}).state
    waiting = graph.execute_program_functional("join", state=left_state)
    assert waiting.joins[0].fired is False
    assert not waiting.nodes

    right_state = graph.execute_program_functional(
        "right", state=waiting.state, input_views={"right_input": _view([[7.0, 11.0]])}
    ).state
    joined = graph.execute_program_functional("join", state=right_state)
    values = {item.spec.resource_id: item.active_view.value for item in joined.state.resources}
    assert joined.joins[0].fired is True
    torch.testing.assert_close(values["output"], _view([[13.0, 19.0]]).value)

    already_consumed = graph.execute_program_functional("join", state=joined.state)
    assert already_consumed.joins[0].fired is False
    with pytest.raises(mechanisms.ResourceGraphCompileError, match="FormulaProgramNode"):
        mechanisms.ResourceGraphCompiler.compile_program(graph, "join")



def test_formula_program_join_lowers_arrivals_to_tensor_masks() -> None:
    value_type = mechanisms.TensorType.axes(
        ("batch", "token", "feature"), sizes=("B", "S", 1)
    )
    left_binding = mechanisms.InputBinding("left", value_type)
    right_binding = mechanisms.InputBinding("right", value_type)
    program = mechanisms.FormulaProgram.build(outputs=(mechanisms.add(left_binding, right_binding),))
    left = mechanisms.TensorResource(_spec("left"), _view([[1.0], [2.0]]))
    right = mechanisms.TensorResource(_spec("right"), _view([[10.0], [20.0]]))
    output = mechanisms.TensorResource(_spec("output"), _view([[0.0], [0.0]]))
    node = mechanisms.FormulaProgramNode(
        "sum",
        mechanisms.FormulaFabricV2(program),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("output")},
        output_slots={"value": program.outputs[0]},
    )
    graph = mechanisms.ProgramGraph(
        (left, right, output), (), nodes=(node,), programs={"join": (mechanisms.ProgramJoin("ready", "sum"),)}
    )
    plan = mechanisms.ResourceGraphCompiler.compile_join(graph, "ready")
    program_plan = mechanisms.ResourceGraphCompiler.compile_program(graph, "join")
    assert isinstance(program_plan, mechanisms.StaticDataflowProgramExecutionPlan)
    arrivals = torch.tensor([[True, True], [True, False]])
    inputs = (left.resolve().view.value, right.resolve().view.value, output.resolve().view.value, arrivals)
    eager = plan(*inputs)
    whole_program = program_plan(*inputs)
    compiled = torch.compile(plan, backend="eager", fullgraph=True)(*inputs)
    exported = torch.export.export(plan, inputs).module()(*inputs)

    torch.testing.assert_close(eager[2], _view([[11.0], [0.0]]).value)
    torch.testing.assert_close(eager[-2], torch.tensor([[False, False], [True, False]]))
    torch.testing.assert_close(eager[-1], torch.tensor([[True], [False]]))
    for eager_value, whole_value in zip(eager[:-1], whole_program, strict=True):
        torch.testing.assert_close(eager_value, whole_value)
    for eager_value, compiled_value in zip(eager, compiled, strict=True):
        torch.testing.assert_close(eager_value, compiled_value)
    for eager_value, exported_value in zip(eager, exported, strict=True):
        torch.testing.assert_close(eager_value, exported_value)
    credit = program_plan.credit_gradient(
        *inputs,
        terminal_cotangents={"output": torch.ones_like(output.resolve().view.value)},
    )
    torch.testing.assert_close(credit.resource_values[2], eager[2])
    torch.testing.assert_close(credit.arrivals, whole_program[-1])
    torch.testing.assert_close(credit.join_ready[0], torch.tensor([True, False]))
    torch.testing.assert_close(credit.resource_cotangents["left"], _view([[1.0], [0.0]]).value)
    torch.testing.assert_close(credit.resource_cotangents["right"], _view([[1.0], [0.0]]).value)


def test_formula_program_join_credit_lowering_uses_only_fired_rows() -> None:
    value_type = mechanisms.TensorType.axes(
        ("batch", "token", "feature"), sizes=("B", "S", 1)
    )
    left_binding = mechanisms.InputBinding("left", value_type)
    right_binding = mechanisms.InputBinding("right", value_type)
    program = mechanisms.FormulaProgram.build(outputs=(mechanisms.add(left_binding, right_binding),))
    left = mechanisms.TensorResource(_spec("left"), _view([[1.0], [2.0]]))
    right = mechanisms.TensorResource(_spec("right"), _view([[10.0], [20.0]]))
    output = mechanisms.TensorResource(_spec("output"), _view([[0.0], [0.0]]))
    node = mechanisms.FormulaProgramNode(
        "sum",
        mechanisms.FormulaFabricV2(program),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("output")},
        output_slots={"value": program.outputs[0]},
    )
    graph = mechanisms.ProgramGraph(
        (left, right, output), (), nodes=(node,), programs={"join": (mechanisms.ProgramJoin("ready", "sum"),)}
    )
    plan = mechanisms.ResourceGraphCompiler.compile_join(graph, "ready")
    arrivals = torch.tensor([[True, True], [True, False]])
    result = plan.credit_gradient(
        left.resolve().view.value,
        right.resolve().view.value,
        output.resolve().view.value,
        arrivals,
        terminal_cotangents={"output": torch.ones(2, 1, 1)},
    )

    torch.testing.assert_close(result.resource_values[2], _view([[11.0], [0.0]]).value)
    torch.testing.assert_close(result.ready, torch.tensor([True, False]))
    torch.testing.assert_close(result.remaining_arrivals, torch.tensor([[False, False], [True, False]]))
    torch.testing.assert_close(result.publications, torch.tensor([[True], [False]]))
    torch.testing.assert_close(result.resource_cotangents["left"], _view([[1.0], [0.0]]).value)
    torch.testing.assert_close(result.resource_cotangents["right"], _view([[1.0], [0.0]]).value)
    torch.testing.assert_close(result.resource_cotangents["output"], _view([[0.0], [1.0]]).value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for compiled join coverage")
def test_formula_program_join_compiles_and_captures_on_cuda() -> None:
    device = torch.device("cuda")
    value_type = mechanisms.TensorType.axes(
        ("batch", "token", "feature"), sizes=("B", "S", 1)
    )
    left_binding = mechanisms.InputBinding("left", value_type)
    right_binding = mechanisms.InputBinding("right", value_type)
    program = mechanisms.FormulaProgram.build(outputs=(mechanisms.add(left_binding, right_binding),))
    left = mechanisms.TensorResource(_spec("left"), _view([[0.0], [0.0]]))
    right = mechanisms.TensorResource(_spec("right"), _view([[0.0], [0.0]]))
    output = mechanisms.TensorResource(_spec("output"), _view([[0.0], [0.0]]))
    node = mechanisms.FormulaProgramNode(
        "sum",
        mechanisms.FormulaFabricV2(program),
        input_ports={"left": mechanisms.ResourcePort("left"), "right": mechanisms.ResourcePort("right")},
        output_ports={"value": mechanisms.ResourcePort("output")},
        output_slots={"value": program.outputs[0]},
    )
    graph = mechanisms.ProgramGraph(
        (left, right, output), (), nodes=(node,), programs={"join": (mechanisms.ProgramJoin("ready", "sum"),)}
    )
    prototype = mechanisms.ResourceGraphCompiler.compile_join(graph, "ready").to(device)
    eager_plan = copy.deepcopy(prototype).to(device)
    compiled_plan = copy.deepcopy(prototype).to(device)
    eager_left = torch.randn(2, 1, 1, device=device, requires_grad=True)
    eager_right = torch.randn(2, 1, 1, device=device, requires_grad=True)
    compiled_left = eager_left.detach().clone().requires_grad_(True)
    compiled_right = eager_right.detach().clone().requires_grad_(True)
    output_value = torch.zeros_like(eager_left)
    arrivals = torch.tensor([[True, True], [True, False]], device=device)

    eager_output = eager_plan(eager_left, eager_right, output_value, arrivals)[2]
    eager_gradients = torch.autograd.grad(eager_output.square().mean(), (eager_left, eager_right))
    compiled_output = torch.compile(compiled_plan, backend="inductor", fullgraph=True)(
        compiled_left, compiled_right, output_value, arrivals
    )[2]
    compiled_gradients = torch.autograd.grad(compiled_output.square().mean(), (compiled_left, compiled_right))
    torch.testing.assert_close(compiled_output, eager_output)
    for actual, expected in zip(compiled_gradients, eager_gradients, strict=True):
        torch.testing.assert_close(actual, expected)

    capture_plan = copy.deepcopy(prototype).to(device)
    capture_inputs = (
        torch.randn(2, 1, 1, device=device),
        torch.randn(2, 1, 1, device=device),
        torch.zeros(2, 1, 1, device=device),
        arrivals,
    )
    captured = capture_plan.capture(*capture_inputs)
    torch.testing.assert_close(captured.replay(*capture_inputs), capture_plan(*capture_inputs))


def test_formula_program_loop_lowers_its_bounded_masked_horizon() -> None:
    value_type = mechanisms.TensorType.axes(
        ("batch", "token", "feature"), sizes=("B", "S", 1)
    )
    continue_type = mechanisms.TensorType.axes(("batch",), sizes=("B",))
    state_binding = mechanisms.InputBinding("state", value_type)
    continue_binding = mechanisms.InputBinding("continue", continue_type)
    program = mechanisms.FormulaProgram.build(
        outputs=(
            mechanisms.add(state_binding, state_binding),
            mechanisms.add(continue_binding, continue_binding),
        )
    )
    state = mechanisms.TensorResource(_spec("state"), _view([[1.0], [3.0]]))
    continuation = mechanisms.TensorResource(
        _continue_spec("continue"), _continue_view(torch.tensor([1.0, 1.0]))
    )
    node = mechanisms.FormulaProgramNode(
        "iterate",
        mechanisms.FormulaFabricV2(program),
        input_ports={
            "state": mechanisms.ResourcePort("state"),
            "continue": mechanisms.ResourcePort("continue"),
        },
        output_ports={
            "state": mechanisms.ResourcePort("state"),
            "continue": mechanisms.ResourcePort("continue"),
        },
        output_slots={"state": "%0", "continue": "%1"},
    )
    loop = mechanisms.ProgramLoop("bounded", "iterate", "continue", max_iterations=3)
    graph = mechanisms.ProgramGraph(
        (state, continuation),
        (),
        nodes=(node,),
        programs={"iterate": ("iterate",)},
        loops=(loop,),
    )

    runtime = graph.execute_loop_functional("bounded")
    plan = mechanisms.ResourceGraphCompiler.compile_loop(graph, "bounded")
    static = torch.compile(plan, backend="eager", fullgraph=True)(
        state.resolve().view.value, continuation.resolve().view.value
    )
    expected = _view([[8.0], [24.0]]).value

    runtime_state = next(
        item.active_view.value for item in runtime.state.resources if item.spec.resource_id == "state"
    )
    torch.testing.assert_close(runtime_state, expected)
    torch.testing.assert_close(static[0], expected)


def test_formula_program_loop_credit_lowering_uses_the_recorded_active_horizon() -> None:
    value_type = mechanisms.TensorType.axes(
        ("batch", "token", "feature"), sizes=("B", "S", 1)
    )
    continue_type = mechanisms.TensorType.axes(("batch",), sizes=("B",))
    state_binding = mechanisms.InputBinding("state", value_type)
    continue_binding = mechanisms.InputBinding("continue", continue_type)
    program = mechanisms.FormulaProgram.build(
        outputs=(
            mechanisms.add(state_binding, state_binding),
            mechanisms.add(continue_binding, continue_binding),
        )
    )
    state = mechanisms.TensorResource(_spec("state"), _view([[1.0], [3.0]]))
    continuation = mechanisms.TensorResource(
        _continue_spec("continue"), _continue_view(torch.tensor([1.0, -1.0]))
    )
    node = mechanisms.FormulaProgramNode(
        "iterate",
        mechanisms.FormulaFabricV2(program),
        input_ports={
            "state": mechanisms.ResourcePort("state"),
            "continue": mechanisms.ResourcePort("continue"),
        },
        output_ports={
            "state": mechanisms.ResourcePort("state"),
            "continue": mechanisms.ResourcePort("continue"),
        },
        output_slots={"state": "%0", "continue": "%1"},
    )
    loop = mechanisms.ProgramLoop("bounded", "iterate", "continue", max_iterations=3)
    graph = mechanisms.ProgramGraph(
        (state, continuation),
        (),
        nodes=(node,),
        programs={"iterate": ("iterate",)},
        loops=(loop,),
    )
    plan = mechanisms.ResourceGraphCompiler.compile_loop(graph, "bounded")
    result = plan.credit_gradient(
        state.resolve().view.value,
        continuation.resolve().view.value,
        terminal_cotangents={"state": torch.ones(2, 1, 1)},
    )

    torch.testing.assert_close(result.resource_values[0], _view([[8.0], [6.0]]).value)
    assert len(result.iteration_active) == 3
    torch.testing.assert_close(result.iteration_active[0], torch.tensor([True, True]))
    torch.testing.assert_close(result.iteration_active[1], torch.tensor([True, False]))
    torch.testing.assert_close(result.iteration_active[2], torch.tensor([True, False]))
    torch.testing.assert_close(result.resource_cotangents["state"], _view([[8.0], [2.0]]).value)


def test_connection_compiler_keeps_mounted_nodes_as_an_explicit_dynamic_boundary(tmp_path) -> None:
    source = mechanisms.TensorResource(_spec("source"), _view([[1.0, 2.0]]))
    workspace = mechanisms.TensorResource(_spec("workspace"), _view([[0.0, 0.0]]))
    output = mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]]))
    graph = mechanisms.ProgramGraph(
        (source, workspace, output),
        (mechanisms.Connection("copy", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("workspace")),),
        nodes=(_ScaleNode(),),
        programs={"mixed": ("copy", "scale")},
    )

    with pytest.raises(mechanisms.ResourceGraphCompileError, match="ProgramNode"):
        mechanisms.ResourceGraphCompiler.compile_program(graph, "mixed")

    saved = mechanisms.save_program_graph(graph, tmp_path / "program-graph-node")
    with pytest.raises(mechanisms.ResourceGraphError, match="nodes do not match"):
        mechanisms.load_program_graph(saved.tensors_path)


def test_federated_region_mounts_as_a_typed_graph_node() -> None:
    source = mechanisms.TensorResource(
        _vector_spec("source"), _vector_view(torch.randn(1, 2))
    )
    output = mechanisms.TensorResource(
        _vector_spec("output"), _vector_view(torch.zeros(1, 2))
    )
    program = _federated_program()
    node = mechanisms.FederatedProgramNode(
        "federated",
        program,
        input_resource_id="source",
        output_resource_id="output",
    )
    graph = mechanisms.ProgramGraph(
        (source, output), (), nodes=(node,), programs={"run": ("federated",)}
    )

    result = graph.execute_program_functional("run")
    output_state = next(item for item in result.state.resources if item.spec.resource_id == "output")

    assert tuple(item.node_id for item in result.nodes) == ("federated",)
    assert tuple(output_state.active_view.value.shape) == (1, 2)
    assert arti.component_ref(graph).startswith("arti/program-graph@sha256:")
    assert arti.component_ref(node).startswith("arti/federated-program-node@sha256:")
    component_graph = arti.component_graph(graph)
    assert arti.validate_component_graph(component_graph) == component_graph
    assert any(
        item["ref"] and item["ref"].startswith("arti/federated-program-node@sha256:")
        for item in component_graph["nodes"]
    )


def test_terminal_routed_program_mounts_without_a_federation_wrapper() -> None:
    source = mechanisms.TensorResource(_spec("source"), _view([[2.0, 4.0]]))
    output = mechanisms.TensorResource(_spec("output"), _view([[0.0, 0.0]]))
    node = mechanisms.RoutedProgramNode(
        "local",
        _TerminalRoutedProgram(),
        input_resource_id="source",
        output_resource_id="output",
    )
    graph = mechanisms.ProgramGraph(
        (source, output), (), nodes=(node,), programs={"run": ("local",)}
    )

    result = graph.execute_program_functional("run")
    output_state = next(item for item in result.state.resources if item.spec.resource_id == "output")

    torch.testing.assert_close(output_state.active_view.value, _view([[6.0, 12.0]]).value)
    assert tuple(item.node_id for item in result.nodes) == ("local",)
    assert arti.component_ref(node).startswith("arti/routed-program-node@sha256:")


def test_program_graph_artifact_restores_a_supplied_federated_node(tmp_path) -> None:
    program = _federated_program()
    source = mechanisms.TensorResource(
        _vector_spec("source"), _vector_view(torch.randn(1, 2))
    )
    output = mechanisms.TensorResource(
        _vector_spec("output"), _vector_view(torch.zeros(1, 2))
    )
    node = mechanisms.FederatedProgramNode(
        "federated",
        program,
        input_resource_id="source",
        output_resource_id="output",
    )
    graph = mechanisms.ProgramGraph(
        (source, output), (), nodes=(node,), programs={"run": ("federated",)}
    )
    expected = graph.execute_program_functional("run")
    expected_value = next(
        item.active_view.value for item in expected.state.resources if item.spec.resource_id == "output"
    )
    saved = mechanisms.save_program_graph(graph, tmp_path / "federated-node")

    fresh_program = copy.deepcopy(program)
    fresh_node = mechanisms.FederatedProgramNode(
        "federated",
        fresh_program,
        input_resource_id="source",
        output_resource_id="output",
    )
    restored = mechanisms.load_program_graph(saved.tensors_path, nodes=(fresh_node,))
    actual = restored.execute_program_functional("run")
    actual_value = next(
        item.active_view.value for item in actual.state.resources if item.spec.resource_id == "output"
    )

    assert restored.contract_fingerprint == graph.contract_fingerprint
    torch.testing.assert_close(actual_value, expected_value)


def test_artilayer_can_host_a_functional_graph_and_attachment_preserves_its_boundary() -> None:
    source = mechanisms.TensorResource(
        _vector_spec("host_input"), _vector_view(torch.zeros(1, 4))
    )
    output = mechanisms.TensorResource(
        _vector_spec("host_output"), _vector_view(torch.zeros(1, 4))
    )
    graph = mechanisms.ProgramGraph(
        (source, output),
        (mechanisms.Connection("copy", mechanisms.ResourcePort("host_input"), mechanisms.ResourcePort("host_output")),),
        programs={"host": ("copy",)},
    )
    layer = arti.ARTILayer(
        graph=graph,
        graph_program_id="host",
        graph_input_resource_id="host_input",
        graph_output_resource_id="host_output",
        axis_names=("batch", "feature"),
        axis_roles=("batch", "feature"),
    )
    x = torch.randn(1, 4, requires_grad=True)

    value, result = layer(x, return_info=True, return_trace=True)

    torch.testing.assert_close(value, x)
    assert isinstance(result.trace, mechanisms.ProgramGraphExecution)
    assert layer.runtime_provenance()["graph_execution"] is True
    assert layer.runtime_provenance()["functional_graph_state"] is True
    torch.testing.assert_close(graph.resource("host_output").resolve().view.value, torch.zeros(1, 4))
    value.sum().backward()
    assert x.grad is not None
    layer_graph = arti.component_graph(layer)
    assert arti.validate_component_graph(layer_graph) == layer_graph
    assert any(
        item["ref"] and item["ref"].startswith("arti/program-graph@sha256:")
        for item in layer_graph["nodes"]
    )

    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(4))
    attached = arti.ARTI.attach(model, layer, layers="0")
    attached_output = attached(torch.randn(1, 4))
    assert attached_output.shape == (1, 4)
