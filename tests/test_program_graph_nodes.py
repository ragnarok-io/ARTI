from __future__ import annotations

import copy

import pytest
import torch

import arti
from arti import mechanisms
from benchmarks.federal_refine_architecture_search import ArchitectureTaskBank, ExperimentConfig
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


def _federated_program() -> mechanisms.FederatedProgram:
    config = ExperimentConfig(
        tasks=2,
        depths=(1, 2),
        steps=2,
        batch_size=2,
        candidates=2,
        eval_size_per_task=2,
        max_seconds=30.0,
    )
    bank = ArchitectureTaskBank(config, task_id=0, seed=17)
    program = bank.sealed_program(refine_steps=2)
    return mechanisms.FederatedProgram(
        {program.program_id: program},
        terminal_abi=bank.terminal_abi(),
        root_program_ids=(program.program_id,),
        max_levels=1,
        max_k=16,
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
    torch.testing.assert_close(output_state.active_view.value, _view([[6.0, 10.0]]).value)
    torch.testing.assert_close(output.resolve().view.value, _view([[0.0, 0.0]]).value)

    output_state.active_view.value.sum().backward()
    assert node.scale.grad is not None
    assert source_view.value.grad is not None


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
        _vector_spec("source"), _vector_view(torch.randn(1, 16))
    )
    output = mechanisms.TensorResource(
        _vector_spec("output"), _vector_view(torch.zeros(1, 16))
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
    assert tuple(output_state.active_view.value.shape) == (1, 16)
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
        _vector_spec("source"), _vector_view(torch.randn(1, 16))
    )
    output = mechanisms.TensorResource(
        _vector_spec("output"), _vector_view(torch.zeros(1, 16))
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
