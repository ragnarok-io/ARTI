from __future__ import annotations

import torch
from torch import Tensor, nn

import arti
from arti import mechanisms


_OBSERVER_REF = "arti/test-resource-graph-phase-observer@1"


class _ResourceGraphPhaseObserver(mechanisms.TensorViewObserver):
    """Deterministically route once through the resource graph, then exit."""

    _component_reference = _OBSERVER_REF

    @property
    def query_dim(self) -> int:
        return 2

    def contract_config(self) -> dict[str, object]:
        return {"query_dim": self.query_dim}

    def forward(self, view: mechanisms.TensorView) -> mechanisms.TensorViewObservation:
        batch = view.value.shape[view.batch_axis]
        advanced = view.value.reshape(batch, -1).mean(dim=-1) > 0.5
        tokens = view.value.new_zeros((batch, 1, self.query_dim))
        tokens[:, :, 0] = torch.where(advanced, 0.0, 10.0).unsqueeze(1)
        tokens[:, :, 1] = torch.where(advanced, 10.0, 0.0).unsqueeze(1)
        return mechanisms.TensorViewObservation(
            tokens,
            torch.ones((batch, 1), dtype=torch.bool, device=view.value.device),
            view.descriptor_fingerprint,
        )


arti.register_component(
    _OBSERVER_REF,
    component_type=_ResourceGraphPhaseObserver,
    lifecycle="alpha",
    constructible=False,
    config_builder=lambda component: component.contract_config(),
)


def _view(values: list[float]) -> mechanisms.TensorView:
    return mechanisms.TensorView.from_tensor(
        torch.tensor([values], dtype=torch.float32),
        axis_names=("batch", "token"),
        axis_roles=("batch", "sequence"),
    )


class _HalfGate(nn.Module):
    def forward(self, context: Tensor) -> Tensor:
        return torch.full_like(context, 0.5)


def _pattern() -> mechanisms.TensorViewPattern:
    return mechanisms.TensorViewPattern(
        min_rank=2,
        max_rank=2,
        allowed_axis_roles=("batch", "sequence"),
    )


def _schema() -> mechanisms.TensorSchema:
    return mechanisms.TensorSchema(
        dtype="float32",
        device_class="any",
        dimensions=("B", 2),
        semantic_axes=("batch", "token"),
        mask_semantics="none",
    )


def _resource(resource_id: str, *, lifetime: mechanisms.ResourceLifetime) -> mechanisms.TensorResource:
    return mechanisms.TensorResource(
        mechanisms.TensorResourceSpec(resource_id, _pattern(), lifetime=lifetime),
        _view([0.0, 0.0]),
    )


def _runtime(
    *,
    max_k: int = 1,
    program_type: type[nn.Module] = mechanisms.RoutedProgram,
    federation_type: type[nn.Module] = mechanisms.FederatedProgram,
) -> tuple[nn.Module, mechanisms.ProgramGraph, mechanisms.TensorViewResourceGraphAction]:
    graph = mechanisms.ProgramGraph(
        (
            _resource("world", lifetime=mechanisms.ResourceLifetime.CALL),
            _resource("memory", lifetime=mechanisms.ResourceLifetime.PERSISTENT),
            _resource("output", lifetime=mechanisms.ResourceLifetime.CALL),
        ),
        (
            mechanisms.Connection(
                "write_memory",
                mechanisms.ResourcePort("world"),
                mechanisms.ResourcePort("memory"),
            ),
            mechanisms.Connection(
                "emit_output",
                mechanisms.ResourcePort("memory"),
                mechanisms.ResourcePort("output"),
                depends_on=("write_memory",),
                transfer=mechanisms.LearnableAffineTransfer(
                    gain=1.0, bias=1.0, learnable=False
                ),
            ),
        ),
        programs={"write_then_emit": ("write_memory", "emit_output")},
    )
    action = mechanisms.TensorViewResourceGraphAction(
        action_id="write-and-return",
        graph=graph,
        connection_ids=("write_memory", "emit_output"),
        input_resource_id="world",
        output_resource_id="output",
    )
    abi = mechanisms.TerminalOutputABI(
        fields=(
            mechanisms.TerminalField("value", _schema(), "terminal-value"),
            mechanisms.TerminalField(
                "validity",
                mechanisms.TensorSchema(
                    dtype="boolean",
                    device_class="any",
                    dimensions=("B",),
                    semantic_axes=("batch",),
                    mask_semantics="boolean-validity",
                ),
                "terminal-validity",
            ),
            mechanisms.TerminalField(
                "score",
                mechanisms.TensorSchema(
                    dtype="float32",
                    device_class="any",
                    dimensions=("B",),
                    semantic_axes=("batch",),
                    mask_semantics="none",
                ),
                "terminal-score",
            ),
        ),
        factor_order=(),
        validity_contract="one validity value per row",
        packing_contract="named terminal tensors",
        score_contract="one terminal score per row",
        consumer_contract="hard one winner",
        gradient_contract=mechanisms.GradientContract.autograd(),
    )
    query = mechanisms.seal_tensor_view_bank_query(
        mechanisms.TensorViewBankQuery(
            pattern=_pattern(),
            observer=_ResourceGraphPhaseObserver(),
            matcher=mechanisms.BankMemberMatcher(
                torch.eye(2), member_ids=("write-and-return", "exit")
            ),
        )
    )
    program = program_type(
        program_id="resource-workshop",
        query=query,
        actions=(action,),
        terminal_action=mechanisms.ProgramTerminalAction("exit", input_schema=_schema()),
        local_refine=mechanisms.ProgramExecutionPolicy(min_steps=2, max_steps=2),
        input_pattern=_pattern(),
        exit_pattern=_pattern(),
        terminal_abi=abi,
    )
    return (
        federation_type(
            {program.program_id: program},
            terminal_abi=abi,
            root_program_ids=(program.program_id,),
            max_levels=1,
            max_k=max_k,
        ),
        graph,
        action,
    )


def test_role_oriented_program_names_preserve_existing_runtime_contracts() -> None:
    runtime, _graph, _action = _runtime(
        program_type=mechanisms.RoutedProgram,
        federation_type=mechanisms.FederatedProgram,
    )

    assert isinstance(runtime, mechanisms.FederatedProgram)
    program = runtime.programs["resource-workshop"]
    assert isinstance(program, mechanisms.RoutedProgram)
    assert arti.component_ref(program).startswith("arti/routed-program@sha256:")
    assert arti.component_ref(runtime).startswith("arti/federated-program@sha256:")

    output, trace = runtime(_view([0.0, 0.0]), return_trace=True)

    torch.testing.assert_close(output["value"], _view([1.0, 1.0]).value)
    assert trace.winner_paths == ("resource-workshop/exit",)


def test_resource_graph_action_is_functional_until_its_terminal_path_wins() -> None:
    runtime, graph, action = _runtime()
    source = _view([0.0, 0.0])

    direct = action(source)

    torch.testing.assert_close(direct.value, _view([1.0, 1.0]).value)
    torch.testing.assert_close(graph.resource("memory").resolve().view.value, _view([0.0, 0.0]).value)
    torch.testing.assert_close(graph.resource("output").resolve().view.value, _view([0.0, 0.0]).value)

    output, trace = runtime(source, return_trace=True)

    torch.testing.assert_close(output["value"], _view([1.0, 1.0]).value)
    torch.testing.assert_close(graph.resource("memory").resolve().view.value, source.value)
    torch.testing.assert_close(graph.resource("output").resolve().view.value, _view([1.0, 1.0]).value)
    assert [item.candidate_id for item in trace.steps[0].local_refine] == [
        "write-and-return",
        "exit",
    ]
    assert "winner-committed-resource-graphs" in runtime.programs["resource-workshop"].signature.execution_capabilities
    assert len(trace.committed_resources) == 1
    receipt = trace.committed_resources[0]
    assert receipt.connection_ids == ("write_memory", "emit_output")
    assert set(receipt.resource_ids) == {"world", "memory", "output"}
    assert receipt.action_id == "write-and-return"
    assert receipt.winner_path.endswith("exit")


def test_resource_graph_action_component_contract_captures_the_declared_subprogram() -> None:
    _, graph, action = _runtime()

    config = action.contract_config()

    assert config["graph_contract_fingerprint"] == graph.contract_fingerprint
    assert config["connection_ids"] == ["write_memory", "emit_output"]
    assert arti.component_ref(action).startswith(
        "arti/tensor-view-resource-graph-action@sha256:"
    )


def test_k_wide_resource_candidates_do_not_publish_the_discarded_branch_state() -> None:
    runtime, graph, _ = _runtime(max_k=2)
    source = _view([0.0, 0.0])

    output, trace = runtime(source, max_k=2, return_trace=True)

    # At the second local step the retained action branch would emit [2, 2],
    # while the terminal branch keeps the first action's [1, 1] state.
    torch.testing.assert_close(output["value"], _view([1.0, 1.0]).value)
    torch.testing.assert_close(graph.resource("memory").resolve().view.value, source.value)
    torch.testing.assert_close(graph.resource("output").resolve().view.value, _view([1.0, 1.0]).value)
    assert trace.maximum_kept_paths == 1


def test_resource_action_can_bind_a_connection_to_its_local_input_without_a_global_query() -> None:
    graph = mechanisms.ProgramGraph(
        (
            _resource("world", lifetime=mechanisms.ResourceLifetime.CALL),
            _resource("output", lifetime=mechanisms.ResourceLifetime.CALL),
        ),
        (
            mechanisms.Connection(
                "local_gate",
                mechanisms.ResourcePort("world"),
                mechanisms.ResourcePort("output"),
                activation=_HalfGate(),
            ),
        ),
    )
    action = mechanisms.TensorViewResourceGraphAction(
        action_id="local-gate",
        graph=graph,
        connection_ids=("local_gate",),
        input_resource_id="world",
        output_resource_id="output",
        use_input_context=True,
    )

    output = action(_view([2.0, 6.0]))

    torch.testing.assert_close(output.value, _view([1.0, 3.0]).value)
    torch.testing.assert_close(graph.resource("output").resolve().view.value, _view([0.0, 0.0]).value)
    assert action.contract_config()["connection_context"] == "current-input-view"


def test_resource_action_can_call_a_declared_graph_subprogram_with_functional_repetition() -> None:
    _, graph, _ = _runtime()
    action = mechanisms.TensorViewResourceGraphAction(
        action_id="repeat-subprogram",
        graph=graph,
        program_id="write_then_emit",
        input_resource_id="world",
        output_resource_id="output",
        iterations=2,
    )

    output = action(_view([0.0, 0.0]))
    invocation = graph.invoke_functional(
        graph.program("write_then_emit"),
        input_resource_id="world",
        input_view=_view([0.0, 0.0]),
        output_resource_id="output",
        iterations=2,
    )

    torch.testing.assert_close(output.value, _view([1.0, 1.0]).value)
    output_state = next(
        item for item in invocation.state.resources if item.spec.resource_id == "output"
    )
    assert output_state.step_index == 2
    assert action.contract_config()["program_id"] == "write_then_emit"


def test_final_loss_trains_a_direct_resource_connection_without_a_route_query() -> None:
    transfer = mechanisms.LearnableAffineTransfer(gain=0.0, bias=0.0)
    graph = mechanisms.ProgramGraph(
        (
            _resource("world", lifetime=mechanisms.ResourceLifetime.CALL),
            _resource("output", lifetime=mechanisms.ResourceLifetime.CALL),
        ),
        (
            mechanisms.Connection(
                "learned_direct",
                mechanisms.ResourcePort("world"),
                mechanisms.ResourcePort("output"),
                transfer=transfer,
            ),
        ),
    )
    action = mechanisms.TensorViewResourceGraphAction(
        action_id="learned-direct",
        graph=graph,
        connection_ids=("learned_direct",),
        input_resource_id="world",
        output_resource_id="output",
    )
    source = _view([1.0, 2.0])
    target = source.value * 3.0
    optimizer = torch.optim.Adam(action.parameters(), lr=0.1)
    initial = torch.nn.functional.mse_loss(action(source).value, target).detach()

    for _ in range(180):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(action(source).value, target)
        loss.backward()
        optimizer.step()

    final = torch.nn.functional.mse_loss(action(source).value, target).detach()
    assert final < initial * 0.01
    torch.testing.assert_close(graph.resource("output").resolve().view.value, _view([0.0, 0.0]).value)


def test_direct_resource_connection_executes_before_candidate_query() -> None:
    template, graph, _ = _runtime()
    direct = mechanisms.TensorViewResourceGraphAction(
        action_id="direct-link",
        graph=graph,
        connection_ids=("write_memory", "emit_output"),
        input_resource_id="world",
        output_resource_id="output",
    )
    unused = mechanisms.TensorViewResourceGraphAction(
        action_id="unused-candidate",
        graph=graph,
        connection_ids=("write_memory", "emit_output"),
        input_resource_id="world",
        output_resource_id="output",
    )
    query = mechanisms.seal_tensor_view_bank_query(
        mechanisms.TensorViewBankQuery(
            pattern=_pattern(),
            observer=_ResourceGraphPhaseObserver(),
            matcher=mechanisms.BankMemberMatcher(
                torch.eye(2), member_ids=("unused-candidate", "exit")
            ),
        )
    )
    program = mechanisms.RoutedProgram(
        program_id="direct-workshop",
        query=query,
        actions=(unused,),
        direct_actions=(direct,),
        terminal_action=mechanisms.ProgramTerminalAction("exit", input_schema=_schema()),
        local_refine=mechanisms.ProgramExecutionPolicy(min_steps=1, max_steps=1),
        input_pattern=_pattern(),
        exit_pattern=_pattern(),
        terminal_abi=template.terminal_abi,
    )
    runtime = mechanisms.FederatedProgram(
        {program.program_id: program},
        terminal_abi=template.terminal_abi,
        root_program_ids=(program.program_id,),
        max_levels=1,
        max_k=1,
    )

    output, trace = runtime(_view([0.0, 0.0]), return_trace=True)

    torch.testing.assert_close(output["value"], _view([1.0, 1.0]).value)
    torch.testing.assert_close(graph.resource("output").resolve().view.value, _view([1.0, 1.0]).value)
    assert [item.candidate_id for item in trace.steps[0].local_refine] == ["exit"]
    assert "direct-resource-connections" in program.signature.execution_capabilities
