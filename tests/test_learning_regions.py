from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from arti import (
    CreditBoundary,
    CreditBoundaryMode,
    CreditGradientField,
    CreditStructureCandidate,
    CreditStructureChoice,
    CreditEdge,
    LearningRegionError,
    OptimizerDomain,
    OptimizerDomainStatus,
    OptimizerExecutionPlan,
    ParameterOwnershipTable,
    ParameterUse,
    PairedCreditUpdate,
    ProgramParameterBinding,
    credit_gradient_field,
    derive_learning_regions,
    derive_learning_regions_from_program_graph,
    program_graph_credit_edges,
)
from arti import mechanisms


def _uses(*items: tuple[str, str, nn.Parameter]) -> tuple[ParameterUse, ...]:
    return tuple(ParameterUse(parameter_id, region_id, parameter) for parameter_id, region_id, parameter in items)


def _plan(
    parameters: dict[str, nn.Parameter],
    uses: tuple[ParameterUse, ...],
    domains: tuple[OptimizerDomain, ...],
) -> OptimizerExecutionPlan:
    return OptimizerExecutionPlan(parameters, ParameterOwnershipTable.from_uses(uses), domains)


def _paired_losses(direct: float, boundary: float) -> PairedCreditUpdate:
    empty_field = CreditGradientField((), (), ())
    return PairedCreditUpdate(
        direct_gradients=empty_field,
        boundary_gradients=empty_field,
        direct_parameters={},
        boundary_parameters={},
        direct_query_loss=torch.tensor(direct),
        boundary_query_loss=torch.tensor(boundary),
    )


def test_sealed_credit_edge_preserves_a_bypass_component() -> None:
    regions = derive_learning_regions(
        ("a", "b", "c"),
        (
            CreditEdge("a-to-b", "a", "b", sealed=True),
            CreditEdge("a-to-c", "a", "c"),
            CreditEdge("c-to-b", "c", "b"),
        ),
        learning_sources={"a": ("task",)},
    )

    assert len(regions) == 1
    assert regions[0].node_ids == ("a", "b", "c")
    assert regions[0].learning_sources == ("task",)


def test_graph_trains_two_sealed_regions_from_independent_losses() -> None:
    pattern = mechanisms.TensorViewPattern(
        min_rank=3,
        max_rank=3,
        allowed_axis_roles=("batch", "sequence", "feature"),
    )
    source = torch.tensor([[[0.5], [1.5]]])
    resources = tuple(
        mechanisms.TensorResource(
            mechanisms.TensorResourceSpec(resource_id, pattern),
            mechanisms.TensorView.from_tensor(
                source if resource_id == "source" else torch.zeros_like(source),
                axis_names=("batch", "token", "feature"),
                axis_roles=("batch", "sequence", "feature"),
            ),
        )
        for resource_id in ("source", "scores", "answer_input", "adapted")
    )
    score_transfer = mechanisms.LearnableAffineTransfer(gain=0.5)
    answer_transfer = mechanisms.LearnableAffineTransfer(gain=1.5)
    graph = mechanisms.ProgramGraph(
        resources,
        (
            mechanisms.Connection(
                "score", mechanisms.ResourcePort("source"), mechanisms.ResourcePort("scores"),
                transfer=score_transfer,
            ),
            mechanisms.Connection(
                "credit_gate", mechanisms.ResourcePort("scores"),
                mechanisms.ResourcePort("answer_input"),
                credit_boundary=CreditBoundary(mode=CreditBoundaryMode.CLOSED),
            ),
            mechanisms.Connection(
                "answer", mechanisms.ResourcePort("answer_input"),
                mechanisms.ResourcePort("adapted"), transfer=answer_transfer,
            ),
        ),
        programs={"adapt": ("score", "credit_gate", "answer")},
    )
    score_weight = score_transfer.gain
    join_mix = answer_transfer.gain
    plan = OptimizerExecutionPlan.from_program_graph(
        graph,
        (
            ProgramParameterBinding("score_weight", "score", score_weight),
            ProgramParameterBinding("join_mix", "answer", join_mix),
        ),
        learning_sources={"score": ("route",), "answer": ("answer",)},
        learning_rate=0.01,
    )
    score_region = plan.ownership.owner_for("score_weight").owner_domain_id
    answer_region = plan.ownership.owner_for("join_mix").owner_domain_id
    assert score_region != answer_region

    execution = graph.execute_program_functional("adapt")
    outputs = {item.spec.resource_id: item.active_view.value for item in execution.state.resources}
    before_score = score_weight.detach().clone()
    before_join = join_mix.detach().clone()
    receipt = plan.step_with_losses(local_objectives={
        score_region: (outputs["scores"] - 1).square().mean(),
        answer_region: (outputs["adapted"] - 3).square().mean(),
    })
    assert receipt.updated_parameter_ids == ("join_mix", "score_weight")
    assert not torch.equal(score_weight, before_score)
    assert not torch.equal(join_mix, before_join)


def test_program_graph_projection_derives_regions_from_declared_resource_flow() -> None:
    pattern = mechanisms.TensorViewPattern(
        min_rank=3,
        max_rank=3,
        allowed_axis_roles=("batch", "sequence", "feature"),
    )
    def resource(resource_id: str) -> mechanisms.TensorResource:
        return mechanisms.TensorResource(
            mechanisms.TensorResourceSpec(resource_id, pattern),
            mechanisms.TensorView.from_tensor(
                torch.zeros(1, 1, 1),
                axis_names=("batch", "token", "feature"),
                axis_roles=("batch", "sequence", "feature"),
            ),
        )

    graph = mechanisms.ProgramGraph(
        (resource("input"), resource("middle"), resource("output")),
        (
            mechanisms.Connection("encode", mechanisms.ResourcePort("input"), mechanisms.ResourcePort("middle")),
            mechanisms.Connection("decode", mechanisms.ResourcePort("middle"), mechanisms.ResourcePort("output")),
        ),
    )
    edges = program_graph_credit_edges(graph)

    assert [edge.edge_id for edge in edges] == ["credit:encode->decode:middle"]
    open_regions = derive_learning_regions_from_program_graph(graph, learning_sources={"encode": ("task",)})
    sealed_regions = derive_learning_regions_from_program_graph(
        graph,
        sealed_credit_edge_ids=(edges[0].edge_id,),
        learning_sources={"encode": ("task",)},
    )

    assert len(open_regions) == 1
    assert open_regions[0].node_ids == ("decode", "encode")
    assert [region.node_ids for region in sealed_regions] == [("decode",), ("encode",)]
    assert [region.is_learnable for region in sealed_regions] == [False, True]


def test_closed_connection_boundary_commits_its_projected_credit_edge() -> None:
    pattern = mechanisms.TensorViewPattern(
        min_rank=3,
        max_rank=3,
        allowed_axis_roles=("batch", "sequence", "feature"),
    )

    def resource(resource_id: str) -> mechanisms.TensorResource:
        return mechanisms.TensorResource(
            mechanisms.TensorResourceSpec(resource_id, pattern),
            mechanisms.TensorView.from_tensor(
                torch.zeros(1, 1, 1),
                axis_names=("batch", "token", "feature"),
                axis_roles=("batch", "sequence", "feature"),
            ),
        )

    graph = mechanisms.ProgramGraph(
        (resource("input"), resource("middle"), resource("output")),
        (
            mechanisms.Connection("encode", mechanisms.ResourcePort("input"), mechanisms.ResourcePort("middle")),
            mechanisms.Connection(
                "decode",
                mechanisms.ResourcePort("middle"),
                mechanisms.ResourcePort("output"),
                credit_boundary=CreditBoundary(mode=CreditBoundaryMode.CLOSED),
            ),
        ),
    )

    edges = program_graph_credit_edges(graph)
    assert [(edge.edge_id, edge.sealed) for edge in edges] == [
        ("credit:encode->decode:middle", True)
    ]
    regions = derive_learning_regions_from_program_graph(graph, learning_sources={"encode": ("task",)})
    assert [region.node_ids for region in regions] == [("decode",), ("encode",)]


def test_graph_structure_commit_derives_domains_and_retains_parameter_state() -> None:
    pattern = mechanisms.TensorViewPattern(
        min_rank=3,
        max_rank=3,
        allowed_axis_roles=("batch", "sequence", "feature"),
    )

    def resource(resource_id: str) -> mechanisms.TensorResource:
        return mechanisms.TensorResource(
            mechanisms.TensorResourceSpec(resource_id, pattern),
            mechanisms.TensorView.from_tensor(
                torch.zeros(1, 1, 1),
                axis_names=("batch", "token", "feature"),
                axis_roles=("batch", "sequence", "feature"),
            ),
        )

    boundary = CreditBoundary(mode=CreditBoundaryMode.MEAN)
    graph = mechanisms.ProgramGraph(
        (resource("input"), resource("middle"), resource("output")),
        (
            mechanisms.Connection("encode", mechanisms.ResourcePort("input"), mechanisms.ResourcePort("middle")),
            mechanisms.Connection(
                "decode",
                mechanisms.ResourcePort("middle"),
                mechanisms.ResourcePort("output"),
                credit_boundary=boundary,
            ),
        ),
    )
    encode = nn.Parameter(torch.tensor([1.0]))
    decode = nn.Parameter(torch.tensor([2.0]))
    shared = nn.Parameter(torch.tensor([3.0]))
    bindings = (
        ProgramParameterBinding("encode_weight", "encode", encode),
        ProgramParameterBinding("decode_weight", "decode", decode),
        ProgramParameterBinding("shared_weight", "encode", shared),
        ProgramParameterBinding("shared_weight", "decode", shared),
    )
    plan = OptimizerExecutionPlan.from_program_graph(
        graph,
        bindings,
        learning_sources={"encode": ("task",)},
        learning_rate=0.1,
    )
    assert len(plan.domains) == 1
    for parameter in (encode, decode, shared):
        parameter.grad = torch.ones_like(parameter)
    plan.step()
    state_before = {
        parameter_id: entry.exp_avg.detach().clone()
        for parameter_id, entry in plan.state_arena._entries.items()
    }

    boundary.mode = CreditBoundaryMode.CLOSED
    next_plan, receipt = plan.commit_program_graph_structure(
        graph,
        bindings,
        learning_sources={"encode": ("task",)},
    )

    assert receipt.generation == 1
    assert receipt.retained_parameter_ids == ("decode_weight", "encode_weight", "shared_weight")
    assert next_plan.ownership.owner_for("shared_weight").owner_domain_id == "shared"
    assert next_plan.domains["shared"].status is OptimizerDomainStatus.ACTIVE
    assert next_plan.ownership.owner_for("decode_weight").owner_domain_id != "shared"
    for parameter_id, expected in state_before.items():
        torch.testing.assert_close(next_plan.state_arena._entries[parameter_id].exp_avg, expected)

    for parameter in (encode, decode, shared):
        parameter.grad = torch.ones_like(parameter)
    update = next_plan.step()
    assert update.updated_parameter_ids == ("encode_weight", "shared_weight")
    assert update.skipped_parameter_ids == ("decode_weight",)


def test_credit_structure_window_commits_paired_outcome_and_sealed_domain_split() -> None:
    pattern = mechanisms.TensorViewPattern(
        min_rank=3,
        max_rank=3,
        allowed_axis_roles=("batch", "sequence", "feature"),
    )

    def resource(resource_id: str) -> mechanisms.TensorResource:
        return mechanisms.TensorResource(
            mechanisms.TensorResourceSpec(resource_id, pattern),
            mechanisms.TensorView.from_tensor(
                torch.zeros(1, 1, 1),
                axis_names=("batch", "token", "feature"),
                axis_roles=("batch", "sequence", "feature"),
            ),
        )

    boundary = CreditBoundary(mode=CreditBoundaryMode.MEAN)
    graph = mechanisms.ProgramGraph(
        (resource("input"), resource("middle"), resource("output")),
        (
            mechanisms.Connection("encode", mechanisms.ResourcePort("input"), mechanisms.ResourcePort("middle")),
            mechanisms.Connection(
                "decode",
                mechanisms.ResourcePort("middle"),
                mechanisms.ResourcePort("output"),
                credit_boundary=boundary,
            ),
        ),
    )
    encode = nn.Parameter(torch.tensor([1.0]))
    decode = nn.Parameter(torch.tensor([2.0]))
    bindings = (
        ProgramParameterBinding("encode_weight", "encode", encode),
        ProgramParameterBinding("decode_weight", "decode", decode),
    )
    plan = OptimizerExecutionPlan.from_program_graph(
        graph,
        bindings,
        learning_sources={"encode": ("task",)},
        learning_rate=0.1,
    )

    next_plan, receipt = plan.commit_credit_structure_window(
        graph,
        bindings,
        (
            CreditStructureCandidate(
                "decode",
                boundary,
                CreditStructureChoice(),
                _paired_losses(direct=2.0, boundary=1.0),
                boundary_mode=CreditBoundaryMode.CLOSED,
                seal_on_boundary=True,
            ),
        ),
        learning_sources={"encode": ("task",)},
    )

    assert boundary.mode is CreditBoundaryMode.CLOSED
    assert receipt.selected_boundary_connection_ids == ("decode",)
    assert receipt.sealed_connection_ids == ("decode",)
    assert receipt.structure_commit.generation == 1
    assert next_plan.domains[next_plan.ownership.owner_for("encode_weight").owner_domain_id].is_active
    assert not next_plan.domains[next_plan.ownership.owner_for("decode_weight").owner_domain_id].is_active


def test_structure_commit_accepts_independent_new_domain_configurations() -> None:
    pattern = mechanisms.TensorViewPattern(
        min_rank=3,
        max_rank=3,
        allowed_axis_roles=("batch", "sequence", "feature"),
    )

    def resource(resource_id: str) -> mechanisms.TensorResource:
        return mechanisms.TensorResource(
            mechanisms.TensorResourceSpec(resource_id, pattern),
            mechanisms.TensorView.from_tensor(
                torch.zeros(1, 1, 1),
                axis_names=("batch", "token", "feature"),
                axis_roles=("batch", "sequence", "feature"),
            ),
        )

    boundary = CreditBoundary(mode=CreditBoundaryMode.CLOSED)
    graph = mechanisms.ProgramGraph(
        (resource("input"), resource("middle"), resource("output")),
        (
            mechanisms.Connection("encode", mechanisms.ResourcePort("input"), mechanisms.ResourcePort("middle")),
            mechanisms.Connection(
                "decode",
                mechanisms.ResourcePort("middle"),
                mechanisms.ResourcePort("output"),
                credit_boundary=boundary,
            ),
        ),
    )
    encode = nn.Parameter(torch.tensor([1.0]))
    decode = nn.Parameter(torch.tensor([2.0]))
    bindings = (
        ProgramParameterBinding("encode_weight", "encode", encode),
        ProgramParameterBinding("decode_weight", "decode", decode),
    )
    plan = _plan(
        {"encode_weight": encode, "decode_weight": decode},
        _uses(("encode_weight", "before_encode", encode), ("decode_weight", "before_decode", decode)),
        (
            OptimizerDomain("before_encode", learning_rate=0.01),
            OptimizerDomain("before_decode", learning_rate=0.03),
        ),
    )
    regions = derive_learning_regions_from_program_graph(
        graph,
        learning_sources={"encode": ("task",)},
    )
    overrides = {
        region.region_id: OptimizerDomain(
            region.region_id,
            learning_rate=0.02 if "encode" in region.node_ids else 0.04,
            weight_decay=0.1 if "encode" in region.node_ids else 0.0,
            status=OptimizerDomainStatus.ACTIVE if region.is_learnable else OptimizerDomainStatus.DORMANT,
        )
        for region in regions
    }

    next_plan, _ = plan.commit_program_graph_structure(
        graph,
        bindings,
        learning_sources={"encode": ("task",)},
        domain_overrides=overrides,
    )

    encode_domain = next_plan.domains[next_plan.ownership.owner_for("encode_weight").owner_domain_id]
    decode_domain = next_plan.domains[next_plan.ownership.owner_for("decode_weight").owner_domain_id]
    assert encode_domain.learning_rate == 0.02
    assert encode_domain.weight_decay == 0.1
    assert decode_domain.learning_rate == 0.04
    assert decode_domain.status is OptimizerDomainStatus.DORMANT


def test_shared_parameter_has_one_explicit_owner_domain() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    ownership = ParameterOwnershipTable.from_uses(
        _uses(("shared.weight", "left", parameter), ("shared.weight", "right", parameter))
    )

    entry = ownership.owner_for("shared.weight")
    assert entry.shared
    assert entry.owner_domain_id == "shared"
    assert entry.use_region_ids == ("left", "right")


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA is required",
    ))],
)
def test_region_losses_project_credit_without_cross_objective_leakage(device: str) -> None:
    left = nn.Parameter(torch.tensor([1.0], device=device))
    right = nn.Parameter(torch.tensor([2.0], device=device))
    shared = nn.Parameter(torch.tensor([0.5], device=device))
    plan = _plan(
        {"left": left, "right": right, "shared": shared},
        _uses(
            ("left", "left_region", left),
            ("right", "right_region", right),
            ("shared", "left_region", shared),
            ("shared", "right_region", shared),
        ),
        (
            OptimizerDomain("left_region", learning_rate=0.1),
            OptimizerDomain("right_region", learning_rate=0.1),
            OptimizerDomain("shared", learning_rate=0.1),
        ),
    )
    left_value = left + shared
    right_value = right * left_value
    task_loss = (right_value - 5.0).square().sum()
    left_loss = (left_value - 3.0).square().sum()
    right_loss = (right_value + 1.0).square().sum()

    expected_task = torch.autograd.grad(task_loss, (left, right, shared), retain_graph=True)
    expected_left = torch.autograd.grad(left_loss, (left, shared), retain_graph=True)
    expected_right = torch.autograd.grad(right_loss, (right, shared), retain_graph=True)
    leaked_left = torch.autograd.grad(right_loss, left, retain_graph=True)[0]
    gradients = plan.gradients_from_losses(
        task_loss,
        local_objectives={"left_region": left_loss, "right_region": right_loss},
    )

    torch.testing.assert_close(gradients["left"], expected_task[0] + expected_left[0])
    torch.testing.assert_close(gradients["right"], expected_task[1] + expected_right[0])
    torch.testing.assert_close(
        gradients["shared"], expected_task[2] + expected_left[1] + expected_right[1]
    )
    assert not torch.equal(gradients["left"], expected_task[0] + expected_left[0] + leaked_left)


def test_distinct_region_objectives_train_task_and_exit_in_one_step() -> None:
    pattern = mechanisms.TensorViewPattern(
        min_rank=3, max_rank=3, allowed_axis_roles=("batch", "sequence", "feature"),
    )

    def resource(resource_id: str) -> mechanisms.TensorResource:
        return mechanisms.TensorResource(
            mechanisms.TensorResourceSpec(resource_id, pattern),
            mechanisms.TensorView.from_tensor(
                torch.ones(1, 1, 1) if resource_id == "input" else torch.zeros(1, 1, 1),
                axis_names=("batch", "token", "feature"),
                axis_roles=("batch", "sequence", "feature"),
            ),
        )

    graph = mechanisms.ProgramGraph(
        (resource("input"), resource("task_output"), resource("exit_output")),
        (
            mechanisms.Connection(
                "task", mechanisms.ResourcePort("input"), mechanisms.ResourcePort("task_output"),
                transfer=mechanisms.LearnableAffineTransfer(gain=0.0),
            ),
            mechanisms.Connection(
                "exit", mechanisms.ResourcePort("input"), mechanisms.ResourcePort("exit_output"),
                transfer=mechanisms.LearnableAffineTransfer(gain=-2.0),
            ),
        ),
        programs={"run": ("task", "exit")},
    )
    task_weight = graph.connections["task"].transfer.gain
    exit_logit = graph.connections["exit"].transfer.gain
    graph.connections["task"].transfer.bias.requires_grad_(False)
    graph.connections["exit"].transfer.bias.requires_grad_(False)
    exit_atom = mechanisms.FormulaRefineExit(input_kind="logit")
    exit_mask = torch.ones(1, 1, dtype=torch.bool)
    assert not bool(exit_atom(exit_logit.reshape(1, 1), mask=exit_mask).requested.item())
    plan = OptimizerExecutionPlan.from_program_graph(
        graph,
        (
            ProgramParameterBinding("task_weight", "task", task_weight),
            ProgramParameterBinding("exit_logit", "exit", exit_logit),
        ),
        learning_sources={"task": ("answer",), "exit": ("stop",)},
        learning_rate=0.3,
    )
    task_region = plan.ownership.owner_for("task_weight").owner_domain_id
    exit_region = plan.ownership.owner_for("exit_logit").owner_domain_id
    assert task_region != exit_region

    for _ in range(20):
        execution = graph.execute_program_functional("run")
        outputs = {item.spec.resource_id: item.active_view.value for item in execution.state.resources}
        task_loss = (outputs["task_output"] - 2.0).square().sum()
        exit_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["exit_output"], torch.ones_like(outputs["exit_output"]),
        )
        receipt = plan.step_with_losses(local_objectives={
            task_region: task_loss, exit_region: exit_loss,
        })
        assert receipt.updated_parameter_ids == ("exit_logit", "task_weight")
    assert abs(task_weight.item() - 2.0) < 0.35
    assert exit_logit.item() > 0.0
    assert bool(exit_atom(exit_logit.reshape(1, 1), mask=exit_mask).requested.item())
    assert plan.state_arena._domain_steps[task_region].item() == 20
    assert plan.state_arena._domain_steps[exit_region].item() == 20

    unchanged = exit_logit.detach().clone()
    execution = graph.execute_program_functional("run")
    task_output = next(item.active_view.value for item in execution.state.resources
                       if item.spec.resource_id == "task_output")
    only_task = (task_output - 2.0).square().sum()
    receipt = plan.step_with_losses(local_objectives={task_region: only_task})
    assert receipt.skipped_parameter_ids == ("exit_logit",)
    torch.testing.assert_close(exit_logit, unchanged)


def test_region_objectives_respect_closed_credit_boundary() -> None:
    upstream = nn.Parameter(torch.tensor([1.0]))
    downstream = nn.Parameter(torch.tensor([2.0]))
    boundary = CreditBoundary(mode=CreditBoundaryMode.CLOSED)
    plan = _plan(
        {"upstream": upstream, "downstream": downstream},
        _uses(("upstream", "upstream_region", upstream),
              ("downstream", "downstream_region", downstream)),
        (
            OptimizerDomain("upstream_region", learning_rate=0.1),
            OptimizerDomain("downstream_region", learning_rate=0.1),
        ),
    )
    output = boundary(upstream) * downstream
    task_loss = (output - 4.0).square().sum()
    upstream_loss = (upstream - 3.0).square().sum()
    downstream_loss = (output - 5.0).square().sum()

    gradients = plan.gradients_from_losses(
        task_loss,
        local_objectives={
            "upstream_region": upstream_loss,
            "downstream_region": downstream_loss,
        },
    )

    torch.testing.assert_close(gradients["upstream"], torch.tensor([-4.0]))
    torch.testing.assert_close(gradients["downstream"], torch.tensor([-10.0]))


def test_region_objectives_accumulate_shared_credit_across_vjp_batches() -> None:
    shared = nn.Parameter(torch.tensor([0.5]))
    individual = {f"weight_{index}": nn.Parameter(torch.tensor([float(index)]))
                  for index in range(10)}
    uses = tuple(
        use
        for index, (parameter_id, parameter) in enumerate(individual.items())
        for use in (
            ParameterUse(parameter_id, f"region_{index}", parameter),
            ParameterUse("shared_weight", f"region_{index}", shared),
        )
    )
    plan = _plan(
        {**individual, "shared_weight": shared}, uses,
        tuple(OptimizerDomain(f"region_{index}", learning_rate=0.1) for index in range(10)) +
        (OptimizerDomain("shared", learning_rate=0.1),),
    )
    losses = {
        f"region_{index}": (parameter + shared - index / 2).square().sum()
        for index, parameter in enumerate(individual.values())
    }
    expected = torch.autograd.grad(
        sum(losses.values()), (*individual.values(), shared), retain_graph=True,
    )

    gradients = plan.gradients_from_losses(local_objectives=losses, vjp_batch_size=3)

    for parameter_id, reference in zip(individual, expected[:-1], strict=True):
        torch.testing.assert_close(gradients[parameter_id], reference)
    torch.testing.assert_close(gradients["shared_weight"], expected[-1])


def test_grouped_plan_matches_one_reference_adamw_for_equal_domains() -> None:
    torch.manual_seed(1)
    left = nn.Parameter(torch.randn(3))
    right = nn.Parameter(torch.randn(2))
    left_reference = nn.Parameter(left.detach().clone())
    right_reference = nn.Parameter(right.detach().clone())
    domain_left = OptimizerDomain("left", learning_rate=0.02, weight_decay=0.03)
    domain_right = OptimizerDomain("right", learning_rate=0.02, weight_decay=0.03)
    plan = _plan(
        {"left": left, "right": right},
        _uses(("left", "left", left), ("right", "right", right)),
        (domain_left, domain_right),
    )
    reference = torch.optim.AdamW([left_reference, right_reference], lr=0.02, weight_decay=0.03, foreach=True)

    for _ in range(3):
        left.grad = torch.randn_like(left)
        right.grad = torch.randn_like(right)
        left_reference.grad = left.grad.detach().clone()
        right_reference.grad = right.grad.detach().clone()
        receipt = plan.step()
        reference.step()
        reference.zero_grad(set_to_none=True)
        assert receipt.updated_parameter_ids == ("left", "right")

    torch.testing.assert_close(left, left_reference)
    torch.testing.assert_close(right, right_reference)
    assert len(plan.state_dict()["state_arena"]["entries"]) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the GPU foreach contract")
def test_grouped_plan_matches_cuda_foreach_adamw_in_one_physical_bucket() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    left = nn.Parameter(torch.randn(17, device=device))
    right = nn.Parameter(torch.randn(13, device=device))
    left_reference = nn.Parameter(left.detach().clone())
    right_reference = nn.Parameter(right.detach().clone())
    plan = _plan(
        {"left": left, "right": right},
        _uses(("left", "left", left), ("right", "right", right)),
        (
            OptimizerDomain("left", learning_rate=0.01, weight_decay=0.02),
            OptimizerDomain("right", learning_rate=0.01, weight_decay=0.02),
        ),
    )
    reference = torch.optim.AdamW([left_reference, right_reference], lr=0.01, weight_decay=0.02, foreach=True)

    for _ in range(3):
        left.grad = torch.randn_like(left)
        right.grad = torch.randn_like(right)
        left_reference.grad = left.grad.detach().clone()
        right_reference.grad = right.grad.detach().clone()
        receipt = plan.step()
        reference.step()
        reference.zero_grad(set_to_none=True)
        assert len(receipt.buckets) == 1
        assert receipt.buckets[0].parameter_ids == ("left", "right")

    torch.testing.assert_close(left, left_reference)
    torch.testing.assert_close(right, right_reference)


def test_zero_gradient_updates_but_none_gradient_and_paused_domain_do_not() -> None:
    parameter = nn.Parameter(torch.tensor([2.0]))
    domain = OptimizerDomain("only", learning_rate=0.1, weight_decay=0.1)
    plan = _plan({"weight": parameter}, _uses(("weight", "only", parameter)), (domain,))

    parameter.grad = torch.zeros_like(parameter)
    first = plan.step()
    assert first.updated_parameter_ids == ("weight",)
    assert parameter.item() < 2.0
    checkpoint = copy.deepcopy(plan.state_dict())

    parameter.grad = None
    second = plan.step()
    assert second.updated_parameter_ids == ()
    assert second.skipped_parameter_ids == ("weight",)
    torch.testing.assert_close(
        plan.state_dict()["state_arena"]["entries"]["weight"]["parameter_step"],
        checkpoint["state_arena"]["entries"]["weight"]["parameter_step"],
    )

    paused = OptimizerExecutionPlan(
        {"weight": parameter},
        ParameterOwnershipTable.from_uses(_uses(("weight", "only", parameter))),
        (OptimizerDomain("only", learning_rate=0.1, status=OptimizerDomainStatus.PAUSED),),
        state_arena=plan.state_arena,
    )
    parameter.grad = torch.ones_like(parameter)
    before = parameter.detach().clone()
    third = paused.step()
    assert third.updated_parameter_ids == ()
    torch.testing.assert_close(parameter, before)


def test_functional_adamw_trial_matches_one_committed_plan_step_without_mutating_snapshot() -> None:
    parameter = nn.Parameter(torch.tensor([2.0, -1.0]))
    plan = _plan(
        {"weight": parameter},
        _uses(("weight", "only", parameter)),
        (OptimizerDomain("only", learning_rate=0.1, weight_decay=0.02),),
    )
    gradient = torch.tensor([0.5, -0.25])
    before_parameter = parameter.detach().clone()

    trial = plan.functional_adamw_trial(gradients={"weight": gradient})

    torch.testing.assert_close(parameter, before_parameter)
    assert plan.state_arena._entries == {}
    assert trial.updated_parameter_ids == ("weight",)
    assert trial.advanced_domain_ids == ("only",)
    plan.step(gradients={"weight": gradient})

    torch.testing.assert_close(parameter, trial.parameters["weight"])
    committed = plan.state_arena._entries["weight"]
    proposed = trial.state_entries["weight"]
    torch.testing.assert_close(committed.exp_avg, proposed.exp_avg)
    torch.testing.assert_close(committed.exp_avg_sq, proposed.exp_avg_sq)
    torch.testing.assert_close(committed.parameter_step, proposed.parameter_step)
    torch.testing.assert_close(plan.state_arena._domain_steps["only"], trial.domain_steps["only"])


def test_prepared_optimizer_step_matches_eager_commit_and_preserves_masked_state() -> None:
    parameter = nn.Parameter(torch.tensor([1.0, -2.0]))
    reference_parameter = nn.Parameter(parameter.detach().clone())
    domain = OptimizerDomain("only", learning_rate=0.1, weight_decay=0.02)
    plan = _plan({"weight": parameter}, _uses(("weight", "only", parameter)), (domain,))
    reference = _plan(
        {"weight": reference_parameter},
        _uses(("weight", "only", reference_parameter)),
        (domain,),
    )
    prepared = plan.prepare_static_step()
    compiled = torch.compile(prepared, backend="eager", fullgraph=True)
    gradient = torch.tensor([0.25, -0.5])

    parameter_updates, domain_updates = compiled(
        torch.tensor([True]),
        torch.tensor([True]),
        gradient,
    )
    reference.step(gradients={"weight": gradient.clone()})

    assert parameter_updates.tolist() == [True]
    assert domain_updates.tolist() == [True]
    torch.testing.assert_close(parameter, reference_parameter)
    for field in ("exp_avg", "exp_avg_sq", "parameter_step"):
        torch.testing.assert_close(
            getattr(plan.state_arena._entries["weight"], field),
            getattr(reference.state_arena._entries["weight"], field),
        )

    before_parameter = parameter.detach().clone()
    before_state = plan.state_arena._entries["weight"].exp_avg.detach().clone()
    parameter_updates, domain_updates = compiled(
        torch.tensor([False]),
        torch.tensor([True]),
        torch.zeros_like(gradient),
    )
    assert parameter_updates.tolist() == [False]
    assert domain_updates.tolist() == [False]
    torch.testing.assert_close(parameter, before_parameter)
    torch.testing.assert_close(plan.state_arena._entries["weight"].exp_avg, before_state)

    prepared.synchronize_control_state()
    assert plan.transaction_index == 2
    rebuilt, _ = plan.commit_structure(
        {"weight": parameter},
        plan.ownership,
        tuple(plan.domains.values()),
    )
    assert rebuilt.transaction_index == 2


def test_prepared_optimizer_step_independently_masks_logical_domains() -> None:
    left = nn.Parameter(torch.tensor([1.0]))
    right = nn.Parameter(torch.tensor([2.0]))
    plan = _plan(
        {"left": left, "right": right},
        _uses(("left", "left", left), ("right", "right", right)),
        (
            OptimizerDomain("left", learning_rate=0.1),
            OptimizerDomain("right", learning_rate=0.1),
        ),
    )
    prepared = torch.compile(plan.prepare_static_step(), backend="eager", fullgraph=True)
    before_right = right.detach().clone()

    parameter_updates, domain_updates = prepared(
        torch.tensor([True, False]),
        torch.tensor([True, True]),
        torch.tensor([0.25]),
        torch.tensor([-0.5]),
    )

    assert parameter_updates.tolist() == [True, False]
    assert domain_updates.tolist() == [True, False]
    assert not torch.equal(left.detach(), torch.tensor([1.0]))
    torch.testing.assert_close(right, before_right)
    assert plan.state_arena._domain_steps["left"].item() == 1
    assert plan.state_arena._domain_steps["right"].item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for compiled optimizer-step coverage")
def test_prepared_optimizer_step_compiles_with_inductor_on_cuda() -> None:
    torch._dynamo.reset()
    try:
        parameter = nn.Parameter(torch.tensor([1.0, -2.0], device="cuda"))
        domain = OptimizerDomain("only", learning_rate=0.1)
        plan = _plan({"weight": parameter}, _uses(("weight", "only", parameter)), (domain,))
        compiled = torch.compile(plan.prepare_static_step(), backend="inductor", fullgraph=True)

        parameter_updates, domain_updates = compiled(
            torch.tensor([True], device="cuda"),
            torch.tensor([True], device="cuda"),
            torch.tensor([0.25, -0.5], device="cuda"),
        )

        assert parameter_updates.tolist() == [True]
        assert domain_updates.tolist() == [True]
        assert not torch.equal(parameter.detach(), torch.tensor([1.0, -2.0], device="cuda"))
    finally:
        torch._dynamo.reset()


def test_domain_cadence_skips_a_transaction_without_advancing_state() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    plan = _plan(
        {"weight": parameter},
        _uses(("weight", "only", parameter)),
        (OptimizerDomain("only", learning_rate=0.1, cadence=2),),
    )

    parameter.grad = torch.tensor([1.0])
    first = plan.step()
    after_first = parameter.detach().clone()
    parameter.grad = torch.tensor([1.0])
    second = plan.step()

    assert first.updated_parameter_ids == ("weight",)
    assert second.updated_parameter_ids == ()
    assert second.skipped_parameter_ids == ("weight",)
    torch.testing.assert_close(parameter, after_first)


def test_shared_parameter_is_updated_once_not_once_per_use_site() -> None:
    parameter = nn.Parameter(torch.tensor([1.0, -2.0]))
    reference = nn.Parameter(parameter.detach().clone())
    plan = _plan(
        {"shared": parameter},
        _uses(("shared", "left", parameter), ("shared", "right", parameter)),
        (OptimizerDomain("shared", learning_rate=0.05),),
    )
    optimizer = torch.optim.AdamW([reference], lr=0.05, weight_decay=0.0, foreach=True)

    gradient = torch.tensor([0.25, -0.5])
    parameter.grad = gradient.clone()
    reference.grad = gradient.clone()
    receipt = plan.step()
    optimizer.step()

    assert receipt.updated_parameter_ids == ("shared",)
    torch.testing.assert_close(parameter, reference)


def test_grouped_plan_consumes_explicit_credit_gradient_field_without_parameter_grad() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    plan = _plan(
        {"weight": parameter},
        _uses(("weight", "only", parameter)),
        (OptimizerDomain("only", learning_rate=0.1),),
    )
    objective = 0.5 * (parameter - 3.0).square().sum()
    field = credit_gradient_field(
        {"task": objective},
        {"task": torch.ones(())},
        {"weight": parameter},
        create_graph=False,
    )

    assert parameter.grad is None
    receipt = plan.step(gradients=field.as_dict())

    assert receipt.updated_parameter_ids == ("weight",)
    assert parameter.grad is None
    assert parameter.item() > 1.0


def test_structure_commit_preserves_moments_for_retained_parameter_identity() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    plan = _plan(
        {"weight": parameter},
        _uses(("weight", "before", parameter)),
        (OptimizerDomain("before", learning_rate=0.01),),
    )
    parameter.grad = torch.tensor([0.5])
    plan.step()
    state_before = plan.state_arena.state_dict()["entries"]["weight"]

    next_plan, receipt = plan.commit_structure(
        {"weight": parameter},
        ParameterOwnershipTable.from_uses(_uses(("weight", "after", parameter))),
        (OptimizerDomain("after", learning_rate=0.01),),
    )

    assert receipt.previous_generation == 0
    assert receipt.generation == 1
    assert receipt.retained_parameter_ids == ("weight",)
    parameter.grad = torch.tensor([0.25])
    next_plan.step()
    state_after = next_plan.state_arena.state_dict()["entries"]["weight"]
    assert state_after["parameter_step"].item() == state_before["parameter_step"].item() + 1


def test_structure_commit_replaced_parameter_id_starts_fresh_optimizer_state() -> None:
    old = nn.Parameter(torch.tensor([1.0]))
    plan = _plan(
        {"weight": old},
        _uses(("weight", "region", old)),
        (OptimizerDomain("region", learning_rate=0.01),),
    )
    old.grad = torch.tensor([0.5])
    plan.step()
    replacement = nn.Parameter(torch.tensor([5.0]))

    next_plan, receipt = plan.commit_structure(
        {"weight": replacement},
        ParameterOwnershipTable.from_uses(_uses(("weight", "region", replacement))),
        tuple(plan.domains.values()),
    )

    assert receipt.retained_parameter_ids == ()
    assert receipt.added_parameter_ids == ("weight",)
    assert receipt.removed_parameter_ids == ("weight",)
    assert "weight" not in next_plan.state_arena._entries
    assert "region" not in next_plan.state_arena._domain_steps
    replacement.grad = torch.tensor([0.25])
    next_plan.step()
    assert next_plan.state_arena._entries["weight"].parameter_step.item() == 1
    assert plan.state_arena._entries["weight"].parameter_step.item() == 1


def test_structure_commit_reused_domain_id_restarts_clock_when_members_change() -> None:
    retained = nn.Parameter(torch.tensor([1.0]))
    added = nn.Parameter(torch.tensor([2.0]))
    plan = _plan(
        {"retained": retained},
        _uses(("retained", "region", retained)),
        (OptimizerDomain("region", learning_rate=0.01),),
    )
    for _ in range(2):
        retained.grad = torch.tensor([0.5])
        plan.step()
    old_entry = plan.state_arena._entries["retained"]

    next_plan, receipt = plan.commit_structure(
        {"retained": retained, "added": added},
        ParameterOwnershipTable.from_uses(
            _uses(("retained", "region", retained), ("added", "region", added))
        ),
        tuple(plan.domains.values()),
    )

    assert receipt.retained_parameter_ids == ("retained",)
    assert next_plan.state_arena._entries["retained"] is old_entry
    assert "region" not in next_plan.state_arena._domain_steps
    assert plan.state_arena._domain_steps["region"].item() == 2
    retained.grad = torch.tensor([0.25])
    added.grad = torch.tensor([0.25])
    next_plan.step()
    assert next_plan.state_arena._domain_steps["region"].item() == 1
    assert next_plan.state_arena._entries["retained"].parameter_step.item() == 3


def test_structure_commit_checkpoint_omits_removed_parameter_and_domain_state() -> None:
    retained = nn.Parameter(torch.tensor([1.0]))
    removed = nn.Parameter(torch.tensor([2.0]))
    plan = _plan(
        {"retained": retained, "removed": removed},
        _uses(("retained", "keep", retained), ("removed", "drop", removed)),
        (OptimizerDomain("keep", learning_rate=0.01), OptimizerDomain("drop", learning_rate=0.01)),
    )
    retained.grad = torch.tensor([0.25])
    removed.grad = torch.tensor([0.5])
    plan.step()
    next_plan, receipt = plan.commit_structure(
        {"retained": retained},
        ParameterOwnershipTable.from_uses(_uses(("retained", "keep", retained))),
        (OptimizerDomain("keep", learning_rate=0.01),),
    )
    assert receipt.removed_parameter_ids == ("removed",)
    assert "removed" in plan.state_arena.state_dict()["entries"]
    assert "removed" not in next_plan.state_arena.state_dict()["entries"]
    assert next_plan.state_arena._entries["retained"].exp_avg is plan.state_arena._entries["retained"].exp_avg
    assert next_plan.state_arena._domain_steps["keep"] is plan.state_arena._domain_steps["keep"]
    saved = next_plan.state_dict()
    assert set(saved["state_arena"]["entries"]) == {"retained"}
    assert set(saved["state_arena"]["domain_steps"]) == {"keep"}

    restored_parameter = nn.Parameter(retained.detach().clone())
    restored = OptimizerExecutionPlan(
        {"retained": restored_parameter},
        ParameterOwnershipTable.from_uses(_uses(("retained", "keep", restored_parameter))),
        (OptimizerDomain("keep", learning_rate=0.01),),
        topology_generation=next_plan.topology_generation,
    )
    restored.load_state_dict(saved)
    assert restored.transaction_index == next_plan.transaction_index
    restored_state = restored.state_arena.state_dict()["entries"]["retained"]
    torch.testing.assert_close(
        restored_state["exp_avg"], next_plan.state_arena.state_dict()["entries"]["retained"]["exp_avg"],
    )
    retained.grad = torch.tensor([0.125])
    restored_parameter.grad = torch.tensor([0.125])
    next_plan.step()
    restored.step()
    torch.testing.assert_close(restored_parameter, retained, atol=0, rtol=0)
    expected_state = next_plan.state_arena.state_dict()["entries"]["retained"]
    actual_state = restored.state_arena.state_dict()["entries"]["retained"]
    for name in ("exp_avg", "exp_avg_sq", "parameter_step"):
        torch.testing.assert_close(actual_state[name], expected_state[name], atol=0, rtol=0)


def test_checkpoint_requires_matching_ownership() -> None:
    parameter = nn.Parameter(torch.tensor([1.0]))
    plan = _plan(
        {"weight": parameter},
        _uses(("weight", "one", parameter)),
        (OptimizerDomain("one", learning_rate=0.1),),
    )
    parameter.grad = torch.tensor([1.0])
    plan.step()
    checkpoint = plan.state_dict()
    mismatched = _plan(
        {"weight": parameter},
        _uses(("weight", "two", parameter)),
        (OptimizerDomain("two", learning_rate=0.1),),
    )

    with pytest.raises(LearningRegionError, match="topology generation|ownership"):
        mismatched.load_state_dict(checkpoint)
