from __future__ import annotations

from unittest import mock

import pytest
import torch

import arti
from arti import mechanisms
from arti import formula_fabric as formula_fabric_module
from arti.vnext_contracts import ContractLimits


def program() -> mechanisms.FormulaFabricProgram:
    return mechanisms.FormulaFabricProgram(
        arena_capacity=4,
        feature_dim=2,
        steps=(
            (mechanisms.FormulaInvocation(mechanisms.FormulaPrimitive.ADD, 2),),
            (mechanisms.FormulaInvocation(mechanisms.FormulaPrimitive.MULTIPLY, 3),),
        ),
    )


def route(
    *,
    batch: int = 2,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> mechanisms.FormulaRoutePlan:
    weights = torch.zeros(batch, 2, 1, 2, 4, device=device, dtype=dtype)
    weights[:, 0, 0, 0, 0] = 1
    weights[:, 0, 0, 1, 1] = 1
    weights[:, 1, 0, 0, 2] = 1
    weights[:, 1, 0, 1, 1] = 1
    enabled = torch.ones(batch, 2, 1, dtype=torch.bool, device=device)
    return mechanisms.FormulaRoutePlan(weights, enabled, enabled, enabled)


def arena(
    *,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> mechanisms.FormulaArenaState:
    value = torch.tensor(
        [
            [[2.0, 3.0], [4.0, 5.0]],
            [[-1.0, 2.0], [3.0, -4.0]],
        ],
        device=device,
        requires_grad=requires_grad,
    )
    mask = torch.ones(2, 2, dtype=torch.bool, device=device)
    state = mechanisms.FormulaArenaState.from_tensor(value, mask, capacity=4)
    if requires_grad:
        state.value.retain_grad()
    return state


def test_hard_fabric_executes_formula_chain_and_ssa_versions() -> None:
    fabric = mechanisms.FormulaFabric(program())
    source = arena()

    result = fabric(source, route())

    expected_add = source.value[:, 0] + source.value[:, 1]
    expected_multiply = expected_add * source.value[:, 1]
    torch.testing.assert_close(result.state.value[:, 2], expected_add)
    torch.testing.assert_close(result.state.value[:, 3], expected_multiply)
    assert result.state.mask.all()
    assert torch.equal(
        result.state.version,
        torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
    )
    assert torch.equal(result.trace.input_version[:, 1, 0, 0], torch.ones(2, dtype=torch.int64))
    assert result.trace.program_fingerprint == program().fingerprint


def test_program_owns_an_immutable_normalized_schedule() -> None:
    cell = mechanisms.FormulaInvocation(mechanisms.FormulaPrimitive.ADD, 2)
    step = [cell]
    steps = [step]
    schedule = mechanisms.FormulaFabricProgram(
        arena_capacity=3,
        feature_dim=2,
        steps=steps,
    )
    fingerprint = schedule.fingerprint

    step.clear()
    steps.clear()

    assert schedule.steps == ((cell,),)
    assert schedule.fingerprint == fingerprint


def test_route_plan_owns_weights_without_breaking_source_gradients() -> None:
    logits = torch.randn(2, 2, 1, 2, 4, requires_grad=True)
    weights = mechanisms.straight_through_route(logits)
    enabled = torch.ones(2, 2, 1, dtype=torch.bool)
    plan = mechanisms.FormulaRoutePlan(
        weights,
        enabled,
        enabled,
        enabled,
        estimator="straight-through",
    )

    weights.detach().zero_()

    assert bool((plan.weights.detach().sum(dim=-1) == 1).all())
    plan.weights.square().sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_route_plan_rejects_ownership_budget_before_cloning(monkeypatch) -> None:
    weights = torch.zeros(1, 1, 1, 1, 1)
    enabled = torch.ones(1, 1, 1, dtype=torch.bool)
    monkeypatch.setattr(
        formula_fabric_module,
        "DEFAULT_CONTRACT_LIMITS",
        ContractLimits(max_operation_bytes=1),
    )

    with mock.patch.object(
        torch.Tensor,
        "clone",
        side_effect=AssertionError("clone must not run before admission"),
    ):
        with pytest.raises(ValueError, match="ownership exceeds max_operation_bytes"):
            mechanisms.FormulaRoutePlan(weights, enabled, enabled, enabled)


def test_hard_executor_matches_independent_reference_interpreter() -> None:
    fabric = mechanisms.FormulaFabric(program())
    source = arena()
    plan = route()

    actual = fabric(source, plan).state
    reference = fabric.reference(source, plan)

    assert torch.equal(actual.value, reference.value)
    assert torch.equal(actual.mask, reference.mask)
    assert torch.equal(actual.version, reference.version)


def test_straight_through_route_is_hard_forward_and_differentiable() -> None:
    fabric = mechanisms.FormulaFabric(program())
    source = arena(requires_grad=True)
    logits = torch.full((2, 2, 1, 2, 4), -4.0, requires_grad=True)
    preferred = torch.tensor([[[[[0], [1]]], [[[2], [1]]]]]).expand(2, -1, -1, -1, -1)
    with torch.no_grad():
        logits.scatter_(-1, preferred, 4.0)
    weights = mechanisms.straight_through_route(logits)
    enabled = torch.ones(2, 2, 1, dtype=torch.bool)
    plan = mechanisms.FormulaRoutePlan(
        weights,
        enabled,
        enabled,
        enabled,
        estimator="straight-through",
    )

    result = fabric(source, plan)
    result.state.value[:, 3].square().mean().backward()

    assert torch.equal(weights.detach().sum(dim=-1), torch.ones_like(weights[..., 0]))
    assert bool(((weights.detach() == 0) | (weights.detach() == 1)).all())
    assert source.value.grad is not None and torch.isfinite(source.value.grad).all()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert bool(logits.grad.abs().sum() > 0)


def test_synchronous_step_does_not_observe_another_cell_write() -> None:
    schedule = mechanisms.FormulaFabricProgram(
        arena_capacity=3,
        feature_dim=1,
        steps=(
            (
                mechanisms.FormulaInvocation(mechanisms.FormulaPrimitive.ADD, 2),
                mechanisms.FormulaInvocation(mechanisms.FormulaPrimitive.MULTIPLY, 1),
            ),
        ),
    )
    value = torch.tensor([[[2.0], [3.0], [10.0]]])
    source = mechanisms.FormulaArenaState(
        value,
        torch.ones(1, 3, dtype=torch.bool),
        torch.zeros(1, 3, dtype=torch.int64),
    )
    weights = torch.zeros(1, 1, 2, 2, 3)
    weights[0, 0, 0, 0, 0] = 1
    weights[0, 0, 0, 1, 1] = 1
    weights[0, 0, 1, 0, 2] = 1
    weights[0, 0, 1, 1, 0] = 1
    enabled = torch.ones(1, 1, 2, dtype=torch.bool)

    result = mechanisms.FormulaFabric(schedule)(
        source,
        mechanisms.FormulaRoutePlan(weights, enabled, enabled, enabled),
    )

    assert result.state.value[0, 2, 0].item() == 5.0
    assert result.state.value[0, 1, 0].item() == 20.0


def test_commit_mask_can_run_a_cost_matched_sham_without_mutation() -> None:
    source = arena()
    schedule = mechanisms.FormulaFabricProgram(
        arena_capacity=4,
        feature_dim=2,
        steps=((mechanisms.FormulaInvocation(mechanisms.FormulaPrimitive.ADD, 2),),),
    )
    weights = route().weights[:, :1]
    enabled = torch.ones(2, 1, 1, dtype=torch.bool)
    base = mechanisms.FormulaRoutePlan(weights, enabled, enabled, enabled)
    commit = torch.zeros_like(base.commit_mask)
    result = mechanisms.FormulaFabric(schedule)(
        source,
        mechanisms.FormulaRoutePlan(
            base.weights,
            base.valid_mask,
            base.fire_mask,
            commit,
        ),
    )

    assert torch.equal(result.state.value, source.value)
    assert torch.equal(result.state.mask, source.mask)
    assert torch.equal(result.state.version, torch.zeros(2, 4, dtype=torch.int64))


def test_ssa_versions_advance_only_for_committed_batch_rows() -> None:
    source = arena()
    schedule = mechanisms.FormulaFabricProgram(
        arena_capacity=4,
        feature_dim=2,
        steps=((mechanisms.FormulaInvocation(mechanisms.FormulaPrimitive.ADD, 2),),),
    )
    weights = route().weights[:, :1]
    enabled = torch.ones(2, 1, 1, dtype=torch.bool)
    commit = torch.tensor([[[True]], [[False]]])

    result = mechanisms.FormulaFabric(schedule)(
        source,
        mechanisms.FormulaRoutePlan(weights, enabled, enabled, commit),
    )

    torch.testing.assert_close(
        result.state.value[0, 2], source.value[0, 0] + source.value[0, 1]
    )
    torch.testing.assert_close(result.state.value[1, 2], source.value[1, 2])
    assert torch.equal(
        result.state.version,
        torch.tensor([[0, 0, 1, 0], [0, 0, 0, 0]]),
    )
    assert torch.equal(result.trace.output_version[:, 0, 0], torch.tensor([1, 0]))


def test_contracts_fail_closed() -> None:
    source = arena()
    fabric = mechanisms.FormulaFabric(program())
    base = route()
    soft = base.weights.clone()
    soft[..., 0, :] = 0.25
    with pytest.raises(ValueError, match="exactly one-hot"):
        fabric(
            source,
            mechanisms.FormulaRoutePlan(
                soft,
                base.valid_mask,
                base.fire_mask,
                base.commit_mask,
            ),
        )

    invalid_source = mechanisms.FormulaArenaState(
        source.value,
        source.mask.clone().scatter(1, torch.tensor([[1], [1]]), False),
        source.version,
    )
    with pytest.raises(ValueError, match="invalid arena value"):
        fabric(invalid_source, base)

    with pytest.raises(ValueError, match="write a slot twice"):
        mechanisms.FormulaFabricProgram(
            arena_capacity=3,
            feature_dim=2,
            steps=(
                (
                    mechanisms.FormulaInvocation(mechanisms.FormulaPrimitive.ADD, 2),
                    mechanisms.FormulaInvocation(mechanisms.FormulaPrimitive.SUBTRACT, 2),
                ),
            ),
        )


def test_invalid_slots_are_numerically_isolated_from_formula_execution() -> None:
    schedule = mechanisms.FormulaFabricProgram(
        arena_capacity=3,
        feature_dim=1,
        steps=((mechanisms.FormulaInvocation(mechanisms.FormulaPrimitive.ADD, 2),),),
    )
    value = torch.tensor([[[2.0], [3.0], [float("nan")]]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    state = mechanisms.FormulaArenaState(
        value,
        mask,
        torch.zeros(1, 3, dtype=torch.int64),
    )
    weights = torch.zeros(1, 1, 1, 2, 3)
    weights[..., 0, 0] = 1
    weights[..., 1, 1] = 1
    enabled = torch.ones(1, 1, 1, dtype=torch.bool)

    result = mechanisms.FormulaCommitBlend(mechanisms.FormulaFabric(schedule))(
        state,
        mechanisms.FormulaRoutePlan(weights, enabled, enabled, enabled),
        torch.ones(1, 1, 1),
    )
    result.state.value[:, 2].sum().backward()

    torch.testing.assert_close(result.state.value[:, 2], torch.tensor([[5.0]]))
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_valid_non_finite_arena_value_fails_closed() -> None:
    source = arena()
    invalid = mechanisms.FormulaArenaState(
        source.value.clone().index_fill(1, torch.tensor([0]), float("inf")),
        source.mask,
        source.version,
    )

    with pytest.raises(ValueError, match="valid Formula arena values must be finite"):
        mechanisms.FormulaFabric(program())(invalid, route())


def test_formula_fabric_has_versioned_alpha_identity() -> None:
    fabric = mechanisms.FormulaFabric(program())

    assert arti.component_ref(fabric) == "arti/formula-fabric@1"
    spec = arti.component_spec(fabric)
    assert spec.lifecycle == "stable"
    assert spec.config_schema_version == 2
    assert spec.config["program_fingerprint"] == program().fingerprint


def test_formula_fabric_rejects_runtime_allocation_over_limit() -> None:
    fabric = mechanisms.FormulaFabric(
        program(),
        limits=ContractLimits(max_operation_bytes=128),
    )

    with pytest.raises(ValueError, match="max_operation_bytes"):
        fabric(arena(), route())

    compiled = torch.compile(fabric, backend="eager", fullgraph=True)
    with pytest.raises(Exception):
        compiled(arena(), route())


def test_formula_fabric_rejects_static_tables_before_registration() -> None:
    schedule = mechanisms.FormulaFabricProgram(
        arena_capacity=4,
        feature_dim=1,
        steps=(
            (
                mechanisms.FormulaInvocation(mechanisms.FormulaPrimitive.ADD, 0),
                mechanisms.FormulaInvocation(mechanisms.FormulaPrimitive.ADD, 1),
            ),
            (
                mechanisms.FormulaInvocation(mechanisms.FormulaPrimitive.ADD, 2),
                mechanisms.FormulaInvocation(mechanisms.FormulaPrimitive.ADD, 3),
            ),
            (
                mechanisms.FormulaInvocation(mechanisms.FormulaPrimitive.ADD, 0),
                mechanisms.FormulaInvocation(mechanisms.FormulaPrimitive.ADD, 1),
            ),
        ),
    )

    with pytest.raises(ValueError, match="static table"):
        mechanisms.FormulaFabric(schedule, limits=ContractLimits(max_elements=4))


def test_formula_trace_is_an_independent_snapshot() -> None:
    fabric = mechanisms.FormulaFabric(program())
    plan = route()
    result = fabric(arena(), plan)
    original_formula = fabric._formula_ids.clone()
    original_fire = plan.fire_mask.clone()

    result.trace.formula_id.fill_(99)
    result.trace.fire_mask.zero_()

    assert torch.equal(fabric._formula_ids, original_formula)
    assert torch.equal(plan.fire_mask, original_fire)


def test_formula_fabric_revalidates_mutated_route_masks() -> None:
    fabric = mechanisms.FormulaFabric(program())
    plan = route()
    plan.fire_mask.zero_()

    with pytest.raises(ValueError, match="commit_mask must be a subset"):
        fabric(arena(), plan)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile is unavailable")
def test_formula_fabric_fullgraph_forward_backward_matches_eager() -> None:
    eager = mechanisms.FormulaFabric(program())
    compiled = torch.compile(
        mechanisms.FormulaFabric(program()), backend="eager", fullgraph=True
    )
    eager_source = arena(requires_grad=True)
    compiled_source = arena(requires_grad=True)
    plan = route()

    expected = eager(eager_source, plan)
    actual = compiled(compiled_source, plan)
    torch.testing.assert_close(actual.state.value, expected.state.value)
    assert torch.equal(actual.state.mask, expected.state.mask)
    assert torch.equal(actual.state.version, expected.state.version)

    expected.state.value[:, 3].sum().backward()
    actual.state.value[:, 3].sum().backward()
    torch.testing.assert_close(compiled_source.value.grad, eager_source.value.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_formula_fabric_cuda_parity_and_gradient() -> None:
    fabric = mechanisms.FormulaFabric(program()).cuda()
    source = arena(device="cuda", requires_grad=True)
    plan = route(device="cuda")

    actual = fabric(source, plan)
    reference = fabric.reference(source, plan)

    assert torch.equal(actual.state.value.detach(), reference.value)
    actual.state.value[:, 3].sum().backward()
    assert source.value.grad is not None and torch.isfinite(source.value.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_formula_fabric_cuda_fullgraph_forward_backward() -> None:
    eager = mechanisms.FormulaFabric(program()).cuda()
    compiled = torch.compile(mechanisms.FormulaFabric(program()).cuda(), fullgraph=True)
    eager_source = arena(device="cuda", requires_grad=True)
    compiled_source = arena(device="cuda", requires_grad=True)
    plan = route(device="cuda")

    expected = eager(eager_source, plan)
    actual = compiled(compiled_source, plan)
    torch.testing.assert_close(actual.state.value, expected.state.value)
    assert torch.equal(actual.state.mask, expected.state.mask)
    assert torch.equal(actual.state.version, expected.state.version)

    expected.state.value[:, 3].sum().backward()
    actual.state.value[:, 3].sum().backward()
    torch.testing.assert_close(compiled_source.value.grad, eager_source.value.grad)
