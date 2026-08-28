from __future__ import annotations

from dataclasses import replace

import torch
import pytest

import arti
from arti.recall_formula import FactorSpec, RecallFormulaContract
from arti.recall_registry import RecallFormulaId


class _NegateStateFormula(torch.nn.Module):
    recall_formula_contract = RecallFormulaContract(
        identity=RecallFormulaId.parse("tests/negate-state@1"),
        factors=(FactorSpec("unused", init="zero"),),
        identity_preserving=False,
    )

    def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        del factors
        return -state


class _SelectiveStateFormula(torch.nn.Module):
    recall_formula_contract = RecallFormulaContract(
        identity=RecallFormulaId.parse("tests/selective-state@1"),
        factors=(FactorSpec("unused", init="zero"),),
        identity_preserving=False,
    )

    def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        del factors
        return torch.where(state[..., :1] > 0, state, state * 2)


def test_refine_policy_is_versioned_but_runtime_only() -> None:
    policy = arti.RefinePolicy(
        max_steps=6,
        min_steps=2,
        tolerance=1e-3,
        checkpoints=(1, 3, 6),
    )

    assert arti.component_ref(policy) == "arti/refine-policy@1"
    assert arti.component_spec(policy).variant == "runtime-only"
    assert arti.component_spec(policy).config["max_steps"] == 6


def test_adaptive_refine_policy_is_composed_and_versioned() -> None:
    policy = arti.RefinePolicy.adaptive(
        max_steps=64,
        min_steps=2,
        scope="token",
        relative_tolerance=1e-3,
        route_tolerance=1e-2,
        patience=2,
        executor="early_break",
    )

    assert isinstance(policy, arti.AdaptiveRefinePolicy)
    assert arti.component_ref(policy) == "arti/refine-policy@2"
    assert arti.component_ref(policy.budget) == "arti/refine-budget@1"
    assert arti.component_ref(policy.stop) == "arti/refine-stop@1"
    assert arti.component_spec(policy).dependencies == (
        "arti/refine-budget@1",
        "arti/refine-stop@1",
    )


def test_token_adaptive_refine_freezes_converged_tokens_independently() -> None:
    recall = arti.Recall(2, 2, formula=_SelectiveStateFormula(), activation="none")
    x = torch.tensor([[[1.0, 1.0], [-1.0, -1.0]]])
    policy = arti.RefinePolicy.adaptive(
        max_steps=4,
        min_steps=1,
        scope="token",
        absolute_tolerance=0.0,
        relative_tolerance=1e-6,
        patience=1,
        trace_level="routes",
    )

    y, trace = recall(x, refine_policy=policy, return_trace=True)

    assert isinstance(trace, arti.RecallTraceV2)
    assert trace.token_steps_attempted.tolist() == [[1, 4]]
    assert trace.token_step_attempted[0, :, 0].tolist() == [True, False, False, False]
    assert trace.token_step_attempted[0, :, 1].tolist() == [True, True, True, True]
    assert trace.token_stop_reason[0, 0].item() == int(arti.RecallStopReason.CONVERGED)
    assert trace.token_stop_reason[0, 1].item() == int(arti.RecallStopReason.MAX_STEPS)
    torch.testing.assert_close(y[0, 0], x[0, 0], rtol=0, atol=0)
    assert not torch.equal(y[0, 1], x[0, 1])
    trace.validate()


def test_early_break_reports_real_kernel_steps() -> None:
    recall = arti.Recall(2, 2, formula=_SelectiveStateFormula(), activation="none")
    x = torch.ones(1, 2, 2)
    common = dict(
        max_steps=8,
        min_steps=1,
        scope="token",
        relative_tolerance=1e-6,
        trace_level="routes",
    )

    static_y, static_trace = recall(
        x,
        refine_policy=arti.RefinePolicy.adaptive(**common, executor="static_masked"),
        return_trace=True,
    )
    eager_y, eager_trace = recall(
        x,
        refine_policy=arti.RefinePolicy.adaptive(**common, executor="early_break"),
        return_trace=True,
    )

    torch.testing.assert_close(eager_y, static_y, rtol=0, atol=0)
    assert static_trace.kernel_steps.item() == 8
    assert eager_trace.kernel_steps.item() == 1
    assert static_trace.logical_token_steps.item() == 2
    assert eager_trace.logical_token_steps.item() == 2


def test_adaptive_patience_starts_after_minimum_depth() -> None:
    recall = arti.Recall(2, 2, formula="arti/delta@1", activation="none")
    x = torch.ones(1, 1, 2)
    memory = torch.zeros(1, 2, 2)
    policy = arti.RefinePolicy.adaptive(
        max_steps=8,
        min_steps=2,
        scope="token",
        absolute_tolerance=0.0,
        relative_tolerance=1e-9,
        patience=2,
        executor="early_break",
        trace_level="routes",
    )

    _, trace = recall(
        x,
        memory=memory,
        refine_policy=policy,
        return_trace=True,
    )

    assert trace.token_steps_attempted.item() == 3
    assert trace.kernel_steps.item() == 3
    assert trace.token_stop_reason.item() == int(arti.RecallStopReason.CONVERGED)


def test_adaptive_trace_preserves_mask_and_gradients() -> None:
    recall = arti.Recall(4, 4, activation="none")
    x = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    policy = arti.RefinePolicy.adaptive(
        max_steps=3,
        min_steps=3,
        scope="token",
        relative_tolerance=1e-6,
        checkpoints=(1, 3),
        checkpoint_mode="gradient",
        trace_level="routes",
    )

    y, trace = recall(x, mask=mask, refine_policy=policy, return_trace=True)
    y[mask].square().mean().backward()

    assert trace.token_steps_attempted.tolist() == [[3, 3, 0], [3, 0, 0]]
    assert torch.all(trace.token_stop_reason[~mask] == int(arti.RecallStopReason.MASKED))
    assert trace.logical_token_steps.item() == 9
    assert x.grad is not None and torch.isfinite(x.grad[mask]).all()
    assert recall.state.recall.bank.grad is not None
    assert torch.isfinite(recall.state.recall.bank.grad).all()
    rebuilt = arti.RecallTraceV2.from_diagnostics(trace.diagnostics())
    torch.testing.assert_close(rebuilt.token_steps_committed, trace.token_steps_committed)


def test_route_stack_is_versioned_recursive_runtime_composition() -> None:
    recall = arti.Recall(4, 4, activation="none")
    x = torch.randn(2, 3, 4)
    _, info = recall(
        x,
        refine_policy=arti.RefinePolicy.fixed(1, trace_level="full"),
        return_info=True,
    )
    plan = recall.route_plan(info)
    block = arti.RecallRouteStack(axis="block", items=(plan,))
    site = arti.RecallRouteStack(axis="site", items=(block,))

    assert arti.component_ref(site) == "arti/recall-route-stack@1"
    spec = arti.component_spec(site)
    assert spec.variant == "runtime-only"
    assert spec.config["schema_version"] == 1
    assert spec.config["axis"] == "site"
    assert spec.config["count"] == 1
    assert spec.config["items"][0]["reference"] == "arti/recall-route-stack@1"
    assert spec.dependencies == ("arti/recall-route-stack@1",)
    assert site.detach().items[0].items[0].weights.grad_fn is None
    assert site.clone().items[0].items[0].weights.data_ptr() != plan.weights.data_ptr()


def test_route_plan_capture_owns_an_independent_tensor_snapshot() -> None:
    recall = arti.Recall(4, 4, activation="none")
    x = torch.randn(2, 3, 4)
    _, info = recall(
        x,
        refine_policy=arti.RefinePolicy.fixed(1, trace_level="full"),
        return_info=True,
    )
    plan = recall.route_plan(info)
    weights = plan.weights.clone()
    routes = plan.route.clone()

    info["recall_bank_weights"].zero_()
    info["recall_route"].zero_()

    torch.testing.assert_close(plan.weights, weights, rtol=0, atol=0)
    torch.testing.assert_close(plan.route, routes, rtol=0, atol=0)


def test_route_stack_fingerprint_includes_ordered_child_layout() -> None:
    recall = arti.Recall(4, 4, activation="none")
    _, info = recall(
        torch.randn(1, 2, 4),
        refine_policy=arti.RefinePolicy.fixed(1, trace_level="full"),
        return_info=True,
    )
    plan = recall.route_plan(info)
    changed = replace(plan, layout_fingerprint="different-layout")

    first = arti.RecallRouteStack(axis="block", items=(plan, changed))
    second = arti.RecallRouteStack(axis="block", items=(changed, plan))

    assert arti.component_spec(first).config_fingerprint != arti.component_spec(
        second
    ).config_fingerprint


def test_route_stack_rejects_ambiguous_structure() -> None:
    with pytest.raises(ValueError, match="axis"):
        arti.RecallRouteStack(axis="", items=(object(),))
    with pytest.raises(ValueError, match="must not be empty"):
        arti.RecallRouteStack(axis="block", items=())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tolerance", float("nan")),
        ("tolerance", float("inf")),
        ("cycle_tolerance", float("nan")),
        ("check_finite", 1),
    ],
)
def test_refine_policy_rejects_ambiguous_runtime_values(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        arti.RefinePolicy(**{field: value})


def test_recall_asset_identity_does_not_include_runtime_budget() -> None:
    recall = arti.Recall(4, 4)
    before = arti.component_spec(recall)
    x = torch.randn(2, 3, 4)
    recall(x, refine_policy=arti.RefinePolicy.fixed(1))
    recall(x, refine_policy=arti.RefinePolicy(max_steps=8, min_steps=2, tolerance=1e-3))
    after = arti.component_spec(recall)

    assert before.config == after.config
    assert before.config_fingerprint == after.config_fingerprint


def test_recall_runtime_state_contract_does_not_include_runtime_budget() -> None:
    updater_a = arti.alpha.NormalizedDeltaRecallValueUpdater(
        hidden_dim=4,
        slots=4,
        workspace_dim=8,
    )
    updater_b = arti.alpha.NormalizedDeltaRecallValueUpdater(
        hidden_dim=4,
        slots=4,
        workspace_dim=8,
    )
    updater_b.load_state_dict(updater_a.state_dict())
    shallow = arti.alpha.RecallRuntime(updater_a, arti.Recall(4, 4))
    deep = arti.alpha.RecallRuntime(updater_b, arti.Recall(4, 4))

    assert shallow.contract_fingerprint == deep.contract_fingerprint
    assert arti.component_ref(shallow) == "arti/recall-runtime@1"
    spec = arti.component_spec(shallow)
    assert spec.variant == "values-only-session"
    assert "arti/recall-state@1" in spec.dependencies
    assert "arti/recall@4" in spec.dependencies
    assert "max_steps" not in spec.config


def test_runtime_policy_controls_depth_and_returns_typed_trace() -> None:
    recall = arti.Recall(4, 4, activation="none")
    x = torch.randn(2, 3, 4)
    policy = arti.RefinePolicy.fixed(3, checkpoints=(1, 3))

    y, trace = recall(
        x,
        refine_policy=policy,
        return_trace=True,
    )

    assert y.shape == x.shape
    assert isinstance(trace, arti.RecallTrace)
    assert trace.step_attempted.shape == (2, 3)
    assert trace.step_committed.shape == (2, 3)
    assert trace.route.shape[:2] == (2, 3)
    assert set(trace.checkpoints) == {1, 3}
    assert trace.diagnostics()["recall_trace_schema"].item() == arti.RECALL_TRACE_SCHEMA_VERSION


def test_zero_step_runtime_policy_is_exact_identity() -> None:
    recall = arti.Recall(4, 4, activation="none")
    x = torch.randn(2, 3, 4)

    y, trace = recall(
        x,
        refine_policy=arti.RefinePolicy.fixed(0),
        return_trace=True,
    )

    torch.testing.assert_close(y, x, rtol=0, atol=0)
    assert trace.max_steps == 0
    assert torch.equal(trace.steps_attempted, torch.zeros(2, dtype=torch.int64))
    assert torch.equal(trace.steps_committed, torch.zeros(2, dtype=torch.int64))


def test_frozen_route_is_versioned_and_reads_current_bank_values() -> None:
    torch.manual_seed(29)
    recall = arti.Recall(
        4,
        8,
        routing="grouped",
        group_size=2,
        group_topk=1,
        activation="none",
    )
    x = torch.randn(2, 3, 4)
    _, info = recall(
        x,
        refine_policy=arti.RefinePolicy.fixed(1, trace_level="full"),
        return_info=True,
    )
    plan = recall.route_plan(info)
    assert arti.component_ref(plan) == "arti/recall-route-plan@1"

    policy = arti.RefinePolicy.fixed(
        2,
        trace_level="full",
    )
    before, before_info = recall(
        x,
        refine_policy=policy,
        route_plan=plan,
        return_info=True,
    )
    with torch.no_grad():
        recall.state.recall.bank.add_(0.25)
    after, after_info = recall(
        x,
        refine_policy=policy,
        route_plan=plan,
        return_info=True,
    )

    assert not torch.equal(after, before)
    torch.testing.assert_close(
        before_info["recall_route_history"][:, 0],
        before_info["recall_route_history"][:, 1],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        after_info["recall_index_history"],
        before_info["recall_index_history"],
        rtol=0,
        atol=0,
    )


def test_frozen_route_keeps_current_bank_gradient_path() -> None:
    recall = arti.Recall(
        4,
        8,
        routing="grouped",
        group_size=2,
        group_topk=1,
        activation="none",
    )
    x = torch.randn(2, 3, 4)
    _, info = recall(
        x,
        refine_policy=arti.RefinePolicy.fixed(1, trace_level="full"),
        return_info=True,
    )
    plan = recall.route_plan(info)
    recall.zero_grad(set_to_none=True)

    y = recall(
        x,
        refine_policy=arti.RefinePolicy.fixed(2),
        route_plan=plan,
    )
    y.square().mean().backward()

    assert recall.state.recall.bank.grad is not None
    assert torch.isfinite(recall.state.recall.bank.grad).all()


def test_frozen_route_detaches_manual_plan_weights_and_route() -> None:
    recall = arti.Recall(
        4,
        8,
        routing="grouped",
        group_size=2,
        group_topk=1,
        activation="none",
    )
    x = torch.randn(2, 3, 4, requires_grad=True)
    _, info = recall(
        x,
        refine_policy=arti.RefinePolicy.fixed(1, trace_level="full"),
        return_info=True,
    )
    original = recall.route_plan(info)
    weights = original.weights.clone().requires_grad_()
    route = original.route.clone().requires_grad_()
    plan = replace(original, weights=weights, route=route)

    recall(x, route_plan=plan).square().mean().backward()

    assert weights.grad is None
    assert route.grad is None
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    ("routing", "formula", "slots", "group_size"),
    [
        ("dense", "arti/delta@1", 8, 2),
        ("dense", "arti/affine@1", 8, 2),
        ("grouped", "arti/delta@1", 8, 2),
        ("grouped", "arti/affine@1", 8, 2),
    ],
)
def test_route_plan_has_one_canonical_layout_and_replays_first_read(
    routing: str,
    formula: str,
    slots: int,
    group_size: int,
) -> None:
    torch.manual_seed(37)
    recall = arti.Recall(
        4,
        slots,
        formula=formula,
        routing=routing,
        group_size=group_size,
        group_topk=1,
        activation="none",
    )
    x = torch.randn(2, 3, 4)
    dynamic, info = recall(
        x,
        refine_policy=arti.RefinePolicy.fixed(1, trace_level="full"),
        return_info=True,
    )
    plan = recall.route_plan(info)
    frozen = recall(x, refine_policy=arti.RefinePolicy.fixed(1), route_plan=plan)

    assert plan.weights.ndim == 4
    assert plan.indices.shape == plan.weights.shape
    assert plan.weights.shape[:3] == (2, 3, recall.state.recall.composition_factor)
    torch.testing.assert_close(frozen, dynamic)


def test_frozen_route_bypasses_query_and_key_but_not_value_bank() -> None:
    torch.manual_seed(41)
    recall = arti.Recall(
        4,
        8,
        routing="grouped",
        group_size=2,
        group_topk=1,
        activation="none",
    )
    x = torch.randn(2, 3, 4)
    _, info = recall(
        x,
        refine_policy=arti.RefinePolicy.fixed(1, trace_level="full"),
        return_info=True,
    )
    plan = recall.route_plan(info)
    before = recall(x, route_plan=plan)
    with torch.no_grad():
        recall.state.recall.query.weight.normal_(std=10.0)
        assert recall.state.recall.key_bank is not None
        recall.state.recall.key_bank.normal_(std=10.0)
    after_routing_change = recall(x, route_plan=plan)
    with torch.no_grad():
        recall.state.recall.bank.add_(0.5)
    after_value_change = recall(x, route_plan=plan)

    torch.testing.assert_close(after_routing_change, before, rtol=0, atol=0)
    assert not torch.equal(after_value_change, before)


def test_dynamic_refine_requeries_after_each_state_change() -> None:
    recall = arti.Recall(2, 2, activation="none")
    with torch.no_grad():
        recall.state.recall.bank.copy_(torch.eye(2))
    x = torch.tensor([[[1.0, 0.0]]])

    _, info = recall(
        x,
        refine_policy=arti.RefinePolicy.fixed(3, trace_level="full"),
        return_info=True,
    )

    routes = info["recall_route_history"][0, :, 0]
    assert not torch.equal(routes[0], routes[1])
    assert not torch.equal(routes[1], routes[2])
    assert torch.all(info["recall_step_route_change"][:, 1:] > 0)


def test_per_sample_convergence_freezes_only_converged_samples() -> None:
    recall = arti.Recall(4, 2, formula="arti/delta@1", activation="none")
    x = torch.ones(2, 1, 4)
    memory = torch.stack((torch.zeros(2, 4), torch.ones(2, 4)))
    policy = arti.RefinePolicy(max_steps=4, min_steps=1, tolerance=0.0)

    y, trace = recall(x, memory=memory, refine_policy=policy, return_trace=True)

    assert trace.steps_attempted.tolist() == [1, 4]
    assert trace.steps_committed.tolist() == [1, 4]
    assert trace.step_attempted[0].tolist() == [True, False, False, False]
    assert trace.step_attempted[1].tolist() == [True, True, True, True]
    torch.testing.assert_close(y[0], x[0], rtol=0, atol=0)
    assert not torch.equal(y[1], x[1])


def test_nonfinite_candidate_is_local_to_one_batch_item() -> None:
    recall = arti.Recall(4, 2, formula="arti/delta@1", activation="none")
    x = torch.ones(2, 1, 4)
    memory = torch.stack((torch.full((2, 4), torch.nan), torch.ones(2, 4)))

    with torch.no_grad():
        y, info = recall(
            x,
            memory=memory,
            refine_policy=arti.RefinePolicy.fixed(
                2,
                check_finite=True,
                trace_level="full",
            ),
            return_info=True,
        )
    trace = arti.RecallTrace.from_diagnostics(info)

    assert trace.stop_reason[0].item() == int(arti.RecallStopReason.NONFINITE)
    assert trace.steps_attempted[0].item() == 1
    assert trace.steps_committed[0].item() == 0
    torch.testing.assert_close(y[0], x[0], rtol=0, atol=0)
    assert torch.isfinite(y[1]).all()
    assert torch.count_nonzero(info["recall_route"][0]) == 0
    assert torch.count_nonzero(info["recall_bank_weights"][0]) == 0
    assert torch.all(info["recall_bank_indices"][0] == -1)
    assert torch.count_nonzero(trace.route[0, 0]) > 0
    committed_mask = trace.step_committed.reshape(2, 2, 1, 1, 1)
    committed_route = torch.where(committed_mask, trace.route, torch.zeros_like(trace.route))
    assert torch.count_nonzero(committed_route[0]) == 0
    assert torch.count_nonzero(committed_route[1]) > 0


def test_disabling_finite_checks_does_not_silently_sanitize_values() -> None:
    recall = arti.Recall(4, 2, formula="arti/delta@1", activation="none")
    x = torch.ones(1, 1, 4)
    memory = torch.full((1, 2, 4), torch.nan)

    y = recall(
        x,
        memory=memory,
        refine_policy=arti.RefinePolicy.fixed(1, check_finite=False),
    )

    assert torch.isnan(y).all()


def test_training_policy_raises_before_nonfinite_backward() -> None:
    recall = arti.Recall(4, 2, formula="arti/delta@1", activation="none")
    memory = torch.full((1, 2, 4), torch.nan, requires_grad=True)

    with pytest.raises(RuntimeError, match="non-finite"):
        recall(
            torch.ones(1, 1, 4),
            memory=memory,
            refine_policy=arti.RefinePolicy.fixed(1, nonfinite_action="raise"),
        )

    assert memory.grad is None


def test_masked_nonfinite_token_does_not_stop_a_healthy_sample() -> None:
    recall = arti.Recall(4, 2, formula="arti/delta@1", activation="none")
    x = torch.ones(1, 2, 4)
    x[:, 1] = torch.nan
    mask = torch.tensor([[True, False]])

    y, trace = recall(
        x,
        mask=mask,
        refine_policy=arti.RefinePolicy.fixed(2),
        return_trace=True,
    )

    assert trace.stop_reason.item() == int(arti.RecallStopReason.MAX_STEPS)
    assert torch.isfinite(y[:, 0]).all()
    assert torch.isnan(y[:, 1]).all()


def test_masked_nan_matches_clean_padding_forward_and_backward() -> None:
    torch.manual_seed(53)
    clean = arti.Recall(
        4,
        8,
        routing="grouped",
        group_size=2,
        group_topk=1,
        activation="none",
    )
    contaminated = arti.Recall(
        4,
        8,
        routing="grouped",
        group_size=2,
        group_topk=1,
        activation="none",
    )
    contaminated.load_state_dict(clean.state_dict())
    x_clean = torch.randn(2, 3, 4, requires_grad=True)
    x_nan = x_clean.detach().clone().requires_grad_()
    with torch.no_grad():
        x_clean[:, 2].zero_()
        x_nan[:, 2].fill_(torch.nan)
    mask = torch.tensor([[True, True, False], [True, True, False]])

    clean_y = clean(x_clean, mask=mask)
    nan_y = contaminated(x_nan, mask=mask)
    clean_y[:, :2].square().mean().backward()
    nan_y[:, :2].square().mean().backward()

    torch.testing.assert_close(nan_y[:, :2], clean_y[:, :2])
    torch.testing.assert_close(x_nan.grad[:, :2], x_clean.grad[:, :2])
    for clean_parameter, nan_parameter in zip(
        clean.parameters(), contaminated.parameters(), strict=True
    ):
        if clean_parameter.grad is None:
            assert nan_parameter.grad is None
        else:
            assert torch.isfinite(nan_parameter.grad).all()
            torch.testing.assert_close(nan_parameter.grad, clean_parameter.grad)


def test_top_level_read_is_last_committed_not_later_speculative() -> None:
    recall = arti.Recall(4, 2, formula="arti/delta@1", activation="none")
    x = torch.ones(2, 1, 4)
    memory = torch.stack((torch.zeros(2, 4), torch.ones(2, 4)))
    _, info = recall(
        x,
        memory=memory,
        refine_policy=arti.RefinePolicy(
            max_steps=4,
            min_steps=1,
            tolerance=0.0,
            trace_level="full",
        ),
        return_info=True,
    )

    assert info["recall_steps_attempted"].tolist() == [1, 4]
    torch.testing.assert_close(
        info["recall_route"][0],
        info["recall_route_history"][0, 0],
        rtol=0,
        atol=0,
    )


def test_trace_rejects_unknown_schema() -> None:
    recall = arti.Recall(4, 2)
    _, info = recall(
        torch.randn(1, 2, 4),
        refine_policy=arti.RefinePolicy(trace_level="routes"),
        return_info=True,
    )
    info["recall_trace_schema"] = torch.tensor(999)

    with pytest.raises(ValueError, match="schema"):
        arti.RecallTrace.from_diagnostics(info)


def test_default_execution_does_not_record_step_history() -> None:
    recall = arti.Recall(4, 2)
    _, info = recall(torch.randn(1, 2, 4), return_info=True)

    assert "recall_route_history" not in info
    assert "recall_context_history" not in info


def test_gradient_checkpoint_supports_sparse_depth_supervision() -> None:
    recall = arti.Recall(4, 4, activation="none")
    x = torch.randn(2, 3, 4)
    policy = arti.RefinePolicy.fixed(
        3,
        checkpoints=(1, 3),
        checkpoint_mode="gradient",
        trace_level="routes",
    )

    _, trace = recall(x, refine_policy=policy, return_trace=True)
    trace.checkpoints[1].square().mean().add(trace.checkpoints[3].square().mean()).backward()

    assert recall.state.recall.bank.grad is not None
    assert torch.isfinite(recall.state.recall.bank.grad).all()


def test_convergence_has_priority_over_cycle_detection() -> None:
    recall = arti.Recall(4, 2, formula="arti/delta@1", activation="none")
    x = torch.ones(1, 1, 4)
    memory = torch.zeros(1, 2, 4)
    policy = arti.RefinePolicy(
        max_steps=4,
        min_steps=1,
        tolerance=0.0,
        cycle_tolerance=0.0,
        trace_level="routes",
    )

    _, trace = recall(x, memory=memory, refine_policy=policy, return_trace=True)

    assert trace.stop_reason.item() == int(arti.RecallStopReason.CONVERGED)
    assert trace.steps_attempted.item() == 1
    assert trace.steps_committed.item() == 1


def test_exact_two_cycle_stops_and_records_committed_steps() -> None:
    recall = arti.Recall(4, 2, formula=_NegateStateFormula(), activation="none")
    x = torch.randn(2, 1, 4)
    policy = arti.RefinePolicy(
        max_steps=8,
        min_steps=2,
        cycle_tolerance=0.0,
        cycle_periods=(2,),
        trace_level="routes",
    )

    _, trace = recall(x, refine_policy=policy, return_trace=True)

    assert trace.stop_reason.tolist() == [int(arti.RecallStopReason.CYCLE)] * 2
    assert trace.steps_attempted.tolist() == [2, 2]
    assert trace.steps_committed.tolist() == [2, 2]


def test_compile_fullgraph_runs_canonical_refine_without_typed_trace() -> None:
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    recall = arti.Recall(
        4,
        8,
        routing="grouped",
        group_size=2,
        group_topk=1,
        activation="none",
    )
    policy = arti.RefinePolicy.fixed(2)
    compiled = torch.compile(recall, backend="eager", fullgraph=True)
    x = torch.randn(2, 3, 4)

    expected = recall(x, refine_policy=policy)
    actual = compiled(x, refine_policy=policy)

    torch.testing.assert_close(actual, expected)


def test_compile_fullgraph_runs_adaptive_static_masked_refine() -> None:
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    recall = arti.Recall(4, 8, activation="none", breadth_mode="mixed")
    policy = arti.RefinePolicy.adaptive(
        max_steps=3,
        min_steps=3,
        scope="token",
        relative_tolerance=1e-6,
        executor="static_masked",
        trace_level="none",
    )
    compiled = torch.compile(recall, backend="eager", fullgraph=True)
    x = torch.randn(2, 3, 4)

    torch.testing.assert_close(
        compiled(x, refine_policy=policy),
        recall(x, refine_policy=policy),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_refine_and_frozen_route_preserve_device_dtype_and_grad(dtype) -> None:
    recall = arti.Recall(
        8,
        16,
        routing="grouped",
        group_size=4,
        group_topk=1,
        activation="none",
    ).cuda().to(dtype=dtype)
    x = torch.randn(2, 4, 8, device="cuda", dtype=dtype)
    _, info = recall(
        x,
        refine_policy=arti.RefinePolicy.fixed(1, trace_level="full"),
        return_info=True,
    )
    plan = recall.route_plan(info)
    recall.zero_grad(set_to_none=True)

    y = recall(x, refine_policy=arti.RefinePolicy.fixed(3), route_plan=plan)
    y.float().square().mean().backward()

    assert y.device.type == "cuda" and y.dtype == dtype
    assert plan.indices.device.type == "cuda" and plan.indices.dtype == torch.long
    assert recall.state.recall.bank.grad is not None
    assert torch.isfinite(recall.state.recall.bank.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_adaptive_refine_preserves_device_dtype_trace_and_grad(dtype) -> None:
    recall = arti.Recall(8, 16, activation="none").cuda().to(dtype=dtype)
    x = torch.randn(2, 4, 8, device="cuda", dtype=dtype, requires_grad=True)
    policy = arti.RefinePolicy.adaptive(
        max_steps=4,
        min_steps=2,
        scope="token",
        relative_tolerance=1e-5,
        patience=2,
        trace_level="routes",
        executor="early_break",
    )

    y, trace = recall(x, refine_policy=policy, return_trace=True)
    y.float().square().mean().backward()

    assert y.device.type == "cuda" and y.dtype == dtype
    assert trace.token_steps_committed.device.type == "cuda"
    assert trace.logical_token_steps.device.type == "cuda"
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert recall.state.recall.bank.grad is not None
    assert torch.isfinite(recall.state.recall.bank.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_amp_dynamic_and_frozen_route_match_for_one_read() -> None:
    torch.manual_seed(67)
    recall = arti.Recall(
        8,
        16,
        routing="grouped",
        group_size=4,
        group_topk=2,
        activation="none",
    ).cuda()
    x = torch.randn(2, 4, 8, device="cuda")
    policy = arti.RefinePolicy.fixed(1, trace_level="full")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        dynamic, info = recall(x, refine_policy=policy, return_info=True)
        plan = recall.route_plan(info)
        frozen = recall(x, refine_policy=policy, route_plan=plan)

    torch.testing.assert_close(frozen.float(), dynamic.float(), rtol=2e-2, atol=2e-3)
    assert plan.indices.device.type == "cuda"
    assert torch.isfinite(frozen).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_fullgraph_refine_matches_eager() -> None:
    recall = arti.Recall(
        8,
        16,
        routing="grouped",
        group_size=4,
        group_topk=1,
        activation="none",
    ).cuda()
    policy = arti.RefinePolicy.fixed(3)
    compiled = torch.compile(recall, backend="eager", fullgraph=True)
    x = torch.randn(2, 4, 8, device="cuda")

    torch.testing.assert_close(
        compiled(x, refine_policy=policy),
        recall(x, refine_policy=policy),
    )
