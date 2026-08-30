from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import arti


class _StepSignal(torch.nn.Module):
    def __init__(self, exit_at: int) -> None:
        super().__init__()
        self.exit_at = exit_at
        self.calls = 0

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        value = 1.0 if self.calls >= self.exit_at else -1.0
        return state.new_full((state.shape[0],), value)


class _RowSignal(torch.nn.Module):
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape != (2, 4):
            raise ValueError("test source expects two flattened branch rows")
        return state.new_tensor([1.0, -1.0])


def _policy(*, max_steps: int, min_steps: int) -> arti.AdaptiveRefinePolicy:
    return arti.RefinePolicy.adaptive(
        max_steps=max_steps,
        min_steps=min_steps,
        scope="token",
        absolute_tolerance=0.0,
        relative_tolerance=1e-12,
        patience=max_steps + 1,
        executor="static_masked",
        trace_level="routes",
    )


def _recall() -> arti.Recall:
    torch.manual_seed(8041)
    return arti.Recall(4, 8, activation="none")


def test_refine_exit_atom_preserves_trainable_score() -> None:
    source = torch.nn.Linear(4, 1)
    control = arti.alpha.RefineExitControl(source)
    state = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    request = control(state, mask=mask)
    request.score[mask].sum().backward()

    assert request.requested.shape == mask.shape
    assert request.requested.dtype == torch.bool
    assert torch.equal(request.requested & ~mask, torch.zeros_like(mask))
    assert source.weight.grad is not None
    assert torch.isfinite(source.weight.grad).all()


def test_refine_exit_atom_requires_declared_signal_kind() -> None:
    mask = torch.ones(1, 2, dtype=torch.bool)
    predicate = arti.alpha.FormulaRefineExit(input_kind="predicate")
    logit = arti.alpha.FormulaRefineExit(input_kind="logit")

    assert torch.all(predicate(torch.ones(1, 2, dtype=torch.bool), mask=mask).requested)
    with pytest.raises(TypeError, match="predicate"):
        predicate(torch.ones(1, 2), mask=mask)
    with pytest.raises(TypeError, match="logit"):
        logit(torch.ones(1, 2, dtype=torch.bool), mask=mask)


def test_refine_exit_components_are_versioned_separately_from_formula_v2() -> None:
    atom = arti.alpha.FormulaRefineExit(
        input_kind="logit",
        scope="branch",
        threshold=0.25,
    )
    control = arti.alpha.RefineExitControl(torch.nn.Linear(4, 1), atom=atom)
    request = atom(
        torch.tensor([1.0]),
        mask=torch.tensor([[True, False]]),
    )

    assert arti.component_ref(atom) == "arti/formula-atom-refine-exit@1"
    assert arti.component_ref(control) == "arti/refine-exit-control@1"
    assert arti.component_ref(request) == "arti/refine-exit-request@1"
    assert arti.component_spec(control).dependencies == (
        "arti/formula-atom-refine-exit@1",
    )
    catalog = {item["ref"]: item for item in arti.component_catalog()}
    assert catalog["arti/refine-exit-control@1"]["artifact_policy"] == "runtime_only"
    assert catalog["arti/refine-exit-request@1"]["artifact_policy"] == "runtime_only"
    assert catalog["arti/recall-trace@3"]["artifact_policy"] == "runtime_only"


def test_model_exit_commits_current_step_and_matches_fixed_depth_prefix() -> None:
    torch.manual_seed(8041)
    recall = arti.Recall(4, 8, activation="none", breadth_mode="mixed")
    x = torch.randn(2, 3, 4)
    expected = recall(x, refine_policy=_policy(max_steps=3, min_steps=3))
    control = arti.alpha.RefineExitControl(_StepSignal(exit_at=3))

    actual, trace = recall(
        x,
        refine_policy=_policy(max_steps=6, min_steps=1),
        refine_exit=control,
        model_exit=True,
        return_trace=True,
    )

    assert isinstance(trace, arti.RecallTraceV3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.all(trace.token_steps_attempted == 3)
    assert torch.all(trace.token_stop_reason == int(arti.RecallStopReason.MODEL_EXIT))
    assert torch.all(trace.model_exit_stop[:, 2])
    assert arti.component_ref(trace) == "arti/recall-trace@3"
    trace.base.validate()
    trace.validate()
    rebuilt = arti.RecallTraceV3.from_diagnostics(trace.diagnostics())
    rebuilt.base.validate()
    rebuilt.validate()
    forged_base = replace(
        trace.base,
        token_stop_reason=torch.full_like(
            trace.base.token_stop_reason,
            int(arti.RecallStopReason.CONVERGED),
        ),
    )
    with pytest.raises(ValueError, match="V2-safe projection"):
        replace(trace, base=forged_base).validate()


def test_legacy_trace_rejects_model_exit_reason() -> None:
    recall = _recall()
    _, trace = recall(
        torch.randn(1, 2, 4),
        refine_policy=arti.RefinePolicy.fixed(1, trace_level="routes"),
        return_trace=True,
    )
    forged = replace(
        trace,
        stop_reason=torch.full_like(
            trace.stop_reason,
            int(arti.RecallStopReason.MODEL_EXIT),
        ),
    )

    with pytest.raises(ValueError, match="unknown reason"):
        forged.validate()


def test_minimum_depth_records_blocked_requests_before_exit() -> None:
    recall = _recall()
    x = torch.randn(1, 2, 4)
    control = arti.alpha.RefineExitControl(_StepSignal(exit_at=1))

    _, trace = recall(
        x,
        refine_policy=_policy(max_steps=6, min_steps=3),
        refine_exit=control,
        model_exit=True,
        return_trace=True,
    )

    assert torch.all(trace.token_steps_attempted == 3)
    assert torch.all(trace.exit_requested[:, :3])
    assert torch.all(trace.blocked_by_min_steps[:, :2])
    assert not torch.any(trace.exit_effective[:, :2])
    assert torch.all(trace.model_exit_stop[:, 2])


def test_equal_minimum_and_maximum_keeps_max_steps_terminal_reason() -> None:
    recall = _recall()
    x = torch.randn(1, 2, 4)
    control = arti.alpha.RefineExitControl(_StepSignal(exit_at=1))

    _, trace = recall(
        x,
        refine_policy=_policy(max_steps=3, min_steps=3),
        refine_exit=control,
        model_exit=True,
        return_trace=True,
    )

    assert torch.all(trace.token_steps_attempted == 3)
    assert torch.all(trace.token_stop_reason == int(arti.RecallStopReason.MAX_STEPS))
    assert not torch.any(trace.model_exit_stop)


def test_nonfinite_exit_signal_stops_after_committed_prefix() -> None:
    recall = _recall()
    x = torch.randn(1, 2, 4)
    one_step = recall(x, refine_policy=_policy(max_steps=1, min_steps=1))
    source = torch.nn.Linear(4, 1)
    with torch.no_grad():
        source.weight.fill_(float("nan"))
    control = arti.alpha.RefineExitControl(source)

    with torch.no_grad():
        actual, trace = recall(
            x,
            refine_policy=_policy(max_steps=4, min_steps=1),
            refine_exit=control,
            model_exit=True,
            return_trace=True,
        )

    torch.testing.assert_close(actual, one_step, rtol=0, atol=0)
    assert torch.all(trace.token_steps_committed == 1)
    assert torch.all(trace.token_stop_reason == int(arti.RecallStopReason.NONFINITE))
    assert not torch.any(trace.exit_requested)
    trace.validate()


def test_nonfinite_exit_signal_stops_during_autograd() -> None:
    recall = _recall()
    x = torch.randn(1, 2, 4, requires_grad=True)
    source = torch.nn.Linear(4, 1)
    with torch.no_grad():
        source.weight.fill_(float("nan"))
    control = arti.alpha.RefineExitControl(source)

    y, trace = recall(
        x,
        refine_policy=_policy(max_steps=4, min_steps=1),
        refine_exit=control,
        model_exit=True,
        return_trace=True,
    )
    y.square().mean().backward()

    assert torch.all(trace.token_stop_reason == int(arti.RecallStopReason.NONFINITE))
    assert torch.isfinite(trace.exit_score).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    trace.validate()


def test_model_exit_false_preserves_existing_trace_and_output() -> None:
    recall = _recall()
    x = torch.randn(1, 2, 4)
    policy = _policy(max_steps=3, min_steps=1)
    expected, expected_trace = recall(x, refine_policy=policy, return_trace=True)
    control = arti.alpha.RefineExitControl(_StepSignal(exit_at=1))

    actual, trace = recall(
        x,
        refine_policy=policy,
        refine_exit=control,
        model_exit=False,
        return_trace=True,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert isinstance(trace, arti.RecallTraceV2)
    torch.testing.assert_close(
        trace.token_steps_attempted,
        expected_trace.token_steps_attempted,
        rtol=0,
        atol=0,
    )
    assert control.source.calls == 0


def test_model_exit_is_disabled_by_default() -> None:
    recall = _recall()
    x = torch.randn(1, 2, 4)
    policy = _policy(max_steps=3, min_steps=1)
    control = arti.alpha.RefineExitControl(_StepSignal(exit_at=1))

    _, trace = recall(
        x,
        refine_policy=policy,
        refine_exit=control,
        return_trace=True,
    )

    assert isinstance(trace, arti.RecallTraceV2)
    assert control.source.calls == 0


def test_refine_exit_requires_static_finite_policy() -> None:
    recall = _recall()
    control = arti.alpha.RefineExitControl(_StepSignal(exit_at=1))
    dynamic = _policy(max_steps=2, min_steps=1).replace(executor="early_break")
    unchecked = _policy(max_steps=2, min_steps=1).replace(check_finite=False)

    with pytest.raises(ValueError, match="static_masked"):
        recall(
            torch.randn(1, 2, 4),
            refine_policy=dynamic,
            refine_exit=control,
            model_exit=True,
        )
    with pytest.raises(ValueError, match="finite-state"):
        recall(
            torch.randn(1, 2, 4),
            refine_policy=unchecked,
            refine_exit=control,
            model_exit=True,
        )


def test_batched_refine_exit_is_branch_local_without_k_aggregation() -> None:
    torch.manual_seed(8042)
    recall = arti.Recall(
        4,
        8,
        activation="none",
        breadth=2,
        breadth_mode="independent",
    )
    x = torch.randn(1, 2, 4)
    control = arti.alpha.RefineExitControl(_RowSignal(), scope="branch")

    _, result = recall(
        x,
        active_k=2,
        refine_policy=_policy(max_steps=4, min_steps=1),
        refine_exit=control,
        model_exit=True,
        return_branches=True,
    )

    attempted = result.branch_diagnostics["recall_token_steps_attempted"]
    assert attempted.shape == (1, 2, 2)
    assert torch.all(attempted[0, 0] == 1)
    assert torch.all(attempted[0, 1] == 4)
    stopped = result.branch_diagnostics["recall_model_exit_stop"]
    assert torch.all(stopped[0, 0, 0])
    assert not torch.any(stopped[0, 1])


def test_batched_refine_exit_is_k3_permutation_equivariant() -> None:
    torch.manual_seed(8043)
    recall = arti.Recall(4, 12, activation="none")
    x = torch.randn(2, 3, 4)
    candidates = arti.alpha.query_recall_branches(recall, x, max_k=3, active_k=3)
    source = torch.nn.Linear(4, 1)
    control = arti.alpha.RefineExitControl(source)
    policy = _policy(max_steps=4, min_steps=1)
    original = arti.alpha.run_batched_refine(
        recall,
        x,
        candidates=candidates,
        refine_policy=policy,
        refine_exit=control,
        model_exit=True,
    )
    order = torch.tensor([[2, 0, 1], [1, 2, 0]])
    permuted = arti.alpha.run_batched_refine(
        recall,
        x,
        candidates=candidates.permute_branches(order),
        refine_policy=policy,
        refine_exit=control,
        model_exit=True,
    )
    batch = torch.arange(x.shape[0]).unsqueeze(1)

    torch.testing.assert_close(permuted.value, original.value[batch, order])
    torch.testing.assert_close(
        permuted.branch_diagnostics["recall_exit_score"],
        original.branch_diagnostics["recall_exit_score"][batch, order],
    )
    assert torch.equal(
        permuted.branch_diagnostics["recall_model_exit_stop"],
        original.branch_diagnostics["recall_model_exit_stop"][batch, order],
    )


def test_packed_empty_branches_still_validate_exit_policy() -> None:
    recall = arti.Recall(4, 8, activation="none")
    x = torch.randn(1, 2, 4)
    candidates = arti.alpha.query_recall_branches(
        recall,
        x,
        mask=torch.zeros(1, 2, dtype=torch.bool),
        max_k=2,
        active_k=2,
    )
    plan = arti.alpha.BatchedRefinePlan.recall_only(execution_layout="packed_active")
    control = arti.alpha.RefineExitControl(torch.nn.Linear(4, 1))

    with pytest.raises(arti.alpha.BatchedRefineContractError, match="Adaptive"):
        arti.alpha.run_batched_refine(
            recall,
            x,
            candidates=candidates,
            plan=plan,
            refine_policy=arti.RefinePolicy.fixed(2),
            refine_exit=control,
            model_exit=True,
        )
    with pytest.raises(arti.alpha.BatchedRefineContractError, match="finite-state"):
        arti.alpha.run_batched_refine(
            recall,
            x,
            candidates=candidates,
            plan=plan,
            refine_policy=_policy(max_steps=2, min_steps=1).replace(
                check_finite=False
            ),
            refine_exit=control,
            model_exit=True,
        )


def test_refine_exit_rejects_legacy_refine_policy() -> None:
    recall = _recall()
    control = arti.alpha.RefineExitControl(_StepSignal(exit_at=1))

    with pytest.raises(ValueError, match="AdaptiveRefinePolicy"):
        recall(
            torch.randn(1, 2, 4),
            refine_policy=arti.RefinePolicy.fixed(2),
            refine_exit=control,
            model_exit=True,
        )


def test_refine_exit_rejects_bare_module_provider() -> None:
    recall = _recall()

    with pytest.raises(TypeError, match="RefineExitControl"):
        recall(
            torch.randn(1, 2, 4),
            refine_policy=_policy(max_steps=2, min_steps=1),
            refine_exit=torch.nn.Identity(),
            model_exit=True,
        )


def test_refine_exit_static_masked_fullgraph_matches_eager() -> None:
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    torch.manual_seed(8041)
    recall = arti.Recall(4, 8, activation="none", breadth_mode="mixed")
    source = torch.nn.Linear(4, 1)
    with torch.no_grad():
        source.weight.zero_()
        source.bias.fill_(1.0)
    control = arti.alpha.RefineExitControl(source)
    policy = _policy(max_steps=4, min_steps=2)
    compiled = torch.compile(recall, backend="eager", fullgraph=True)
    x = torch.randn(2, 3, 4)

    torch.testing.assert_close(
        compiled(
            x,
            refine_policy=policy,
            refine_exit=control,
            model_exit=True,
        ),
        recall(x, refine_policy=policy, refine_exit=control, model_exit=True),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_refine_exit_cuda_static_masked_smoke() -> None:
    recall = _recall().cuda()
    control = arti.alpha.RefineExitControl(torch.nn.Linear(4, 1)).cuda()
    x = torch.randn(2, 3, 4, device="cuda", requires_grad=True)

    y, trace = recall(
        x,
        refine_policy=_policy(max_steps=4, min_steps=2),
        refine_exit=control,
        model_exit=True,
        return_trace=True,
    )
    (y.square().mean() + trace.exit_score.square().mean()).backward()

    assert y.is_cuda and trace.exit_requested.is_cuda
    assert control.source.weight.grad is not None
    assert torch.isfinite(control.source.weight.grad).all()
