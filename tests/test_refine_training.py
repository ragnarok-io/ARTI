from __future__ import annotations

import pytest
import torch
from unittest.mock import patch

import arti
from arti import alpha
from arti.recall_formula import FactorSpec, RecallFormulaContract
from arti.recall_experts import canonical_tensor_state_sha256


def _policy(steps: int) -> arti.AdaptiveRefinePolicy:
    return arti.RefinePolicy.adaptive(
        max_steps=steps,
        min_steps=steps,
        scope="token",
        relative_tolerance=1e-12,
        trace_level="routes",
    )


def _query_hash(recall: arti.Recall) -> str:
    return canonical_tensor_state_sha256(
        {"query.weight": recall.state.recall.query.weight}
    )


class _NonfiniteFormula(torch.nn.Module):
    recall_formula_contract = RecallFormulaContract(
        factors=(FactorSpec("value"),),
        identity_preserving=False,
    )

    def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        del factors
        triggered = state.abs().sum(dim=-1, keepdim=True) > 0
        return torch.where(
            triggered,
            torch.full_like(state, float("nan")),
            torch.zeros_like(state),
        )


class _Program:
    def __init__(self, fingerprint: str) -> None:
        self.fingerprint = fingerprint


class _ProgramTaggedFormula(torch.nn.Module):
    recall_formula_contract = RecallFormulaContract(
        factors=(FactorSpec("content"),),
        identity_preserving=False,
    )

    def __init__(self, fingerprint: str) -> None:
        super().__init__()
        self.program = _Program(fingerprint)

    def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        return state + factors[..., 0, :]


def test_refine_training_components_are_alpha_runtime_contracts() -> None:
    trainer = alpha.RefineStepTraining(max_snapshot_staleness=2)
    recall = arti.Recall(4, 8, breadth=1, activation="none")
    rollout = trainer.capture(
        recall,
        torch.randn(1, 2, 4),
        policy=_policy(2),
    )

    assert arti.component_ref(trainer) == "arti/refine-step-training@1"
    assert arti.component_ref(rollout) == "arti/refine-rollout@1"
    assert arti.get_component_registry().registration_for(trainer).artifact_policy == (
        "runtime_only"
    )
    assert arti.get_component_registry().registration_for(rollout).artifact_policy == (
        "runtime_only"
    )
    assert arti.component_spec(rollout).dependencies == ("arti/recall@4",)


def test_capture_flattens_detached_adjacent_states_and_masks() -> None:
    torch.manual_seed(3101)
    recall = arti.Recall(4, 8, breadth=1, activation="none")
    trainer = alpha.RefineStepTraining()
    x = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    rollout = trainer.capture(
        recall,
        x,
        mask=mask,
        policy=_policy(3),
        snapshot_generation=7,
    )

    assert rollout.hidden_state.shape == (6, 3, 4)
    assert rollout.mask.shape == (6, 3)
    assert rollout.trajectory_id.tolist() == [0, 0, 0, 1, 1, 1]
    assert rollout.step_index.tolist() == [0, 1, 2, 0, 1, 2]
    assert rollout.sample_id.tolist() == [0, 0, 0, 1, 1, 1]
    assert rollout.branch_id.tolist() == [0] * 6
    assert not rollout.hidden_state.requires_grad
    assert rollout.hidden_state.grad_fn is None
    assert not hasattr(rollout, "next_hidden")
    assert not hasattr(rollout, "teacher_hidden")
    torch.testing.assert_close(rollout.hidden_state[0], x[0].detach())
    torch.testing.assert_close(rollout.hidden_state[3], x[1].detach())
    assert not rollout.valid_token_mask[:, 2].any()
    assert rollout.snapshot_generation == 7


def test_capture_requires_fixed_depth_sampling() -> None:
    recall = arti.Recall(4, 8, breadth=1, activation="none")
    adaptive = arti.RefinePolicy.adaptive(
        max_steps=4,
        min_steps=1,
        scope="token",
        relative_tolerance=1e-4,
    )

    with pytest.raises(alpha.RefineTrainingContractError, match="fixed depth"):
        alpha.RefineStepTraining().capture(
            recall,
            torch.randn(1, 2, 4),
            policy=adaptive,
        )


def test_capture_rejects_stochastic_recall_until_rng_identity_is_supported() -> None:
    recall = arti.Recall(4, 8)

    with pytest.raises(alpha.RefineTrainingContractError, match="deterministic"):
        alpha.RefineStepTraining().capture(
            recall,
            torch.randn(1, 2, 4),
            policy=_policy(2),
        )


def test_replay_is_one_step_permutation_equivalent_and_freshly_queried() -> None:
    torch.manual_seed(3102)
    recall = arti.Recall(4, 12, breadth=1, activation="none")
    trainer = alpha.RefineStepTraining()
    rollout = trainer.capture(
        recall,
        torch.randn(2, 2, 4),
        policy=_policy(4),
    )
    field = recall.state.recall
    with patch.object(field, "_project_query", wraps=field._project_query) as query_spy:
        result = trainer.replay(recall, rollout)
    seen = [call.args[0].detach().clone() for call in query_spy.call_args_list]

    assert seen
    seen_rows = torch.cat(seen, dim=0)
    expected_rows = rollout.hidden_state[rollout.valid_item_mask]
    assert seen_rows.shape == expected_rows.shape
    assert torch.equal(
        torch.sort(seen_rows.flatten(1).sum(dim=1)).values,
        torch.sort(expected_rows.flatten(1).sum(dim=1)).values,
    )

    order = torch.tensor([7, 0, 5, 2, 6, 3, 1, 4])
    permuted = rollout.permute(order)
    permuted_result = trainer.replay(recall, permuted)
    inverse = torch.argsort(order)
    torch.testing.assert_close(
        permuted_result.value.index_select(0, inverse),
        result.value,
    )
    torch.testing.assert_close(
        permuted_result.route.index_select(0, inverse),
        result.route,
    )


def test_query_is_fixed_excluded_from_optimizer_and_unchanged_by_training() -> None:
    torch.manual_seed(3103)
    recall = arti.Recall(4, 8, breadth=1, activation="none")
    trainer = alpha.RefineStepTraining()
    forbidden = torch.optim.SGD(recall.parameters(), lr=0.01)
    with pytest.raises(alpha.RefineTrainingContractError, match="optimizer"):
        trainer.assert_optimizer_contract(recall, forbidden)

    optimizer = torch.optim.SGD([recall.state.recall.bank], lr=0.05)
    trainer.assert_optimizer_contract(recall, optimizer)
    rollout = trainer.capture(recall, torch.randn(2, 3, 4), policy=_policy(3))
    query_before = _query_hash(recall)
    bank_before = recall.state.recall.bank.detach().clone()
    result = trainer.replay(recall, rollout)
    per_token = result.value.square().mean(dim=-1)
    loss = trainer.reduce_task_loss(
        per_token,
        result,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    assert recall.state.recall.query.weight.grad is None
    optimizer.step()

    assert _query_hash(recall) == query_before
    assert not torch.equal(recall.state.recall.bank, bank_before)
    assert loss.valid_trajectories.item() == 2


def test_real_downstream_classification_loss_trains_bank_not_query() -> None:
    torch.manual_seed(3105)
    recall = arti.Recall(4, 12, breadth=1, activation="none")
    task_head = torch.nn.Linear(4, 3, bias=False)
    task_head.requires_grad_(False)
    labels = torch.tensor([0, 2])
    trainer = alpha.RefineStepTraining()
    rollout = trainer.capture(recall, torch.randn(2, 2, 4), policy=_policy(3))
    result = trainer.replay(recall, rollout)
    logits = task_head(result.value.mean(dim=1))
    row_labels = labels.index_select(0, result.sample_id)
    task_loss = torch.nn.functional.cross_entropy(logits, row_labels, reduction="none")
    loss = trainer.reduce_task_loss(task_loss, result)
    loss.total.backward()

    assert recall.state.recall.bank.grad is not None
    assert torch.isfinite(recall.state.recall.bank.grad).all()
    assert recall.state.recall.query.weight.grad is None
    assert all(parameter.grad is None for parameter in task_head.parameters())


def test_loss_is_balanced_per_trajectory_branch_and_source_sample() -> None:
    trainer = alpha.RefineStepTraining()
    valid = torch.ones(8, 1, dtype=torch.bool)
    result = alpha.RefineStepTrainingResult(
        value=torch.zeros(8, 1, 2),
        valid_token_mask=valid,
        trajectory_id=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        sample_id=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        branch_id=torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]),
        step_index=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
        trajectory_count=4,
        sample_count=2,
        route=torch.empty(8, 1, 0),
        indices=torch.empty(8, 1, 0, dtype=torch.int64),
        weights=torch.empty(8, 1, 0),
    )
    loss = trainer.reduce_task_loss(
        torch.tensor([0.0, 2.0, 4.0, 4.0, 10.0, 10.0, 10.0, 10.0]),
        result,
        baseline_loss=torch.tensor([1.0, 1.0, 5.0, 5.0, 12.0, 12.0, 12.0, 12.0]),
        improvement_weight=0.25,
    )

    torch.testing.assert_close(loss.task, torch.tensor(6.25))
    torch.testing.assert_close(loss.improvement, torch.tensor(0.125))
    torch.testing.assert_close(loss.total, torch.tensor(6.28125))


def test_loss_rejects_mixed_sample_identity_inside_one_trajectory() -> None:
    with pytest.raises(alpha.RefineTrainingContractError, match="canonical sample"):
        alpha.RefineStepTrainingResult(
            value=torch.zeros(2, 1, 2),
            valid_token_mask=torch.ones(2, 1, dtype=torch.bool),
            trajectory_id=torch.zeros(2, dtype=torch.int64),
            sample_id=torch.tensor([0, 1]),
            branch_id=torch.zeros(2, dtype=torch.int64),
            step_index=torch.zeros(2, dtype=torch.int64),
            trajectory_count=2,
            sample_count=2,
            route=torch.empty(2, 1, 0),
            indices=torch.empty(2, 1, 0, dtype=torch.int64),
            weights=torch.empty(2, 1, 0),
        )


def test_loss_rejects_partial_surviving_branch_sets() -> None:
    result = alpha.RefineStepTrainingResult(
        value=torch.zeros(2, 1, 2),
        valid_token_mask=torch.tensor([[True], [False]]),
        trajectory_id=torch.tensor([0, 1]),
        sample_id=torch.zeros(2, dtype=torch.int64),
        branch_id=torch.tensor([0, 1]),
        step_index=torch.zeros(2, dtype=torch.int64),
        trajectory_count=2,
        sample_count=1,
        route=torch.empty(2, 1, 0),
        indices=torch.empty(2, 1, 0, dtype=torch.int64),
        weights=torch.empty(2, 1, 0),
    )

    with pytest.raises(alpha.RefineTrainingContractError, match="partial set"):
        alpha.RefineStepTraining().reduce_task_loss(torch.ones(2), result)


def test_loss_masks_nonfinite_padding_and_rejects_nonfinite_live_values() -> None:
    result = alpha.RefineStepTrainingResult(
        value=torch.zeros(1, 2, 2),
        valid_token_mask=torch.tensor([[True, False]]),
        trajectory_id=torch.zeros(1, dtype=torch.int64),
        sample_id=torch.zeros(1, dtype=torch.int64),
        branch_id=torch.zeros(1, dtype=torch.int64),
        step_index=torch.zeros(1, dtype=torch.int64),
        trajectory_count=1,
        sample_count=1,
        route=torch.empty(1, 2, 0),
        indices=torch.empty(1, 2, 0, dtype=torch.int64),
        weights=torch.empty(1, 2, 0),
    )
    padded_nan = torch.tensor([[1.0, float("nan")]])
    loss = alpha.RefineStepTraining().reduce_task_loss(padded_nan, result)
    torch.testing.assert_close(loss.total, torch.tensor(1.0))

    live_nan = padded_nan.clone()
    live_nan[0, 0] = float("nan")
    with pytest.raises(alpha.RefineTrainingContractError, match="non-finite"):
        alpha.RefineStepTraining().reduce_task_loss(live_nan, result)


def test_loss_rejects_partial_depth_and_masks_invalid_row_nan_before_reduction() -> None:
    result = alpha.RefineStepTrainingResult(
        value=torch.zeros(2, 1, 2),
        valid_token_mask=torch.tensor([[True], [False]]),
        trajectory_id=torch.zeros(2, dtype=torch.int64),
        sample_id=torch.zeros(2, dtype=torch.int64),
        branch_id=torch.zeros(2, dtype=torch.int64),
        step_index=torch.tensor([0, 1]),
        trajectory_count=1,
        sample_count=1,
        route=torch.empty(2, 1, 0),
        indices=torch.empty(2, 1, 0, dtype=torch.int64),
        weights=torch.empty(2, 1, 0),
    )

    with pytest.raises(alpha.RefineTrainingContractError, match="partial set"):
        alpha.RefineStepTraining().reduce_task_loss(
            torch.tensor([1.0, float("nan")]), result
        )


def test_loss_gives_a_fully_inactive_source_sample_zero_weight() -> None:
    result = alpha.RefineStepTrainingResult(
        value=torch.zeros(2, 1, 2),
        valid_token_mask=torch.tensor([[True], [False]]),
        trajectory_id=torch.tensor([0, 1]),
        sample_id=torch.tensor([0, 1]),
        branch_id=torch.zeros(2, dtype=torch.int64),
        step_index=torch.zeros(2, dtype=torch.int64),
        trajectory_count=2,
        sample_count=2,
        route=torch.empty(2, 1, 0),
        indices=torch.empty(2, 1, 0, dtype=torch.int64),
        weights=torch.empty(2, 1, 0),
    )

    loss = alpha.RefineStepTraining().reduce_task_loss(
        torch.tensor([1.0, float("nan")]), result
    )
    torch.testing.assert_close(loss.total, torch.tensor(1.0))
    assert loss.valid_trajectories.item() == 1


def test_snapshot_staleness_and_query_mutation_fail_closed() -> None:
    recall = arti.Recall(4, 8, breadth=1, activation="none")
    trainer = alpha.RefineStepTraining(max_snapshot_staleness=1)
    rollout = trainer.capture(
        recall,
        torch.randn(1, 2, 4),
        policy=_policy(2),
        snapshot_generation=4,
    )
    with torch.no_grad():
        recall.state.recall.bank.add_(0.01)
    with pytest.raises(alpha.RefineTrainingContractError, match="same-generation"):
        trainer.replay(recall, rollout, current_generation=4)
    trainer.replay(recall, rollout, current_generation=5)
    with torch.no_grad():
        recall.state.recall.query.weight.add_(0.01)
    with pytest.raises(alpha.RefineTrainingContractError, match="fixed Query changed"):
        trainer.replay(recall, rollout, current_generation=5)


def test_k_wide_replay_rejects_cross_generation_candidate_identity() -> None:
    recall = arti.Recall(
        4,
        16,
        group_topk=4,
        breadth=3,
        breadth_mode="independent",
        activation="none",
    )
    trainer = alpha.RefineStepTraining(max_snapshot_staleness=1)
    rollout = trainer.capture(
        recall,
        torch.randn(1, 2, 4),
        policy=_policy(2),
        breadth=3,
        snapshot_generation=7,
    )

    with pytest.raises(alpha.RefineTrainingContractError, match="exact capture generation"):
        trainer.replay(recall, rollout, current_generation=8)


def test_formula_program_and_execution_mode_are_snapshot_bound() -> None:
    recall = arti.Recall(
        4,
        8,
        formula=_ProgramTaggedFormula("a" * 64),
        activation="none",
    )
    trainer = alpha.RefineStepTraining()
    rollout = trainer.capture(recall, torch.randn(1, 2, 4), policy=_policy(2))
    recall.state.recall.formula.program = _Program("b" * 64)
    with pytest.raises(alpha.RefineTrainingContractError, match="execution configuration"):
        trainer.replay(recall, rollout)

    recall = arti.Recall(4, 8, activation="none")
    rollout = trainer.capture(recall, torch.randn(1, 2, 4), policy=_policy(2))
    recall.eval()
    with pytest.raises(alpha.RefineTrainingContractError, match="execution configuration"):
        trainer.replay(recall, rollout)


def test_nonfinite_and_uncommitted_transitions_are_never_supervised() -> None:
    recall = arti.Recall(4, 8, formula=_NonfiniteFormula(), activation="none")
    trainer = alpha.RefineStepTraining()
    rollout = trainer.capture(
        recall,
        torch.randn(1, 2, 4),
        policy=_policy(2),
    )

    assert rollout.attempted[0].all()
    assert not rollout.committed.any()
    assert not rollout.valid_token_mask.any()
    result = trainer.replay(recall, rollout)
    torch.testing.assert_close(result.value, rollout.hidden_state)
    with pytest.raises(alpha.RefineTrainingContractError, match="no live"):
        trainer.reduce_task_loss(
            torch.ones(result.value.shape[:2]),
            result,
        )


def test_nonfinite_transition_is_excluded_even_when_runtime_commits_it() -> None:
    recall = arti.Recall(4, 8, formula=_NonfiniteFormula(), activation="none")
    policy = arti.RefinePolicy.adaptive(
        max_steps=1,
        min_steps=1,
        scope="token",
        relative_tolerance=1e-12,
        check_finite=False,
    )
    rollout = alpha.RefineStepTraining().capture(
        recall,
        torch.randn(1, 2, 4),
        policy=policy,
    )

    assert rollout.attempted.all()
    assert rollout.committed.all()
    assert not rollout.finite.any()
    assert not rollout.valid_token_mask.any()


def test_capture_rejects_trainable_query() -> None:
    recall = arti.Recall(4, 8, breadth=1)
    recall.state.recall.query.weight.requires_grad_(True)

    with pytest.raises(alpha.RefineTrainingContractError, match="fixed Query"):
        alpha.RefineStepTraining().capture(
            recall,
            torch.randn(1, 2, 4),
            policy=_policy(2),
        )


def test_k_wide_rollout_preserves_branch_lineage_and_requeries() -> None:
    torch.manual_seed(3104)
    recall = arti.Recall(
        4,
        16,
        group_topk=4,
        breadth=3,
        breadth_mode="independent",
        activation="none",
    )
    trainer = alpha.RefineStepTraining()
    rollout = trainer.capture(
        recall,
        torch.randn(2, 2, 4),
        policy=_policy(3),
        breadth=3,
    )

    assert rollout.hidden_state.shape == (18, 2, 4)
    assert rollout.trajectory_count == 6
    assert rollout.branch_id.tolist() == [
        0, 0, 0, 1, 1, 1, 2, 2, 2,
        0, 0, 0, 1, 1, 1, 2, 2, 2,
    ]
    from arti import batched_refine

    with patch.object(
        batched_refine,
        "query_recall_branches",
        wraps=batched_refine.query_recall_branches,
    ) as query_spy:
        result = trainer.replay(recall, rollout)
    assert query_spy.call_count == 1
    assert query_spy.call_args.args[1].shape[0] == 2
    assert result.value.shape == rollout.hidden_state.shape
    assert result.route.shape[:2] == rollout.hidden_state.shape[:2]
    assert torch.isfinite(result.value[result.valid_token_mask]).all()
    assert not torch.equal(
        result.value[rollout.trajectory_id == 0],
        result.value[rollout.trajectory_id == 1],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_refine_step_training_cuda_backward() -> None:
    recall = arti.Recall(8, 16, breadth=1, activation="none").cuda()
    trainer = alpha.RefineStepTraining()
    rollout = trainer.capture(
        recall,
        torch.randn(2, 3, 8, device="cuda"),
        policy=_policy(3),
    )
    result = trainer.replay(recall, rollout)
    loss = trainer.reduce_task_loss(
        result.value.float().square().mean(dim=-1),
        result,
    )
    loss.total.backward()

    assert recall.state.recall.bank.grad is not None
    assert torch.isfinite(recall.state.recall.bank.grad).all()
