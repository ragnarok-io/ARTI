from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

import arti
from arti import mechanisms


def _policy(depth: int) -> arti.AdaptiveRefinePolicy:
    return arti.RefinePolicy.adaptive(
        max_steps=depth,
        min_steps=depth,
        scope="token",
        relative_tolerance=1e-12,
        trace_level="routes",
        executor="static_masked",
    )


class _BiasLogit(nn.Module):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(float(value)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.bias.expand(value.shape[0])


def _capture(
    *,
    depth: int = 3,
    breadth: int = 1,
) -> tuple[arti.Recall, mechanisms.RefineRollout, mechanisms.RefineStepTrainingResult]:
    torch.manual_seed(4101)
    recall = arti.Recall(
        4,
        16,
        group_topk=4,
        breadth=breadth,
        breadth_mode="independent",
        activation="none",
    )
    trainer = mechanisms.RefineStepTraining()
    rollout = trainer.capture(
        recall,
        torch.randn(2, 3, 4),
        policy=_policy(depth),
        breadth=breadth,
        snapshot_generation=9,
    )
    result = trainer.replay(recall, rollout, current_generation=9)
    return recall, rollout, result


def test_builds_detached_canonical_token_curve() -> None:
    _recall, rollout, result = _capture(depth=4)
    task_loss = result.value.square().mean(dim=-1)
    curve = mechanisms.RefineExitTraining.build_curve(
        rollout,
        result,
        task_loss,
        scope="token",
    )

    assert curve.post_state.shape == (2, 4, 3, 4)
    assert curve.task_loss.shape == (2, 4, 3)
    assert curve.depth == 4
    assert curve.trajectory_count == 2
    assert not curve.post_state.requires_grad
    assert not curve.task_loss.requires_grad
    assert curve.sample_id.tolist() == [0, 1]
    assert curve.branch_id.tolist() == [0, 0]
    assert arti.component_ref(curve) == "arti/refine-exit-curve@1"


def test_curve_is_permutation_invariant_and_keeps_k_lineage() -> None:
    _recall, rollout, result = _capture(depth=3, breadth=3)
    task_loss = result.value.square().mean(dim=-1)
    expected = mechanisms.RefineExitTraining.build_curve(rollout, result, task_loss)
    order = torch.randperm(rollout.hidden_state.shape[0])
    permuted_rollout = rollout.permute(order)
    permuted_result = replace(
        result,
        value=result.value.index_select(0, order),
        valid_token_mask=result.valid_token_mask.index_select(0, order),
        trajectory_id=result.trajectory_id.index_select(0, order),
        sample_id=result.sample_id.index_select(0, order),
        branch_id=result.branch_id.index_select(0, order),
        step_index=result.step_index.index_select(0, order),
        route=result.route.index_select(0, order),
        indices=result.indices.index_select(0, order),
        weights=result.weights.index_select(0, order),
    )
    actual = mechanisms.RefineExitTraining.build_curve(
        permuted_rollout,
        permuted_result,
        task_loss.index_select(0, order),
    )

    torch.testing.assert_close(actual.post_state, expected.post_state)
    torch.testing.assert_close(actual.task_loss, expected.task_loss)
    assert actual.sample_id.tolist() == [0, 0, 0, 1, 1, 1]
    assert actual.branch_id.tolist() == [0, 1, 2, 0, 1, 2]


def test_curve_rejects_partial_depth_and_ambiguous_loss_shape() -> None:
    _recall, rollout, result = _capture(depth=3)
    invalid = result.valid_token_mask.clone()
    invalid[1, 0] = False
    with pytest.raises(mechanisms.RefineTrainingContractError, match="complete full depth"):
        mechanisms.RefineExitTraining.build_curve(
            rollout,
            replace(result, valid_token_mask=invalid),
            torch.ones_like(invalid, dtype=torch.float32),
        )
    with pytest.raises(TypeError, match="token task_loss"):
        mechanisms.RefineExitTraining.build_curve(
            rollout,
            result,
            torch.ones(result.value.shape[0]),
            scope="token",
        )


def test_branch_hazard_matches_exact_quality_constrained_objective() -> None:
    _recall, rollout, result = _capture(depth=3)
    task_loss = torch.tensor([4.0, 2.0, 1.0, 4.0, 2.0, 1.0])
    curve = mechanisms.RefineExitTraining.build_curve(
        rollout,
        result,
        task_loss,
        scope="branch",
    )
    control = mechanisms.RefineExitControl(_BiasLogit(), scope="branch")
    training = mechanisms.RefineExitTraining(
        compute_weight=0.1,
        quality_tolerance=0.0,
        quality_weight=1.0,
    )
    loss = training.loss(control, curve)

    torch.testing.assert_close(
        loss.stop_probability[0],
        torch.tensor([0.5, 0.5, 1.0]),
    )
    torch.testing.assert_close(
        loss.terminal_probability[0],
        torch.tensor([0.5, 0.25, 0.25]),
    )
    torch.testing.assert_close(loss.expected_task, torch.tensor(2.75))
    torch.testing.assert_close(loss.full_depth_task, torch.tensor(1.0))
    torch.testing.assert_close(loss.expected_logical_depth, torch.tensor(1.75))
    torch.testing.assert_close(loss.quality_violation, torch.tensor(1.75))
    torch.testing.assert_close(loss.total, torch.tensor(4.675))


def test_gradient_chooses_continue_for_bad_shallow_and_stop_for_good_shallow() -> None:
    _recall, rollout, result = _capture(depth=2)
    control = mechanisms.RefineExitControl(_BiasLogit(), scope="branch")
    training = mechanisms.RefineExitTraining(
        compute_weight=0.1,
        quality_weight=1.0,
    )
    bad_shallow = torch.tensor([5.0, 1.0, 5.0, 1.0])
    curve = training.build_curve(rollout, result, bad_shallow, scope="branch")
    loss = training.loss(control, curve)
    loss.total.backward()
    assert control.source.bias.grad is not None
    assert control.source.bias.grad.item() > 0

    control.source.bias.grad = None
    good_shallow = torch.tensor([0.0, 1.0, 0.0, 1.0])
    curve = training.build_curve(rollout, result, good_shallow, scope="branch")
    loss = training.loss(control, curve)
    loss.total.backward()
    assert control.source.bias.grad is not None
    assert control.source.bias.grad.item() < 0


def test_token_exit_training_detaches_recall_and_task_path() -> None:
    recall, rollout, result = _capture(depth=3)
    control = mechanisms.RefineExitControl(nn.Linear(4, 1), scope="token")
    curve = mechanisms.RefineExitTraining.build_curve(
        rollout,
        result,
        result.value.square().mean(dim=-1),
        scope="token",
    )
    loss = mechanisms.RefineExitTraining(compute_weight=0.01).loss(control, curve)
    loss.total.backward()

    assert all(parameter.grad is None for parameter in recall.parameters())
    assert all(parameter.grad is not None for parameter in control.parameters())
    assert torch.isfinite(loss.total)


def test_min_steps_can_force_the_full_depth_without_teacher_labels() -> None:
    _recall, rollout, result = _capture(depth=4)
    curve = mechanisms.RefineExitTraining.build_curve(
        rollout,
        result,
        torch.arange(8, dtype=torch.float32),
        scope="branch",
    )
    control = mechanisms.RefineExitControl(_BiasLogit(8.0), scope="branch")
    loss = mechanisms.RefineExitTraining(compute_weight=1.0).loss(
        control,
        curve,
        min_steps=4,
    )

    torch.testing.assert_close(
        loss.stop_probability,
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(2, -1),
    )
    torch.testing.assert_close(loss.expected_logical_depth, torch.tensor(4.0))
    torch.testing.assert_close(loss.expected_task, loss.full_depth_task)


def test_exit_training_rejects_predicate_scope_mismatch_and_nonfinite_logits() -> None:
    _recall, rollout, result = _capture(depth=2)
    curve = mechanisms.RefineExitTraining.build_curve(
        rollout,
        result,
        torch.ones(result.value.shape[:2]),
    )
    predicate = mechanisms.RefineExitControl(
        nn.Identity(),
        input_kind="predicate",
        scope="token",
    )
    with pytest.raises(mechanisms.RefineTrainingContractError, match="logit"):
        mechanisms.RefineExitTraining().loss(predicate, curve)
    branch = mechanisms.RefineExitControl(_BiasLogit(), scope="branch")
    with pytest.raises(mechanisms.RefineTrainingContractError, match="scope"):
        mechanisms.RefineExitTraining().loss(branch, curve)
    nonfinite = mechanisms.RefineExitControl(_BiasLogit(float("nan")), scope="token")
    with pytest.raises(mechanisms.RefineTrainingContractError, match="non-finite"):
        mechanisms.RefineExitTraining().loss(nonfinite, curve)


def test_optimizer_contract_is_controller_only() -> None:
    recall, _rollout, _result = _capture(depth=2)
    control = mechanisms.RefineExitControl(nn.Linear(4, 1))
    controller_optimizer = torch.optim.AdamW(control.parameters(), lr=1e-3)
    mechanisms.RefineExitTraining.assert_optimizer_contract(
        recall,
        control,
        controller_optimizer,
    )
    mixed = torch.optim.AdamW(
        [*control.parameters(), recall.state.recall.bank],
        lr=1e-3,
    )
    with pytest.raises(mechanisms.RefineTrainingContractError, match="exactly"):
        mechanisms.RefineExitTraining.assert_optimizer_contract(recall, control, mixed)


def test_exit_training_components_are_alpha_runtime_contracts() -> None:
    _recall, rollout, result = _capture(depth=2)
    curve = mechanisms.RefineExitTraining.build_curve(
        rollout,
        result,
        torch.ones(result.value.shape[:2]),
    )
    training = mechanisms.RefineExitTraining()

    assert arti.component_ref(training) == "arti/refine-exit-training@1"
    assert arti.component_ref(curve) == "arti/refine-exit-curve@1"
    registry = arti.get_component_registry()
    assert registry.registration_for(training).artifact_policy == "runtime_only"
    assert registry.registration_for(curve).artifact_policy == "runtime_only"
    assert arti.component_spec(training).dependencies == (
        "arti/refine-exit-control@1",
        "arti/refine-exit-curve@1",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_exit_training_cuda_backward() -> None:
    recall = arti.Recall(8, 16, activation="none").cuda()
    step = mechanisms.RefineStepTraining()
    rollout = step.capture(
        recall,
        torch.randn(3, 4, 8, device="cuda"),
        policy=_policy(4),
    )
    result = step.replay(recall, rollout)
    curve = mechanisms.RefineExitTraining.build_curve(
        rollout,
        result,
        result.value.float().square().mean(dim=-1),
    )
    control = mechanisms.RefineExitControl(nn.Linear(8, 1).cuda())
    loss = mechanisms.RefineExitTraining(compute_weight=0.01).loss(control, curve)
    loss.total.backward()

    assert all(parameter.grad is None for parameter in recall.parameters())
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in control.parameters()
    )
