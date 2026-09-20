from __future__ import annotations

import pytest
import torch
from torch import nn

import arti
from arti import mechanisms
from benchmarks.train_refine_exit_combined import _quality_gate


class _BiasLogit(nn.Module):
    def __init__(self, value: float = 3.0) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(float(value)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.bias.expand(value.shape[0])


def test_combined_benchmark_quality_gate_compares_full_depth_loss() -> None:
    passed = _quality_gate(
        learned_loss=0.104,
        fixed_full_depth_loss=0.100,
        tolerance=0.005,
    )
    failed = _quality_gate(
        learned_loss=0.106,
        fixed_full_depth_loss=0.100,
        tolerance=0.005,
    )

    assert passed["passed"] is True
    assert passed["loss_delta"] == pytest.approx(0.004)
    assert failed["passed"] is False
    with pytest.raises(ValueError, match="non-negative"):
        _quality_gate(
            learned_loss=0.1,
            fixed_full_depth_loss=0.1,
            tolerance=-0.1,
        )


def _policy(depth: int, *, minimum: int | None = None) -> arti.AdaptiveRefinePolicy:
    return arti.RefinePolicy.adaptive(
        max_steps=depth,
        min_steps=depth if minimum is None else minimum,
        scope="token",
        relative_tolerance=1e-12,
        trace_level="routes",
        executor="static_masked",
    )


def _train_control_once(
    recall: arti.Recall,
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    depth: int,
    breadth: int = 1,
) -> mechanisms.RefineExitControl:
    step_training = mechanisms.RefineStepTraining()
    rollout = step_training.capture(
        recall,
        value,
        mask=mask,
        policy=_policy(depth),
        breadth=breadth,
    )
    result = step_training.replay(recall, rollout)
    task_loss = result.value.square().mean(dim=(1, 2))
    curve = mechanisms.RefineExitTraining.build_curve(
        rollout,
        result,
        task_loss,
        scope="branch",
    )
    control = mechanisms.RefineExitControl(
        _BiasLogit().to(value.device),
        scope="branch",
    )
    optimizer = torch.optim.SGD(control.parameters(), lr=0.01)
    training = mechanisms.RefineExitTraining(compute_weight=0.01)
    training.assert_optimizer_contract(recall, control, optimizer)
    optimizer.zero_grad(set_to_none=True)
    loss = training.loss(control, curve, min_steps=2)
    loss.total.backward()
    optimizer.step()
    assert all(parameter.grad is None for parameter in recall.parameters())
    return control


def test_training_curve_to_hard_runtime_round_trip_with_ragged_mask() -> None:
    torch.manual_seed(6101)
    recall = arti.Recall(4, 12, activation="none")
    value = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    control = _train_control_once(recall, value, mask, depth=3)
    fixed = recall(value, mask=mask, refine_policy=_policy(2))
    actual, trace = recall(
        value,
        mask=mask,
        refine_policy=_policy(3, minimum=2),
        refine_exit=control,
        model_exit=True,
        return_trace=True,
    )

    torch.testing.assert_close(actual, fixed, rtol=0, atol=0)
    assert isinstance(trace, arti.RecallTraceV3)
    assert torch.all(trace.token_steps_attempted[mask] == 2)
    assert torch.all(
        trace.token_stop_reason[mask] == int(arti.RecallStopReason.MODEL_EXIT)
    )
    assert not torch.any(trace.exit_requested & ~mask.unsqueeze(1))

    restored = mechanisms.RefineExitControl(_BiasLogit(), scope="branch")
    restored.load_state_dict(control.state_dict())
    restored_value, restored_trace = recall(
        value,
        mask=mask,
        refine_policy=_policy(3, minimum=2),
        refine_exit=restored,
        model_exit=True,
        return_trace=True,
    )
    torch.testing.assert_close(restored_value, actual, rtol=0, atol=0)
    torch.testing.assert_close(
        restored_trace.exit_score,
        trace.exit_score,
        rtol=0,
        atol=0,
    )
    assert torch.equal(restored_trace.token_stop_reason, trace.token_stop_reason)


def test_k3_training_curve_keeps_branch_local_hard_exit() -> None:
    torch.manual_seed(6102)
    recall = arti.Recall(
        4,
        16,
        group_topk=4,
        breadth=3,
        breadth_mode="independent",
        activation="none",
    )
    value = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, False, False], [True, True, True]])
    control = _train_control_once(
        recall,
        value,
        mask,
        depth=3,
        breadth=3,
    )
    _merged, result = recall(
        value,
        mask=mask,
        active_k=3,
        refine_policy=_policy(3, minimum=2),
        refine_exit=control,
        model_exit=True,
        return_branches=True,
    )

    attempted = result.branch_diagnostics["recall_token_steps_attempted"]
    branch_mask = result.candidates.candidate_mask.permute(0, 2, 1)
    assert attempted.shape == (2, 3, 3)
    assert torch.all(attempted[branch_mask] == 2)
    stopped = result.branch_diagnostics["recall_model_exit_stop"]
    assert stopped.shape == (2, 3, 3, 3)
    assert torch.all(stopped[:, :, 1][branch_mask])
    assert not torch.any(stopped[:, :, 0][branch_mask])
    assert not torch.any(stopped[:, :, 2][branch_mask])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_training_curve_to_hard_runtime() -> None:
    torch.manual_seed(6103)
    recall = arti.Recall(8, 16, activation="none").cuda()
    value = torch.randn(3, 4, 8, device="cuda")
    mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True], [True, False, False, False]],
        device="cuda",
    )
    control = _train_control_once(recall, value, mask, depth=3)
    fixed = recall(value, mask=mask, refine_policy=_policy(2))
    actual, trace = recall(
        value,
        mask=mask,
        refine_policy=_policy(3, minimum=2),
        refine_exit=control,
        model_exit=True,
        return_trace=True,
    )

    torch.testing.assert_close(actual, fixed, rtol=0, atol=0)
    assert torch.all(trace.token_steps_attempted[mask] == 2)
