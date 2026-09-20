from __future__ import annotations

import pytest
import torch

from benchmarks.train_qwen_federated_autonomous_federation import _retained_task_risk_credit


@pytest.mark.parametrize("scale", (1.0, 1000.0))
def test_risk_credit_matches_frozen_temperature_gradient(scale) -> None:
    scores = (torch.tensor([-2.0, -1.0, 0.0], dtype=torch.float64) * scale).requires_grad_()
    losses = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64, requires_grad=True)
    credit, risk, temperature = _retained_task_risk_credit(scores, losses)
    gradient, loss_gradient = torch.autograd.grad(credit, (scores, losses), allow_unused=True)
    p = (scores.detach() / temperature).softmax(0)
    torch.testing.assert_close(gradient, p * (losses.detach() - risk))
    assert gradient[0] < 0 < gradient[-1]
    assert torch.count_nonzero(gradient) == scores.numel()
    assert credit.item() == 0.0
    assert loss_gradient is None
    assert not temperature.requires_grad and not risk.requires_grad


@pytest.mark.parametrize("scores,losses", (([0.0], [2.0]), ([-4.0, -1.0, 0.0], [2.0] * 3)))
def test_equal_losses_and_single_candidate_have_zero_credit(scores, losses) -> None:
    scores = torch.tensor(scores, dtype=torch.float64, requires_grad=True)
    credit, _, _ = _retained_task_risk_credit(scores, torch.tensor(losses, dtype=torch.float64))
    (gradient,) = torch.autograd.grad(credit, (scores,))
    torch.testing.assert_close(gradient, torch.zeros_like(scores), atol=1e-14, rtol=0)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("gradient_scale", (1.0, 2.0**-64))
def test_huge_equal_losses_have_exact_zero_credit_through_large_query_inputs(
    device, dtype, gradient_scale,
) -> None:
    generator = torch.Generator().manual_seed(1)
    parameter = (torch.randn(16, generator=generator).to(device=device, dtype=dtype) / 1e18).requires_grad_()
    losses = torch.full((16,), 1e37, device=device, dtype=dtype, requires_grad=True)
    credit, risk, _ = _retained_task_risk_credit(parameter * 1e18, losses, gradient_scale=gradient_scale)
    gradient, loss_gradient = torch.autograd.grad(credit, (parameter, losses), allow_unused=True)
    assert torch.equal(gradient, torch.zeros_like(parameter))
    assert torch.equal(risk, losses.detach()[0])
    assert credit.item() == 0.0
    assert loss_gradient is None


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_common_representable_loss_offset_does_not_change_route_gradient(device) -> None:
    losses = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64, device=device) * 2.0**34
    gradients, risks = [], []
    for offset in (0.0, 2.0**80):
        scores = torch.tensor([-4.0, -1.0, 0.0], dtype=torch.float64, device=device, requires_grad=True)
        credit, risk, _ = _retained_task_risk_credit(scores, losses + offset)
        gradients.append(torch.autograd.grad(credit, scores)[0])
        risks.append(risk)
    torch.testing.assert_close(*gradients, rtol=1e-14, atol=0.0)
    torch.testing.assert_close(risks[1], risks[0] + 2.0**80, rtol=0.0, atol=0.0)


def test_risk_credit_preserves_hard_task_forward_and_gradient() -> None:
    scores = torch.tensor([-3000.0, -1500.0, -800.0], requires_grad=True)
    operands = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    losses = operands.square()
    credit, _, _ = _retained_task_risk_credit(scores, losses)
    hard_task = losses[scores.detach().argmax()]
    combined = hard_task + 0.1 * credit
    assert torch.equal(hard_task, combined)
    (gradient,) = torch.autograd.grad(combined, (operands,))
    torch.testing.assert_close(gradient, torch.tensor([0.0, 0.0, 6.0]))


def test_risk_credit_is_score_shift_and_permutation_invariant() -> None:
    original = torch.tensor([-20.0, -10.0, 0.0], dtype=torch.float64)
    losses = torch.tensor([2.0, 3.0, 5.0], dtype=torch.float64)
    permutation = torch.tensor([2, 0, 1])
    gradients = []
    for shift, order in ((0.0, torch.arange(3)), (10000.0, permutation)):
        scores = (original[order] + shift).requires_grad_()
        credit, _, _ = _retained_task_risk_credit(scores, losses[order])
        (gradient,) = torch.autograd.grad(credit, (scores,))
        gradients.append(gradient[order.argsort()])
    torch.testing.assert_close(*gradients, atol=1e-14, rtol=1e-14)


def test_risk_credit_keeps_score_input_chain_rule_not_parameter_isolation() -> None:
    operands = torch.tensor([-20.0, 5.0, 10.0], dtype=torch.float64, requires_grad=True)
    scores = operands * 2.0
    losses = operands.square()
    credit, risk, temperature = _retained_task_risk_credit(scores, losses)
    (gradient,) = torch.autograd.grad(credit, (operands,))
    p = (scores.detach() / temperature).softmax(0)
    torch.testing.assert_close(gradient, 2.0 * p * (losses.detach() - risk))
    assert torch.count_nonzero(gradient) == operands.numel()


@pytest.mark.parametrize("scores,losses", (([], []), ([0.0], [1.0, 2.0])))
def test_risk_credit_requires_aligned_nonempty_candidates(scores, losses) -> None:
    with pytest.raises(ValueError, match="aligned"):
        _retained_task_risk_credit(torch.tensor(scores), torch.tensor(losses))
