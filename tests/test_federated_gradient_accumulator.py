import pytest
import torch

from benchmarks.train_qwen_federated_autonomous_federation import (
    _clip_training_gradients, _retained_task_risk_credit,
)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("magnitude", (0.2, 1e20, 1e37, 3e38))
def test_finite_large_gradients_keep_true_norm_direction_and_unused_none(device, magnitude):
    first = torch.nn.Parameter(torch.zeros(4, device=device))
    second = torch.nn.Parameter(torch.zeros(2, device=device))
    unused = torch.nn.Parameter(torch.zeros(1, device=device))
    first.grad = torch.tensor([1.0, -1.0, 0.5, -0.5], device=device) * magnitude
    second.grad = torch.tensor([1.0, -1.0], device=device) * magnitude
    before = torch.cat((first.grad.double(), second.grad.double()))
    expected_norm = before.norm()
    expected = before * (1.0 / (expected_norm + 1e-6)).clamp(max=1)
    result = _clip_training_gradients((first, second, unused), (), norm=1, mode="global")
    assert result["all"] == pytest.approx(float(expected_norm))
    torch.testing.assert_close(torch.cat((first.grad, second.grad)).double(), expected, rtol=1e-6, atol=1e-7)
    assert unused.grad is None
    assert bool(torch.isfinite(first.grad).all())


@pytest.mark.parametrize("value", (float("inf"), float("nan")))
def test_real_nonfinite_gradients_are_not_zeroed_or_suppressed(value):
    parameter = torch.nn.Parameter(torch.zeros(1))
    parameter.grad = torch.tensor([value])
    with pytest.raises(RuntimeError, match="non-finite"):
        _clip_training_gradients((parameter,), (), norm=1, mode="global")
    assert not bool(torch.isfinite(parameter.grad).all())


def test_large_routing_gradient_does_not_shrink_task_or_count_partitions():
    routing, task, count = (torch.nn.Parameter(torch.zeros(2)) for _ in range(3))
    routing.grad, task.grad, count.grad = torch.full((2,), 1e37), torch.tensor([0.2, 0.3]), torch.ones(2)
    expected_task = task.grad.clone()
    norms = _clip_training_gradients((routing, task, count), (routing,), norm=1,
                                     mode="routing-count-task", execution_counts=(count,))
    assert norms["routing"] > 1e37 and norms["count"] == pytest.approx(2**0.5)
    torch.testing.assert_close(task.grad, expected_task, rtol=0, atol=0)
    torch.testing.assert_close(routing.grad.norm(), torch.tensor(1.0))
    torch.testing.assert_close(count.grad.norm(), torch.tensor(1.0))


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_uniform_scale_handles_finite_forward_with_out_of_range_backward(device):
    scale = 2.0**-64
    parameter = torch.nn.Parameter(torch.tensor([1e-20, -1e-20], device=device))
    reference = parameter.detach().double().requires_grad_()
    reference_loss = ((reference * 1e20) * 1e10).square().mean()
    expected, = torch.autograd.grad(reference_loss, reference)
    nominal_norm = expected.norm()
    expected = expected * (1.0 / (nominal_norm + 1e-6)).clamp(max=1)
    loss = ((parameter * 1e20) * 1e10).square().mean()
    assert bool(torch.isfinite(loss))
    (loss * scale).backward()
    assert bool(torch.isfinite(parameter.grad).all())
    norm = _clip_training_gradients((parameter,), (), norm=1, mode="global", gradient_scale=scale)["all"]
    assert norm == pytest.approx(float(nominal_norm), rel=2e-6)
    torch.testing.assert_close(parameter.grad.double(), expected, rtol=2e-6, atol=1e-7)


def test_retained_risk_gradient_scale_is_applied_once_without_changing_measured_risk():
    scores = torch.tensor([-0.1, -1.2, -2.3], requires_grad=True)
    losses = torch.tensor([1e36, 1e37, 2e37])
    normal, risk, _ = _retained_task_risk_credit(scores, losses)
    scale = 2.0**-64
    scaled, scaled_risk, _ = _retained_task_risk_credit(scores, losses, gradient_scale=scale)
    first, = torch.autograd.grad(normal, scores)
    second, = torch.autograd.grad(scaled, scores)
    torch.testing.assert_close(first.double(), second.double() / scale, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(risk, scaled_risk, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ("ordered-operators", "parallel-delta"))
def test_scaled_episode_mean_preserves_effect_task_route_count_and_adam_update(mode):
    from test_federated_retained_training import _learners, _episode
    from benchmarks.train_federated_interacting_learners import train_minibatch

    torch.manual_seed(782)
    normal = _learners()
    torch.manual_seed(782)
    scaled = _learners()
    episodes = (_episode(), _episode(factor=-0.7))
    kwargs = dict(width=2, route_credit="retained-vjp", message_credit="trajectory-input-vjp",
                  plasticity_composition=mode)
    _, before = train_minibatch(normal, episodes, **kwargs)
    _, after = train_minibatch(scaled, episodes, gradient_scale=2.0**-64, **kwargs)
    for left, right, ln, rn in zip(normal, scaled, before["gradient_norms"], after["gradient_norms"], strict=True):
        for group in ln:
            assert ln[group] == pytest.approx(rn[group], rel=2e-5, abs=1e-7)
        for a, b in zip(left.query.parameters(), right.query.parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-7)
            assert (a.grad is None) == (b.grad is None)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=1e-7)
            for key, value in left.optimizer.state.get(a, {}).items():
                torch.testing.assert_close(value, right.optimizer.state[b][key], rtol=2e-5, atol=1e-7)


def test_peer_vjp_scale_is_not_squared_and_single_episode_entry_matches():
    from test_federated_interacting_learners import _tiny_learners, _episode
    from benchmarks.train_federated_interacting_learners import train_round

    torch.manual_seed(625)
    normal = _tiny_learners()
    torch.manual_seed(625)
    scaled = _tiny_learners()
    kwargs = dict(width=4, route_credit="retained-vjp", message_credit="trajectory-input-vjp")
    _, before = train_round(normal, _episode(), **kwargs)
    _, after = train_round(scaled, _episode(), gradient_scale=2.0**-64, **kwargs)
    assert any(row["message_input_gradient_norm"] > 0 for row in before["learners"])
    for left, right in zip(normal, scaled, strict=True):
        for a, b in zip(left.query.parameters(), right.query.parameters(), strict=True):
            assert (a.grad is None) == (b.grad is None)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=1e-7)
            torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-7)
    assert after["gradient_scale"] == 2.0**-64


def test_resume_inherits_numerical_scale_unless_explicitly_overridden():
    from benchmarks.train_federated_interacting_learners import _resume_gradient_scale

    saved = {"metrics": {"gradient_scale": 2.0**-64}}
    assert _resume_gradient_scale(None, saved, "retained-vjp") == 2.0**-64
    assert _resume_gradient_scale(1.0, saved, "retained-vjp") == 1.0
    assert _resume_gradient_scale(None, {}, "retained-vjp") == 1.0
    with pytest.raises(ValueError, match="retained-vjp"):
        _resume_gradient_scale(None, saved, "episode-beam")
    for invalid in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="gradient_scale"):
            _resume_gradient_scale(invalid, saved, "retained-vjp")
