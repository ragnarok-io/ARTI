from __future__ import annotations

import torch
import torch.nn as nn

import arti
import arti.nn as arti_nn
import arti.torch as arti_torch


def test_learned_pulse_can_disable_half_independently() -> None:
    pulse = arti_nn.LearnedPulse(k=3, dim=8, use_half=False)
    output = pulse(torch.randn(2, 5, 8))

    assert isinstance(pulse.half_act, nn.Identity)
    assert output.shape == (2, 3, 8)


def test_learned_pulse_compacts_to_fixed_pulses() -> None:
    pulse = arti_nn.LearnedPulse(k=4)
    x = torch.randn(2, 10, 6)

    z = pulse(x)

    assert z.shape == (2, 4, 6)


def test_pulse_alias_is_default_learned_pulse() -> None:
    pulse = arti_nn.Pulse(k=4)
    x = torch.randn(2, 10, 6)

    z = pulse(x)

    assert isinstance(pulse, arti_nn.LearnedPulse)
    assert z.shape == (2, 4, 6)


def test_learned_pulse_return_info() -> None:
    pulse = arti_nn.LearnedPulse(k=3, dim=5)
    x = torch.randn(2, 8, 5)

    z, info = pulse(x, return_info=True)

    assert z.shape == (2, 3, 5)
    for key in ("survival_mean", "survival_min", "survival_max", "fragment_norm", "pulse_norm"):
        assert key in info
        assert info[key].shape == ()
        assert torch.isfinite(info[key])


def test_learned_pulse_is_differentiable() -> None:
    pulse = arti_nn.LearnedPulse(k=3, dim=5, hidden_dim=7)
    x = torch.randn(2, 9, 5, requires_grad=True)

    loss = pulse(x).square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    grads = [param.grad for param in pulse.parameters() if param.requires_grad and param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_learned_pulse_accepts_external_survival_guidance() -> None:
    pulse = arti_nn.LearnedPulse(k=2)
    x = torch.zeros(1, 5, 3)
    x[:, 0] = torch.tensor([8.0, 0.0, 0.0])
    q_low = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]])
    q_high = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])

    z_low = pulse(x, q=q_low)
    z_high = pulse(x, q=q_high)

    assert z_high.norm(dim=-1).mean() > z_low.norm(dim=-1).mean()


def test_learned_pulse_accepts_mask_separately_from_survival_guidance() -> None:
    pulse = arti_nn.LearnedPulse(k=2)
    x = torch.zeros(1, 5, 3)
    x[:, 0] = torch.tensor([8.0, 0.0, 0.0])
    mask_off = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]])
    mask_on = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])

    z_off = pulse(x, mask=mask_off)
    z_on = pulse(x, mask=mask_on)

    assert z_on.norm(dim=-1).mean() > z_off.norm(dim=-1).mean()


def test_learned_pulse_q_topk_uses_explicit_q_not_padding_mask() -> None:
    pulse = arti_nn.LearnedPulse(k=2, dim=3, hidden_dim=5, q_topk=1)
    x = torch.randn(1, 5, 3, requires_grad=True)
    mask = torch.ones(1, 5)

    z = pulse(x, mask=mask)
    z.mean().backward()

    assert z.shape == (1, 2, 3)
    assert x.grad is not None
    touched = x.grad.abs().sum(dim=-1) > 0
    assert int(touched.sum().item()) > 1


def test_learned_pulse_refinement_block() -> None:
    pulse = arti_nn.LearnedPulse(k=4, dim=6, hidden_dim=8, refine=True)
    x = torch.randn(2, 11, 6, requires_grad=True)

    z = pulse(x)
    z.mean().backward()

    assert z.shape == (2, 4, 6)
    assert x.grad is not None


def test_learned_pulse_gated_refine_and_topk_options() -> None:
    pulse = arti_nn.LearnedPulse(k=3, dim=6, hidden_dim=8, refine=True, refine_mode="gated", fold_topk=2, q_topk=5)
    x = torch.randn(2, 11, 6, requires_grad=True)
    q = torch.rand(2, 11)

    z = pulse(x, q=q)
    z.mean().backward()

    assert z.shape == (2, 3, 6)
    assert x.grad is not None


def test_learned_pulse_attention_fold_mode() -> None:
    pulse = arti_nn.LearnedPulse(k=3, dim=8, hidden_dim=12, fold_mode="attention", fold_heads=2)
    x = torch.randn(2, 10, 8, requires_grad=True)
    q = torch.rand(2, 10)

    z = pulse(x, q=q)
    z.square().mean().backward()

    assert z.shape == (2, 3, 8)
    assert x.grad is not None


def test_learned_pulse_works_with_downstream_layers() -> None:
    pulse = arti_nn.LearnedPulse(k=4, dim=6)
    conv = nn.Conv1d(6, 8, kernel_size=3, padding=1)
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(4 * 6, 5))
    x = torch.randn(2, 12, 6, requires_grad=True)

    z = pulse(x)
    conv_out = conv(z.transpose(1, 2))
    mlp_out = mlp(z)
    loss = conv_out.mean() + mlp_out.mean()
    loss.backward()

    assert conv_out.shape == (2, 8, 4)
    assert mlp_out.shape == (2, 5)
    assert x.grad is not None


def test_learned_pulse_public_namespaces() -> None:
    assert arti.Pulse is arti_nn.Pulse
    assert arti.Pulse is arti_nn.LearnedPulse
    assert arti_torch.Pulse is arti_nn.Pulse
    assert arti.LearnedPulse is arti_nn.LearnedPulse
    assert arti_torch.LearnedPulse is arti_nn.LearnedPulse
