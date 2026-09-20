from __future__ import annotations

import pytest
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from benchmarks._qwen_numerics import SCALED_RMSNORM, enable_scaled_qwen_rmsnorm


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_scaled_rmsnorm_native_range_is_exact(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(491)
    norm = Qwen3RMSNorm(64).to(device=device, dtype=dtype)
    value = torch.randn(3, 7, 64, device=device, dtype=dtype, requires_grad=True)
    weights = torch.randn_like(value)
    native = norm(value)
    native_grad = torch.autograd.grad((native * weights).sum(), (value, norm.weight))
    assert enable_scaled_qwen_rmsnorm(norm) == 1
    actual = norm(value)
    actual_grad = torch.autograd.grad((actual * weights).sum(), (value, norm.weight))
    assert torch.equal(native, actual)
    for expected, gradient in zip(native_grad, actual_grad, strict=True):
        assert torch.equal(expected, gradient)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("magnitude", [0.0, 1e-20, 1e19, 1e29, 1e37])
def test_scaled_rmsnorm_large_and_small_values_match_fp64(device, dtype, magnitude):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    norm = Qwen3RMSNorm(64).to(device=device, dtype=dtype)
    value = (torch.linspace(-1, 1, 64, device=device).reshape(1, 1, -1) * magnitude).to(dtype)
    value.requires_grad_()
    probe = torch.linspace(0.25, 1.5, 64, device=device, dtype=dtype)
    enable_scaled_qwen_rmsnorm(norm)
    actual = norm(value)
    gradient, weight_gradient = torch.autograd.grad((actual * probe).sum(), (value, norm.weight))
    reference_value = value.detach().double().requires_grad_()
    reference_weight = norm.weight.detach().double().requires_grad_()
    reference = reference_weight * reference_value * torch.rsqrt(
        reference_value.square().mean(-1, keepdim=True) + norm.variance_epsilon
    )
    reference_gradient, reference_weight_gradient = torch.autograd.grad(
        (reference * probe.double()).sum(), (reference_value, reference_weight)
    )
    assert torch.isfinite(actual).all() and torch.isfinite(gradient).all()
    torch.testing.assert_close(actual.float(), reference.float(), rtol=0.005, atol=0.004)
    torch.testing.assert_close(
        weight_gradient.float(), reference_weight_gradient.float(), rtol=0.008, atol=0.006,
    )
    # Compare relative gradients without masking tiny but meaningful derivatives
    # behind a fixed absolute tolerance.
    gradient_scale = max(magnitude, norm.variance_epsilon ** 0.5)
    torch.testing.assert_close(
        gradient.double() * gradient_scale, reference_gradient * gradient_scale,
        rtol=0.02 if dtype == torch.bfloat16 else 1e-4,
        atol=0.008 if dtype == torch.bfloat16 else 3e-6,
    )


def test_scaled_rmsnorm_is_model_local_and_preserves_parameters():
    model = nn.Sequential(Qwen3RMSNorm(4), nn.Linear(4, 4), Qwen3RMSNorm(4))
    untouched = Qwen3RMSNorm(4)
    ids = [id(parameter) for parameter in model.parameters()]
    state = {name: value.clone() for name, value in model.state_dict().items()}
    assert enable_scaled_qwen_rmsnorm(model) == 2
    assert enable_scaled_qwen_rmsnorm(model) == 2
    assert "forward" not in untouched.__dict__
    assert ids == [id(parameter) for parameter in model.parameters()]
    assert model._arti_qwen_numerics == SCALED_RMSNORM
    for name, value in model.state_dict().items():
        assert torch.equal(state[name], value)


def test_native_finite_large_input_can_silently_zero_the_output():
    norm = Qwen3RMSNorm(64)
    value = torch.linspace(-1, 1, 64).reshape(1, 1, -1) * 1e29
    assert torch.isfinite(value).all()
    assert torch.count_nonzero(norm(value)) == 0
    enable_scaled_qwen_rmsnorm(norm)
    output = norm(value)
    assert torch.isfinite(output).all() and torch.count_nonzero(output) == output.numel()


def test_scale_is_token_local_with_zero_and_large_rows_together():
    norm = Qwen3RMSNorm(8)
    values = torch.tensor([0.0, 1e29, 1.0, 0.0, 1e-20, -1e29]).reshape(2, 3, 1)
    values = (values * torch.linspace(-1, 1, 8)).requires_grad_()
    enable_scaled_qwen_rmsnorm(norm)
    actual = norm(values)
    gradient = torch.autograd.grad(actual.sum(), values)[0]
    reference_value = values.detach().double().requires_grad_()
    reference = reference_value * torch.rsqrt(
        reference_value.square().mean(-1, keepdim=True) + norm.variance_epsilon
    )
    expected_gradient = torch.autograd.grad(reference.sum(), reference_value)[0]
    assert torch.isfinite(actual).all() and torch.isfinite(gradient).all()
    torch.testing.assert_close(actual.double(), reference, rtol=1e-6, atol=1e-7)
    scale = values.detach().double().abs().amax(-1, keepdim=True).clamp_min(1e-3)
    torch.testing.assert_close(gradient.double() * scale, expected_gradient * scale, rtol=1e-5, atol=1e-6)
