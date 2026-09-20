from __future__ import annotations

import copy

import pytest
import torch

from arti.formula_program_query_v4 import (
    FormulaProgramQueryTensorEncoderV1,
    _scaled_population_std,
    _scaled_root_mean_square,
)


DEVICES = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_population_std_keeps_large_finite_values_and_gradients(device, dtype):
    magnitude = 1000.0 if dtype == torch.float16 else 1e21
    generator = torch.Generator().manual_seed(947)
    value = (torch.randn(2, 49, 16, generator=generator) * magnitude).to(device, dtype)
    value[:, :, 0] = magnitude
    value.requires_grad_()
    reference = value.detach().double().requires_grad_()
    actual = _scaled_population_std(value, dim=1)
    expected = reference.std(dim=1, unbiased=False)
    tolerance = {torch.float32: 2e-6, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, rtol=tolerance, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    assert torch.isfinite(value.grad).all()
    torch.testing.assert_close(
        value.grad.double(), reference.grad, rtol=tolerance, atol=tolerance / 49,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("length", (1, 49))
def test_population_std_normal_range_is_exact(device, length):
    value = torch.linspace(-2, 2, 2 * length * 8, device=device).reshape(2, length, 8)
    value.requires_grad_()
    reference = value.detach().clone().requires_grad_()
    actual = _scaled_population_std(value, dim=1)
    expected = reference.std(dim=1, unbiased=False)
    assert torch.equal(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(value.grad, reference.grad, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_tensor_encoder_preserves_summary_and_parameter_gradients(device):
    with torch.random.fork_rng():
        torch.manual_seed(47)
        encoder = FormulaProgramQueryTensorEncoderV1(8, 16).to(device)
    reference = copy.deepcopy(encoder)
    value = torch.linspace(-2, 2, 2 * 49 * 8, device=device).reshape(2, 49, 8)
    actual = encoder(value)
    encoded = reference.network(value)
    expected = torch.cat((
        encoded.new_ones((2, 1)), encoded[:, 0], encoded[:, -1], encoded.mean(dim=1),
        encoded.std(dim=1, unbiased=False), actual[:, -1:].detach(),
    ), dim=-1)
    assert torch.equal(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    for left, right in zip(encoder.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_large_upstream_gradient_does_not_multiply_by_forward_scale(device):
    value = torch.tensor([[-1e30, 1e30]], device=device, requires_grad=True)
    result = _scaled_population_std(value, dim=1)
    result.backward(torch.tensor([1e10], device=device))
    assert torch.isfinite(value.grad).all()
    torch.testing.assert_close(value.grad, torch.tensor([[-5e9, 5e9]], device=device))


def test_population_std_supports_second_order_gradients():
    value = torch.tensor([[1.0, -2.0, 4.0]], dtype=torch.float64, requires_grad=True)
    def function(x):
        return _scaled_population_std(x, dim=1)

    assert torch.autograd.gradcheck(function, (value,))
    assert torch.autograd.gradgradcheck(function, (value,))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("magnitude", (1.0, 1e21, 1e30))
def test_root_mean_square_preserves_units_and_large_upstream_gradient(device, magnitude):
    value = torch.tensor([[-1, 2, 0, 4]], device=device, dtype=torch.float32).mul_(magnitude).requires_grad_()
    reference = value.detach().double().requires_grad_()
    result = _scaled_root_mean_square(value, dim=-1)
    expected = reference.square().mean(-1).sqrt()
    torch.testing.assert_close(result.double(), expected, rtol=2e-6, atol=0)
    result.backward(torch.full_like(result, 1e10))
    expected.backward(torch.full_like(expected, 1e10))
    assert torch.isfinite(value.grad).all()
    torch.testing.assert_close(value.grad.double(), reference.grad, rtol=2e-6, atol=0)


def test_root_mean_square_zero_and_second_order_gradients():
    zero = torch.zeros(2, 4, requires_grad=True)
    _scaled_root_mean_square(zero, dim=-1).sum().backward()
    assert torch.equal(zero.grad, torch.zeros_like(zero))
    value = torch.tensor([[1.0, -2.0, 4.0]], dtype=torch.float64, requires_grad=True)
    def function(x):
        return _scaled_root_mean_square(x, dim=-1)
    assert torch.autograd.gradcheck(function, (value,))
    assert torch.autograd.gradgradcheck(function, (value,))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("function", [_scaled_population_std, _scaled_root_mean_square])
def test_statistics_vmap_preserves_values_and_gradients(device, function):
    value = torch.linspace(-3, 4, 60, device=device, dtype=torch.float64).reshape(3, 4, 5).requires_grad_()
    reference = value.detach().clone().requires_grad_()
    actual = torch.vmap(lambda x: function(x, dim=1))(value)
    expected = torch.stack([function(x, dim=1) for x in reference])
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(value.grad, reference.grad)
    transformed = torch.vmap(torch.func.grad(lambda x: function(x, dim=1).sum()))(value.detach())
    torch.testing.assert_close(transformed, reference.grad)
