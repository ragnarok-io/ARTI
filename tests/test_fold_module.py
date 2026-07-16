from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import arti
import arti.nn as arti_nn
import arti.torch as arti_torch


def test_fold_compacts_sequence_shape() -> None:
    fold = arti_nn.Fold(k=4)
    x = torch.randn(2, 9, 6)

    z = fold(x)

    assert z.shape == (2, 4, 6)


def test_fold_is_differentiable() -> None:
    fold = arti_nn.Fold(k=3, dim=5)
    x = torch.randn(2, 7, 5, requires_grad=True)

    loss = fold(x).square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    grads = [param.grad for param in fold.parameters()]
    assert grads
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)


def test_fold_eval_is_deterministic() -> None:
    fold = arti_nn.Fold(k=3, dim=4, dropout=0.25)
    fold.eval()
    x = torch.randn(2, 8, 4)
    q = torch.rand(2, 8)

    z1 = fold(x, q=q)
    z2 = fold(x, q=q)

    assert torch.allclose(z1, z2)


def test_fold_q_guides_survival() -> None:
    fold = arti_nn.Fold(k=2, dim=3)
    x = torch.zeros(1, 4, 3)
    x[:, 0] = torch.tensor([10.0, 0.0, 0.0])
    q_low = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    q_high = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    z_low = fold(x, q=q_low)
    z_high = fold(x, q=q_high)

    assert z_high[..., 0].abs().mean() > z_low[..., 0].abs().mean() + 1.0


def test_fold_accepts_q_with_trailing_dimension() -> None:
    fold = arti_nn.Fold(k=5)
    x = torch.randn(3, 6, 7)
    q = torch.rand(3, 6, 1)

    z = fold(x, q=q)

    assert z.shape == (3, 5, 7)


def test_fold_accepts_mask_separately_from_q() -> None:
    fold = arti_nn.Fold(k=2, dim=3)
    x = torch.zeros(1, 4, 3)
    x[:, 0] = torch.tensor([9.0, 0.0, 0.0])
    mask_off = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    mask_on = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

    z_off = fold(x, mask=mask_off)
    z_on = fold(x, mask=mask_on)

    assert z_off[..., 0].abs().max() < 1e-5
    assert z_on[..., 0].abs().max() > 0.1


def test_fold_combines_q_and_mask() -> None:
    fold = arti_nn.Fold(k=2, dim=3)
    x = torch.zeros(1, 4, 3)
    x[:, 0] = torch.tensor([9.0, 0.0, 0.0])
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])

    z = fold(x, q=q, mask=mask)

    assert z[..., 0].abs().mean() < 1.0


def test_fold_topk_sparse_path_is_differentiable() -> None:
    fold = arti_nn.Fold(k=3, dim=5, topk=2)
    x = torch.randn(2, 8, 5, requires_grad=True)
    q = torch.rand(2, 8)

    loss = fold(x, q=q).square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_fold_attention_mode_is_differentiable() -> None:
    fold = arti_nn.Fold(k=3, dim=8, hidden_dim=12, mode="attention", heads=2)
    x = torch.randn(2, 9, 8, requires_grad=True)
    q = torch.rand(2, 9)

    z = fold(x, q=q)
    z.mean().backward()

    assert z.shape == (2, 3, 8)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_fold_works_with_conv1d_and_flatten_mlp() -> None:
    fold = arti_nn.Fold(k=4, dim=6)
    conv = nn.Conv1d(6, 8, kernel_size=3, padding=1)
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(4 * 6, 5))
    x = torch.randn(2, 10, 6, requires_grad=True)

    z = fold(x)
    conv_out = conv(z.transpose(1, 2))
    mlp_out = mlp(z)
    loss = conv_out.mean() + mlp_out.mean()
    loss.backward()

    assert conv_out.shape == (2, 8, 4)
    assert mlp_out.shape == (2, 5)
    assert x.grad is not None


@pytest.mark.parametrize("temperature", [float("nan"), float("inf")])
def test_fold_rejects_nonfinite_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        arti_nn.Fold(k=2, temperature=temperature)


def test_fold_public_namespaces() -> None:
    assert arti.Fold is arti_nn.Fold
    assert arti_torch.Fold is arti_nn.Fold
