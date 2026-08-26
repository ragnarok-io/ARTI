from __future__ import annotations

import pytest
import torch

from arti import alpha


def _program(dim: int) -> alpha.FormulaFabricProgram:
    return alpha.FormulaFabricProgram(
        arena_capacity=3,
        feature_dim=dim,
        steps=(
            (alpha.FormulaInvocation(alpha.FormulaPrimitive.ADD, 0),),
        ),
        domain="topology-active",
    )


def _route(value: torch.Tensor) -> alpha.FormulaRoutePlan:
    weights = value.new_zeros((value.shape[0], 1, 1, 2, 3))
    weights[:, 0, 0, 0, 1] = 1
    weights[:, 0, 0, 1, 2] = 1
    enabled = torch.ones(
        value.shape[0], 1, 1, dtype=torch.bool, device=value.device
    )
    return alpha.FormulaRoutePlan(weights, enabled, enabled, enabled)


def _run(value: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    topology = alpha.ReversibleTopology(active_count=3)
    folded = topology.fold(value, mask)
    arena = alpha.FormulaArenaState(
        folded.active,
        folded.active_mask,
        torch.zeros_like(folded.active_mask, dtype=torch.int64),
        "topology-active",
    )
    fabric = alpha.FormulaFabric(_program(value.shape[-1])).to(value.device)
    result = fabric(arena, _route(arena.value))
    restored = topology.unfold(folded.replace(active=result.state.value))
    return restored.value, folded.record.active_index


def test_formula_fabric_operates_only_on_folded_active_payload() -> None:
    value = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]],
        requires_grad=True,
    )
    mask = torch.ones(1, 5, dtype=torch.bool)

    actual, active_index = _run(value, mask)

    expected = value.detach().clone()
    expected[:, 0] = value.detach()[:, 1] + value.detach()[:, 2]
    torch.testing.assert_close(actual, expected)
    assert torch.equal(active_index, torch.tensor([[0, 1, 2]]))
    assert torch.equal(actual[:, 3:], value.detach()[:, 3:])

    actual.sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    torch.testing.assert_close(value.grad[:, 3:], torch.ones_like(value.grad[:, 3:]))


def test_masked_payload_remains_masked_and_unmodified() -> None:
    value = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
    mask = torch.tensor([[True, True, True, False, False]])

    actual, _ = _run(value, mask)

    assert torch.equal(actual[:, 3:], value[:, 3:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_formula_fabric_topology_cuda_forward_backward(dtype: torch.dtype) -> None:
    value = torch.randn(2, 6, 4, device="cuda", dtype=dtype, requires_grad=True)
    mask = torch.ones(2, 6, dtype=torch.bool, device="cuda")

    actual, _ = _run(value, mask)
    actual.float().square().mean().backward()

    assert actual.is_cuda and actual.dtype is dtype
    assert value.grad is not None and torch.isfinite(value.grad).all()
