from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from arti._formula_device_query import FormulaDeviceQuery


class _FixedLogits(nn.Module):
    def __init__(self, logits: Tensor, input_dim: int) -> None:
        super().__init__()
        self.register_buffer("logits", logits)
        self.register_buffer("dtype_anchor", torch.zeros(input_dim))

    def forward(self, summary: Tensor) -> Tensor:
        return self.logits.to(dtype=summary.dtype).expand(summary.shape[0], -1)


def _module(
    logits: list[float],
    *,
    width: int = 2,
    families: list[int] | None = None,
    priority: list[int] | None = None,
    preserve: bool = True,
    device: str | torch.device = "cpu",
) -> FormulaDeviceQuery:
    actions = len(logits)
    module = FormulaDeviceQuery(
        _FixedLogits(torch.tensor(logits), input_dim=16),
        slot_count=2,
        candidate_family_ids=families or list(range(actions)),
        width=width,
        action_priority=priority,
        preserve_family_coverage=preserve,
    )
    return module.to(device)


def _inputs(
    rows: int = 1, device: str | torch.device = "cpu"
) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor], Tensor]:
    values = (
        torch.arange(rows * 4, dtype=torch.float32, device=device).reshape(rows, 2, 2),
        torch.ones(rows, 3, dtype=torch.float32, device=device),
    )
    occupied = torch.ones(rows, 2, dtype=torch.bool, device=device)
    eligible = torch.ones(rows, 4, dtype=torch.bool, device=device)
    parent_scores = torch.zeros(rows, device=device)
    return (values, occupied, eligible, parent_scores), occupied


def test_legal_logsoftmax_ignores_ineligible_nonfinite_logits() -> None:
    module = _module([1.0, math.nan, 2.0, math.nan], width=2, preserve=False)
    (values, occupied, eligible, parent_scores), _ = _inputs()
    eligible[0, 1] = False
    eligible[0, 3] = False

    result = module(values, occupied, eligible, parent_scores)

    expected = torch.log_softmax(torch.tensor([1.0, 2.0]), dim=0)
    torch.testing.assert_close(result.scores[0, [0, 2]], expected)
    assert torch.isneginf(result.scores[0, 1])
    assert torch.isneginf(result.scores[0, 3])
    assert bool(result.finite[0])


def test_family_coverage_matches_ranked_representatives() -> None:
    module = _module(
        [4.0, 3.0, 2.0, 1.0],
        families=[10, 10, 20, 20],
        width=2,
    )
    (values, occupied, eligible, parent_scores), _ = _inputs()

    result = module(values, occupied, eligible, parent_scores)

    assert result.order[0].tolist() == [0, 1, 2, 3]
    assert result.local_selected[0].tolist() == [True, False, True, False]
    assert result.representative[0].tolist() == [True, False, True, False]
    assert bool(result.coverage_satisfied[0])


def test_stable_tie_break_uses_action_priority_without_host_selection() -> None:
    module = _module(
        [1.0, 1.0, 1.0, 1.0],
        width=2,
        priority=[2, 0, 3, 1],
        preserve=False,
    )
    (values, occupied, eligible, parent_scores), _ = _inputs()

    result = module(values, occupied, eligible, parent_scores)

    assert result.order[0].tolist() == [1, 3, 0, 2]
    assert result.local_selected[0].tolist() == [False, True, False, True]


def test_nonfinite_occupied_row_and_dead_row_are_sanitized_and_not_selected() -> None:
    module = _module([4.0, 3.0, 2.0, 1.0], preserve=False)
    (values, occupied, eligible, parent_scores), _ = _inputs(rows=3)
    values[0][0, 0, 0] = math.nan
    occupied[0, 0] = True
    occupied[1] = False
    values[1][0, 0] = math.nan
    eligible[2] = False

    result = module(values, occupied, eligible, parent_scores)

    assert result.input_finite.tolist() == [False, True, True]
    assert result.score_finite.tolist() == [True, True, True]
    assert result.finite.tolist() == [False, True, True]
    assert not bool(result.local_selected[0].any())
    assert bool(result.local_selected[1].any())
    assert not bool(result.local_selected[2].any())
    assert torch.isfinite(result.scores).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_kernel_is_device_local(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    module = _module(
        [4.0, 3.0, 2.0, 1.0],
        families=[10, 10, 20, 20],
        width=2,
        device=device,
    )
    (values, occupied, eligible, parent_scores), _ = _inputs(rows=3, device=device)

    result = module(values, occupied, eligible, parent_scores)

    assert result.scores.device.type == device
    assert result.local_selected.device.type == device
    assert result.order.device.type == device
    assert module._family_membership.device.type == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_constructor_metadata_migrates_with_module() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    network = _FixedLogits(torch.tensor([4.0, 3.0, 2.0, 1.0]), input_dim=16).to(device)
    module = FormulaDeviceQuery(
        network,
        slot_count=2,
        candidate_family_ids=torch.tensor([10, 10, 20, 20], device=device),
        action_priority=torch.tensor([2, 0, 3, 1], device=device),
        width=2,
    ).to(device)

    assert module.candidate_family_ids.device == device
    assert module.action_priority.device == device
    assert module._family_membership.device == device


@pytest.mark.parametrize(("device", "backend"), [("cpu", "aot_eager"), ("cuda", "inductor")])
def test_compiled_fullgraph_matches_eager(device: str, backend: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    module = _module(
        [4.0, 3.0, 2.0, 1.0],
        families=[10, 10, 20, 20],
        width=2,
        device=device,
    )
    inputs, _ = _inputs(rows=3, device=device)
    exported = torch.export.export(module, inputs).module()
    compiled = torch.compile(exported, backend=backend, fullgraph=True)

    expected = module(*inputs)
    actual = compiled(*inputs)

    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, equal_nan=True)
