from __future__ import annotations

import pytest
import torch

from arti import ExecutionPolicy, alpha
from arti.nn import Recall


def _recall() -> Recall:
    return Recall(
        dim=8,
        slots=24,
        formula="arti/delta@1",
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=3,
        key_dim=8,
    )


def _reference(recall: Recall, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    candidates = alpha.query_recall_branches(recall, value, mask=mask, max_k=3)
    return alpha.run_branch_search(
        recall,
        value,
        candidates=candidates,
        execution_policy=ExecutionPolicy.fixed(3),
    ).value


def test_static_branch_kernel_matches_true_k_wide_requery_and_gradients() -> None:
    torch.manual_seed(9321)
    reference_recall = _recall()
    kernel_recall = _recall()
    kernel_recall.load_state_dict(reference_recall.state_dict())
    reference_value = torch.randn(3, 5, 8, requires_grad=True)
    kernel_value = reference_value.detach().clone().requires_grad_(True)
    mask = torch.tensor(
        [[True, True, True, True, True], [True, True, False, False, False], [True, False, True, False, True]]
    )

    expected = _reference(reference_recall, reference_value, mask)
    actual, _delta, _score = alpha.StaticBranchSearchKernel(
        kernel_recall,
        branches=3,
        iteration_steps=3,
    )(kernel_value, mask)

    torch.testing.assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(kernel_value.grad, reference_value.grad)
    torch.testing.assert_close(
        kernel_recall.state.recall.bank.grad,
        reference_recall.state.recall.bank.grad,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compiled_recall_forward_keeps_default_k_wide_execution() -> None:
    torch.manual_seed(9323)
    device = torch.device("cuda", torch.cuda.current_device())
    recall = _recall().to(device)
    value = torch.randn(2, 4, 8, device=device)
    mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True]],
        device=device,
    )
    expected = recall(value, mask=mask)
    compiled = torch.compile(recall, backend="inductor", fullgraph=True)
    actual = compiled(value, mask=mask)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_static_branch_kernel_inductor_keeps_batch_bucket_and_k_wide_semantics() -> None:
    torch.manual_seed(9322)
    device = torch.device("cuda", torch.cuda.current_device())
    recall = _recall().to(device)
    kernel = alpha.StaticBranchSearchKernel(
        recall,
        branches=3,
        iteration_steps=3,
    )

    # Complete one eager call before compilation so one-time state calibration
    # cannot be mistaken for compiled execution work.
    warm_value = torch.randn(2, 4, 8, device=device)
    warm_mask = torch.ones(2, 4, dtype=torch.bool, device=device)
    kernel(warm_value, warm_mask)
    # Each bucket has a static shape. This avoids Inductor's Windows
    # symbolic-shape C++ fallback while still allowing the runtime to select a
    # compiled graph for every supported batch bucket.
    automatic_dynamic_shapes = torch._dynamo.config.automatic_dynamic_shapes
    torch._dynamo.config.automatic_dynamic_shapes = False
    try:
        for batch in (2, 7):
            torch._dynamo.reset()
            value = torch.randn(batch, 4, 8, device=device, requires_grad=True)
            mask = torch.ones(batch, 4, dtype=torch.bool, device=device)
            mask[0, -1] = False
            expected, _, _ = kernel(value, mask)
            compiled = torch.compile(kernel, backend="inductor", fullgraph=True)
            actual, _, _ = compiled(value, mask)
            torch.testing.assert_close(actual, expected)
            actual.square().mean().backward()
            assert value.grad is not None
            assert torch.isfinite(value.grad).all()
    finally:
        torch._dynamo.config.automatic_dynamic_shapes = automatic_dynamic_shapes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_static_branch_kernel_cuda_graph_replay_accepts_new_bucket_values() -> None:
    torch.manual_seed(9324)
    device = torch.device("cuda", torch.cuda.current_device())
    kernel = alpha.StaticBranchSearchKernel(
        _recall().to(device),
        branches=3,
        iteration_steps=3,
    ).eval()
    capture_value = torch.randn(2, 4, 8, device=device)
    capture_mask = torch.ones(2, 4, dtype=torch.bool, device=device)
    captured = kernel.capture(capture_value, capture_mask)

    for seed in (419, 420):
        generator = torch.Generator(device=device).manual_seed(seed)
        value = torch.randn(2, 4, 8, device=device, generator=generator)
        mask = torch.ones(2, 4, dtype=torch.bool, device=device)
        mask[0, seed % 4] = False
        expected = kernel(value, mask)
        actual = captured.replay(value, mask)
        for actual_part, expected_part in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_part, expected_part)
