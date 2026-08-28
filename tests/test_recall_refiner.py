from __future__ import annotations

import pytest
import torch

import arti
import arti.nn as arti_nn
import arti.torch as arti_torch


def test_recall_refiner_delegates_to_one_canonical_loop() -> None:
    recall = arti_nn.Recall(5, 10, activation="none")
    refiner = arti_nn.RecallRefiner(recall)
    h = torch.randn(2, 3, 5)
    policy = arti.RefinePolicy.fixed(3, trace_level="summary")

    expected, expected_info = recall(h, refine_policy=policy, return_info=True)
    actual, actual_info = refiner(h, policy=policy, return_info=True)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        actual_info["recall_step_update_ratio"],
        expected_info["recall_step_update_ratio"],
    )
    assert actual_info["recall_step_attempted"].shape == (2, 3)


def test_recall_refiner_preserves_recall_gradients() -> None:
    recall = arti_nn.Recall(6, 12, activation="none")
    refiner = arti_nn.RecallRefiner(recall)
    h = torch.randn(3, 2, 6, requires_grad=True)

    refiner(h, policy=arti.RefinePolicy.fixed(2)).square().mean().backward()

    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert recall.state.recall.bank.grad is not None
    assert torch.isfinite(recall.state.recall.bank.grad).all()


def test_recall_refiner_requires_explicit_policy_and_recall() -> None:
    recall = arti_nn.Recall(4, 4)
    refiner = arti_nn.RecallRefiner(recall)
    h = torch.randn(1, 2, 4)

    with pytest.raises(TypeError, match="explicit RefinePolicy"):
        refiner(h)
    with pytest.raises(TypeError, match="arti.nn.Recall"):
        arti_nn.RecallRefiner(torch.nn.Identity())
    with pytest.raises(ValueError, match="policy"):
        refiner(
            h,
            policy=arti.RefinePolicy.fixed(1),
            refine_policy=arti.RefinePolicy.fixed(1),
        )


def test_recall_refiner_is_versioned_runtime_adapter() -> None:
    refiner = arti_nn.RecallRefiner(arti_nn.Recall(4, 4))

    assert arti.component_ref(refiner) == "arti/recall-refiner@2"
    spec = arti.component_spec(refiner)
    assert spec.variant == "runtime-policy-adapter"
    assert spec.config == {}
    assert "arti/recall@4" in spec.dependencies


def test_recall_refiner_public_namespaces() -> None:
    assert arti.RecallRefiner is arti_nn.RecallRefiner
    assert arti_torch.RecallRefiner is arti_nn.RecallRefiner
