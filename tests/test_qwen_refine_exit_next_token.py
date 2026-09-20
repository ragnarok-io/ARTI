from __future__ import annotations

import pytest
import torch

from benchmarks.train_qwen_refine_exit_next_token import (
    CONTROL_TRAIN_TEXTS,
    RECALL_TRAIN_TEXTS,
    TEST_TEXTS,
    VALIDATION_TEXTS,
    _corrupt_hidden,
    _position_bands,
    _quality_gate,
)


def test_qwen_refine_exit_uses_complete_disjoint_text_splits() -> None:
    train = set(RECALL_TRAIN_TEXTS)
    control = set(CONTROL_TRAIN_TEXTS)
    validation = set(VALIDATION_TEXTS)
    test = set(TEST_TEXTS)

    assert len(train) == len(RECALL_TRAIN_TEXTS)
    assert len(control) == len(CONTROL_TRAIN_TEXTS)
    assert len(validation) == len(VALIDATION_TEXTS)
    assert len(test) == len(TEST_TEXTS)
    assert train.isdisjoint(control)
    assert train.isdisjoint(validation)
    assert control.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert control.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_qwen_refine_exit_corruption_is_deterministic_and_nonidentity() -> None:
    value = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    first = _corrupt_hidden(value, seed=71, drop_probability=0.25, noise=0.1)
    second = _corrupt_hidden(value, seed=71, drop_probability=0.25, noise=0.1)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first, value)


def test_qwen_refine_exit_uses_disjoint_position_id_bands() -> None:
    bands = _position_bands(64)
    assert tuple(bands) == (
        "recall_train",
        "control_train",
        "validation",
        "test",
    )
    intervals = [
        set(range(band["start"], band["stop_exclusive"]))
        for band in bands.values()
    ]
    for index, left in enumerate(intervals):
        for right in intervals[index + 1 :]:
            assert left.isdisjoint(right)
    with pytest.raises(ValueError, match="positive integer"):
        _position_bands(0)


def test_qwen_refine_exit_quality_gate_requires_quality_and_step_saving() -> None:
    passed = _quality_gate(
        learned_loss=1.02,
        fixed_full_depth_loss=1.0,
        tolerance=0.03,
        mean_steps=3.0,
        depth=8,
        minimum_step_saving=0.25,
    )
    quality_failed = _quality_gate(
        learned_loss=1.04,
        fixed_full_depth_loss=1.0,
        tolerance=0.03,
        mean_steps=3.0,
        depth=8,
        minimum_step_saving=0.25,
    )
    step_failed = _quality_gate(
        learned_loss=1.0,
        fixed_full_depth_loss=1.0,
        tolerance=0.03,
        mean_steps=7.0,
        depth=8,
        minimum_step_saving=0.25,
    )

    assert passed["passed"] is True
    assert quality_failed["quality_passed"] is False
    assert quality_failed["passed"] is False
    assert step_failed["active_step_passed"] is False
    assert step_failed["passed"] is False
    with pytest.raises(ValueError, match="non-negative"):
        _quality_gate(
            learned_loss=1.0,
            fixed_full_depth_loss=1.0,
            tolerance=-0.1,
            mean_steps=1.0,
            depth=8,
            minimum_step_saving=0.25,
        )
    with pytest.raises(ValueError, match="minimum step saving"):
        _quality_gate(
            learned_loss=1.0,
            fixed_full_depth_loss=1.0,
            tolerance=0.0,
            mean_steps=1.0,
            depth=8,
            minimum_step_saving=1.0,
        )
