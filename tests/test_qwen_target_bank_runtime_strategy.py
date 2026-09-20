from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))

from run_qwen_target_bank_content_write import (
    QwenTargetBankStrategy,
    checkpoint_binding_status,
    load_checkpoint_payload,
    numeric_tree_is_finite,
)
from run_qwen_target_bank_updater import _TargetBankWriter
from run_qwen_federal_runtime_engineering_gate import QwenFederalRuntimePath
from arti.mechanisms import TargetBankUpdater, WriteRefinePolicy


def test_qwen_target_bank_strategy_builds_fixed_policy() -> None:
    strategy = QwenTargetBankStrategy(write_steps=4, read_steps=2)

    policy = strategy.write_policy()

    assert policy.budget.max_steps == 4
    assert policy.budget.min_steps == 4
    assert strategy.receipt()["write_policy"] == "fixed"


def test_qwen_target_bank_strategy_builds_adaptive_policy() -> None:
    strategy = QwenTargetBankStrategy(
        write_steps=8,
        read_steps=3,
        adaptive_relative_tolerance=1e-3,
        adaptive_min_steps=2,
    )

    policy = strategy.write_policy()

    assert policy.budget.max_steps == 8
    assert policy.budget.min_steps == 2
    assert policy.stop is not None
    assert policy.stop.relative_tolerance == pytest.approx(1e-3)
    assert strategy.receipt()["write_policy"] == "adaptive"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"write_steps": 0, "read_steps": 1}, "write_steps"),
        ({"write_steps": 1, "read_steps": -1}, "read_steps"),
        (
            {"write_steps": 2, "read_steps": 1, "adaptive_min_steps": 3},
            "adaptive_min_steps",
        ),
    ],
)
def test_qwen_target_bank_strategy_rejects_invalid_depths(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        QwenTargetBankStrategy(**kwargs)


def test_qwen_checkpoint_validation_is_fail_closed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format": "arti.qwen-target-bank-updater.v1",
            "config": {"updater_version": 1},
            "runtime": {},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="TargetBankUpdater@2"):
        load_checkpoint_payload(checkpoint)


def test_qwen_checkpoint_binding_rejects_mismatched_assets() -> None:
    expected = {"model_id": "qwen", "slots": 32}

    assert checkpoint_binding_status({}, expected) == "legacy_partial"
    assert (
        checkpoint_binding_status(
            {"contract_version": 2, "model_id": "qwen", "slots": 32},
            expected,
        )
        == "bound"
    )
    with pytest.raises(ValueError, match="slots"):
        checkpoint_binding_status(
            {"contract_version": 2, "model_id": "qwen", "slots": 64},
            expected,
        )


def test_qwen_receipt_finiteness_is_recursive() -> None:
    assert numeric_tree_is_finite({"metrics": [1.0, 2, None], "status": "ok"})
    assert not numeric_tree_is_finite({"metrics": {"kl": float("nan")}})
    assert not numeric_tree_is_finite([float("inf")])


def test_qwen_target_writer_forwards_zero_exposure() -> None:
    writer = _TargetBankWriter(
        TargetBankUpdater(
            hidden_dim=4,
            slots=4,
            policy=WriteRefinePolicy.fixed(2),
        )
    )
    trace = torch.randn(2, 3, 4)
    bank = torch.randn(2, 4, 4)

    result = writer(trace, bank, exposure=torch.zeros(2))

    assert torch.equal(result, bank)


def test_qwen_federal_runtime_path_composes_existing_components() -> None:
    torch.manual_seed(17)
    path = QwenFederalRuntimePath(8, active_count=4).eval()
    bank = torch.randn(3, 1, 8, 8)
    trace = torch.randn(3, 1, 5, 8)
    mask = torch.ones(3, 5, dtype=torch.bool)

    result, exposure, active = path(bank, trace, mask)

    assert result.shape == bank.shape
    assert exposure.shape == (3,)
    assert active.shape == (3, 1, 4)
    assert torch.isfinite(result).all()
    assert (exposure >= 0.25).all() and (exposure <= 1.0).all()
    assert not torch.equal(result, bank)
