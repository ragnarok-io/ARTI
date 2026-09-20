from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from benchmarks.evaluate_qwen_ttt_opposite_history import (
    LogitBatch,
    answer_margins,
    bootstrap_lower_bound,
    categorical_kl,
    gate_decision,
    load_protocol,
    paired_dialogue_indices,
    rebase_round_positions,
    state_similarity,
    summarize_seed,
    swap_pair_rows,
)
from benchmarks.run_qwen_self_teacher_affine_ttt import MessageRound


def test_protocol_is_retired_and_binds_legacy_controls() -> None:
    protocol = load_protocol(Path("benchmarks/qwen_ttt_opposite_history_protocol.json"))

    assert protocol["status"] == "retired-diagnostic"
    assert len(protocol["seeds"]) >= 3
    assert set(protocol["controls"]) == {
        "correct_bank",
        "wrong_history_bank",
        "reversed_history_bank",
        "zero_bank",
        "latest_message_only",
        "full_context_teacher",
    }
    assert len(protocol["artifact_sha256"]) == 64
    assert protocol["success_thresholds"]["correct_vs_wrong_kl_win_rate"] == 0.8


def test_protocol_rejects_unlocked_payload(tmp_path: Path) -> None:
    source = json.loads(
        Path("benchmarks/qwen_ttt_opposite_history_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    source["status"] = "locked-before-first-run"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="retired"):
        load_protocol(path)


def test_pairing_is_deterministic_unique_and_heldout() -> None:
    first = paired_dialogue_indices(224, 256, pairs=8, seed=63001)
    second = paired_dialogue_indices(224, 256, pairs=8, seed=63001)

    assert first == second
    assert len(first) == len(set(first)) == 16
    assert min(first) >= 224 and max(first) < 256


def test_swap_pair_rows_is_involutive() -> None:
    value = torch.arange(24).reshape(4, 2, 3)
    swapped = swap_pair_rows(value)

    assert torch.equal(swapped[0], value[1])
    assert torch.equal(swapped[2], value[3])
    assert torch.equal(swap_pair_rows(swapped), value)


def test_rebase_round_positions_preserves_tokens_after_exchange_reordering() -> None:
    rows = torch.arange(2)
    first = MessageRound(
        rows,
        torch.tensor([[10, 11, 0], [12, 0, 0]]),
        torch.zeros(2, 3, dtype=torch.int64),
        torch.tensor([[True, True, False], [True, False, False]]),
    )
    second = MessageRound(
        rows,
        torch.tensor([[20, 21, 22], [23, 24, 0]]),
        torch.zeros(2, 3, dtype=torch.int64),
        torch.tensor([[True, True, True], [True, True, False]]),
    )

    rebased = rebase_round_positions((second, first))

    assert torch.equal(rebased[0].input_ids, second.input_ids)
    assert rebased[0].position_ids.tolist() == [[0, 1, 2], [0, 1, 0]]
    assert rebased[1].position_ids.tolist() == [[3, 4, 0], [2, 0, 0]]


def test_categorical_kl_and_state_similarity_have_expected_identity() -> None:
    logits = torch.tensor([[2.0, -1.0], [0.5, 0.25]])
    shifted = torch.tensor([[-1.0, 2.0], [0.25, 0.5]])
    state = torch.randn(3, 2, 4)

    assert categorical_kl(logits, logits, temperature=1.0) == pytest.approx(0.0, abs=1e-7)
    assert categorical_kl(logits, shifted, temperature=1.0) > 0
    assert state_similarity(state, state)["cosine"] == pytest.approx(1.0)
    assert state_similarity(state, state)["relative_rmse"] == 0.0


def test_bootstrap_lower_bound_is_deterministic_and_discriminating() -> None:
    strong = bootstrap_lower_bound([True] * 19 + [False], samples=1000)
    weak = bootstrap_lower_bound([True, False] * 10, samples=1000)

    assert strong > 0.5
    assert weak <= 0.5
    assert strong == bootstrap_lower_bound([True] * 19 + [False], samples=1000)


def test_summary_detects_correct_bank_and_causal_swap() -> None:
    teacher = LogitBatch(
        (
            torch.tensor([[5.0, -3.0]]),
            torch.tensor([[-3.0, 5.0]]),
        )
    )
    correct = teacher
    wrong = LogitBatch((teacher.rows[1], teacher.rows[0]))
    zero = LogitBatch((torch.zeros(1, 2), torch.zeros(1, 2)))
    summary = summarize_seed(
        teacher=teacher,
        conditions={
            "correct_bank": correct,
            "wrong_history_bank": wrong,
            "reversed_history_bank": zero,
            "zero_bank": zero,
            "latest_message_only": zero,
        },
        target_ids=(torch.tensor([0]), torch.tensor([1])),
        temperature=1.0,
    )

    assert all(summary["correct_vs_wrong_wins"])
    assert all(summary["correct_vs_reset_wins"])
    assert all(summary["correct_vs_reversed_wins"])
    assert all(summary["bank_swap_answer_flips"])
    assert answer_margins(correct, ((0, 0, 1), (0, 1, 0))) == [8.0, 8.0]


def test_gate_decision_classifies_write_specificity_failure() -> None:
    protocol = load_protocol(Path("benchmarks/qwen_ttt_opposite_history_protocol.json"))
    aggregate = {
        "correct_vs_wrong_kl_win_rate": 0.4,
        "correct_vs_reset_kl_win_rate": 0.9,
        "correct_vs_wrong_bootstrap_95_lower": 0.2,
        "correct_vs_reset_bootstrap_95_lower": 0.7,
        "bank_swap_answer_flip_rate": 0.8,
        "teacher_answer_margin_positive_rate": 1.0,
        "sequential_chunk_state_cosine": 0.99,
        "sequential_chunk_output_kl": 0.01,
        "zero_vs_latest_output_kl": 0.0,
    }
    report = {
        "aggregate": aggregate,
        "seeds": {
            "1": {"mean_kl": {"correct_bank": 1.0, "wrong_history_bank": 0.9, "zero_bank": 2.0}},
            "2": {"mean_kl": {"correct_bank": 1.0, "wrong_history_bank": 1.1, "zero_bank": 2.0}},
            "3": {"mean_kl": {"correct_bank": 1.0, "wrong_history_bank": 1.1, "zero_bank": 2.0}},
        },
    }

    decision = gate_decision(report, protocol)

    assert not decision["passed"]
    assert decision["classification"] == "history_write_specificity_failure"
    assert decision["failure_modes"] == ["history_write_specificity_failure"]
