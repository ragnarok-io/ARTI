from __future__ import annotations

from pathlib import Path

import pytest
import torch

from benchmarks.probe_qwen_ttt_reachability import (
    _first_token_batch,
    load_probe_protocol,
)


def test_research_loop_protocol_has_valid_lifecycle_status() -> None:
    protocol = load_probe_protocol(Path("benchmarks/qwen_ttt_research_loop_protocol.json"))
    assert protocol["status"] in {
        "design-locked-no-training-run",
        "gate1-training-in-progress",
    }
    assert protocol["trainable"] == ["updater"]
    assert "teacher_response_prefix" in protocol["forbidden_student_inputs"]


def test_hybrid_research_loop_protocol_is_supported() -> None:
    protocol = load_probe_protocol(
        Path("benchmarks/qwen_ttt_binding_swap_research_loop_protocol_v8.json")
    )
    assert protocol["format"] == "arti.qwen-ttt-research-loop.v2"
    assert protocol["training_schedule"]["default_response_mode"] == "hybrid"
    assert protocol["training_schedule"]["response_gate_path"] == "free_rollout_only"


def test_unknown_research_loop_protocol_version_is_rejected() -> None:
    source = Path("benchmarks/qwen_ttt_research_loop_protocol.json")
    payload = source.read_text(encoding="utf-8").replace(
        "arti.qwen-ttt-research-loop.v1",
        "arti.qwen-ttt-research-loop.v99",
        1,
    )
    class InMemoryProtocol:
        def read_text(self, *, encoding: str) -> str:
            assert encoding == "utf-8"
            return payload

    with pytest.raises(ValueError, match="unsupported TTT research-loop protocol"):
        load_probe_protocol(InMemoryProtocol())


def test_first_token_batch_has_no_answer_prefix_and_preserves_positions() -> None:
    batch = _first_token_batch(
        [torch.tensor([10, 11]), torch.tensor([20])],
        [torch.tensor([4, 5]), torch.tensor([9])],
        pad_token_id=0,
    )
    assert batch.input_ids.tolist() == [[10, 11], [20, 0]]
    assert batch.position_ids.tolist() == [[4, 5], [9, 0]]
    assert batch.attention_mask.tolist() == [[True, True], [True, False]]


def test_first_token_batch_rejects_misaligned_rows() -> None:
    with pytest.raises(ValueError, match="aligned"):
        _first_token_batch(
            [torch.tensor([10, 11])],
            [torch.tensor([4])],
            pad_token_id=0,
        )
