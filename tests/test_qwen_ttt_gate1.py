from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from benchmarks.train_qwen_ttt_gate1 import (
    HistoryTraceRound,
    _canonical_positions,
    _chunk_spans,
    _find_subsequence,
    _teacher_frame_positions,
    behavior_criterion_met,
    consistency_criterion_met,
    criterion_met,
    fresh_reload_criterion_met,
    response_criterion_met,
    train_overfit_criterion_met,
    MessageRound,
    first_token_kl,
    per_row_kl,
    response_kl,
    swap_flattened_response_rows,
    norm_matched_random_state,
    online_recapture_history_state,
    replay_history_state,
    shuffle_state_slots,
    select_gate_batch,
    swap_pair_rows,
    swap_history_trace_rounds,
    swap_history_rounds,
    validate_manifest_latest_tensors,
)


def test_deterministic_execution_configuration_is_noop_when_disabled() -> None:
    from benchmarks.train_qwen_ttt_gate1 import configure_deterministic_execution

    configure_deterministic_execution(False)


def test_fresh_reload_gate_requires_independent_parity() -> None:
    metrics = _gate_metrics()
    parity = {"pass": True}
    assert fresh_reload_criterion_met(metrics, parity, fresh_process=True)
    assert not fresh_reload_criterion_met(
        metrics,
        {"pass": False},
        fresh_process=True,
    )
    assert not fresh_reload_criterion_met(metrics, parity, fresh_process=False)


class _FakeTraceRuntime:
    fields = [SimpleNamespace(bank=torch.zeros(2, 3))]

    def initial_values(self, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
        return torch.zeros(batch_size, 1, 2, 3, dtype=reference.dtype)

    def update_values(
        self,
        traces: torch.Tensor,
        previous_values: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del mask
        contribution = (
            traces[:, 0]
            .sum(dim=1)
            .unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, 1, 2, -1)
        )
        return previous_values + contribution


class _OnlineFakeBackbone(torch.nn.Module):
    def __init__(self, owner: "_OnlineFakeModel") -> None:
        super().__init__()
        self.owner = owner

    def forward(self, input_ids, **_kwargs):
        self.owner.last_ids = input_ids.detach().clone()
        return SimpleNamespace(past_key_values=object())


class _OnlineFakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_ids = torch.empty(0, dtype=torch.long)
        self.model = _OnlineFakeBackbone(self)


class _OnlineFakeCapture:
    def __init__(self, model: _OnlineFakeModel, *, site: bool) -> None:
        self.model = model
        self.site = site

    def pop(self) -> torch.Tensor:
        values = self.model.last_ids.float()
        if self.site:
            return values.unsqueeze(1).unsqueeze(-1)
        return values


class _OnlineFakeRuntime(_FakeTraceRuntime):
    def __init__(self) -> None:
        self.used_states: list[torch.Tensor] = []

    @contextmanager
    def use(self, values: torch.Tensor, _mask: torch.Tensor):
        self.used_states.append(values.detach().clone())
        yield

    def update_values(
        self,
        traces: torch.Tensor,
        previous_values: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is None:
            raise AssertionError("online test requires a write mask")
        contribution = (traces[:, 0, :, 0] * mask).sum(dim=1)
        return previous_values + contribution[:, None, None, None]


def _gate_metrics() -> dict[str, float]:
    return {
        "correct_wrong_gap": 0.5,
        "correct_zero_gap": 0.5,
        "answer_flip_rate": 1.0,
        "causal_delta_cosine": 0.9,
        "causal_delta_relative_error": 0.5,
        "response_kl": 1.0,
        "reset_response_kl": 2.0,
        "response_rollout": 1.0,
        "response_teacher_top1_agreement": 1.0,
        "chunk_loss": 0.5,
    }


def test_behavior_and_consistency_gates_are_separate() -> None:
    metrics = _gate_metrics()
    assert behavior_criterion_met(metrics)
    assert consistency_criterion_met(metrics)
    assert criterion_met(metrics)

    metrics["chunk_loss"] = 2.0
    assert behavior_criterion_met(metrics)
    assert not consistency_criterion_met(metrics)
    assert not criterion_met(metrics)


def test_complete_gate_cannot_pass_before_consistency_stage() -> None:
    assert not criterion_met(_gate_metrics(), consistency_active=False)


def test_teacher_forced_response_cannot_satisfy_latest_only_gate() -> None:
    metrics = _gate_metrics()
    metrics["response_rollout"] = 0.0
    metrics["response_teacher_forced"] = 1.0
    assert not response_criterion_met(metrics)
    assert not criterion_met(metrics)


def test_relative_response_improvement_without_teacher_agreement_fails() -> None:
    metrics = _gate_metrics()
    metrics["response_teacher_top1_agreement"] = 0.0625
    assert not response_criterion_met(metrics)
    assert not criterion_met(metrics)


def test_norm_matched_random_state_preserves_each_bank_norm() -> None:
    state = torch.randn(3, 2, 4, 5)
    control = norm_matched_random_state(state, seed=1729)
    assert control.shape == state.shape
    assert torch.allclose(
        control.float().flatten(1).norm(dim=1),
        state.float().flatten(1).norm(dim=1),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.equal(control, norm_matched_random_state(state, seed=1729))


def test_shuffle_state_slots_preserves_values_and_is_deterministic() -> None:
    state = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    shuffled = shuffle_state_slots(state)
    assert torch.equal(shuffled, shuffle_state_slots(state))
    assert torch.equal(
        torch.sort(shuffled, dim=-2).values,
        torch.sort(state, dim=-2).values,
    )


def test_sparse_response_overfit_gate_only_counts_active_rollout_steps() -> None:
    metrics = _gate_metrics()
    metrics["response_teacher_top1_agreement"] = 0.9
    assert train_overfit_criterion_met(metrics, response_active=True)
    assert not train_overfit_criterion_met(metrics, response_active=False)


def test_canonical_positions_remove_history_offsets() -> None:
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    assert _canonical_positions(mask).tolist() == [[0, 1, 2, 0], [0, 1, 0, 0]]


def test_teacher_frame_positions_preserve_latest_token_values() -> None:
    latest = torch.tensor([[7, 8, 9, 10], [7, 8, 9, 10]])
    latest_mask = torch.ones(2, 4, dtype=torch.bool)
    teacher = torch.tensor([[1, 2, 7, 8, 9, 10, 11], [3, 7, 8, 9, 10, 12, 13]])
    teacher_mask = torch.ones(2, 7, dtype=torch.bool)
    result = _teacher_frame_positions(latest, latest_mask, teacher, teacher_mask)
    assert result.tolist() == [[2, 3, 4, 5], [1, 2, 3, 4]]
    assert _find_subsequence(teacher[0], latest[0]) == 2


def test_swap_pair_rows_is_involution() -> None:
    value = torch.arange(24).reshape(4, 2, 3)
    assert torch.equal(swap_pair_rows(swap_pair_rows(value)), value)


def test_per_row_kl_is_zero_for_identical_logits() -> None:
    logits = torch.randn(4, 5, 7)
    mask = torch.tensor([[True, True, False, False, False], [True, False, False, False, False], [True, True, True, False, False], [True, False, False, False, False]])
    flat = logits[mask]
    assert per_row_kl(flat, flat, mask, temperature=1.0).abs().max().item() < 1e-6


def test_first_token_kl_is_zero_for_identical_logits() -> None:
    logits = torch.randn(4, 11)
    assert first_token_kl(logits, logits, temperature=1.0).abs().max().item() < 1e-6


def test_response_kl_is_zero_for_identical_logits() -> None:
    logits = torch.randn(5, 11)
    mask = torch.tensor([[True, True], [True, True], [True, False]])
    flat = logits
    assert response_kl(flat, flat, mask, temperature=1.0).abs().item() < 1e-6


def test_swap_flattened_response_rows_preserves_pair_token_layout() -> None:
    values = torch.tensor(
        [[10.0], [11.0], [20.0], [21.0], [30.0], [40.0]],
    )
    mask = torch.tensor(
        [[True, True], [True, True], [True, False], [True, False]],
    )
    swapped = swap_flattened_response_rows(values, mask)
    assert swapped[:, 0].tolist() == [20.0, 21.0, 10.0, 11.0, 40.0, 30.0]


def test_swap_flattened_response_rows_rejects_unmatched_pair_lengths() -> None:
    values = torch.ones(5, 1)
    mask = torch.tensor(
        [[True, True], [True, False], [True, False], [True, False]],
    )
    with pytest.raises(ValueError, match="equal token counts"):
        swap_flattened_response_rows(values, mask)


def test_swap_history_rounds_rebuilds_opposite_inputs() -> None:
    message = MessageRound(
        row_indices=torch.tensor([0, 1, 2, 3]),
        input_ids=torch.tensor([[10], [11], [20], [21]]),
        position_ids=torch.zeros(4, 1, dtype=torch.long),
        active_mask=torch.ones(4, 1, dtype=torch.bool),
    )
    swapped = swap_history_rounds((message,))[0]
    assert swapped.input_ids.squeeze(1).tolist() == [11, 10, 21, 20]


def test_shared_trace_replay_is_partition_invariant_and_uses_global_rows() -> None:
    rounds = (
        HistoryTraceRound(
            row_indices=torch.tensor([2, 3]),
            traces=torch.tensor(
                [
                    [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                    [[[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]],
                ]
            ),
            active_mask=torch.ones(2, 2, dtype=torch.bool),
        ),
    )
    runtime = _FakeTraceRuntime()
    message = replay_history_state(runtime, rounds, mode="message")
    token = replay_history_state(runtime, rounds, mode="token")
    random = replay_history_state(runtime, rounds, mode="random", chunk_seed=1729)
    assert torch.equal(message, token)
    assert torch.equal(message, random)
    assert torch.equal(message[:2], torch.zeros_like(message[:2]))


def test_shared_trace_swap_reorders_trace_rows_without_model_reexecution() -> None:
    rounds = (
        HistoryTraceRound(
            row_indices=torch.tensor([0, 1]),
            traces=torch.arange(12, dtype=torch.float32).reshape(2, 1, 2, 3),
            active_mask=torch.ones(2, 2, dtype=torch.bool),
        ),
    )
    swapped = swap_history_trace_rounds(rounds)
    assert torch.equal(swapped[0].traces[0], rounds[0].traces[1])
    assert torch.equal(swapped[0].traces[1], rounds[0].traces[0])


def test_online_recapture_updates_bank_between_token_forwards() -> None:
    model = _OnlineFakeModel()
    runtime = _OnlineFakeRuntime()
    final_capture = _OnlineFakeCapture(model, site=False)
    site_capture = _OnlineFakeCapture(model, site=True)
    rounds = (
        MessageRound(
            row_indices=torch.tensor([0]),
            input_ids=torch.tensor([[10, 20, 30]]),
            position_ids=torch.tensor([[4, 5, 6]]),
            active_mask=torch.ones(1, 3, dtype=torch.bool),
        ),
    )

    message = online_recapture_history_state(
        model,
        runtime,
        final_capture,
        site_capture,
        rounds,
        mode="message",
    )
    assert message[:, 0, 0, 0].tolist() == [60.0]
    assert len(runtime.used_states) == 1

    runtime.used_states.clear()
    token = online_recapture_history_state(
        model,
        runtime,
        final_capture,
        site_capture,
        rounds,
        mode="token",
    )
    assert token[:, 0, 0, 0].tolist() == [60.0]
    assert [state[:, 0, 0, 0].item() for state in runtime.used_states] == [0.0, 10.0, 30.0]


def test_online_recapture_random_chunks_are_deterministic() -> None:
    model = _OnlineFakeModel()
    runtime = _OnlineFakeRuntime()
    final_capture = _OnlineFakeCapture(model, site=False)
    site_capture = _OnlineFakeCapture(model, site=True)
    rounds = (
        MessageRound(
            row_indices=torch.tensor([0]),
            input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            position_ids=torch.arange(5).reshape(1, 5),
            active_mask=torch.ones(1, 5, dtype=torch.bool),
        ),
    )
    first = online_recapture_history_state(
        model,
        runtime,
        final_capture,
        site_capture,
        rounds,
        mode="random",
        chunk_seed=1729,
    )
    runtime.used_states.clear()
    second = online_recapture_history_state(
        model,
        runtime,
        final_capture,
        site_capture,
        rounds,
        mode="random",
        chunk_seed=1729,
    )
    assert torch.equal(first, second)


def test_select_gate_batch_renumbers_history_rows() -> None:
    from benchmarks.train_qwen_ttt_gate1 import GateBatch

    ids = torch.arange(8).reshape(4, 2)
    mask = torch.ones_like(ids, dtype=torch.bool)
    message = MessageRound(
        row_indices=torch.arange(4),
        input_ids=ids.clone(),
        position_ids=torch.zeros_like(ids),
        active_mask=mask.clone(),
    )
    batch = GateBatch(
        history_rounds=(message,),
        student_input_ids=ids,
        student_attention_mask=mask,
        student_position_ids=torch.zeros_like(ids),
        teacher_input_ids=ids,
        teacher_attention_mask=mask,
        teacher_position_ids=torch.zeros_like(ids),
        teacher_target_mask=mask,
        target_ids=ids,
        response_ids=ids,
        response_mask=mask,
    )
    selected = select_gate_batch(batch, 1, 2)
    assert selected.history_rounds[0].row_indices.tolist() == [0, 1]
    assert selected.history_rounds[0].input_ids[:, 0].tolist() == [4, 6]


def test_random_chunk_spans_are_deterministic_and_cover_input() -> None:
    first = _chunk_spans(17, "random", 1729)
    second = _chunk_spans(17, "random", 1729)
    assert first == second
    assert first[0][0] == 0
    assert first[-1][1] == 17
    assert all(start < end for start, end in first)
    assert all(left[1] == right[0] for left, right in zip(first, first[1:]))


def test_manifest_latest_tensor_validation_rejects_changed_input() -> None:
    from benchmarks.train_qwen_ttt_gate1 import GateBatch, tensor_digest

    ids = torch.tensor([[1, 2], [1, 2]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    positions = torch.tensor([[0, 1], [0, 1]])
    batch = GateBatch(
        history_rounds=(),
        student_input_ids=ids,
        student_attention_mask=mask,
        student_position_ids=positions,
        teacher_input_ids=torch.ones(2, 1, dtype=torch.long),
        teacher_attention_mask=torch.ones(2, 1, dtype=torch.bool),
        teacher_position_ids=torch.zeros(2, 1, dtype=torch.long),
        teacher_target_mask=torch.ones(2, 1, dtype=torch.bool),
        target_ids=torch.ones(2, 1, dtype=torch.long),
        response_ids=torch.ones(2, 1, dtype=torch.long),
        response_mask=torch.ones(2, 1, dtype=torch.bool),
    )
    records = [
        {
            "latest_input_hash": tensor_digest(ids[:1]),
            "latest_mask_hash": tensor_digest(mask[:1]),
            "latest_position_hash": tensor_digest(positions[:1]),
        }
    ]
    validate_manifest_latest_tensors(batch, records)
    records[0]["latest_input_hash"] = "wrong"
    try:
        validate_manifest_latest_tensors(batch, records)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("changed manifest hash must be rejected")
