from __future__ import annotations

import copy

import pytest
import torch

import arti
from arti.formula_program_query_v3 import FormulaProgramEffectCandidateV2
from benchmarks import probe_qwen_federated_online_meta as probe
from benchmarks.probe_qwen_federated_online_meta import (
    DEFAULT_PROTOCOL,
    _episodes,
    _load_protocol,
    build_program_query,
    independent_bank_roots,
    qwen_suffix_logits_batched,
)


def test_protocol_uses_paired_counterfactual_latest_messages() -> None:
    protocol = _load_protocol(DEFAULT_PROTOCOL)
    episodes = _episodes(protocol)
    grouped: dict[str, list[object]] = {}
    for episode in episodes:
        grouped.setdefault(episode.pair_id, []).append(episode)

    assert protocol["scope"] == "private-research"
    assert protocol["model"]["revision"] == (
        "c1899de289a04d12100db370d81485cdf75e47ca"
    )
    assert all(len(rows) == 2 for rows in grouped.values())
    for rows in grouped.values():
        left, right = rows
        assert left.event_2 == right.event_2
        assert left.event_1 != right.event_1
        assert left.answer != right.answer


def test_protocol_rejects_a_non_counterfactual_pair(tmp_path) -> None:
    protocol = _load_protocol(DEFAULT_PROTOCOL)
    damaged = copy.deepcopy(protocol)
    damaged["episodes"][1]["event_2"] = "A different latest question"
    path = tmp_path / "protocol.json"
    path.write_text(__import__("json").dumps(damaged), encoding="utf-8")

    with pytest.raises(ValueError, match="keep event_2 fixed"):
        _load_protocol(path)


def test_low_rank_predecessor_starts_zero_and_effect_updates_only_its_bank() -> None:
    query, producer, effect = build_program_query(
        hidden_dim=8,
        rank=3,
        seed=17,
        device=torch.device("cpu"),
    )
    assert producer.bank_slot_ref is not None
    initial = query.initial_bank_state()
    x = torch.randn(1, 1, 8)

    execution = query({"x": x}, bank_state=initial)
    torch.testing.assert_close(execution.value, x)
    assert [step.candidate_id for step in execution.trace.steps] == [
        "plastic-lora",
        "outer-write",
        "stop",
    ]
    assert len(execution.proposals) == 1
    proposal = execution.proposals[0]
    assert proposal.target == producer.bank_slot_ref
    assert proposal.predecessor_id == producer.candidate_id
    assert torch.count_nonzero(initial.value(producer.bank_slot_ref)) == 0
    assert torch.count_nonzero(execution.bank_state.value(producer.bank_slot_ref)) > 0

    query_input = torch.randn(1, 4, 8)
    before = query.reexecute(
        producer.candidate_id,
        {"x": query_input},
        bank_state=initial,
    )
    after = query.reexecute(
        producer.candidate_id,
        {"x": query_input},
        bank_state=execution.bank_state,
    )
    torch.testing.assert_close(before, query_input)
    assert not torch.equal(after, before)
    after.square().mean().backward()
    assert effect.operand_store.tensor("writer").grad is not None
    assert effect.operand_store.tensor("rate").grad is not None
    assert arti.component_ref(producer.candidate) == "arti/formula-program-candidate@2"


def test_program_query_can_federate_every_neural_plasticity_family() -> None:
    query, producer, _primary_effect = build_program_query(
        hidden_dim=8,
        rank=3,
        seed=17,
        device=torch.device("cpu"),
        effect_execution_count=1.0,
        effect_families=("all",),
    )
    effects = tuple(
        candidate
        for candidate in query.candidates
        if isinstance(candidate, FormulaProgramEffectCandidateV2)
    )
    assert {candidate.candidate_id for candidate in effects} == {
        "affine-write",
        "blend-write",
        "outer-write",
        "transport-write",
        "polynomial-write",
        "proximal-write",
    }
    assert {candidate.atom_ref for candidate in effects} == {
        "arti/formula-atom-neural-plasticity@1",
        "arti/formula-atom-neural-plasticity-blend@1",
        "arti/formula-atom-neural-plasticity-outer@2",
        "arti/formula-atom-neural-plasticity-transport@1",
        "arti/formula-atom-neural-plasticity-polynomial@1",
        "arti/formula-atom-neural-plasticity-proximal@1",
    }

    initial = query.initial_bank_state()
    for effect in effects:
        arena = query._arena({"x": torch.randn(1, 4, 8)}, bank_state=initial)
        produced = producer(arena)
        current = produced.values.get("adapted")
        assert current is not None
        assert effect.accepts(produced)
        executed = effect(produced)
        assert executed.values.get("done") is current
        assert len(executed.proposals) == 1
        proposal = executed.proposals[0]
        assert proposal.target == producer.bank_slot_ref
        assert proposal.predecessor_id == producer.candidate_id
        assert proposal.effect_atom_ref == effect.atom_ref


def test_episode_bank_roots_do_not_share_storage_or_commits() -> None:
    query, producer, _effect = build_program_query(
        hidden_dim=8,
        rank=3,
        seed=17,
        device=torch.device("cpu"),
    )
    assert producer.bank_slot_ref is not None
    roots = independent_bank_roots(query, 3)
    pointers = {
        root.value(producer.bank_slot_ref).untyped_storage().data_ptr()
        for root in roots
    }
    assert len(pointers) == len(roots)

    updated = query({"x": torch.randn(1, 1, 8)}, bank_state=roots[0]).bank_state
    assert torch.count_nonzero(updated.value(producer.bank_slot_ref)) > 0
    assert all(
        torch.count_nonzero(root.value(producer.bank_slot_ref)) == 0
        for root in roots
    )


class _SuffixProbeModel(torch.nn.Module):
    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return hidden + attention_mask.unsqueeze(-1)


def test_batched_suffix_helper_rejects_incompatible_rows() -> None:
    model = _SuffixProbeModel()
    with pytest.raises(ValueError, match="non-empty"):
        qwen_suffix_logits_batched(model, (), layer_index=0)
    with pytest.raises(ValueError, match="feature"):
        qwen_suffix_logits_batched(
            model,
            (torch.zeros(1, 2, 3), torch.zeros(1, 2, 4)),
            layer_index=0,
        )


def test_batched_suffix_helper_preserves_order_and_gradient(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []

    def fake_suffix(
        _model: torch.nn.Module,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        assert layer_index == 4
        assert torch.all(attention_mask == 1)
        calls.append((hidden.shape[0], hidden.shape[1]))
        return hidden * 2

    monkeypatch.setattr(probe, "qwen_suffix_logits", fake_suffix)
    rows = (
        torch.full((1, 2, 3), 1.0, requires_grad=True),
        torch.full((1, 3, 3), 2.0, requires_grad=True),
        torch.full((1, 2, 3), 3.0, requires_grad=True),
    )
    outputs = qwen_suffix_logits_batched(
        _SuffixProbeModel(),
        rows,
        layer_index=4,
    )

    assert calls == [(2, 2), (1, 3)]
    for output, row in zip(outputs, rows, strict=True):
        torch.testing.assert_close(output, row * 2)
    sum(output.sum() for output in outputs).backward()
    assert all(torch.equal(row.grad, torch.full_like(row, 2.0)) for row in rows)
