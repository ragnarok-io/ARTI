from __future__ import annotations

import pytest
import torch

from arti import mechanisms
from benchmarks._federated_v4_federation import (
    build_autonomous_effect_federation,
    build_branch_visible_effect_federation,
)
from benchmarks.train_qwen_federated_effect_federation import EFFECT_FAMILIES


def _federation() -> object:
    return build_branch_visible_effect_federation(
        hidden_dim=4,
        rank=4,
        seed=731,
        device=torch.device("cpu"),
        plastic_branches=4,
        write_stages=2,
    )


def _autonomous_federation() -> object:
    return build_autonomous_effect_federation(
        hidden_dim=4,
        rank=4,
        seed=743,
        device=torch.device("cpu"),
        plastic_branches=4,
        min_operations=1,
        max_operations=3,
    )


def test_v4_federation_contains_large_search_dimensions_without_duplicate_fast_state() -> None:
    federation = _federation()
    query = federation.query

    assert len(query.owner_states) == 4
    assert len(federation.producer_stages) == 2
    assert all(len(stage) == 4 for stage in federation.producer_stages)
    assert len(federation.terminal_producers) == 4
    assert all(len(stage) == len(EFFECT_FAMILIES) for stage in federation.effect_stages)
    assert {
        effect.candidate_id.split("-write-stage-", maxsplit=1)[0]
        for effect in federation.effects
    } == set(EFFECT_FAMILIES)
    assert all(
        effect.atom_ref.startswith("arti/formula-atom-neural-plasticity")
        for effect in federation.effects
    )
    assert len(query.initial_bank_state().slot_refs) == 4
    assert len([name for name in query.state_dict() if name.endswith(".value")]) == 4

    owners = {
        candidate.bank_owner_id: candidate.bank_owner
        for candidate in federation.producer_stages[0]
    }
    for candidate in federation.producers:
        assert candidate.bank_owner is owners[candidate.bank_owner_id]


def test_v4_federation_effect_is_visible_to_later_ordinary_occurrence() -> None:
    federation = _federation()
    first = federation.producer_stages[0][0]
    effect = federation.effect_stages[0][2]
    reread = federation.producer_stages[1][0]
    x = torch.randn(1, 2, 4)
    entry = federation.query._arena({"x": x})
    produced = first(entry)
    modified = effect(produced)
    result = reread(modified)

    assert modified.values.get(effect.output_slot) is produced.values.get(first.output_slot)
    assert result.producer(reread.output_slot).plastic_revision == 1
    current, revision = result.effect_state(first.bank_slot_ref)
    assert revision == 1
    assert current is modified.proposals[-1].successor
    assert result.values.get(reread.output_slot) is not None


def test_v4_federation_task_bank_is_fast_state_not_optimizer_parameter() -> None:
    federation = _federation()
    query = federation.query
    bank_ids = {id(owner.value) for owner in query.owner_states}
    parameter_ids = {id(parameter) for parameter in query.parameters()}

    assert bank_ids.isdisjoint(parameter_ids)
    assert all(not isinstance(owner.value, torch.nn.Parameter) for owner in query.owner_states)
    assert all(
        isinstance(candidate, mechanisms.FormulaProgramEffectCandidateV3)
        for candidate in federation.effects
    )


def test_autonomous_effects_all_expose_trainable_execution_count() -> None:
    federation = _autonomous_federation()
    counts = tuple(candidate.execution_count_tensor() for candidate in federation.effects)

    assert len(federation.effects) == 18
    assert all(count is not None and count.requires_grad for count in counts)
    assert all(float(count.detach()) == 2.0 for count in counts)
    assert {candidate.hard_execution_count() for candidate in federation.effects} == {2}
    assert all(
        candidate.contract_config()["execution_count"]["semantics"]  # type: ignore[index]
        == "bounded-hard-repeat-predecessor-bank-slot"
        for candidate in federation.effects
    )


def test_autonomous_tail_effects_start_with_two_trainable_executions() -> None:
    federation = build_autonomous_effect_federation(
        hidden_dim=4,
        rank=4,
        seed=743,
        device=torch.device("cpu"),
        plastic_branches=4,
        min_operations=1,
        max_operations=3,
        max_effect_operations=2,
    )
    for layer in federation.tail_layers:
        assert len(layer) == len(EFFECT_FAMILIES)
        for effect in layer:
            count = effect.execution_count_tensor()
            assert count is not None and count.requires_grad
            assert float(count.detach()) == 2.0
            assert effect.hard_execution_count() == 2


def test_autonomous_checkpoint_keeps_learned_counts_instead_of_reinitializing() -> None:
    source = _autonomous_federation()
    with torch.no_grad():
        for effect in source.effects:
            effect.execution_count_tensor().fill_(1.125)
    target = _autonomous_federation()
    target.query.load_state_dict(source.query.state_dict(), strict=True)
    for effect in target.effects:
        assert float(effect.execution_count_tensor().detach()) == pytest.approx(1.125)
        assert effect.hard_execution_count() == 1


def test_autonomous_federation_searches_effect_position_and_exit_depth() -> None:
    federation = _autonomous_federation()
    query = federation.query
    arena = query._arena({"x": torch.randn(1, 2, 4)})

    initial_eligible = query.eligible(arena, steps=0)
    initial_ids = {
        candidate.candidate_id
        for candidate, eligible in zip(query.candidates, initial_eligible[:-1], strict=True)
        if bool(eligible)
    }
    assert initial_ids == {
        candidate.candidate_id
        for candidate in federation.transition_layers[0]
        if isinstance(candidate, mechanisms.FormulaProgramTensorCandidateV3)
    }

    first = next(
        candidate
        for candidate in federation.transition_layers[0]
        if isinstance(candidate, mechanisms.FormulaProgramTensorCandidateV3)
    )
    after_first = first(arena)
    next_eligible = query.eligible(after_first, steps=1)
    next_ids = {
        candidate.candidate_id
        for candidate, eligible in zip(query.candidates, next_eligible[:-1], strict=True)
        if bool(eligible)
    }
    assert {
        candidate.candidate_id for candidate in federation.transition_layers[1]
    }.issubset(next_ids)
    assert {
        candidate.candidate_id for candidate in federation.terminal_layers[0]
    }.issubset(next_ids)


def test_autonomous_federation_can_chain_effects_before_learned_exit() -> None:
    federation = _autonomous_federation()
    query = federation.query
    arena = query._arena({"x": torch.randn(1, 2, 4)})
    first = next(
        candidate
        for candidate in federation.transition_layers[0]
        if isinstance(candidate, mechanisms.FormulaProgramTensorCandidateV3)
    )
    effect_one = next(
        candidate
        for candidate in federation.transition_layers[1]
        if isinstance(candidate, mechanisms.FormulaProgramEffectCandidateV3)
    )
    effect_two = next(
        candidate
        for candidate in federation.transition_layers[2]
        if isinstance(candidate, mechanisms.FormulaProgramEffectCandidateV3)
    )

    produced = first(arena)
    modified_once = effect_one(produced)
    modified_twice = effect_two(modified_once)
    assert modified_once.values.get(effect_one.output_slot) is produced.values.get(
        first.output_slot
    )
    assert modified_twice.values.get(effect_two.output_slot) is modified_once.values.get(
        effect_one.output_slot
    )
    assert [proposal.predecessor_owner_id for proposal in modified_twice.proposals] == [
        first.bank_owner_id,
        first.bank_owner_id,
    ]
    assert [proposal.successor_revision for proposal in modified_twice.proposals] == [1, 2]

    terminal = next(
        candidate
        for candidate in federation.terminal_layers[-1]
        if candidate.bank_owner_id == first.bank_owner_id
    )
    completed = terminal(modified_twice)
    eligible = query.eligible(completed, steps=4)
    assert not bool(eligible[:-1].any())
    assert bool(eligible[-1])
