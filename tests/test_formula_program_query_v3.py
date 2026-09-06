from __future__ import annotations

import copy
from dataclasses import fields

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

import arti
from arti import mechanisms


SOURCE_REF = "arti/test-predecessor-bank@1"
SLOTS = ("x", "primary", "decoy", "after-effect")


def _type() -> mechanisms.TensorType:
    return mechanisms.TensorType(
        ("B", "D"),
        ("B", 3),
        dtype="float32",
        domain="activation",
    )


def _bank(name: str) -> mechanisms.BankBinding:
    return mechanisms.BankBinding(name, SOURCE_REF, name, _type())


def _producer(
    candidate_id: str,
    output_slot: str,
    weight: Tensor,
) -> mechanisms.FormulaProgramTensorCandidateV2:
    value = mechanisms.InputBinding("value", _type())
    bank = _bank("weight")
    candidate = mechanisms.FormulaProgramCandidate(
        candidate_id,
        mechanisms.FormulaProgram.build(outputs=(mechanisms.scale(value, bank),)),
        input_slots={"value": "x"},
        output_slot=output_slot,
        operands={"weight": weight},
    )
    return mechanisms.FormulaProgramTensorCandidateV2(
        candidate,
        plastic_bank_slot="weight",
    )


def _effect() -> mechanisms.FormulaProgramEffectCandidateV2:
    value = mechanisms.InputBinding("value", _type())
    writer = _bank("writer")
    gain = _bank("gain")
    effect = mechanisms.neural_plasticity(
        value,
        mechanisms.scale(value, writer),
        mechanisms.scale(value, gain),
    )
    return mechanisms.FormulaProgramEffectCandidateV2(
        "self-operate",
        mechanisms.FormulaEffectProgramV2(
            mechanisms.FormulaProgram.build(outputs=(effect,)),
            data_input_name="value",
            state_type=_type(),
        ),
        input_slot="primary",
        output_slot="after-effect",
        operands={
            "writer": torch.ones(1, 3),
            "gain": torch.zeros(1, 3),
        },
        trainable_operands=("writer",),
    )


def _passthrough() -> mechanisms.FormulaProgramTensorCandidateV2:
    value = mechanisms.InputBinding("value", _type())
    weight = _bank("passthrough")
    candidate = mechanisms.FormulaProgramCandidate(
        "no-effect",
        mechanisms.FormulaProgram.build(outputs=(mechanisms.scale(value, weight),)),
        input_slots={"value": "primary"},
        output_slot="after-effect",
        operands={"passthrough": torch.ones(1, 3)},
    )
    return mechanisms.FormulaProgramTensorCandidateV2(candidate)


def _query() -> tuple[
    mechanisms.FormulaProgramQueryV3,
    mechanisms.FormulaProgramTensorCandidateV2,
    mechanisms.FormulaProgramTensorCandidateV2,
    mechanisms.FormulaProgramEffectCandidateV2,
]:
    primary = _producer("primary-producer", "primary", torch.tensor([[0.5, 1.0, -0.5]]))
    decoy = _producer("decoy-producer", "decoy", torch.tensor([[3.0, -2.0, 4.0]]))
    effect = _effect()
    no_effect = _passthrough()
    query = mechanisms.FormulaProgramQueryV3(
        slot_ids=SLOTS,
        candidates=(primary, decoy, effect, no_effect),
        terminal_slot="after-effect",
        min_steps=1,
        max_steps=3,
        hidden_dim=16,
    )
    return query, primary, decoy, effect


def test_effect_targets_actual_predecessor_bank_and_preserves_data_identity() -> None:
    query, primary, decoy, effect = _query()
    x = torch.tensor([[2.0, -1.0, 3.0]])
    entry = query._arena({"x": x})
    with_primary = primary(entry)
    with_decoy = decoy(with_primary)
    before = with_decoy.values.get("primary")
    assert before is not None

    after = effect(with_decoy)
    assert after.values.get("after-effect") is before
    assert after.bank_state is entry.bank_state
    assert len(after.proposals) == 1

    proposal = after.proposals[0]
    assert proposal.target == primary.bank_slot_ref
    assert proposal.predecessor_id == primary.candidate_id
    assert proposal.target != decoy.bank_slot_ref
    assert proposal.previous_revision == 0
    assert proposal.successor_revision == 1
    torch.testing.assert_close(proposal.previous, torch.tensor([[0.5, 1.0, -0.5]]))
    torch.testing.assert_close(proposal.successor, proposal.previous + before)

    committed = after.committed_state()
    assert committed.revision(primary.bank_slot_ref) == 1
    assert committed.revision(decoy.bank_slot_ref) == 0
    torch.testing.assert_close(
        committed.value(decoy.bank_slot_ref),
        entry.bank_state.value(decoy.bank_slot_ref),
    )


def test_pending_effect_is_write_only_and_changes_only_next_invocation() -> None:
    query, primary, _decoy, effect = _query()
    x = torch.tensor([[2.0, -1.0, 3.0]])
    entry = query._arena({"x": x})
    first = primary(entry)
    first_value = first.values.get("primary")
    assert first_value is not None
    pending = effect(first)

    _inputs, same_call_banks = primary._bindings(pending)
    torch.testing.assert_close(
        same_call_banks["weight"].value,
        entry.bank_state.value(primary.bank_slot_ref),
    )

    successor_state = pending.committed_state()
    next_entry = query._arena({"x": x}, bank_state=successor_state)
    second = primary(next_entry)
    second_value = second.values.get("primary")
    assert second_value is not None
    assert not torch.equal(second_value, first_value)
    torch.testing.assert_close(second_value, x * successor_state.value(primary.bank_slot_ref))

    with torch.no_grad():
        for parameter in query.network.parameters():
            parameter.zero_()
        query.network[-1].bias.copy_(torch.tensor([4.0, 0.0, 3.0, 1.0, 2.0]))
    execution = query({"x": x})
    query.commit_(execution)
    reloaded = query.initial_bank_state()
    assert reloaded.revision(primary.bank_slot_ref) == 1
    torch.testing.assert_close(
        reloaded.value(primary.bank_slot_ref),
        execution.bank_state.value(primary.bank_slot_ref),
    )


def test_corrected_contract_has_no_effect_owned_state_or_explicit_target() -> None:
    query, primary, _decoy, effect = _query()

    assert arti.component_ref(query) == "arti/formula-program-query@3"
    assert arti.component_ref(primary) == "arti/formula-program-tensor-candidate@2"
    assert arti.component_ref(effect) == "arti/formula-program-effect-candidate@2"
    config = effect.contract_config()
    assert config["target_resolution"] == "dynamic-immediate-predecessor-bank-slot"
    assert config["data_lane"] == "identity"
    assert "state_id" not in config
    assert "state_operands" not in config
    assert "target" not in config


def test_composed_lora_producer_can_own_zero_initialized_plastic_bank() -> None:
    hidden_type = mechanisms.TensorType(
        ("B", "S", "D"),
        ("B", "S", 4),
        dtype="float32",
        domain="activation",
    )
    a_type = mechanisms.TensorType(
        ("R", "D"),
        (2, 4),
        dtype="float32",
        domain="activation",
    )
    b_type = mechanisms.TensorType(
        ("D", "R"),
        (4, 2),
        dtype="float32",
        domain="activation",
    )
    scalar_type = mechanisms.TensorType.scalar(dtype="float32", domain="activation")
    value = mechanisms.InputBinding("value", hidden_type)
    base = mechanisms.InputBinding("base", hidden_type)
    projection = mechanisms.BankBinding("lora.a", SOURCE_REF, "a", a_type)
    memory = mechanisms.BankBinding("lora.b", SOURCE_REF, "b", b_type)
    gain = mechanisms.BankBinding("lora.gain", SOURCE_REF, "gain", scalar_type)
    low = mechanisms.contract(
        value,
        projection,
        reduce_axes=(("D", "D"),),
        output_axes=("B", "S", "R"),
    )
    delta = mechanisms.contract(
        low,
        memory,
        reduce_axes=(("R", "R"),),
        output_axes=("B", "S", "D"),
    )
    residual = mechanisms.add(base, mechanisms.scale(delta, gain))
    program = mechanisms.FormulaProgram.build(outputs=(residual,))
    a = torch.randn(2, 4) / 2.0
    producer = mechanisms.FormulaProgramTensorCandidateV2(
        mechanisms.FormulaProgramCandidateV2(
            "lora-producer",
            program,
            input_slots={"value": "x", "base": "x"},
            output_slot="adapted",
            operands={
                "lora.a": a,
                "lora.b": torch.zeros(4, 2),
                "lora.gain": torch.tensor(1.0),
            },
        ),
        plastic_bank_slot="lora.b",
    )

    effect_value = mechanisms.InputBinding("value", hidden_type)
    writer = mechanisms.BankBinding("writer", SOURCE_REF, "writer", a_type)
    rate = mechanisms.BankBinding("rate", SOURCE_REF, "rate", scalar_type)
    summary = mechanisms.reduce_sum(
        mechanisms.reduce_sum(effect_value, axis="B"),
        axis="S",
    )
    key = mechanisms.contract(
        summary,
        writer,
        reduce_axes=(("D", "D"),),
        output_axes=("R",),
    )
    effect_expr = mechanisms.neural_plasticity_outer(effect_value, summary, key, rate)
    effect = mechanisms.FormulaProgramEffectCandidateV2(
        "outer-write",
        mechanisms.FormulaEffectProgramV2(
            mechanisms.FormulaProgram.build(outputs=(effect_expr,)),
            data_input_name="value",
            state_type=b_type,
        ),
        input_slot="adapted",
        output_slot="done",
        operands={"writer": a, "rate": torch.tensor(0.05)},
        trainable_operands=("writer", "rate"),
    )
    query = mechanisms.FormulaProgramQueryV3(
        slot_ids=("x", "adapted", "done"),
        candidates=(producer, effect),
        terminal_slot="done",
        min_steps=2,
        max_steps=2,
        hidden_dim=8,
    )
    x = torch.randn(1, 3, 4)
    initial = query.initial_bank_state()
    assert producer.bank_slot_ref is not None
    assert torch.count_nonzero(initial.value(producer.bank_slot_ref)) == 0

    produced = producer(query._arena({"x": x}, bank_state=initial))
    adapted = produced.values.get("adapted")
    assert adapted is not None
    effected = effect(produced)
    assert effected.values.get("done") is adapted

    execution = query({"x": x}, bank_state=initial)
    assert [step.candidate_id for step in execution.trace.steps] == [
        "lora-producer",
        "outer-write",
        "stop",
    ]
    torch.testing.assert_close(execution.value, x)
    updated = execution.bank_state
    assert torch.count_nonzero(updated.value(producer.bank_slot_ref)) > 0
    reread = query.reexecute("lora-producer", {"x": x}, bank_state=updated)
    assert not torch.equal(reread, x)

    reread.square().mean().backward()
    assert effect.operand_store.tensor("writer").grad is not None
    assert effect.operand_store.tensor("rate").grad is not None
    assert arti.component_ref(producer.candidate) == "arti/formula-program-candidate@2"


def test_two_event_training_uses_only_reexecuted_predecessor_output() -> None:
    query, primary, _decoy, effect = _query()
    support = torch.tensor([[0.8, -0.4, 1.2]])
    query_input = torch.tensor([[1.0, 0.0, 1.0]])
    initial = query.initial_bank_state().value(primary.bank_slot_ref)
    support_output = support * initial
    target = query_input * (initial + 2.0 * support_output)
    trainer = mechanisms.ExactFormulaProgramQueryTrainingV3(
        invalid_weight=2.0,
        max_states=64,
        exploration_probability=0.1,
    )

    loss = trainer.loss(
        query,
        event1={"x": support},
        event2={"x": query_input},
        event2_candidate_id="primary-producer",
        target=target,
        task_loss=lambda output, expected: F.mse_loss(
            output,
            expected,
            reduction="none",
        ).mean(dim=-1),
    )
    loss.total.backward()

    assert loss.visited_states > 1
    assert loss.success_probability > 0
    assert all(parameter.grad is not None for parameter in query.network.parameters())
    writer = effect.operand_store.tensor("writer")
    assert writer.grad is not None
    assert torch.isfinite(writer.grad).all()
    contract = trainer.contract_config()
    assert contract["supervision"] == "event-2-final-task-loss-only"
    assert contract["state_readout"] is False
    assert contract["route_teacher"] is False
    assert (
        arti.component_ref(trainer)
        == "arti/exact-formula-program-query-training@3"
    )


def test_exact_training_rejects_rows_that_would_share_one_bank_root() -> None:
    query, _primary, _decoy, _effect_candidate = _query()
    trainer = mechanisms.ExactFormulaProgramQueryTrainingV3(max_states=64)
    event = {"x": torch.ones(2, 3)}

    with pytest.raises(ValueError, match="batch size one"):
        trainer.loss(
            query,
            event1=event,
            event2=event,
            event2_candidate_id="primary-producer",
            target=torch.ones(2, 3),
            task_loss=lambda output, expected: F.mse_loss(
                output,
                expected,
                reduction="none",
            ).mean(dim=-1),
        )


def test_bank_state_is_canonical_and_commit_survives_fresh_reload() -> None:
    query, primary, decoy, _effect_candidate = _query()
    initial = query.initial_bank_state()
    reversed_state = mechanisms.FormulaProgramBankState(
        tuple(reversed(initial.slot_refs)),
        tuple(reversed(initial.values)),
        tuple(reversed(initial.revisions)),
    )
    assert reversed_state.slot_refs == initial.slot_refs
    for slot_ref in initial.slot_refs:
        torch.testing.assert_close(reversed_state.value(slot_ref), initial.value(slot_ref))

    x = torch.tensor([[2.0, -1.0, 3.0]])
    with torch.no_grad():
        for parameter in query.network.parameters():
            parameter.zero_()
        action_index = query.candidate_ids.index("primary-producer")
        effect_index = query.candidate_ids.index("self-operate")
        query.network[-1].bias[action_index] = 4.0
        query.network[-1].bias[effect_index] = 3.0
    execution = query({"x": x})
    query.commit_(execution)
    expected = query.reexecute(
        primary.candidate_id,
        {"x": x},
        bank_state=query.initial_bank_state(),
    )

    restored, restored_primary, restored_decoy, _restored_effect = _query()
    restored.load_state_dict(copy.deepcopy(query.state_dict()))
    actual = restored.reexecute(
        restored_primary.candidate_id,
        {"x": x},
        bank_state=restored.initial_bank_state(),
    )
    torch.testing.assert_close(actual, expected)
    assert restored.initial_bank_state().revision(restored_primary.bank_slot_ref) == 1
    assert restored.initial_bank_state().revision(restored_decoy.bank_slot_ref) == 0
    assert restored_primary.bank_slot_ref == primary.bank_slot_ref
    assert restored_decoy.bank_slot_ref == decoy.bank_slot_ref


def test_bank_position_index_does_not_cache_values_or_revisions() -> None:
    query, primary, decoy, _effect_candidate = _query()
    original = query.initial_bank_state()
    ref = primary.bank_slot_ref
    before = original.value(ref)
    assert original._index(ref) == original.slot_refs.index(ref)
    successor = before + 1
    changed = original.replace(ref, successor, revision=original.revision(ref) + 1)
    assert changed.value(ref) is successor
    assert original.value(ref) is before
    assert changed.revision(ref) == original.revision(ref) + 1
    assert changed.value(decoy.bank_slot_ref) is original.value(decoy.bank_slot_ref)
    assert changed._slot_positions is not original._slot_positions
    assert tuple(field.name for field in fields(original)) == ("slot_refs", "values", "revisions")
    with pytest.raises(KeyError):
        mechanisms.FormulaProgramBankState.empty().value(ref)


def test_helpful_losing_effect_cannot_leak_into_committed_winner() -> None:
    query, primary, _decoy, effect = _query()
    support = torch.tensor([[1.5, -0.75, 0.5]])
    event2 = torch.tensor([[1.0, 1.0, 1.0]])
    entry = query._arena({"x": support})
    produced = primary(entry)
    helpful_branch = effect(produced).committed_state()
    helpful_output = query.reexecute(
        primary.candidate_id,
        {"x": event2},
        bank_state=helpful_branch,
    )

    with torch.no_grad():
        for parameter in query.network.parameters():
            parameter.zero_()
        priorities = {
            "primary-producer": 5.0,
            "no-effect": 4.0,
            "self-operate": 3.0,
            "decoy-producer": 1.0,
            "stop": 2.0,
        }
        query.network[-1].bias.copy_(
            torch.tensor([priorities[action_id] for action_id in query.action_ids])
        )
    winner = query({"x": support})
    assert [step.candidate_id for step in winner.trace.steps] == [
        "primary-producer",
        "no-effect",
        "stop",
    ]
    assert winner.proposals == ()
    query.commit_(winner)
    committed = query.initial_bank_state()
    torch.testing.assert_close(
        committed.value(primary.bank_slot_ref),
        entry.bank_state.value(primary.bank_slot_ref),
    )
    winner_output = query.reexecute(
        primary.candidate_id,
        {"x": event2},
        bank_state=committed,
    )
    assert F.mse_loss(helpful_output, helpful_output) < F.mse_loss(
        winner_output,
        helpful_output,
    )


def test_bank_replacement_sequence_keeps_order_identity_and_gradients() -> None:
    query, primary, decoy, _effect = _query()
    initial = query.initial_bank_state()
    leaves = tuple(value.detach().clone().requires_grad_() for value in initial.values)
    initial = mechanisms.FormulaProgramBankState(initial.slot_refs, leaves, initial.revisions)
    p_ref, d_ref = primary.bank_slot_ref, decoy.bank_slot_ref
    p, d = initial.value(p_ref), initial.value(d_ref)
    first = 2 * p
    other = d * 0.5
    last = first.square() + other.sum()
    updates = ((p_ref, first, 1), (d_ref, other, 1), (p_ref, last, 2))
    serial = initial
    for ref, value, revision in updates:
        serial = serial.replace(ref, value, revision=revision)
    together = initial._replace_sequence(iter(updates))

    assert together.slot_refs == serial.slot_refs == initial.slot_refs
    assert together.revisions == serial.revisions
    assert together.revision(p_ref) == 2
    assert together.revision(d_ref) == 1
    for ref in initial.slot_refs:
        assert together.value(ref) is serial.value(ref)
        if ref not in (p_ref, d_ref):
            assert together.value(ref) is initial.value(ref)
    assert together.value(p_ref) is last
    assert together.value(d_ref) is other
    assert initial.value(p_ref) is p
    assert initial.value(d_ref) is d
    assert initial.revision(p_ref) == initial.revision(d_ref) == 0

    gradients = torch.autograd.grad(together.value(p_ref).sum(), (p, d))
    torch.testing.assert_close(gradients[0], 8 * p)
    torch.testing.assert_close(gradients[1], torch.full_like(d, 0.5 * p.numel()))


@pytest.mark.parametrize("invalid", ("shape", "dtype", "revision", "decreasing"))
def test_bank_replacement_sequence_validates_intermediate_updates(invalid: str) -> None:
    query, primary, _decoy, _effect = _query()
    initial = query.initial_bank_state()
    ref = primary.bank_slot_ref
    value = initial.value(ref)
    updates = [(ref, value + 1, 1), (ref, value + 2, 2)]
    if invalid == "shape":
        updates[0] = (ref, value.flatten(), 1)
    elif invalid == "dtype":
        updates[0] = (ref, value.double(), 1)
    elif invalid == "revision":
        updates[0] = (ref, value + 1, True)
    else:
        updates[0] = (ref, value + 1, 3)
    with pytest.raises(ValueError, match="successor"):
        initial._replace_sequence(iter(updates))
    assert initial.value(ref) is value
    assert initial.revision(ref) == 0


def test_plastic_program_slot_cannot_also_be_optimizer_trainable() -> None:
    value = mechanisms.InputBinding("value", _type())
    weight = _bank("weight")
    candidate = mechanisms.FormulaProgramCandidateV2(
        "producer",
        mechanisms.FormulaProgram.build(outputs=(mechanisms.scale(value, weight),)),
        input_slots={"value": "x"},
        output_slot="primary",
        operands={"weight": torch.ones(1, 3)},
        trainable_operands=("weight",),
    )

    with pytest.raises(ValueError, match="forward-written state"):
        mechanisms.FormulaProgramTensorCandidateV2(
            candidate,
            plastic_bank_slot="weight",
        )
