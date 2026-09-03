from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import Tensor

import arti
from arti import mechanisms


SOURCE_REF = "arti/vouroboros-branch-visible-bank@1"


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
    input_slot: str,
    output_slot: str,
    *,
    owner_id: str = "shared-producer",
    weight: Tensor | None = None,
) -> mechanisms.FormulaProgramTensorCandidateV3:
    value = mechanisms.InputBinding("value", _type())
    bank = _bank("weight")
    candidate = mechanisms.FormulaProgramCandidate(
        candidate_id,
        mechanisms.FormulaProgram.build(outputs=(mechanisms.scale(value, bank),)),
        input_slots={"value": input_slot},
        output_slot=output_slot,
        operands={"weight": torch.tensor([[2.0, 2.0, 2.0]]) if weight is None else weight},
    )
    return mechanisms.FormulaProgramTensorCandidateV3(
        candidate,
        plastic_bank_slot="weight",
        bank_owner_id=owner_id,
    )


def _effect(
    candidate_id: str,
    input_slot: str,
    output_slot: str,
    *,
    writer_scale: float = 1.0,
) -> mechanisms.FormulaProgramEffectCandidateV3:
    value = mechanisms.InputBinding("value", _type())
    writer = _bank("writer")
    gain = _bank("gain")
    effect = mechanisms.neural_plasticity(
        value,
        mechanisms.scale(value, writer),
        mechanisms.scale(value, gain),
    )
    return mechanisms.FormulaProgramEffectCandidateV3(
        candidate_id,
        mechanisms.FormulaEffectProgramV2(
            mechanisms.FormulaProgram.build(outputs=(effect,)),
            data_input_name="value",
            state_type=_type(),
        ),
        input_slot=input_slot,
        output_slot=output_slot,
        operands={
            "writer": torch.full((1, 3), writer_scale),
            "gain": torch.zeros(1, 3),
        },
        trainable_operands=("writer",),
    )


def _read_after_write_query(
    *,
    content_encoder: bool = False,
) -> tuple[
    mechanisms.FormulaProgramQueryV4,
    mechanisms.FormulaProgramTensorCandidateV3,
    mechanisms.FormulaProgramEffectCandidateV3,
    mechanisms.FormulaProgramTensorCandidateV3,
]:
    first = _producer("producer-first", "x", "produced")
    effect = _effect("effect-first", "produced", "effected")
    second = _producer("producer-second", "effected", "terminal")
    query = mechanisms.FormulaProgramQueryV4(
        slot_ids=("x", "produced", "effected", "terminal"),
        candidates=(first, effect, second),
        terminal_slot="terminal",
        min_steps=3,
        max_steps=3,
        hidden_dim=8,
        tensor_encoder=(
            mechanisms.FormulaProgramQueryTensorEncoderV1(3, 4)
            if content_encoder
            else None
        ),
    )
    return query, first, effect, second


def test_effect_then_same_owner_reexecution_reads_branch_overlay() -> None:
    query, first, effect, second = _read_after_write_query()
    x = torch.tensor([[3.0, -1.0, 0.5]], requires_grad=True)
    entry = query._arena({"x": x})
    produced = first(entry)
    before = produced.values.get("produced")
    assert before is not None
    effected = effect(produced)
    assert effected.values.get("effected") is before
    reread = second(effected)
    actual = reread.values.get("terminal")
    assert actual is not None

    slot_ref = first.bank_slot_ref
    assert slot_ref is not None and slot_ref == second.bank_slot_ref
    assert len(query.initial_bank_state().slot_refs) == 1
    fast_state_keys = [name for name in query.state_dict() if name.endswith(".value")]
    assert fast_state_keys == ["owner_states.0.value"]
    successor, revision = effected.effect_state(slot_ref)
    torch.testing.assert_close(actual, before * successor)
    assert revision == 1
    assert not torch.equal(actual, before * entry.bank_state.value(slot_ref))
    assert entry.bank_state.revision(slot_ref) == 0
    assert torch.equal(entry.bank_state.value(slot_ref), torch.tensor([[2.0, 2.0, 2.0]]))

    actual.square().mean().backward()
    assert effect.operand_store.tensor("writer").grad is not None
    assert "candidates.0.candidate.operand_store.parameters_by_name.weight" not in dict(
        query.named_parameters()
    )


@pytest.mark.parametrize("content_encoder", [False, True])
def test_batched_value_summary_matches_independent_branches(content_encoder: bool) -> None:
    query, first, effect, _second = _read_after_write_query(content_encoder=content_encoder)
    inputs = tuple(torch.randn(1, 3, requires_grad=True) for _ in range(3))
    branches = tuple(effect(first(query._arena({"x": value}))) for value in inputs)
    serial = torch.cat(tuple(query._summarize(branch) for branch in branches))
    batched_values = mechanisms.FormulaProgramArena(
        branches[0].values.slot_ids,
        tuple(
            None
            if value is None
            else torch.cat(tuple(branch.values.values[index] for branch in branches))
            for index, value in enumerate(branches[0].values.values)
        ),
    )
    batched = query._summarize_values(batched_values)

    torch.testing.assert_close(batched, serial)
    parameters = inputs + (() if query.tensor_encoder is None else tuple(query.tensor_encoder.parameters()))
    weights = torch.linspace(0.1, 1.0, serial.numel()).reshape_as(serial)
    serial_gradients = torch.autograd.grad((serial * weights).sum(), parameters, retain_graph=True)
    batched_gradients = torch.autograd.grad((batched * weights).sum(), parameters)
    for actual, expected in zip(batched_gradients, serial_gradients, strict=True):
        torch.testing.assert_close(actual, expected)
    assert all(branch.bank_state.revisions == branches[0].bank_state.revisions for branch in branches)
    assert len({id(branch.proposals[0].successor) for branch in branches}) == len(branches)


def test_consecutive_effects_advance_lineage_and_bank_revision() -> None:
    first = _producer("producer-first", "x", "produced")
    effect1 = _effect("effect-one", "produced", "effected-one", writer_scale=0.5)
    effect2 = _effect("effect-two", "effected-one", "effected-two", writer_scale=0.25)
    second = _producer("producer-second", "effected-two", "terminal")
    query = mechanisms.FormulaProgramQueryV4(
        slot_ids=("x", "produced", "effected-one", "effected-two", "terminal"),
        candidates=(first, effect1, effect2, second),
        terminal_slot="terminal",
        min_steps=4,
        max_steps=4,
        hidden_dim=8,
    )
    execution = query({"x": torch.tensor([[1.0, 2.0, -1.0]])})

    assert [proposal.successor_revision for proposal in execution.proposals] == [1, 2]
    assert execution.proposals[1].previous is execution.proposals[0].successor
    assert execution.proposals[0].predecessor_execution_id == "producer-first"
    assert execution.proposals[0].predecessor_owner_id == "shared-producer"
    assert execution.bank_state.revision(first.bank_slot_ref) == 2
    assert [step.candidate_id for step in execution.trace.steps] == [
        "producer-first",
        "effect-one",
        "effect-two",
        "producer-second",
        "stop",
    ]
    assert execution.trace.steps[0].execution_id == "producer-first"
    assert execution.trace.steps[2].execution_id == "producer-first"
    assert execution.trace.steps[3].execution_id == "producer-second"
    assert execution.trace.steps[3].bank_owner_id == "shared-producer"


def test_stale_lineage_and_cross_branch_proposals_are_rejected_or_isolated() -> None:
    query, first, effect, _second = _read_after_write_query()
    stale_effect = _effect("stale-effect", "produced", "terminal", writer_scale=2.0)
    x = torch.tensor([[1.0, 0.5, -0.25]])
    entry = query._arena({"x": x})
    produced = first(entry)
    first_branch = effect(produced)

    assert not stale_effect.accepts(first_branch)
    with pytest.raises(ValueError, match="lineage is stale"):
        stale_effect(first_branch)

    other_effect = _effect("other-effect", "produced", "terminal", writer_scale=2.0)
    second_branch = other_effect(produced)
    assert len(first_branch.proposals) == len(second_branch.proposals) == 1
    assert first_branch.bank_state is second_branch.bank_state is entry.bank_state
    assert not torch.equal(
        first_branch.committed_state().value(first.bank_slot_ref),
        second_branch.committed_state().value(first.bank_slot_ref),
    )


def test_only_stopped_owner_execution_can_install_shared_bank_state() -> None:
    query, first, _effect_candidate, second = _read_after_write_query()
    execution = query({"x": torch.tensor([[1.0, 2.0, 3.0]])})
    entry = query.initial_bank_state()
    assert entry.revision(first.bank_slot_ref) == 0

    query.commit_(execution)
    committed = query.initial_bank_state()
    assert committed.revision(first.bank_slot_ref) == 1
    assert committed.revision(second.bank_slot_ref) == 1

    restored, restored_first, _restored_effect, restored_second = _read_after_write_query()
    restored.load_state_dict(copy.deepcopy(query.state_dict()))
    restored_state = restored.initial_bank_state()
    torch.testing.assert_close(
        restored_state.value(restored_first.bank_slot_ref),
        committed.value(first.bank_slot_ref),
    )
    assert restored_state.revision(restored_second.bank_slot_ref) == 1

    alien, *_ = _read_after_write_query()
    with pytest.raises(ValueError, match="only this ProgramQuery"):
        alien.commit_(execution)


def test_v4_component_contracts_are_new_and_v3_contract_remains_unchanged() -> None:
    query, first, effect, _second = _read_after_write_query()

    assert arti.component_ref(query) == "arti/formula-program-query@4"
    assert arti.component_ref(first) == "arti/formula-program-tensor-candidate@3"
    assert arti.component_ref(effect) == "arti/formula-program-effect-candidate@3"
    assert query.contract_config()["pending_visibility"] == "branch-local-proposal-overlay"
    assert first.contract_config()["bank_read"] == "latest-branch-local-proposal-overlay"
    assert effect.contract_config()["lineage_ref"] == "arti/formula-producer-lineage@2"

    provenance = arti.component_provenance(query)
    owner_nodes = [
        node
        for node in provenance["components"]
        if node["ref"] == "arti/formula-program-bank-owner@1"
    ]
    assert len(owner_nodes) == 1
    producer_nodes = [
        node
        for node in provenance["components"]
        if node["ref"] == "arti/formula-program-tensor-candidate@3"
    ]
    assert len(producer_nodes) == 2
    assert all(
        "arti/formula-program-bank-owner@1" in node["dependencies"]
        for node in producer_nodes
    )
    assert arti.validate_component_provenance(provenance) == provenance

    value = mechanisms.InputBinding("value", _type())
    bank = _bank("weight")
    old = mechanisms.FormulaProgramTensorCandidateV2(
        mechanisms.FormulaProgramCandidate(
            "old-producer",
            mechanisms.FormulaProgram.build(outputs=(mechanisms.scale(value, bank),)),
            input_slots={"value": "x"},
            output_slot="old-output",
            operands={"weight": torch.ones(1, 3)},
        ),
        plastic_bank_slot="weight",
    )
    assert arti.component_ref(old) == "arti/formula-program-tensor-candidate@2"
    with pytest.raises(TypeError, match="corrected Formula program candidates"):
        mechanisms.FormulaProgramQueryV3(
            slot_ids=("x", "produced", "effected", "terminal"),
            candidates=(first, effect),
            terminal_slot="terminal",
        )


def test_shared_owner_rejects_incompatible_or_divergent_occurrences() -> None:
    first = _producer("producer-first", "x", "first")
    divergent = _producer(
        "producer-second",
        "first",
        "second",
        weight=torch.tensor([[3.0, 3.0, 3.0]]),
    )
    with pytest.raises(ValueError, match="identical state"):
        mechanisms.FormulaProgramQueryV4(
            slot_ids=("x", "first", "second"),
            candidates=(first, divergent),
            terminal_slot="second",
        )


def test_effect_data_lane_is_same_tensor_object() -> None:
    query, first, effect, _second = _read_after_write_query()
    entry = query._arena({"x": torch.randn(1, 3)})
    produced = first(entry)
    before = produced.values.get("produced")
    after = effect(produced).values.get("effected")
    assert before is after


def test_query_observes_effect_only_after_an_ordinary_formula_reread() -> None:
    query, first, effect, _second = _read_after_write_query()
    alternative = _effect("alternative", "produced", "effected", writer_scale=3.0)
    entry = query._arena({"x": torch.tensor([[1.0, -2.0, 0.5]])})
    produced = first(entry)
    left = effect(produced)
    right = alternative(produced)

    assert not torch.equal(left.proposals[-1].successor, right.proposals[-1].successor)
    torch.testing.assert_close(query._summarize(left), query._summarize(right))
    torch.testing.assert_close(
        query.query(left, steps=2).masked_logits,
        query.query(right, steps=2).masked_logits,
    )


def test_content_encoder_preserves_dynamic_length_content_for_query_routing() -> None:
    query, _first, _effect_candidate, _second = _read_after_write_query(
        content_encoder=True
    )
    encoder = query.tensor_encoder
    assert encoder is not None
    one = torch.tensor([[[1.0, 0.0, -1.0]]], requires_grad=True)
    two = torch.tensor([[[1.0, 0.0, -1.0], [-2.0, 1.0, 0.5]]])
    encoded_one = encoder(one)
    encoded_two = encoder(two)

    assert encoded_one.shape == encoded_two.shape == (1, encoder.output_width)
    assert not torch.equal(encoded_one, encoded_two)
    encoded_one.square().mean().backward()
    assert one.grad is not None
    assert any(parameter.grad is not None for parameter in encoder.parameters())
    assert arti.component_ref(encoder) == "arti/formula-program-query-tensor-encoder@1"

    provenance = arti.component_provenance(query)
    root = next(node for node in provenance["components"] if node["path"] == "$")
    assert "arti/formula-program-query-tensor-encoder@1" in root["dependencies"]


def _floating_contract_query(candidate_class, *, shared_owner: bool, dtype: torch.dtype):
    value_type = mechanisms.TensorType(
        ("B", "D"), ("B", 3), dtype="floating", domain="activation",
    )
    value = mechanisms.InputBinding("value", value_type)
    weight = mechanisms.BankBinding("weight", SOURCE_REF, "weight", value_type)
    program = mechanisms.FormulaProgram.build(outputs=(mechanisms.scale(value, weight),))
    edges = (("x", "middle"), ("middle", "terminal")) if shared_owner else (("x", "terminal"),)
    occurrences = tuple(
        mechanisms.FormulaProgramTensorCandidateV3(
            candidate_class(
                f"producer-{index}",
                program,
                input_slots={"value": source},
                output_slot=target,
                operands={"weight": torch.full((1, 3), 2.0, dtype=dtype)},
                trainable_operands=() if shared_owner else ("weight",),
            ),
            plastic_bank_slot="weight" if shared_owner else None,
            bank_owner_id="shared-floating-owner" if shared_owner else None,
        )
        for index, (source, target) in enumerate(edges)
    )
    query = mechanisms.FormulaProgramQueryV4(
        slot_ids=("x", "middle", "terminal") if shared_owner else ("x", "terminal"),
        candidates=occurrences,
        terminal_slot="terminal",
        min_steps=len(occurrences),
        max_steps=len(occurrences),
        hidden_dim=8,
    ).to(dtype=dtype)
    return query, occurrences


@pytest.mark.parametrize(
    "candidate_class",
    (mechanisms.FormulaProgramCandidate, mechanisms.FormulaProgramCandidateV2),
    ids=("single-atom", "subprogram"),
)
@pytest.mark.parametrize("shared_owner", (False, True), ids=("ordinary", "shared-owner"))
def test_floating_contracts_follow_dtype_and_strict_roundtrip(
    candidate_class, shared_owner: bool, tmp_path: Path,
) -> None:
    query, occurrences = _floating_contract_query(
        candidate_class, shared_owner=shared_owner, dtype=torch.float32,
    )
    first = occurrences[0]
    before = copy.deepcopy(first.contract_config())
    slot_refs = tuple(item.bank_slot_ref for item in occurrences)
    owner = first.bank_owner if shared_owner else None
    query.to(dtype=torch.float64)

    assert before["candidate"]["operands"]["weight"]["dtype"] == "torch.float32"
    expected_operand = {
        "shape": [1, 3], "dtype": "torch.float64", "trainable": not shared_owner,
    }
    for occurrence, slot_ref in zip(occurrences, slot_refs, strict=True):
        assert occurrence.bank_slot_ref == slot_ref
        assert occurrence.candidate.operand_store.contract_config()["weight"] == expected_operand
        assert occurrence.candidate.contract_config()["operands"]["weight"] == expected_operand
        assert occurrence.contract_config()["candidate"]["operands"]["weight"] == expected_operand

    if shared_owner:
        assert owner is not None
        assert query.owner_states[0] is owner
        assert all(item.bank_owner is owner for item in occurrences)
        assert all(
            item.candidate.operand_store.tensor("weight") is owner.value for item in occurrences
        )
        assert owner.contract_config()["dtype"] == "torch.float64"
        assert [name for name in query.state_dict() if name.endswith(".value")] == [
            "owner_states.0.value",
        ]
        assert not any(
            "operand_store.operand_" in name for name in query.state_dict()
        )
        state = query.initial_bank_state()
        owner.install_(state.replace(owner.slot_ref, torch.full((1, 3), 3.0, dtype=torch.float64),
                                     revision=1))

    x = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
    expected = query({"x": x}).value
    assert expected.dtype == torch.float64
    saved = arti.save(query, tmp_path / "floating-query.arti.st")
    restored, restored_occurrences = _floating_contract_query(
        candidate_class, shared_owner=shared_owner, dtype=torch.float64,
    )
    arti.load(saved.weights_path, model=restored, strict=True, verify_architecture=True)
    assert arti.component_provenance(restored) == arti.component_provenance(query)
    for name, tensor in query.state_dict().items():
        assert torch.equal(restored.state_dict()[name], tensor)
    for source, target in zip(occurrences, restored_occurrences, strict=True):
        assert target.contract_config() == source.contract_config()
    torch.testing.assert_close(restored({"x": x}).value, expected, rtol=0, atol=0)
    if shared_owner:
        restored_owner = restored.owner_states[0]
        assert all(item.bank_owner is restored_owner for item in restored_occurrences)
        assert all(
            item.candidate.operand_store.tensor("weight") is restored_owner.value
            for item in restored_occurrences
        )
        assert restored.initial_bank_state().revision(restored_owner.slot_ref) == 1
