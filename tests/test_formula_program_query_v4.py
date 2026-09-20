from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import Tensor

import arti
from arti import mechanisms
from arti.component_registry import canonical_contract_reference


SOURCE_REF = "arti/federated-branch-visible-bank@1"


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
    execution_count: float | None = None,
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
        execution_count=None if execution_count is None else torch.tensor(execution_count),
        trainable_execution_count=execution_count is not None,
        max_executions=4,
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


@pytest.mark.parametrize("reverse_candidates", (False, True))
def test_cooperating_ssa_branches_keep_both_banks_and_final_loss_gradients(
    reverse_candidates: bool,
) -> None:
    left = _producer("left", "x", "left-value", owner_id="left-bank")
    right = _producer(
        "right", "x", "right-value", owner_id="right-bank", weight=torch.full((1, 3), 3.0),
    )
    left_write = _effect(
        "left-write", "left-value", "left-written", writer_scale=0.1, execution_count=2.0,
    )
    right_write = _effect(
        "right-write", "right-value", "right-written", writer_scale=0.2, execution_count=2.0,
    )
    left_read = _producer("left-read", "left-written", "left-output", owner_id="left-bank")
    right_read = _producer(
        "right-read", "right-written", "right-output", owner_id="right-bank",
        weight=torch.full((1, 3), 3.0),
    )
    a, b = mechanisms.InputBinding("a", _type()), mechanisms.InputBinding("b", _type())
    join = mechanisms.FormulaProgramTensorCandidateV3(mechanisms.FormulaProgramCandidate(
        "join", mechanisms.FormulaProgram.build(outputs=(mechanisms.add(a, b),)),
        input_slots={"a": "left-output", "b": "right-output"}, output_slot="terminal",
    ))
    candidates = (left, right, left_write, right_write, left_read, right_read, join)
    query = mechanisms.FormulaProgramQueryV4(
        slot_ids=(
            "x", "left-value", "right-value", "left-written", "right-written",
            "left-output", "right-output", "terminal",
        ),
        candidates=tuple(reversed(candidates)) if reverse_candidates else candidates,
        terminal_slot="terminal", max_steps=7, hidden_dim=8,
    )
    x = torch.tensor([[0.25, -0.5, 1.0]], requires_grad=True)
    before = query.initial_bank_state()
    execution = query({"x": x}, bank_state=before)
    expected_left = 2.0 + 0.4 * x
    expected_right = 3.0 + 1.2 * x
    torch.testing.assert_close(execution.value, 2.0 * x * expected_left + 3.0 * x * expected_right)
    assert {step.candidate_id for step in execution.trace.steps} == {
        *(candidate.candidate_id for candidate in candidates), "stop",
    }
    assert {proposal.predecessor_owner_id for proposal in execution.proposals} == {
        "left-bank", "right-bank",
    }
    for producer, expected in ((left, expected_left), (right, expected_right)):
        slot = producer.bank_slot_ref
        assert slot is not None
        torch.testing.assert_close(execution.bank_state.value(slot), expected)
        assert execution.bank_state.revision(slot) == 1
        assert before.revision(slot) == query.initial_bank_state().revision(slot) == 0
        assert not producer.bank_owner.value.requires_grad
    execution.value.sum().backward()
    torch.testing.assert_close(x.grad, 13.0 + 8.8 * x.detach())
    for effect, factor in ((left_write, 2.0), (right_write, 3.0)):
        torch.testing.assert_close(
            effect.operand_store.tensor("writer").grad, 2.0 * (factor * x.detach()).square(),
        )
        assert effect.hard_execution_count() == 2
    query.commit_(execution)
    for slot in before.slot_refs:
        assert query.initial_bank_state().revision(slot) == 1
        torch.testing.assert_close(query.initial_bank_state().value(slot), execution.bank_state.value(slot))

    query.requires_grad_(False)
    with torch.no_grad():
        next_execution = query({"x": x.detach()})
        left_next = expected_left * (1.0 + 0.2 * x)
        right_next = expected_right * (1.0 + 0.4 * x)
        torch.testing.assert_close(
            next_execution.value,
            x * expected_left * left_next + x * expected_right * right_next,
        )
    assert not next_execution.value.requires_grad
    for slot in before.slot_refs:
        assert next_execution.bank_state.revision(slot) == 2
        assert query.initial_bank_state().revision(slot) == 1


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


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("magnitude", (1.0, 1e21, 1e38))
def test_default_summary_preserves_finite_statistics_and_gradients(device, magnitude):
    query, *_ = _read_after_write_query()
    query.to(device)
    value = (torch.tensor([[0.5, 0.75, 1.0]], device=device) * magnitude).requires_grad_()
    reference = value.detach().double().requires_grad_()
    actual = query._summarize(query._arena({"x": value}))[:, :8]
    expected = torch.stack((
        reference.new_ones(1), reference.mean(-1), reference.std(-1, unbiased=False),
        reference.abs().mean(-1), reference.amax(-1), reference.amin(-1),
        reference.square().mean(-1).sqrt(), actual[:, -1].detach().double(),
    ), dim=-1)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, rtol=2e-6, atol=1e-7)
    actual.backward(torch.ones_like(actual) * 1e10)
    expected.backward(torch.ones_like(expected) * 1e10)
    assert torch.isfinite(value.grad).all()
    torch.testing.assert_close(value.grad.double(), reference.grad, rtol=2e-6, atol=1e-7)


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


def test_many_post_output_effects_keep_data_but_receive_future_credit() -> None:
    first = _producer("emit", "x", "terminal")
    effects = tuple(
        _effect(
            f"effect-{index}", "terminal" if index == 0 else f"tail-{index - 1}",
            f"tail-{index}", writer_scale=0.01,
        )
        for index in range(32)
    )
    query = mechanisms.FormulaProgramQueryV4(
        slot_ids=("x", "terminal", *(f"tail-{index}" for index in range(32))),
        candidates=(first, *effects), terminal_slot="terminal",
        min_steps=1, max_steps=33, min_tensor_steps=1,
        max_tensor_steps=1, max_effect_steps=32, hidden_dim=8,
    )
    with torch.no_grad():
        query.network[-1].weight.zero_()
        query.network[-1].bias.fill_(2.0)
        query.network[-1].bias[-1] = 0.0
    x = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
    execution = query({"x": x})
    assert len(execution.proposals) == 32
    assert [item.successor_revision for item in execution.proposals] == list(range(1, 33))
    assert all(item.predecessor_execution_id == "emit" for item in execution.proposals)
    torch.testing.assert_close(execution.value, x * 2, rtol=0, atol=0)
    assert execution.trace.steps[-1].candidate_id == "stop"
    writers = tuple(effect.operand_store.tensor("writer") for effect in effects)
    current_grads = torch.autograd.grad(
        execution.value.sum(), writers, allow_unused=True, retain_graph=True,
    )
    assert all(value is None for value in current_grads)
    future = first(query._arena({"x": x + 1}, bank_state=execution.bank_state))
    future_grads = torch.autograd.grad(future.values.get("terminal").sum(), writers)
    assert all(torch.isfinite(value).all() and value.abs().sum() > 0 for value in future_grads)
    assert query.initial_bank_state().revision(first.bank_slot_ref) == 0
    with torch.no_grad():
        deployed = query({"x": x.detach()})
    torch.testing.assert_close(deployed.value, execution.value)
    for left, right in zip(deployed.bank_state.values, execution.bank_state.values, strict=True):
        torch.testing.assert_close(left, right)


def test_effect_budget_is_independent_of_tensor_budget_and_stop_is_optional() -> None:
    first = _producer("emit", "x", "terminal")
    effects = (_effect("effect-1", "terminal", "tail-1"), _effect("effect-2", "tail-1", "tail-2"))
    query = mechanisms.FormulaProgramQueryV4(
        slot_ids=("x", "terminal", "tail-1", "tail-2"), candidates=(first, *effects),
        terminal_slot="terminal", max_steps=3, min_tensor_steps=1,
        max_tensor_steps=1, max_effect_steps=1,
    )
    arena = first(query._arena({"x": torch.ones(1, 3)}))
    assert arena.tensor_steps == 1 and arena.effect_steps == 0
    assert query.eligible(arena, steps=1).tolist() == [False, True, False, True]
    effected = effects[0](arena)
    assert effected.tensor_steps == 1 and effected.effect_steps == 1
    assert query.eligible(effected, steps=2).tolist() == [False, False, False, True]


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

    assert arti.component_ref(query) == canonical_contract_reference(
        "arti/formula-program-query@4"
    )
    assert arti.component_ref(first) == canonical_contract_reference(
        "arti/formula-program-tensor-candidate@3"
    )
    assert arti.component_ref(effect) == canonical_contract_reference(
        "arti/formula-program-effect-candidate@3"
    )
    assert query.contract_config()["pending_visibility"] == "branch-local-proposal-overlay"
    assert first.contract_config()["bank_read"] == "latest-branch-local-proposal-overlay"
    assert effect.contract_config()["lineage_ref"] == canonical_contract_reference(
        "arti/formula-producer-lineage@2"
    )

    provenance = arti.component_provenance(query)
    owner_nodes = [
        node
        for node in provenance["components"]
        if node["ref"]
        == canonical_contract_reference("arti/formula-program-bank-owner@1")
    ]
    assert len(owner_nodes) == 1
    producer_nodes = [
        node
        for node in provenance["components"]
        if node["ref"]
        == canonical_contract_reference("arti/formula-program-tensor-candidate@3")
    ]
    assert len(producer_nodes) == 2
    assert all(
        canonical_contract_reference("arti/formula-program-bank-owner@1")
        in node["dependencies"]
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
    assert arti.component_ref(old) == canonical_contract_reference(
        "arti/formula-program-tensor-candidate@2"
    )
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


def test_early_terminal_is_allowed_when_later_tensor_dispatches_remain() -> None:
    first = _producer("terminal-first", "x", "terminal")
    later = _producer("later", "terminal", "later")
    query = mechanisms.FormulaProgramQueryV4(
        slot_ids=("x", "terminal", "later"), candidates=(first, later),
        terminal_slot="terminal", min_steps=2, max_steps=2,
        min_tensor_steps=2, max_tensor_steps=2, hidden_dim=8,
    )
    arena = query._arena({"x": torch.randn(1, 3)})
    assert query._candidate_eligible(first, arena, steps=0)
    arena = first(arena)
    selected = arena.values.get("terminal")
    assert not query._stop_eligible(arena, steps=1)
    assert query._candidate_eligible(later, arena, steps=1)
    arena = later(arena)
    assert query._stop_eligible(arena, steps=2)
    assert arena.values.get("terminal") is selected


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
    assert arti.component_ref(encoder) == canonical_contract_reference(
        "arti/formula-program-query-tensor-encoder@1"
    )

    provenance = arti.component_provenance(query)
    root = next(node for node in provenance["components"] if node["path"] == "$")
    assert canonical_contract_reference("arti/formula-program-query-tensor-encoder@1") in root[
        "dependencies"
    ]


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
