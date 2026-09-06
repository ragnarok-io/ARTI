from __future__ import annotations

import copy

import pytest
import torch

import arti
from arti import mechanisms as m


def _type():
    return m.TensorType(("B", "D"), ("B", 3), dtype="floating", domain="activation")


def _multi(candidate_id="producer", *, owner_id="memory"):
    x = m.InputBinding("x", _type())
    weight = m.BankBinding("weight", "arti/named-output-test@1", "weight", _type())
    gain = m.BankBinding("gain", "arti/named-output-test@1", "gain", _type())
    program = m.FormulaProgram.build(outputs=(m.add(x, x), m.scale(m.scale(x, weight), gain)))
    candidate = m.FormulaProgramCandidateV3(
        candidate_id, program, input_slots={"x": "x"},
        output_slots=dict(zip(program.outputs, ("plain", "owned"), strict=True)),
        operands={"weight": torch.full((1, 3), 2.0), "gain": torch.ones(1, 3)},
        trainable_operands=("gain",),
    )
    return m.FormulaProgramTensorCandidateV4(
        candidate, plastic_bank_slot="weight", bank_owner_id=owner_id,
    )


def _effect(input_slot="owned"):
    x = m.InputBinding("x", _type())
    rate = m.BankBinding("rate", "arti/named-output-test@1", "rate", _type())
    zero = m.BankBinding("zero", "arti/named-output-test@1", "zero", _type())
    effect = m.neural_plasticity(x, m.scale(x, rate), m.scale(x, zero))
    return m.FormulaProgramEffectCandidateV3(
        "write", m.FormulaEffectProgramV2(
            m.FormulaProgram.build(outputs=(effect,)), data_input_name="x", state_type=_type(),
        ),
        input_slot=input_slot, output_slot="tail", operands={
            "rate": torch.full((1, 3), 0.1), "zero": torch.zeros(1, 3),
        }, trainable_operands=("rate",), execution_count=torch.tensor(2.0),
        trainable_execution_count=True, max_executions=4,
    )


def _query(*, write=False):
    producer = _multi()
    query = m.FormulaProgramQueryV5(
        slot_ids=("x", "plain", "owned", "tail"),
        candidates=(producer, _effect()) if write else (producer,),
        terminal_slots={"plain-head": "plain", "memory-head": "owned"},
        max_steps=2 if write else 1, hidden_dim=8,
    )
    return query, producer


def test_named_outputs_publish_together_and_keep_per_head_lineage():
    query, producer = _query()
    x = torch.tensor([[1.0, -2.0, 0.5]], requires_grad=True)
    entry = query._arena({"x": x})
    result = producer(entry)
    assert result.tensor_steps == 1 and result.effect_steps == 0
    assert entry.values.get("plain") is entry.values.get("owned") is None
    assert result.producer("plain").plastic_slot is None
    assert result.producer("owned").plastic_slot == producer.bank_slot_ref
    execution = query({"x": x})
    assert tuple(execution.outputs) == ("plain-head", "memory-head")
    assert execution.output_producers["plain-head"].plastic_slot is None
    assert execution.output_producers["memory-head"].plastic_slot == producer.bank_slot_ref
    assert execution.trace.steps[0].output_slots == ("plain", "owned")
    assert execution.trace.steps[0].bank_owner_ids == (None, "memory")
    sum(value.sum() for value in execution.outputs.values()).backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 4.0))
    torch.testing.assert_close(producer.candidate.operand_store.tensor("gain").grad, 2.0 * x.detach())
    assert not producer.bank_owner.value.requires_grad


def test_stop_requires_every_named_head_not_just_the_first():
    x = m.InputBinding("x", _type())
    program = m.FormulaProgram.build(outputs=(m.add(x, x),))
    producers = tuple(m.FormulaProgramTensorCandidateV3(m.FormulaProgramCandidate(
        f"head-{slot}", program, input_slots={"x": "x"}, output_slot=slot,
    )) for slot in ("a", "b"))
    query = m.FormulaProgramQueryV5(
        slot_ids=("x", "a", "b"), candidates=producers,
        terminal_slots={"head-a": "a", "head-b": "b"}, max_steps=2, hidden_dim=8,
    )
    with torch.no_grad():
        query.network[-1].weight.zero_()
        query.network[-1].bias.fill_(0)
        query.network[-1].bias[-1] = 100.0
    initial = query._arena({"x": torch.ones(1, 3)})
    first = producers[0](initial)
    assert not query._stop_eligible(first, steps=1)
    assert query._stop_eligible(producers[1](first), steps=2)
    result = query({"x": torch.ones(1, 3)})
    assert len(result.trace.steps) == 3 and result.trace.steps[-1].candidate_id == "stop"
    assert set(result.outputs) == {"head-a", "head-b"}


def test_plain_head_loss_does_not_train_sibling_bank_factors():
    query, producer = _query()
    x = torch.tensor([[1.0, -2.0, 0.5]], requires_grad=True)
    result = query({"x": x})
    result.outputs["plain-head"].sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 2.0))
    assert producer.candidate.operand_store.tensor("gain").grad is None


def test_named_heads_allow_effect_tail_and_future_loss_before_joint_commit():
    query, producer = _query(write=True)
    with torch.no_grad():
        query.network[-1].weight.zero_()
        query.network[-1].bias[:] = torch.tensor([0.0, 1.0, -1.0])
    x = torch.tensor([[1.0, -2.0, 0.5]])
    initial = query.initial_bank_state()
    result = query({"x": x}, bank_state=initial)
    assert [step.candidate_id for step in result.trace.steps] == ["producer", "write", "stop"]
    slot = producer.bank_slot_ref
    torch.testing.assert_close(result.bank_state.value(slot), 2.0 + 0.4 * x)
    torch.testing.assert_close(result.outputs["memory-head"], 2.0 * x)
    assert result.output_producers["memory-head"].plastic_revision == 0
    assert result.bank_state.revision(slot) == 1
    assert query.initial_bank_state().revision(slot) == 0
    rate = query.candidates[1].operand_store.tensor("rate")
    current_vjp = torch.autograd.grad(
        result.outputs["memory-head"].sum(), rate, allow_unused=True, retain_graph=True,
    )[0]
    assert current_vjp is None
    replay = query.reexecute("producer", {"x": x}, bank_state=result.bank_state)
    replay["owned"].sum().backward()
    torch.testing.assert_close(rate.grad, 4.0 * x.square())
    query.commit_(result)
    assert query.initial_bank_state().revision(slot) == 1
    assert not query.initial_bank_state().value(slot).requires_grad


def test_data_only_head_cannot_guess_a_sibling_heads_bank_target():
    query, producer = _query()
    arena = producer(query._arena({"x": torch.ones(1, 3)}))
    assert _effect("owned").accepts(arena)
    assert not _effect("plain").accepts(arena)


def test_old_contracts_still_require_a_single_public_output():
    producer = _multi()
    with pytest.raises(ValueError, match="exactly one public output"):
        m.FormulaProgramCandidateV2(
            "old", producer.candidate.program, input_slots={"x": "x"}, output_slot="owned",
        )
    with pytest.raises(ValueError, match="named-output"):
        m.FormulaProgramTensorCandidateV3(producer.candidate)
    with pytest.raises(ValueError, match="ProgramQuery@5"):
        m.FormulaProgramQueryV4(
            slot_ids=("x", "plain", "owned"), candidates=(producer,), terminal_slot="owned",
        )


@pytest.mark.parametrize("outputs", ({"missing": "plain"}, {"a": "same", "b": "same"}))
def test_candidate_rejects_missing_or_colliding_output_wiring(outputs):
    producer = _multi()
    program = producer.candidate.program
    if "a" in outputs:
        outputs = dict(zip(program.outputs, outputs.values(), strict=True))
    with pytest.raises(ValueError, match="every Formula output|distinct SSA"):
        m.FormulaProgramCandidateV3("bad", program, input_slots={"x": "x"}, output_slots=outputs)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_multi_head_candidate_batch_uses_existing_checked_plan(device, monkeypatch):
    from arti import _formula_candidate_batch as batch

    producer = _multi()
    query = m.FormulaProgramQueryV5(
        slot_ids=("x", "plain", "owned"), candidates=(producer,),
        terminal_slots={"a": "plain", "b": "owned"}, max_steps=1,
    ).to(device)
    values = (torch.randn(1, 3, device=device), torch.randn(1, 3, device=device))
    requests = tuple((producer, query._arena({"x": value})) for value in values)
    calls = []
    original = batch._run_group

    def record(*args, **kwargs):
        calls.append(len(args[1]))
        return original(*args, **kwargs)

    monkeypatch.setattr(batch, "_run_group", record)
    with torch.no_grad():
        results = query.execute_many(requests)
        serial = query.execute_many(requests, serial=True)
    assert calls == [2]
    for actual, expected in zip(results, serial, strict=True):
        assert actual.tensor_steps == expected.tensor_steps == 1
        for slot in producer.output_slot_ids:
            torch.testing.assert_close(actual.values.get(slot), expected.values.get(slot))
            assert actual.producer(slot).plastic_slot == expected.producer(slot).plastic_slot


def test_batched_gradients_do_not_reach_unused_candidates():
    producers = (_multi("first", owner_id="first"), _multi("unused", owner_id="unused"))
    query = m.FormulaProgramQueryV5(
        slot_ids=("x", "plain", "owned"), candidates=producers,
        terminal_slots={"a": "plain", "b": "owned"}, max_steps=1,
    )
    xs = tuple(torch.randn(1, 3, requires_grad=True) for _ in producers)
    requests = tuple((producer, query._arena({"x": x})) for producer, x in zip(producers, xs, strict=True))
    executed = query.execute_many(requests)
    executed[0].values.get("owned").sum().backward()
    assert xs[0].grad is not None and xs[1].grad is None
    assert producers[0].candidate.operand_store.tensor("gain").grad is not None
    assert producers[1].candidate.operand_store.tensor("gain").grad is None


def test_named_outputs_strict_save_reload_and_clone(tmp_path):
    query, producer = _query(write=True)
    with torch.no_grad():
        query.network[-1].weight.zero_()
        query.network[-1].bias[:] = torch.tensor([0.0, 1.0, -1.0])
    x = torch.tensor([[1.0, -2.0, 0.5]])
    query.commit_(query({"x": x}))
    assert arti.component_ref(query) == "arti/formula-program-query@5"
    assert arti.component_ref(producer) == "arti/formula-program-tensor-candidate@4"
    assert arti.component_ref(producer.candidate) == "arti/formula-program-candidate@3"
    saved = arti.save(query, tmp_path / "named.arti.st")
    restored, _ = _query(write=True)
    arti.load(saved.weights_path, model=restored, strict=True, verify_architecture=True)
    assert arti.component_provenance(query) == arti.component_provenance(restored)
    cloned = copy.deepcopy(query).double()
    expected = query({"x": x})
    for actual in (restored({"x": x}), cloned({"x": x.double()})):
        assert tuple(actual.outputs) == tuple(expected.outputs)
        for name, value in actual.outputs.items():
            torch.testing.assert_close(value.float(), expected.outputs[name])
        assert actual.bank_state.revisions == expected.bank_state.revisions
    query.requires_grad_(False)
    with torch.no_grad():
        frozen = query({"x": x})
    assert all(not value.requires_grad for value in frozen.outputs.values())
    mismatched, _ = _query(write=True)
    mismatched.terminal_slots = {"plain-head": "owned", "memory-head": "plain"}
    with pytest.raises(ValueError):
        arti.load(saved.weights_path, model=mismatched, strict=True, verify_architecture=True)


def test_multi_output_occurrences_share_one_real_bank_owner():
    producers = (_multi("first"), _multi("second"))
    query = m.FormulaProgramQueryV5(
        slot_ids=("x", "plain", "owned"), candidates=producers,
        terminal_slots={"plain": "plain", "owned": "owned"}, max_steps=1,
    )
    assert len(query.owner_states) == 1
    assert producers[0].bank_owner is producers[1].bank_owner
    owner = query.owner_states[0]
    changed = query.initial_bank_state().replace(owner.slot_ref, torch.full((1, 3), 5.0), revision=1)
    owner.install_(changed)
    x = torch.ones(1, 3)
    for producer in producers:
        result = query.reexecute(producer.candidate_id, {"x": x}, bank_state=query.initial_bank_state())
        torch.testing.assert_close(result["plain"], x * 2)
        torch.testing.assert_close(result["owned"], x * 5)
    assert arti.alpha.FormulaProgramQueryV5 is m.FormulaProgramQueryV5


def test_different_head_shapes_survive_native_and_batched_execution():
    x = m.InputBinding("x", _type())
    program = m.FormulaProgram.build(outputs=(m.add(x, x), m.concat(x, x, axis="D")))
    producer = m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
        "heads", program, input_slots={"x": "x"},
        output_slots=dict(zip(program.outputs, ("narrow", "wide"), strict=True)),
    ))
    query = m.FormulaProgramQueryV5(
        slot_ids=("x", "narrow", "wide"), candidates=(producer,),
        terminal_slots={"a": "narrow", "b": "wide"}, max_steps=1,
    )
    value = torch.randn(2, 3)
    requests = tuple((producer, query._arena({"x": row[None]})) for row in value)
    with torch.no_grad():
        batched = query.execute_many(requests)
        native = query.execute_many(requests, serial=True)
    for actual, expected in zip(batched, native, strict=True):
        assert actual.values.get("narrow").shape == (1, 3)
        assert actual.values.get("wide").shape == (1, 6)
        for slot in producer.output_slot_ids:
            torch.testing.assert_close(actual.values.get(slot), expected.values.get(slot))
    outputs = query({"x": value[:1]}).outputs
    assert outputs["a"].shape == (1, 3) and outputs["b"].shape == (1, 6)


def test_standalone_candidate_preserves_all_named_outputs():
    producer = _multi()
    candidate = producer.candidate
    initial = m.FormulaProgramArena.from_mapping(("x", "plain", "owned"), {"x": torch.ones(1, 3)})
    assert candidate.accepts(initial)
    result = candidate(initial)
    assert result.get("plain") is not None and result.get("owned") is not None
    assert not candidate.accepts(result)
    with pytest.raises(ValueError, match="empty"):
        candidate(result)
