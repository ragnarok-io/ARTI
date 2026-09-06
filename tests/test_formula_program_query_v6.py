from __future__ import annotations

import copy

import pytest
import torch

import arti
from arti import mechanisms as m
from arti._formula_candidate_admission import candidate_mask
from arti._formula_admission_table import indexed_candidate_admission


def _type():
    return m.TensorType(("B", "D"), ("B", 1), dtype="floating", domain="activation")


def member(name, source, target, *, owner=None, weight=1.0):
    x = m.InputBinding("x", _type())
    w = m.BankBinding("weight", "arti/response-test@1", "weight", _type())
    minus = m.BankBinding("minus", "arti/response-test@1", "minus", _type())
    value = m.scale(x, w)
    program = m.FormulaProgram.build(outputs=(value, m.scale(value, minus)))
    candidate = m.FormulaProgramCandidateV3(
        name, program, input_slots={"x": source},
        output_slots=dict(zip(program.outputs, (target, target + "_negative"), strict=True)),
        operands={"weight": torch.full((1, 1), weight), "minus": -torch.ones(1, 1)},
        trainable_operands=() if owner else ("weight",),
        batch_broadcast_operands=("weight", "minus"),
    )
    return m.FormulaProgramTensorCandidateV4(
        candidate, plastic_bank_slot=None if owner is None else "weight", bank_owner_id=owner,
    )


def effect():
    x = m.InputBinding("x", _type())
    rate = m.BankBinding("rate", "arti/response-test@1", "rate", _type())
    zero = m.BankBinding("zero", "arti/response-test@1", "zero", _type())
    program = m.FormulaProgram.build(outputs=(m.neural_plasticity(
        x, m.scale(x, rate), m.scale(x, zero),
    ),))
    return m.FormulaProgramEffectCandidateV3(
        "write", m.FormulaEffectProgramV2(program, data_input_name="x", state_type=_type()),
        input_slot="h", output_slot="tail",
        operands={"rate": torch.full((1, 1), -1.0), "zero": torch.zeros(1, 1)},
        trainable_operands=("rate",), execution_count=torch.tensor(2.0), max_executions=4,
    )


def federation(*, write=False, shared=False):
    a = member("a", "x", "h", owner="a")
    b = member("b", "tail" if write else "h", "u", owner="a" if shared else "b")
    candidates = (a, effect(), b) if write else (a, b)
    candidates += (member("left", "u", "y", weight=2.0), member("right", "u", "y", weight=-2.0))
    slots = tuple(dict.fromkeys(("x", *(slot for c in candidates for slot in c.output_slot_ids))))
    edges = {"a": {"b": "h"}, "b": {"left": "u", "right": "u_negative"}}
    if write:
        edges["a"]["write"] = "h"
    query = m.FormulaProgramQueryV6(
        slot_ids=slots, candidates=candidates, terminal_slots={"result": "y"},
        entry_candidates=("a",), continuations=edges, max_steps=4 if write else 3,
    )
    return query


def test_query_is_execution_not_a_separate_network_or_bank():
    query = federation()
    assert not hasattr(query, "network") and query.tensor_encoder is None
    assert not any("network" in name for name, _ in query.named_parameters())
    assert len(query.owner_states) == 2
    for x, choice in ((torch.ones(1, 1), "left"), (-torch.ones(1, 1), "right")):
        result = query({"x": x})
        assert [s.candidate_id for s in result.trace.steps] == ["a", "b", choice, "stop"]
        torch.testing.assert_close(result.outputs["result"], torch.full((1, 1), 2.0))


def test_write_changes_later_response_within_the_same_invocation():
    query = federation(write=True, shared=True)
    assert len(query.owner_states) == 1
    entry = query._arena({"x": torch.ones(1, 1)})
    a = query.candidates[0](entry)
    written = query.candidates[1](a)
    torch.testing.assert_close(written.values.get("tail"), a.values.get("h"), rtol=0, atol=0)
    # The later occurrence reads the changed owner, not the old SSA response.
    b = query.candidates[2](written)
    logits = query.query_logits(b)
    assert logits[0, query.action_ids.index("right")] > logits[0, query.action_ids.index("left")]
    gradient = torch.autograd.grad(logits[0, query.action_ids.index("left")],
                                   query.candidates[1].operand_store.tensor("rate"))[0]
    torch.testing.assert_close(gradient, torch.full_like(gradient, 2.0))
    result = query({"x": torch.ones(1, 1)})
    assert [s.candidate_id for s in result.trace.steps] == ["a", "write", "b", "right", "stop"]
    assert len(result.proposals) == 1


def test_frontier_uses_real_producer_and_indexed_admission_matches():
    query = federation()
    entry = query._arena({"x": torch.ones(1, 1)})
    assert query.eligible(entry, steps=0).tolist() == [True, False, False, False, False]
    a = query.candidates[0](entry)
    assert query.eligible(a, steps=1).tolist() == [False, True, False, False, False]
    b = query.candidates[1](a)
    expected = [False, False, True, True, False]
    assert query.eligible(b, steps=2).tolist() == expected
    with indexed_candidate_admission():
        assert candidate_mask(query, (b,), tuple(query.candidates), steps=2, include_stop=True)[0].tolist() == expected
    supplied = query._arena({"x": torch.ones(1, 1), "u": torch.ones(1, 1), "u_negative": torch.ones(1, 1)})
    assert not query.routing_mask(supplied, steps=1)[:4].any()


def test_real_predecessor_write_changes_later_query_without_changing_current_data():
    query = federation(write=True)
    values = {"x": torch.ones(1, 1)}
    state = query.initial_bank_state()
    first = query(values, bank_state=state)
    assert [s.candidate_id for s in first.trace.steps] == ["a", "write", "b", "left", "stop"]
    assert first.proposals[0].target == query.candidates[0].bank_slot_ref
    torch.testing.assert_close(first.outputs["result"], torch.tensor([[2.0]]))
    second = query(values, bank_state=first.bank_state)
    assert [s.candidate_id for s in second.trace.steps] == ["a", "write", "b", "right", "stop"]
    # Each query starts from its specified state; scoring does not perform writes.
    replay = query(values, bank_state=state)
    assert len(replay.proposals) == len(first.proposals) == len(second.proposals) == 1
    assert replay.trace == first.trace


def test_score_gradient_reaches_previous_write_and_multiple_ordinary_members():
    query = federation(write=True)
    values = {"x": torch.ones(1, 1)}
    first = query(values)
    entry = query._arena(values, bank_state=first.bank_state)
    a = query.candidates[0](entry)
    tail = query.candidates[1](a)
    b = query.candidates[2](tail)
    before = len(b.proposals)
    logits = query.query_logits(b)
    rate = query.candidates[1].operand_store.tensor("rate")
    gradient = torch.autograd.grad(logits[0, query.action_ids.index("left")], rate)[0]
    torch.testing.assert_close(gradient, torch.tensor([[2.0]]))
    assert len(b.proposals) == before


def test_child_returns_remain_real_federation_responses():
    child = federation()
    call = m.FormulaProgramCallCandidateV1(
        "child", child, input_slots={"x": "x"}, output_slots={"result": "response"},
    )
    a = member("finish", "response", "y")
    parent = m.FormulaProgramQueryV6(
        slot_ids=("x", "response", "y", "y_negative"), candidates=(call, a),
        terminal_slots={"result": "y"}, entry_candidates=("child",),
        continuations={"child": {"finish": "response"}}, max_steps=2,
    )
    result = parent({"x": torch.ones(1, 1)})
    assert result.trace.total_dispatches == 5
    assert result.trace.steps[0].child_trace.stopped
    torch.testing.assert_close(result.outputs["result"], torch.tensor([[2.0]]))


def test_versioned_save_reload_and_frozen_replay(tmp_path):
    query = federation(write=True)
    assert arti.component_ref(query) == "arti/formula-program-query@6"
    assert arti.alpha.FormulaProgramQueryV6 is m.FormulaProgramQueryV6
    saved = arti.save(query, tmp_path / "response.arti.st")
    restored = federation(write=True)
    arti.load(saved.weights_path, model=restored, strict=True, verify_architecture=True)
    for model in (restored, copy.deepcopy(query).double()):
        x = torch.ones(1, 1, dtype=model.owner_states[0].value.dtype)
        model.requires_grad_(False)
        with torch.no_grad():
            actual = model({"x": x})
            repeat = model({"x": x})
        torch.testing.assert_close(actual.outputs["result"], repeat.outputs["result"])
        assert actual.trace == repeat.trace


def test_continuation_contract_rejects_nonproducer_score():
    query = federation()
    with pytest.raises(ValueError, match="producer output"):
        m.FormulaProgramQueryV6(
            slot_ids=query.slot_ids, candidates=tuple(query.candidates), terminal_slots=query.terminal_slots,
            entry_candidates=("a",), continuations={"a": {"b": "x"}},
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_device_dtype_and_batch_responses(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    query = federation().to(device=device, dtype=dtype)
    entry = query._arena({"x": torch.tensor([[1.0], [-1.0]], device=device, dtype=dtype)})
    b = query.candidates[1](query.candidates[0](entry))
    logits = query.query_logits(b)
    assert logits.dtype == torch.float32 and logits.shape == (2, 5)
    torch.testing.assert_close(logits[:, 2], entry.values.get("x")[:, 0].float())


def response_sum(weights, *, reverse=False, dtype=torch.float32):
    producers = tuple(member(name, "x", name, weight=weight) for name, weight in zip("abc", weights))
    candidates = (*producers, member("finish", "x", "y"))
    edges = {c.candidate_id: {"finish": c.candidate_id} for c in producers}
    if reverse:
        edges = dict(reversed(tuple(edges.items())))
    query = m.FormulaProgramQueryV6(
        slot_ids=tuple(dict.fromkeys(("x", *(s for c in candidates for s in c.output_slot_ids)))),
        candidates=candidates, terminal_slots={"result": "y"}, entry_candidates=("a",),
        continuations=edges, max_steps=5,
    ).to(dtype=dtype)
    arena = query._arena({"x": torch.ones(1, 1, dtype=dtype)})
    for candidate in producers:
        arena = candidate(arena)
    return query, arena


def test_contribution_order_is_canonical_across_mapping_reconstruction():
    a, aa = response_sum((1e8, -1e8, 1.0))
    b, bb = response_sum((1e8, -1e8, 1.0), reverse=True)
    assert a.contract_config() == b.contract_config()
    assert a._response_columns == b._response_columns
    torch.testing.assert_close(a.query_logits(aa), b.query_logits(bb), rtol=0, atol=0)
    assert a.query_logits(aa)[0, 3] == 1


def test_half_precision_responses_accumulate_in_float32():
    query, arena = response_sum((40000.0, 40000.0), dtype=torch.float16)
    logits = query.query(arena, steps=2).logits
    assert logits.dtype == torch.float32 and logits[0, 2] == 80000


def test_overflowing_legal_response_is_rejected_not_selected():
    query, arena = response_sum((3e38, 3e38))
    with pytest.raises(ValueError, match="non-finite aggregated"):
        query.query(arena, steps=2)
