from __future__ import annotations

import copy

import pytest
import torch

import arti
from arti import mechanisms as m


def _type():
    return m.TensorType(("B", "D"), ("B", 3), dtype="floating", domain="activation")


def _producer(owner="memory", *, name="producer", source="x", outputs=("raw", "owned")):
    x = m.InputBinding("x", _type())
    weight = m.BankBinding("weight", "arti/call-test@1", "weight", _type())
    gain = m.BankBinding("gain", "arti/call-test@1", "gain", _type())
    program = m.FormulaProgram.build(outputs=(m.add(x, x), m.scale(m.scale(x, weight), gain)))
    return m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
        name, program, input_slots={"x": source},
        output_slots=dict(zip(program.outputs, outputs, strict=True)),
        operands={"weight": torch.full((1, 3), 2.0), "gain": torch.ones(1, 3)},
        trainable_operands=("gain",),
    ), plastic_bank_slot="weight", bank_owner_id=owner)


def _effect(*, name="write", source="owned", output="tail"):
    x = m.InputBinding("x", _type())
    rate = m.BankBinding("rate", "arti/call-test@1", "rate", _type())
    zero = m.BankBinding("zero", "arti/call-test@1", "zero", _type())
    program = m.FormulaProgram.build(outputs=(m.neural_plasticity(x, m.scale(x, rate), m.scale(x, zero)),))
    return m.FormulaProgramEffectCandidateV3(
        name, m.FormulaEffectProgramV2(program, data_input_name="x", state_type=_type()),
        input_slot=source, output_slot=output,
        operands={"rate": torch.full((1, 3), 0.1), "zero": torch.zeros(1, 3)},
        trainable_operands=("rate",), execution_count=torch.tensor(2.0),
        trainable_execution_count=True, max_executions=4,
    )


def _prefer_order(query):
    with torch.no_grad():
        query.network[-1].weight.zero_()
        query.network[-1].bias.copy_(torch.arange(len(query.action_ids), 0, -1))
    return query


def _child(owner="memory"):
    return _prefer_order(m.FormulaProgramQueryV5(
        slot_ids=("x", "raw", "owned", "tail"), candidates=(_producer(owner), _effect()),
        terminal_slots={"data": "raw", "memory": "tail"}, max_steps=2, hidden_dim=8,
    ))


def _call(child, name="call", *, source="x", prefix="out"):
    return m.FormulaProgramCallCandidateV1(
        name, child, input_slots={"x": source},
        output_slots={"data": f"{prefix}-data", "memory": f"{prefix}-memory"},
    )


def _parent(child=None):
    child = _child() if child is None else child
    return _prefer_order(m.FormulaProgramQueryV5(
        slot_ids=("x", "out-data", "out-memory"), candidates=(_call(child),),
        terminal_slots={"data": "out-data", "memory": "out-memory"}, max_steps=1, hidden_dim=8,
    ))


def test_child_call_returns_all_heads_real_lineage_and_uninstalled_state():
    parent = _parent()
    child = parent.candidates[0].child
    x = torch.tensor([[1.0, -2.0, 0.5]], requires_grad=True)
    initial = parent.initial_bank_state()
    result = parent({"x": x})
    for name in ("data", "memory"):
        torch.testing.assert_close(result.outputs[name], 2.0 * x)
    slot = child.plastic_candidates[0].bank_slot_ref
    torch.testing.assert_close(result.bank_state.value(slot), 2.0 + 0.4 * x)
    assert result.output_producers["memory"].execution_id == "call/producer"
    assert result.output_producers["memory"].plastic_slot == slot
    assert result.output_producers["memory"].plastic_revision == 1
    assert result.output_producers["data"].plastic_slot is None
    assert result.proposals[0].effect_candidate_id == "call/write"
    assert result.trace.total_dispatches == 3
    assert result.trace.steps[0].child_trace.invocation_path == ("call",)
    assert [step.candidate_id for step in result.trace.steps[0].child_trace.steps] == ["producer", "write", "stop"]
    assert parent.owner_states[0] is child.owner_states[0]
    assert parent.initial_bank_state().revision(slot) == child.initial_bank_state().revision(slot) == 0
    torch.testing.assert_close(parent.initial_bank_state().value(slot), initial.value(slot))
    assert all(id(parent.owner_states[0].value) != id(parameter) for parameter in parent.parameters())


def test_nested_call_matches_expanded_graph_outputs_successors_and_vjp():
    nested = _parent()
    expanded = _prefer_order(m.FormulaProgramQueryV5(
        slot_ids=("x", "raw", "owned", "tail"), candidates=(_producer(), _effect()),
        terminal_slots={"data": "raw", "memory": "tail"}, max_steps=2, hidden_dim=8,
    ))
    values = []
    gradients = []
    for query in (nested, expanded):
        leaf = query.candidates[0].child if query is nested else query
        with torch.no_grad():
            leaf.candidates[0].candidate.operand_store.tensor("gain").copy_(torch.tensor([[1.1, 0.8, 1.5]]))
        x = torch.tensor([[1.0, -2.0, 0.5]], requires_grad=True)
        result = query({"x": x})
        loss = sum((i + 1.25) * t.square().sum() for i, t in enumerate(result.outputs.values()))
        loss = loss + result.bank_state.values[0].square().sum()
        loss.backward()
        gradients.append((x.grad, leaf.candidates[0].candidate.operand_store.tensor("gain").grad,
                          leaf.candidates[1].operand_store.tensor("rate").grad,
                          leaf.candidates[1].execution_count.grad))
        values.append(result)
    for name in values[0].outputs:
        torch.testing.assert_close(values[0].outputs[name], values[1].outputs[name])
    torch.testing.assert_close(values[0].bank_state.values, values[1].bank_state.values)
    assert values[0].bank_state.revisions == values[1].bank_state.revisions
    for left, right in zip(*gradients, strict=True):
        assert left is not None and right is not None
        torch.testing.assert_close(left, right)


def test_repeated_calls_share_current_bank_and_ordered_gradients():
    child = _child()
    first, second = _call(child, "first", prefix="a"), _call(child, "second", prefix="b")
    parent = _prefer_order(m.FormulaProgramQueryV5(
        slot_ids=("x", *first.output_slot_ids, *second.output_slot_ids), candidates=(first, second),
        terminal_slots={"a": "a-memory", "b": "b-memory"}, max_steps=2, hidden_dim=8,
    ))
    x = torch.tensor([[1.0, -2.0, 0.5]])
    result = parent({"x": x})
    torch.testing.assert_close(result.outputs["a"], 2 * x)
    torch.testing.assert_close(result.outputs["b"], (2 + 0.4 * x) * x)
    torch.testing.assert_close(result.bank_state.values[0], (2 + 0.4 * x) * (1 + 0.2 * x))
    assert result.bank_state.revisions == (2,)
    assert len(parent.owner_states) == 1
    assert result.proposals[1].previous is result.proposals[0].successor
    assert [p.predecessor_execution_id for p in result.proposals] == ["first/producer", "second/producer"]
    assert result.trace.total_dispatches == 6
    result.outputs["b"].sum().backward()
    torch.testing.assert_close(child.candidates[1].operand_store.tensor("rate").grad, 4 * x.square())
    assert child.initial_bank_state().revisions == (0,)
    parent.commit_(result)
    assert child.initial_bank_state().revisions == (2,)
    assert not parent.owner_states[0].value.requires_grad


def test_child_sees_parent_proposal_and_parent_tail_keeps_child_producer():
    child = _child()
    producer = _producer(name="outer", outputs=("raw", "owned"))
    before = _effect(name="before", output="before-out")
    call = _call(child, source="before-out")
    after = _effect(name="after", source="out-memory", output="after-out")
    parent = _prefer_order(m.FormulaProgramQueryV5(
        slot_ids=("x", "raw", "owned", "before-out", "out-data", "out-memory", "after-out"),
        candidates=(producer, before, call, after),
        terminal_slots={"output": "after-out"}, max_steps=4, hidden_dim=8,
    ))
    x = torch.tensor([[1.0, -2.0, 0.5]])
    result = parent({"x": x})
    assert len(parent.owner_states) == 1 and parent.owner_states[0] is child.owner_states[0]
    assert [p.previous_revision for p in result.proposals] == [0, 1, 2]
    assert [p.predecessor_execution_id for p in result.proposals] == ["outer", "call/producer", "call/producer"]
    assert all(b.previous is a.successor for a, b in zip(result.proposals, result.proposals[1:]))
    torch.testing.assert_close(result.outputs["output"], (2 + 0.4 * x) * (2 * x))


def test_effect_only_child_can_modify_incoming_real_bank_not_an_artificial_call_bank():
    child = m.FormulaProgramQueryV5(
        slot_ids=("owned", "tail"), candidates=(_effect(),),
        terminal_slots={"edited": "tail"}, max_steps=1, hidden_dim=8,
    )
    call = m.FormulaProgramCallCandidateV1(
        "edit", child, input_slots={"owned": "owned"}, output_slots={"edited": "result"},
    )
    parent = m.FormulaProgramQueryV5(
        slot_ids=("x", "raw", "owned", "result"), candidates=(_producer(), call),
        terminal_slots={"answer": "result"}, max_steps=2, hidden_dim=8,
    )
    result = parent({"x": torch.ones(1, 3)})
    assert len(child.owner_states) == 0 and len(parent.owner_states) == 1
    assert result.proposals[0].predecessor_execution_id == "producer"
    assert result.proposals[0].effect_candidate_id == "edit/write"
    torch.testing.assert_close(result.bank_state.values[0], torch.full((1, 3), 2.4))


def test_returning_an_older_head_does_not_refresh_its_write_target():
    child = _child()
    child.terminal_slots = {"data": "raw", "memory": "owned"}
    child.min_steps = 2
    parent = m.FormulaProgramQueryV5(
        slot_ids=("x", "out-data", "out-memory", "followup"), candidates=(_call(child),),
        terminal_slots={"data": "out-data", "memory": "out-memory"}, max_steps=1,
    )
    call = parent.candidates[0]
    returned = call(parent._arena({"x": torch.ones(1, 3)}))
    assert returned.producer("out-memory").plastic_revision == 0
    assert returned.committed_state().revisions == (1,)
    after = _effect(source="out-memory", output="followup")
    assert returned.values.get("followup") is None
    with pytest.raises(ValueError, match="stale"):
        after._target(returned)
    assert not after.accepts(returned)
    assert returned.producer("out-memory").plastic_value is not returned.committed_state().values[0]


def test_nested_calls_preserve_full_path_and_child_local_query():
    parent = _parent(_parent())
    child = parent.candidates[0].child.candidates[0].child
    with torch.no_grad():
        child.network[-1].bias.copy_(torch.tensor([2.0, 1.0, 10.0]))
    result = parent({"x": torch.ones(1, 3)})
    assert result.proposals[0].predecessor_execution_id == "call/call/producer"
    assert result.proposals[0].effect_candidate_id == "call/call/write"
    assert result.trace.steps[0].child_trace.steps[0].child_trace.invocation_path == ("call", "call")
    assert result.trace.total_dispatches == 4


def test_child_module_call_hooks_are_not_bypassed():
    parent = _parent()
    child = parent.candidates[0].child
    calls = []

    def pre_hook(module, args):
        calls.append("entry")
        return ({"x": args[0]["x"] * 2},)

    def post_hook(module, args, result):
        calls.append("return")

    before = child.register_forward_pre_hook(pre_hook)
    after = child.register_forward_hook(post_hook)
    try:
        result = parent({"x": torch.ones(1, 3)})
    finally:
        before.remove()
        after.remove()
    assert calls == ["entry", "return"]
    torch.testing.assert_close(result.outputs["memory"], torch.full((1, 3), 4.0))
    torch.testing.assert_close(result.bank_state.values[0], torch.full((1, 3), 2.8))


def test_noncommuting_parent_child_effects_compose_in_order_with_exact_vjp():
    def constant_effect(name, source, output, additive, multiplicative, *, trainable=False):
        x = m.InputBinding("x", _type())
        a = m.BankBinding("a", "arti/call-test@1", "a", _type())
        g = m.BankBinding("g", "arti/call-test@1", "g", _type())
        zero = m.BankBinding("zero", "arti/call-test@1", "zero", _type())
        # Constant arithmetic fixture within the data-input effect contract.
        drive = m.add(a, m.scale(x, zero))
        program = m.FormulaProgram.build(outputs=(m.neural_plasticity(x, drive, g),))
        return m.FormulaProgramEffectCandidateV3(
            name, m.FormulaEffectProgramV2(program, data_input_name="x", state_type=_type()),
            input_slot=source, output_slot=output,
            operands={"a": torch.full((1, 3), additive), "g": torch.full((1, 3), multiplicative),
                      "zero": torch.zeros(1, 3)},
            trainable_operands=("g",) if trainable else (), execution_count=torch.tensor(2.0),
            max_executions=4,
        )

    multiply = constant_effect("multiply", "owned", "tail", 0.0, 1.0, trainable=True)
    child = _prefer_order(m.FormulaProgramQueryV5(
        slot_ids=("x", "raw", "owned", "tail"), candidates=(_producer(), multiply),
        terminal_slots={"data": "raw", "memory": "tail"}, max_steps=2, hidden_dim=8,
    ))
    first = _call(child, "first", source="before", prefix="a")
    second = _call(child, "second", source="after", prefix="b")
    parent = _prefer_order(m.FormulaProgramQueryV5(
        slot_ids=("x", "raw", "owned", "before", "a-data", "a-memory", "after", "b-data", "b-memory"),
        candidates=(
            _producer(name="outer"), constant_effect("before", "owned", "before", 0.5, 0.0),
            first, constant_effect("after", "a-memory", "after", 1.5, 0.0), second,
        ), terminal_slots={"result": "b-memory"}, max_steps=5, hidden_dim=8,
    ))
    result = parent({"x": torch.ones(1, 3)})
    torch.testing.assert_close(result.bank_state.values[0], torch.full((1, 3), 60.0))
    assert result.bank_state.revisions == (4,)
    result.bank_state.values[0].sum().backward()
    torch.testing.assert_close(multiply.operand_store.tensor("g").grad, torch.full((1, 3), 108.0))


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_independent_child_candidates_do_not_commit_losers_or_mix_gradients(device):
    first, second = _call(_child("a"), "first"), _call(_child("b"), "second")
    parent = _prefer_order(m.FormulaProgramQueryV5(
        slot_ids=("x", "out-data", "out-memory"), candidates=(first, second),
        terminal_slots={"answer": "out-memory"}, max_steps=1, hidden_dim=8,
    )).to(device)
    x = torch.ones(1, 3, device=device)
    root = parent._arena({"x": x})
    branches = parent.execute_many(((first, root), (second, root)))
    assert root.proposals == ()
    assert all(owner.revision.item() == 0 for owner in parent.owner_states)
    branches[0].committed_state().value(first.child.owner_states[0].slot_ref).sum().backward()
    assert first.child.candidates[1].operand_store.tensor("rate").grad is not None
    assert second.child.candidates[1].operand_store.tensor("rate").grad is None
    parent.requires_grad_(False)
    with torch.no_grad():
        winner = parent({"x": x})
    parent.commit_(winner)
    assert first.child.owner_states[0].revision.item() == 1
    assert second.child.owner_states[0].revision.item() == 0


def test_nested_save_restore_clone_dtype_and_shared_child_identity(tmp_path):
    child = _child()
    calls = (_call(child, "first", prefix="a"), _call(child, "second", prefix="b"))
    parent = _prefer_order(m.FormulaProgramQueryV5(
        slot_ids=("x", "a-data", "a-memory", "b-data", "b-memory"), candidates=calls,
        terminal_slots={"a": "a-memory", "b": "b-memory"}, max_steps=2, hidden_dim=8,
    ))
    saved = arti.save(parent, tmp_path / "nested.arti.st")
    restored = copy.deepcopy(parent)
    arti.load(saved.weights_path, model=restored, strict=True, verify_architecture=True)
    assert restored.candidates[0].child is restored.candidates[1].child
    assert restored.owner_states[0] is restored.candidates[0].child.owner_states[0]
    assert arti.component_graph(parent) == arti.component_graph(restored)
    moved = copy.deepcopy(restored).double()
    assert moved.owner_states[0] is moved.candidates[0].child.owner_states[0]
    assert moved.candidates[0].child.plastic_candidates[0].candidate.operand_store.tensor("weight").dtype == torch.float64
    for model in (restored, moved):
        model.requires_grad_(False)
        with torch.no_grad():
            result = model({"x": torch.ones(1, 3, dtype=next(model.parameters()).dtype)})
        model.commit_(result)
        assert model.candidates[0].child.owner_states[0].revision.item() == 2


def test_call_mapping_and_old_query_boundaries():
    child = _child()
    with pytest.raises(ValueError, match="entry slots"):
        m.FormulaProgramCallCandidateV1("call", child, input_slots={"owned": "x"}, output_slots={"data": "a", "memory": "b"})
    with pytest.raises(ValueError, match="every child terminal"):
        m.FormulaProgramCallCandidateV1("call", child, input_slots={"x": "x"}, output_slots={"memory": "b"})
    with pytest.raises(TypeError, match="Query@4 candidates"):
        m.FormulaProgramQueryV4(slot_ids=("x", "out-data", "out-memory"), candidates=(_call(child),), terminal_slot="out-memory")
    assert arti.component_ref(_call(child)) == "arti/formula-program-call-candidate@1"


def test_parent_reexecute_looks_up_only_local_occurrences():
    child = _child()
    producer = _producer(name="producer")
    call = _call(child)
    parent = m.FormulaProgramQueryV5(
        slot_ids=("x", "raw", "owned", "out-data", "out-memory"), candidates=(producer, call),
        terminal_slots={"answer": "out-memory"}, max_steps=2,
    )
    outputs = parent.reexecute("producer", {"x": torch.ones(1, 3)}, bank_state=parent.initial_bank_state())
    assert set(outputs) == {"raw", "owned"}
    calls = (_call(child, "first", prefix="a"), _call(child, "second", prefix="b"))
    only_calls = m.FormulaProgramQueryV5(
        slot_ids=("x", "a-data", "a-memory", "b-data", "b-memory"), candidates=calls,
        terminal_slots={"a": "a-memory", "b": "b-memory"}, max_steps=2,
    )
    with pytest.raises(ValueError, match="one plastic ordinary occurrence"):
        only_calls.reexecute("producer", {"x": torch.ones(1, 3)}, bank_state=only_calls.initial_bank_state())


def test_new_parent_preserves_bank_shared_with_an_existing_parent():
    child = _child()
    first_parent = _parent(child)
    owner = first_parent.owner_states[0]
    new_producer = _producer(name="outer")
    second_parent = m.FormulaProgramQueryV5(
        slot_ids=("x", "raw", "owned", "out-data", "out-memory"),
        candidates=(new_producer, _call(child)), terminal_slots={"answer": "out-memory"}, max_steps=2,
    )
    assert first_parent.owner_states[0] is second_parent.owner_states[0] is child.owner_states[0] is owner
    assert new_producer.bank_owner is owner
    first_parent.commit_(first_parent({"x": torch.ones(1, 3)}))
    assert second_parent.initial_bank_state().revisions == child.initial_bank_state().revisions == (1,)
    torch.testing.assert_close(second_parent.initial_bank_state().values[0], torch.full((1, 3), 2.4))


def test_distinct_already_mounted_owners_are_not_silently_rebound():
    first, second = _child(), _child()
    owners = (first.owner_states[0], second.owner_states[0])
    with pytest.raises(ValueError, match="distinct mounted Bank owners"):
        m.FormulaProgramQueryV5(
            slot_ids=("x", "a-data", "a-memory", "b-data", "b-memory"),
            candidates=(_call(first, "a", prefix="a"), _call(second, "b", prefix="b")),
            terminal_slots={"a": "a-memory", "b": "b-memory"}, max_steps=2,
        )
    assert first.owner_states[0] is owners[0] and second.owner_states[0] is owners[1]
