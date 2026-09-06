from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import arti
from arti import mechanisms as m
from test_formula_program_query_v6 import effect, federation, member


def join(name, left, right, output):
    kind = m.TensorType(("B", "D"), ("B", 1), dtype="floating", domain="activation")
    a, b = (m.InputBinding(s, kind) for s in ("a", "b"))
    program = m.FormulaProgram.build(outputs=(m.add(a, b),))
    return m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
        name, program, input_slots={"a": left, "b": right},
        output_slots={program.outputs[0]: output},
    ))


def graph(width=2):
    a = member("a", "x", "a_out", weight=2.0)
    b = member("b", "x", "b_out", weight=3.0)
    c = member("c", "a_out", "c_out", weight=4.0)
    d = join("d", "a_out", "b_out", "d_out")
    final = join("final", "c_out", "d_out", "y")
    candidates = (a, b, c, d, final)
    slots = ("x", *(s for c in candidates for s in c.output_slot_ids))
    return m.FormulaProgramQueryV7(
        slot_ids=slots, candidates=candidates, terminal_slots={"y": "y"},
        entry_candidates=("a", "b"),
        continuations={"a": {"b": "a_out", "c": "a_out", "d": "a_out"},
                       "b": {"final": "b_out"}},
        cooperation_width=width, max_steps=5,
    )


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shared_products_form_real_multi_parent_graph_and_sum_gradients(device, dtype):
    model = graph().to(device=device, dtype=dtype)
    x = torch.tensor([[2.0]], device=device, dtype=dtype, requires_grad=True)
    calls = []
    handles = [c.register_forward_hook(lambda c, args, out: calls.append(c.candidate_id))
               for c in model.candidates]
    result = model({"x": x})
    for h in handles:
        h.remove()
    assert [tuple(n.candidate_id for n in f.nodes) for f in result.frontiers] == [
        ("a", "b"), ("c", "d"), ("final",), ("stop",),
    ]
    assert sorted(calls) == ["a", "b", "c", "d", "final"]
    assert result.frontiers[1].nodes[1].parents == ("a", "b")
    assert result.frontiers[2].nodes[0].parents == ("c", "d")
    assert result.products["x"] is x
    assert result.trace.total_dispatches == 5
    torch.testing.assert_close(result.outputs["y"], 13 * x)
    dw, dx = torch.autograd.grad(
        result.outputs["y"].sum(), (model.candidates[0].candidate.operand_store.tensor("weight"), x),
    )
    torch.testing.assert_close(dw, x.new_tensor([[10.0]]))
    torch.testing.assert_close(dx, x.new_tensor([[13.0]]))


def test_replay_rebuilds_each_shared_producer_once_and_matches_gradients():
    model = graph()
    x = torch.tensor([[1.0]], requires_grad=True)
    first = model({"x": x})
    replay = model.replay({"x": x}, first.frontiers)
    assert replay.frontiers == first.frontiers
    torch.testing.assert_close(first.outputs["y"], replay.outputs["y"])
    g1 = torch.autograd.grad(first.outputs["y"].sum(), tuple(model.parameters()), allow_unused=True)
    g2 = torch.autograd.grad(replay.outputs["y"].sum(), tuple(model.parameters()), allow_unused=True)
    for a, b in zip(g1, g2, strict=True):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b)


def test_binding_alternatives_share_weights_and_selection_uses_real_responses():
    a, b = member("a", "x", "left"), member("b", "x", "right", weight=-1.0)
    ordinary = member("use_left", "left", "y", owner="shared")
    alternative = ordinary.with_bindings(
        "use_right", input_slots={"x": "right"},
        output_slots=dict(zip(ordinary.candidate.program.outputs, ("y", "y_negative"), strict=True)),
    )
    assert ordinary.bank_owner is alternative.bank_owner
    assert ordinary.candidate.fabric is alternative.candidate.fabric
    assert ordinary.candidate.operand_store is alternative.candidate.operand_store
    model = m.FormulaProgramQueryV7(
        slot_ids=("x", "left", "left_negative", "right", "right_negative", "y", "y_negative"),
        candidates=(a, b, ordinary, alternative), terminal_slots={"y": "y"},
        entry_candidates=("a", "b"),
        continuations={"a": {"use_left": "left", "use_right": "left_negative"}},
        cooperation_width=2, max_steps=3,
    )
    for value, choice, source in ((2.0, "use_left", "left"), (-2.0, "use_right", "right")):
        result = model({"x": torch.tensor([[value]])})
        assert result.frontiers[1].nodes[0].candidate_id == choice
        assert result.frontiers[1].nodes[0].inputs == (("x", source),)
        assert len(result.frontiers[1].nodes) == 1
        torch.testing.assert_close(result.outputs["y"], torch.tensor([[2.0]]))
        score_gradient = torch.autograd.grad(
            result.decision_log_score, a.candidate.operand_store.tensor("weight"),
        )[0]
        assert torch.isfinite(score_gradient).all() and score_gradient.abs().sum() > 0


def test_effects_remain_real_predecessor_updates_and_replay_does_not_multiply_them():
    old = federation(write=True, shared=True)
    model = m.FormulaProgramQueryV7(
        slot_ids=old.slot_ids, candidates=tuple(old.candidates), terminal_slots=old.terminal_slots,
        entry_candidates=old.entry_candidates, continuations=old.continuations,
        max_steps=4, cooperation_width=4,
    )
    state = model.initial_bank_state()
    result = model({"x": torch.ones(1, 1)}, bank_state=state)
    repeated = model.replay({"x": torch.ones(1, 1)}, result.frontiers, bank_state=state)
    assert len(result.proposals) == len(repeated.proposals) == 1
    assert result.proposals[0].target == old.candidates[0].bank_slot_ref
    torch.testing.assert_close(result.products["h"], result.products["tail"], rtol=0, atol=0)
    torch.testing.assert_close(result.bank_state.values[0], repeated.bank_state.values[0])
    assert result.products["h"] is result.products["tail"]


def test_stale_product_can_feed_ordinary_consumer_but_not_forge_write_lineage():
    model = federation(write=True, shared=True)
    a, write, b = model.candidates[:3]
    arena = a(model._arena({"x": torch.ones(1, 1)}))
    changed = write(arena)
    ordinary = b.with_bindings(
        "read_old", input_slots={"x": "h"}, output_slots=b.candidate.output_slots,
    )
    assert ordinary.accepts(changed)
    assert not write.accepts(changed)
    assert changed.values.get("h") is arena.values.get("h")


def test_answer_ready_does_not_disable_learning_before_return():
    producer = member("answer", "x", "y", owner="answer")
    write = effect()
    write.input_slot = "y"
    model = m.FormulaProgramQueryV7(
        slot_ids=("x", "y", "y_negative", "tail"), candidates=(producer, write),
        terminal_slots={"answer": "y"}, entry_candidates=("answer",),
        continuations={"answer": {"write": "y", "stop": "y_negative"}},
        max_steps=2, cooperation_width=4,
    )
    result = model({"x": torch.ones(1, 1)})
    assert [f.nodes[0].candidate_id for f in result.frontiers] == ["answer", "write", "stop"]
    torch.testing.assert_close(result.outputs["answer"], torch.ones(1, 1), rtol=0, atol=0)
    assert len(result.proposals) == 1
    assert not torch.equal(result.bank_state.values[0], producer.bank_owner.value)


def test_width_and_work_budgets_and_same_snapshot_readiness():
    one = graph(width=1)({"x": torch.ones(1, 1)})
    assert all(len(f.nodes) == 1 for f in one.frontiers)
    model = graph()
    model.max_tensor_steps = 4
    with pytest.raises((RuntimeError, ValueError), match="no .*candidate|no admissible"):
        model({"x": torch.ones(1, 1)})
    model = graph()
    result = model({"x": torch.ones(1, 1)})
    malformed = (replace(result.frontiers[0], nodes=(result.frontiers[0].nodes[0], result.frontiers[1].nodes[0])),)
    with pytest.raises(ValueError, match="not ready"):
        model.replay({"x": torch.ones(1, 1)}, malformed)
    with pytest.raises(ValueError, match="before STOP"):
        model.replay({"x": torch.ones(1, 1)}, result.frontiers[:-1])


def test_versioned_save_reload(tmp_path):
    model = graph()
    assert arti.component_ref(model) == "arti/formula-program-query@7"
    assert arti.alpha.FormulaProgramQueryV7 is m.FormulaProgramQueryV7
    saved = arti.save(model, tmp_path / "cooperative.arti.st")
    restored = graph()
    arti.load(saved.weights_path, model=restored, strict=True, verify_architecture=True)
    expected = model({"x": torch.ones(1, 1)})
    with torch.no_grad():
        actual = restored({"x": torch.ones(1, 1)})
    torch.testing.assert_close(actual.outputs["y"], expected.outputs["y"])
    assert actual.frontiers == expected.frontiers


def test_call_replay_preserves_child_route_even_when_new_search_would_change_it():
    child = federation()
    call = m.FormulaProgramCallCandidateV1(
        "child", child, input_slots={"x": "x"}, output_slots={"result": "response"},
    )
    final = member("finish", "response", "y")
    model = m.FormulaProgramQueryV7(
        slot_ids=("x", "response", "y", "y_negative"), candidates=(call, final),
        terminal_slots={"y": "y"}, entry_candidates=("child",),
        continuations={"child": {"finish": "response"}}, max_steps=2, cooperation_width=4,
    )
    result = model({"x": torch.ones(1, 1)})
    assert len(result.frontiers[0].nodes) == 1
    assert result.trace.total_dispatches == 5
    torch.testing.assert_close(result.outputs["y"], torch.tensor([[2.0]]))
    replayed = model.replay({"x": -torch.ones(1, 1)}, result.frontiers)
    fresh = model({"x": -torch.ones(1, 1)})
    assert replayed.trace == result.trace
    assert fresh.trace != result.trace
    torch.testing.assert_close(replayed.outputs["y"], torch.tensor([[-2.0]]))
    torch.testing.assert_close(fresh.outputs["y"], torch.tensor([[2.0]]))
    incomplete = replace(result.frontiers[0], nodes=(replace(result.frontiers[0].nodes[0], child_trace=None),))
    with pytest.raises(ValueError, match="child decision trace"):
        model.replay({"x": torch.ones(1, 1)}, (incomplete, *result.frontiers[1:]))


def cooperative(model):
    return m.FormulaProgramQueryV7(
        slot_ids=model.slot_ids, candidates=tuple(model.candidates),
        terminal_slots=model.terminal_slots, entry_candidates=model.entry_candidates,
        continuations=model.continuations, cooperation_width=4, max_steps=model.max_steps,
    )


def wrap_child(child, kind=m.FormulaProgramQueryV7):
    call = m.FormulaProgramCallCandidateV1(
        "child", child, input_slots={"x": "x"},
        output_slots={name: "out_" + name for name in child.terminal_slots},
    )
    options = {} if kind is m.FormulaProgramQueryV5 else dict(entry_candidates=("child",), continuations={})
    return kind(
        slot_ids=("x", *call.output_slot_ids), candidates=(call,),
        terminal_slots=call.output_slots, max_steps=1, **options,
    )


@pytest.mark.parametrize("middle_kind", [m.FormulaProgramQueryV5, m.FormulaProgramQueryV6, m.FormulaProgramQueryV7])
def test_nested_replay_keeps_cooperative_grandchild_frontiers_and_hooks(middle_kind):
    leaf = graph()
    model = wrap_child(wrap_child(leaf, middle_kind))
    x = torch.ones(1, 1, requires_grad=True)
    calls = []
    handles = [module.register_forward_hook(lambda mod, args, out: calls.append(id(mod)))
               for module in (leaf, model.candidates[0].child, *leaf.candidates)]
    first = model({"x": x})
    counts = list(calls)
    calls.clear()
    replay = model.replay({"x": x}, first.frontiers)
    for h in handles:
        h.remove()
    assert calls == counts and len(calls) == 7
    assert replay.trace == first.trace
    assert replay.trace.total_dispatches == 7
    leaf_trace = replay.frontiers[0].nodes[0].child_trace.steps[0].child_trace
    assert isinstance(leaf_trace, m.FormulaProgramGraphTraceV1)
    assert [len(f.nodes) for f in leaf_trace.frontiers] == [2, 2, 1, 1]
    params = (x, *leaf.parameters())
    a = torch.autograd.grad(first.outputs["y"].sum(), params, allow_unused=True)
    b = torch.autograd.grad(replay.outputs["y"].sum(), params, allow_unused=True)
    for original, repeated in zip(a, b, strict=True):
        if original is None:
            assert repeated is None
        else:
            torch.testing.assert_close(original, repeated)


def test_two_calls_share_current_bank_but_replay_each_actual_effect_once():
    child = cooperative(federation(write=True, shared=True))
    calls = tuple(m.FormulaProgramCallCandidateV1(
        name, child, input_slots={"x": "x"}, output_slots={"result": name + "_out"},
    ) for name in ("first", "second"))
    model = m.FormulaProgramQueryV7(
        slot_ids=("x", "first_out", "second_out"), candidates=calls,
        terminal_slots={"a": "first_out", "b": "second_out"}, entry_candidates=("first",),
        continuations={"first": {"second": "first_out"}}, max_steps=2,
    )
    state = model.initial_bank_state()
    x = torch.ones(1, 1, requires_grad=True)
    first = model({"x": x}, bank_state=state)
    replay = model.replay({"x": x}, first.frontiers, bank_state=state)
    assert len(first.proposals) == len(replay.proposals) == 2
    assert first.proposals[1].previous is first.proposals[0].successor
    assert replay.proposals[1].previous is replay.proposals[0].successor
    assert first.bank_state.revisions == replay.bank_state.revisions == (2,)
    assert torch.equal(model.owner_states[0].value, state.values[0])
    assert first.frontiers[0].nodes[0].child_trace.invocation_path == ("first",)
    assert first.frontiers[1].nodes[0].child_trace.invocation_path == ("second",)
    rate = child.candidates[1].operand_store.tensor("rate")
    a = torch.autograd.grad(first.outputs["b"].sum() + first.bank_state.values[0].sum(), rate)[0]
    b = torch.autograd.grad(replay.outputs["b"].sum() + replay.bank_state.values[0].sum(), rate)[0]
    torch.testing.assert_close(a, b)
    assert torch.isfinite(a).all() and a.abs().sum() > 0


def test_cooperative_child_replay_uses_old_choices_with_new_operands():
    child = cooperative(federation())
    model = wrap_child(child)
    x = torch.ones(1, 1)
    with torch.no_grad():
        old = model({"x": x})
    # Change only an ordinary operand, leaving the task Bank snapshot unchanged.
    minus = child.candidates[1].candidate.operand_store.tensor("minus")
    with torch.no_grad():
        minus.fill_(2.0)
    fresh = model({"x": x})
    replayed = model.replay({"x": x}, old.frontiers)
    assert fresh.trace != old.trace
    assert replayed.trace == old.trace
    torch.testing.assert_close(replayed.outputs["result"], torch.tensor([[2.0]]))
    torch.testing.assert_close(fresh.outputs["result"], torch.tensor([[-2.0]]))
    weight = child.candidates[2].candidate.operand_store.tensor("weight")
    gradient = torch.autograd.grad(replayed.outputs["result"].sum(), weight)[0]
    torch.testing.assert_close(gradient, torch.ones_like(gradient))


def test_nested_selection_scores_are_counted_once():
    child = cooperative(federation())
    model = wrap_child(wrap_child(child, m.FormulaProgramQueryV5))
    result = model({"x": torch.ones(1, 1)})
    grandchild = result.frontiers[0].nodes[0].child_trace.steps[0].child_trace
    expected = sum(f.selection_log_score for f in result.frontiers)
    expected = expected + sum(f.selection_log_score for f in grandchild.frontiers)
    torch.testing.assert_close(result.decision_log_score, expected)


def test_replay_rejects_mismatched_child_invocation_without_search():
    model = wrap_child(graph())
    first = model({"x": torch.ones(1, 1)})
    node = first.frontiers[0].nodes[0]
    modified = replace(node, child_trace=replace(node.child_trace, invocation_path=("different",)))
    rows = (replace(first.frontiers[0], nodes=(modified,)), *first.frontiers[1:])
    with pytest.raises(ValueError, match="invocation path"):
        model.replay({"x": torch.ones(1, 1)}, rows)


def test_serial_child_replay_preserves_named_ports_not_only_source_order():
    original = graph()
    serial = m.FormulaProgramQueryV5(
        slot_ids=original.slot_ids, candidates=tuple(original.candidates),
        terminal_slots=original.terminal_slots, max_steps=5,
    )
    model = wrap_child(serial)
    result = model({"x": torch.ones(1, 1)})
    consumer = serial.candidates[3].candidate
    consumer.input_slots = {"b": "a_out", "a": "b_out"}
    # The tuple of source slot values is unchanged, but port roles differ.
    with pytest.raises(ValueError, match="binding differs"):
        model.replay({"x": torch.ones(1, 1)}, result.frontiers)


def test_fourier_observations_share_original_tensor_and_keep_old_view_alive():
    image_type = m.TensorType(("B", "N", "D"), ("B", 6, 1), dtype="floating")
    state_type = m.TensorType(("B", "T", "S"), ("B", 1, 2), dtype="floating")
    x = m.InputBinding("x", image_type)
    displacement = m.InputBinding("displacement", state_type)
    mask = m.InputBinding("mask", m.TensorType(("B", "N"), ("B", 6), dtype="boolean"))
    active = m.InputBinding("active", m.TensorType(("B", "T"), ("B", 1), dtype="floating"))
    observed = m.observe_fourier(x, displacement, mask, active, spatial_shape=(2, 3), state_mode="cartesian")
    view = m.reshape(observed, output_axes=("B", "N", "D"), output_sizes=("B", 6, 1))
    score = m.reduce_sum(m.reduce_sum(view, axis="N"), axis="D")
    program = m.FormulaProgram.build(outputs=(view, score))
    first = m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
        "observe1", program,
        input_slots={"x": "x0", "displacement": "d1", "mask": "mask", "active": "active"},
        output_slots=dict(zip(program.outputs, ("view1", "score1"), strict=True)),
    ))
    second = first.with_bindings(
        "observe2", input_slots={"x": "x0", "displacement": "d2", "mask": "mask", "active": "active"},
        output_slots=dict(zip(program.outputs, ("view2", "score2"), strict=True)),
    )
    previous = m.InputBinding("previous", image_type)
    rate = m.BankBinding("rate", "arti/observation-shift-test@1", "rate", state_type)
    coordinates = m.reshape(m.slice_tensor(previous, axis="N", start=0, stop=2),
                            output_axes=("B", "T", "S"), output_sizes=("B", 1, 2))
    new_displacement = m.scale(coordinates, rate)
    shift_score = m.reduce_sum(m.reduce_sum(new_displacement, axis="S"), axis="T")
    shift_program = m.FormulaProgram.build(outputs=(new_displacement, shift_score))
    shift = m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
        "shift", shift_program, input_slots={"previous": "view1"},
        output_slots=dict(zip(shift_program.outputs, ("d2", "shift_score"), strict=True)),
        operands={"rate": torch.full((1, 1, 2), 0.07)}, trainable_operands=("rate",),
        batch_broadcast_operands=("rate",),
    ))
    left, right = m.InputBinding("left", image_type), m.InputBinding("right", image_type)
    combined = m.FormulaProgram.build(outputs=(m.add(left, right),))
    relation = m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
        "relation", combined, input_slots={"left": "view1", "right": "view2"},
        output_slots={combined.outputs[0]: "y"},
    ))
    model = m.FormulaProgramQueryV7(
        slot_ids=("x0", "d1", "d2", "mask", "active", "view1", "view2", "score1", "score2", "shift_score", "y"),
        candidates=(first, shift, second, relation), terminal_slots={"y": "y"},
        entry_candidates=("observe1",),
        continuations={"observe1": {"shift": "score1", "relation": "score1"},
                       "shift": {"observe2": "shift_score"}},
        cooperation_width=2, max_steps=4,
    )
    values = {"x0": torch.arange(1.0, 7.0).reshape(1, 6, 1).requires_grad_(),
              "d1": torch.tensor([[[0.13, -0.2]]], requires_grad=True),
              "mask": torch.ones(1, 6, dtype=torch.bool), "active": torch.ones(1, 1)}
    result = model(values)
    from arti.observation import fourier_observation
    options = dict(spatial_shape=(2, 3), state_mode="cartesian", direction_epsilon=1e-6,
                   compile_policy="safe_training")
    expected1 = fourier_observation(values["x0"], values["d1"], **options).squeeze(1)
    learned_rate = shift.candidate.operand_store.tensor("rate")
    expected_shift = expected1[:, :2].reshape(1, 1, 2) * learned_rate
    expected2 = fourier_observation(values["x0"], expected_shift, **options).squeeze(1)
    cascaded = fourier_observation(expected1, expected_shift, **options).squeeze(1)
    assert not torch.allclose(result.products["view2"], cascaded)
    torch.testing.assert_close(result.products["view1"], expected1)
    torch.testing.assert_close(result.products["view2"], expected2)
    torch.testing.assert_close(result.outputs["y"], expected1 + expected2)
    assert result.frontiers[3].nodes[0].parents == ("observe1", "observe2")
    variables = (values["x0"], values["d1"], learned_rate)
    weights = torch.tensor([1.0, -2.0, 0.3, 4.0, -0.7, 2.2]).reshape(1, 6, 1)
    gradients = torch.autograd.grad((result.outputs["y"] * weights).sum(), variables, retain_graph=True)
    reference = torch.autograd.grad(((expected1 + expected2) * weights).sum(), variables, retain_graph=True)
    for actual, expected in zip(gradients, reference, strict=True):
        torch.testing.assert_close(actual, expected)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    # No direct view1 readout: its displacement still receives credit through
    # view1 -> new coordinates -> Observe(x0, coordinates) -> view2.
    indirect = torch.autograd.grad((result.products["view2"] * weights).sum(), values["d1"])[0]
    expected_indirect = torch.autograd.grad((expected2 * weights).sum(), values["d1"])[0]
    torch.testing.assert_close(indirect, expected_indirect)
    assert indirect.abs().sum() > 0
