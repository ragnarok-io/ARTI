from dataclasses import replace

import pytest
import torch

from arti import mechanisms as m
from benchmarks._federated_recursive_search import (
    CompletedFormulaProduct,
    _bind_completed_products,
    replay_recursive_graph,
    search_cooperative_graphs,
    start_recursive_search,
)
from test_formula_program_query_v6 import effect, federation, member
from test_formula_program_query_v7 import graph, join


def sharing_graph():
    receiver = member("a_receiver", "x", "h")
    donor = member("z_donor", "x", "p", weight=4.0)
    receiver.candidate.requires_empty_slots = ("p",)
    donor.candidate.requires_empty_slots = ("h",)
    c = member("c", "shared", "u", weight=2.0)
    d = member("d", "shared", "v", weight=3.0)
    final = join("final", "u", "v", "y")
    candidates = (receiver, donor, c, d, final)
    return m.FormulaProgramQueryV7(
        slot_ids=("x", "shared", *(s for c in candidates for s in c.output_slot_ids)),
        candidates=candidates, terminal_slots={"y": "y"},
        entry_candidates=("a_receiver", "z_donor"),
        continuations={"a_receiver": {"c": "h", "d": "h", "final": "h"}},
        max_steps=4, cooperation_width=2,
    )


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_pruned_unstopped_donor_products_share_live_graph_and_gradient(device):
    model = sharing_graph().to(device)
    x = torch.ones(1, 1, device=device, requires_grad=True)
    calls, shared_inputs = [], []
    handles = []
    for candidate in model.candidates:
        def observe(module, args, output):
            calls.append(module.candidate_id)
            if module.candidate_id in ("c", "d"):
                shared_inputs.append(args[0].values.get("shared"))
        handles.append(candidate.register_forward_hook(observe))
    result = search_cooperative_graphs(
        (start_recursive_search(model, {"x": x}),),
        product_slots=("shared",), publish_slots=("p",), width=2, beam_width=1,
    )
    for handle in handles:
        handle.remove()
    branch = result.winner
    assert [s.candidate_id for s in branch.execution.trace.steps] == ["a_receiver", "c", "d", "final", "stop"]
    assert "z_donor" not in [r["candidate_id"] for r in branch.route]
    assert calls.count("z_donor") == 1
    product, = result.completed_products
    assert len(shared_inputs) == 2 and all(value is product.value for value in shared_inputs)
    assert branch.arena.values.get("shared") is product.value
    assert branch.arena.producer("shared") is product.lineage
    assert branch.arena.tensor_steps == 4  # Adoption does not count as a dispatch.
    torch.testing.assert_close(branch.execution.outputs["y"], 20 * x)
    dw, dx = torch.autograd.grad(branch.execution.outputs["y"].sum(),
                                (model.candidates[1].candidate.operand_store.tensor("weight"), x))
    torch.testing.assert_close(dw, 5 * x)
    torch.testing.assert_close(dx, torch.full_like(x, 20))
    c, d = branch.execution.frontiers[1].nodes
    assert c.external_inputs == d.external_inputs == (("x", product.occurrence_id, product.output_port),)
    assert c.occurrence_id != d.occurrence_id != product.occurrence_id
    with pytest.raises(NotImplementedError, match="dependency-graph replay"):
        model.replay({"x": x}, branch.execution.frontiers)
    with pytest.raises(NotImplementedError, match="dependency-graph replay"):
        replay_recursive_graph(model, {"x": x}, branch.route)


def test_import_preserves_private_bank_version_without_adopting_writes():
    model = federation(write=True, shared=True)
    # Use the real predecessor write and re-read its new value for the product.
    a, write, read = model.candidates[:3]
    x = torch.ones(1, 1, requires_grad=True)
    original = model._arena({"x": x})
    donor = read(write(a(original)))
    product = CompletedFormulaProduct(7, "value", "u", "shared", donor.values.get("u"),
                                      donor.producer("u"), None)
    consumer = replace(original, values=m.FormulaProgramArena.from_mapping(
        (*model.slot_ids, "shared"), {"x": x}), producers=(*original.producers, None))
    imported = _bind_completed_products(consumer, (product,))
    assert imported.bank_state is original.bank_state
    assert imported.proposals == ()
    assert imported.producer("shared").plastic_revision == 1
    assert imported.effect_state(a.bank_slot_ref)[1] == 0
    assert imported.values.get("shared") is donor.values.get("u")
    consume = member("consume", "shared", "y", weight=2.0)
    result = consume(imported)
    expected = 2 * donor.values.get("u")
    torch.testing.assert_close(result.values.get("y"), expected)
    rate = write.operand_store.tensor("rate")
    actual_vjp = torch.autograd.grad(result.values.get("y").sum(), rate, retain_graph=True)[0]
    direct_vjp = torch.autograd.grad(expected.sum(), rate)[0]
    torch.testing.assert_close(actual_vjp, direct_vjp)
    assert actual_vjp.abs().sum() > 0
    # An old branch's tensor is not promoted into a current write target.
    stale_write = effect()
    stale_write.input_slot = "shared"
    assert not stale_write.accepts(imported)


def test_reserved_directory_and_joint_shape_are_real_admission():
    model = sharing_graph()
    start = start_recursive_search(model, {"x": torch.ones(1, 1)})
    with pytest.raises(ValueError, match="reserved"):
        search_cooperative_graphs((start,), product_slots=("p",))
    with pytest.raises(RuntimeError, match="stopped root"):
        search_cooperative_graphs((start,), product_slots=(), publish_slots=("p",), width=2, beam_width=1)
    with pytest.raises(ValueError, match="same input interaction"):
        search_cooperative_graphs((start, start_recursive_search(model, {"x": torch.ones(1, 1)})),
                                  product_slots=("shared",))
    wrong_shape = CompletedFormulaProduct(12, "value", "p", "shared", torch.ones(1, 2), None, None)
    imported = _bind_completed_products(start.arena, (wrong_shape,))
    assert not model.candidates[2].accepts(imported)


def test_equal_revision_donors_keep_distinct_sources_and_response_selected_bindings():
    producer = member("producer", "x", "p", owner="memory")
    left = member("left", "shared0", "y")
    right = left.with_bindings("right", input_slots={"x": "shared1"}, output_slots=left.candidate.output_slots)
    model = m.FormulaProgramQueryV7(
        slot_ids=("x", "p", "p_negative", "shared0", "shared1", "y", "y_negative"),
        candidates=(producer, left, right), terminal_slots={"y": "y"}, entry_candidates=("producer",),
        continuations={"producer": {"left": "p", "right": "p_negative"}},
        max_steps=2, cooperation_width=2,
    )
    x = torch.ones(1, 1)
    state = model.initial_bank_state()
    other = replace(state, values=(-state.values[0],))
    starts = tuple(start_recursive_search(model, {"x": x}, bank_state=s) for s in (state, other))
    result = search_cooperative_graphs(starts, product_slots=("shared0", "shared1"),
                                        publish_slots=("p",), width=1, beam_width=2)
    a, b = result.completed_products
    assert a.occurrence_id != b.occurrence_id
    assert a.lineage.plastic_slot == b.lineage.plastic_slot
    assert a.lineage.plastic_revision == b.lineage.plastic_revision == 0
    assert a.lineage.plastic_value is not b.lineage.plastic_value
    assert {branch.execution.frontiers[1].nodes[0].candidate_id for branch in result.branches} == {"left", "right"}
    for branch in result.branches:
        node = branch.execution.frontiers[1].nodes[0]
        product = a if node.candidate_id == "left" else b
        assert node.external_inputs == (("x", product.occurrence_id, product.output_port),)
        torch.testing.assert_close(branch.execution.outputs["y"], product.value)


def private_write_graph():
    donor = member("donor", "x", "h", owner="memory")
    receiver = member("receiver", "x", "r")
    donor.candidate.requires_empty_slots = ("r",)
    receiver.candidate.requires_empty_slots = ("h",)
    write = effect()
    read = member("reread", "tail", "u", owner="memory")
    first = member("wait1", "r", "w1")
    second = member("wait2", "w1", "w2")
    c, d = member("c", "shared", "v", weight=2.0), member("d", "shared", "z", weight=3.0)
    final = join("final", "v", "z", "y")
    candidates = (donor, receiver, write, read, first, second, c, d, final)
    model = m.FormulaProgramQueryV7(
        slot_ids=("x", "shared", *(s for c in candidates for s in c.output_slot_ids)),
        candidates=candidates, terminal_slots={"y": "y"},
        entry_candidates=("donor", "receiver"),
        continuations={"donor": {"write": "h", "reread": "h"},
                       "receiver": {"wait1": "r"}, "wait1": {"wait2": "w1"},
                       "wait2": {"c": "w2", "d": "w2", "final": "w2"}},
        cooperation_width=2, max_steps=6,
    )
    return model, write


def test_search_consumes_readdressed_donor_without_installing_private_effect():
    model, write = private_write_graph()
    x = torch.ones(1, 1, requires_grad=True)
    state = model.initial_bank_state()
    result = search_cooperative_graphs((start_recursive_search(model, {"x": x}, bank_state=state),),
                                        product_slots=("shared",), publish_slots=("u",),
                                        width=2, beam_width=2)
    product, = result.completed_products
    assert product.lineage.plastic_revision == 1
    assert product.lineage.plastic_value is not state.values[0]
    answer = result.winner.execution
    assert answer.proposals == ()
    assert answer.bank_state.values[0] is state.values[0]
    assert answer.bank_state.revisions == state.revisions
    expected = 5 * x * (1 + 2 * x * write.operand_store.tensor("rate"))
    torch.testing.assert_close(answer.outputs["y"], expected)
    rate = write.operand_store.tensor("rate")
    grads = torch.autograd.grad(answer.outputs["y"].sum(), (rate, x), retain_graph=True)
    direct = torch.autograd.grad(expected.sum(), (rate, x))
    for actual, wanted in zip(grads, direct, strict=True):
        torch.testing.assert_close(actual, wanted)
    assert grads[0].abs().sum() > 0


def test_frontier_step_publishes_before_stop_and_native_uses_same_entry():
    model = sharing_graph()
    entry = model._arena({"x": torch.ones(1, 1)})
    advance = model.advance_frontier(entry, steps=0, first="z_donor")
    assert not advance.stopped
    assert advance.arena.values.get("p") is not None
    assert advance.arena.values.get("y") is None
    assert len(advance.trace_steps) == 1


def test_cooperative_search_without_external_inputs_replays_stop_and_gradients():
    model = graph()
    x = torch.ones(1, 1, requires_grad=True)
    search = search_cooperative_graphs((start_recursive_search(model, {"x": x}),),
                                       product_slots=(), width=2, beam_width=2)
    first = search.winner.execution
    replay = model.replay({"x": x}, first.frontiers)
    assert first.frontiers == replay.frontiers
    torch.testing.assert_close(first.outputs["y"], replay.outputs["y"])
    params = (*model.parameters(), x)
    actual = torch.autograd.grad(first.outputs["y"].sum(), params, allow_unused=True)
    repeated = torch.autograd.grad(replay.outputs["y"].sum(), params, allow_unused=True)
    for left, right in zip(actual, repeated, strict=True):
        if left is None:
            assert right is None
        else:
            torch.testing.assert_close(left, right)


def test_nonfinite_alternative_does_not_publish_or_kill_valid_branch():
    good = member("good", "x", "y")
    bad = member("bad", "x", "p", weight=1e30)
    good.candidate.requires_empty_slots = ("p",)
    bad.candidate.requires_empty_slots = ("y",)
    model = m.FormulaProgramQueryV7(
        slot_ids=("x", "shared", "y", "y_negative", "p", "p_negative"),
        candidates=(bad, good), terminal_slots={"y": "y"}, entry_candidates=("bad", "good"),
        continuations={}, max_steps=1, cooperation_width=2,
    )
    x = torch.full((1, 1), 1e20)
    result = search_cooperative_graphs((start_recursive_search(model, {"x": x}),),
                                       product_slots=("shared",), publish_slots=("p",),
                                       width=2, beam_width=1)
    assert result.completed_products == ()
    assert result.numerical_rejections == ({"candidate_ids": ("bad",), "step": 0, "code": "FF2_NONFINITE"},)
    torch.testing.assert_close(result.winner.execution.outputs["y"], x)
