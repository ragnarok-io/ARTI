from dataclasses import replace

import pytest
import torch

from arti import mechanisms as m
from benchmarks._federated_product_replay import replay_cooperative_dependencies, replay_cooperative_dependencies_many
from benchmarks._federated_recursive_search import search_cooperative_graphs, start_recursive_search
from test_formula_completed_products import private_write_graph, sharing_graph
from test_formula_program_query_v6 import effect, federation, member
from test_formula_program_query_v7 import join
from test_formula_program_query_v7 import cooperative, wrap_child


def search(model, x, *, product_slots=(), publish_slots=None, beam_width=2):
    with torch.no_grad():
        return search_cooperative_graphs((start_recursive_search(model, {"x": x}),),
                                         product_slots=product_slots, publish_slots=publish_slots,
                                         width=2, beam_width=beam_width)


def replay(model, x, searched):
    return replay_cooperative_dependencies(model, {"x": x}, tape=searched.dependency_tape,
                                           endpoint=searched.winner.dependency_endpoint)


def test_attention_input_adapter_keeps_native_binding_during_replay():
    from test_federated_ordinary_programs import _fixture, _candidate

    federation = _fixture(families=("causal-attention",))
    candidate = _candidate(federation, "causal-attention")
    model = m.FormulaProgramQueryV7(
        slot_ids=federation.query.slot_ids, candidates=(candidate,),
        terminal_slots={"y": candidate.output_slot}, entry_candidates=(candidate.candidate_id,),
        continuations={}, cooperation_width=2, max_steps=1)
    x = torch.randn(1, 3, 4, requires_grad=True)
    native = candidate(model._arena({"x": x})).values.get(candidate.output_slot)
    searched = search(model, x, beam_width=1)
    rebuilt = replay(model, x, searched)
    torch.testing.assert_close(rebuilt.outputs["y"], native)
    actual, = torch.autograd.grad(rebuilt.outputs["y"].sum(), (x,))
    expected, = torch.autograd.grad(native.sum(), (x,))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_no_grad_search_recomputes_pruned_shared_producer_once(device):
    model = sharing_graph().to(device)
    x = torch.ones(1, 1, device=device, requires_grad=True)
    searched = search(model, x, product_slots=("shared",), publish_slots=("p",), beam_width=1)
    assert not searched.completed_products[0].value.requires_grad
    calls = []
    handles = [c.register_forward_hook(lambda c, args, out: calls.append(c.candidate_id)) for c in model.candidates]
    rebuilt = replay(model, x, searched)
    for handle in handles:
        handle.remove()
    assert calls.count("z_donor") == 1
    assert calls == ["z_donor", "c", "d", "final"]
    assert "a_receiver" not in calls  # Only routed the chosen graph; no numerical dependency.
    assert len(rebuilt.executed_occurrences) == 4
    torch.testing.assert_close(rebuilt.outputs["y"], 20 * x)
    dw, dx = torch.autograd.grad(rebuilt.outputs["y"].sum(),
                                (model.candidates[1].candidate.operand_store.tensor("weight"), x))
    torch.testing.assert_close(dw, 5 * x)
    torch.testing.assert_close(dx, torch.full_like(x, 20))


def test_private_effect_ancestor_is_recomputed_but_not_retained_with_meta_gradient():
    model, write = private_write_graph()
    x = torch.ones(1, 1, requires_grad=True)
    searched = search(model, x, product_slots=("shared",), publish_slots=("u",))
    calls = []
    handles = [c.register_forward_hook(lambda c, args, out: calls.append(c.candidate_id)) for c in model.candidates]
    rebuilt = replay(model, x, searched)
    for handle in handles:
        handle.remove()
    assert calls == ["donor", "write", "reread", "c", "d", "final"]
    assert rebuilt.proposals == ()
    assert rebuilt.bank_state.values[0] is searched.dependency_tape.initial_states[0].values[0]
    rate = write.operand_store.tensor("rate")
    expected = 5 * x * (1 + 2 * x * rate)
    torch.testing.assert_close(rebuilt.outputs["y"], expected)
    actual = torch.autograd.grad(rebuilt.outputs["y"].sum(), (rate, x), create_graph=True)
    direct = torch.autograd.grad(expected.sum(), (rate, x), create_graph=True)
    for got, want in zip(actual, direct, strict=True):
        torch.testing.assert_close(got, want)
    torch.testing.assert_close(torch.autograd.grad(actual[0].sum(), x)[0],
                               torch.autograd.grad(direct[0].sum(), x)[0])


def test_replay_uses_new_values_and_current_operands_without_search_or_cache():
    model = sharing_graph()
    searched = search(model, torch.ones(1, 1), product_slots=("shared",), publish_slots=("p",), beam_width=1)
    with torch.no_grad():
        model.candidates[1].candidate.operand_store.tensor("weight").fill_(6.0)
        model.candidates[0].candidate.operand_store.tensor("weight").fill_(-100.0)
    x = torch.full((1, 1), 2.0, requires_grad=True)
    rebuilt = replay(model, x, searched)
    torch.testing.assert_close(rebuilt.outputs["y"], 30 * x)
    torch.testing.assert_close(torch.autograd.grad(rebuilt.outputs["y"].sum(), x)[0], torch.full_like(x, 30))
    assert searched.completed_products[0].value.item() == 4


def test_retained_effect_is_applied_once_and_state_reuse_remains_differentiable():
    model = cooperative(federation(write=True, shared=True))
    x = torch.ones(1, 1, requires_grad=True)
    searched = search(model, x, beam_width=1)
    rebuilt = replay(model, x, searched)
    branch = searched.winner
    assert len(rebuilt.proposals) == len(branch.execution.proposals) == 1
    assert rebuilt.bank_state.revisions == branch.execution.bank_state.revisions
    torch.testing.assert_close(rebuilt.outputs["result"], branch.execution.outputs["result"])
    torch.testing.assert_close(rebuilt.bank_state.values[0], branch.execution.bank_state.values[0])
    state_loss = rebuilt.bank_state.values[0].square().sum()
    rate = model.candidates[1].operand_store.tensor("rate")
    assert torch.autograd.grad(state_loss, rate)[0].abs().sum() > 0


def test_complete_call_replay_keeps_child_choices_and_actual_bank_write():
    child = cooperative(federation(write=True, shared=True))
    parent = wrap_child(child)
    x = torch.ones(1, 1, requires_grad=True)
    searched = search(parent, x, beam_width=1)
    hooks = []
    handle = child.register_forward_hook(lambda *args: hooks.append("child"))
    rebuilt = replay(parent, x, searched)
    handle.remove()
    assert hooks == ["child"]
    assert len(rebuilt.proposals) == 1
    torch.testing.assert_close(rebuilt.outputs["result"], searched.winner.execution.outputs["result"])
    torch.testing.assert_close(rebuilt.bank_state.values[0], searched.winner.execution.bank_state.values[0])


def test_numerically_unused_donor_suffix_is_not_replayed():
    model, write = private_write_graph()
    # A legal donor suffix happens after publication, but is not a data ancestor.
    suffix = effect()
    suffix.candidate_id, suffix.input_slot, suffix.output_slot = "suffix", "u", "unused"
    candidates = (*model.candidates, suffix)
    edges = {k: dict(v) for k, v in model.continuations.items()}
    edges["reread"] = {"suffix": "u_negative"}
    model = m.FormulaProgramQueryV7(
        slot_ids=(*model.slot_ids, "unused"), candidates=candidates,
        terminal_slots=model.terminal_slots, entry_candidates=model.entry_candidates,
        continuations=edges, max_steps=6, cooperation_width=2,
    )
    x = torch.ones(1, 1, requires_grad=True)
    searched = search(model, x, product_slots=("shared",), publish_slots=("u",))
    suffix_ids = {r.node.occurrence_id for r in searched.dependency_tape.occurrences if r.node.candidate_id == "suffix"}
    assert suffix_ids
    rebuilt = replay(model, x, searched)
    assert not suffix_ids.intersection(rebuilt.executed_occurrences)
    assert torch.autograd.grad(rebuilt.outputs["y"].sum(), suffix.operand_store.tensor("rate"),
                               allow_unused=True)[0] is None


def test_distinct_effect_occurrences_at_same_revision_are_both_reconstructed():
    producer = member("producer", "x", "h", owner="memory")
    first, second = effect(), effect()
    first.candidate_id, second.candidate_id = "first", "second"
    with torch.no_grad():
        second.operand_store.tensor("rate").fill_(1)
    read = member("read", "tail", "p", owner="memory")
    combine = join("combine", "shared0", "shared1", "y")
    candidates = (producer, first, second, read, combine)
    model = m.FormulaProgramQueryV7(
        slot_ids=("x", "h", "h_negative", "tail", "p", "p_negative", "shared0", "shared1", "y"),
        candidates=candidates, terminal_slots={"y": "y"}, entry_candidates=("producer",),
        continuations={"producer": {"first": "h", "second": "h", "read": "h"},
                       "read": {"combine": "p"}}, max_steps=4, cooperation_width=2,
    )
    x = torch.ones(1, 1, requires_grad=True)
    searched = search(model, x, product_slots=("shared0", "shared1"), publish_slots=("p",))
    a, b = searched.completed_products
    assert a.lineage.plastic_revision == b.lineage.plastic_revision == 1
    assert a.lineage.plastic_value is not b.lineage.plastic_value
    results = replay_cooperative_dependencies_many(
        model, {"x": x}, tape=searched.dependency_tape,
        endpoints=tuple(branch.dependency_endpoint for branch in searched.branches),
    )
    for result in results:
        assert len(result.proposals) == 1  # Each endpoint retains only its own write.
        torch.testing.assert_close(result.outputs["y"], 2 * x)
    ids = [i for result in results for i in result.executed_occurrences]
    assert len(ids) == len(set(ids))
    total = sum(result.outputs["y"].sum() for result in results)
    grads = torch.autograd.grad(total, (first.operand_store.tensor("rate"), second.operand_store.tensor("rate")))
    for gradient in grads:
        torch.testing.assert_close(gradient, 4 * x)


def test_multiple_endpoints_reconstruct_shared_occurrence_once():
    original = sharing_graph()
    left, right = member("left", "shared", "y", weight=2.0), member("right", "shared", "y", weight=3.0)
    model = m.FormulaProgramQueryV7(
        slot_ids=("x", "shared", "h", "h_negative", "p", "p_negative", "y", "y_negative"),
        candidates=(*original.candidates[:2], left, right), terminal_slots={"y": "y"},
        entry_candidates=original.entry_candidates,
        continuations={"a_receiver": {"left": "h", "right": "h"}},
        max_steps=2, cooperation_width=2,
    )
    x = torch.ones(1, 1, requires_grad=True)
    searched = search(model, x, product_slots=("shared",), publish_slots=("p",))
    assert len(searched.branches) == 2
    calls = []
    hook = model.candidates[1].register_forward_hook(lambda *args: calls.append("donor"))
    results = replay_cooperative_dependencies_many(
        model, {"x": x}, tape=searched.dependency_tape,
        endpoints=tuple(b.dependency_endpoint for b in searched.branches),
    )
    hook.remove()
    assert calls == ["donor"]
    ids = [i for result in results for i in result.executed_occurrences]
    assert len(ids) == len(set(ids)) == 3
    combined = sum(result.outputs["y"].sum() for result in results)
    torch.testing.assert_close(combined, (20 * x).sum())
    dw, dx = torch.autograd.grad(combined, (model.candidates[1].candidate.operand_store.tensor("weight"), x))
    torch.testing.assert_close(dw, 5 * x)
    torch.testing.assert_close(dx, torch.full_like(x, 20))


def test_replay_distinguishes_equal_revision_roots_and_accepts_fresh_fast_states():
    producer = member("producer", "x", "p", owner="memory")
    left = member("left", "source0", "y")
    right = left.with_bindings("right", input_slots={"x": "source1"}, output_slots=left.candidate.output_slots)
    model = m.FormulaProgramQueryV7(
        slot_ids=("x", "p", "p_negative", "source0", "source1", "y", "y_negative"),
        candidates=(producer, left, right), terminal_slots={"y": "y"}, entry_candidates=("producer",),
        continuations={"producer": {"left": "p", "right": "p_negative"}},
        max_steps=2, cooperation_width=2,
    )
    x = torch.ones(1, 1, requires_grad=True)
    initial = model.initial_bank_state()
    states = (initial, replace(initial, values=(-initial.values[0],)))
    with torch.no_grad():
        searched = search_cooperative_graphs(
            tuple(start_recursive_search(model, {"x": x}, bank_state=s) for s in states),
            product_slots=("source0", "source1"), publish_slots=("p",), width=1, beam_width=2,
        )
    a = torch.full((1, 1), 5.0, requires_grad=True)
    b = torch.full((1, 1), -7.0, requires_grad=True)
    fresh = (replace(initial, values=(a,)), replace(initial, values=(b,)))
    results = replay_cooperative_dependencies_many(
        model, {"x": x}, tape=searched.dependency_tape,
        endpoints=tuple(branch.dependency_endpoint for branch in searched.branches), initial_states=fresh,
    )
    outputs = {endpoint.root: result.outputs["y"] for endpoint, result in
               zip((b.dependency_endpoint for b in searched.branches), results, strict=True)}
    torch.testing.assert_close(outputs[0], 5 * x)
    torch.testing.assert_close(outputs[1], -7 * x)
    ga, gb = torch.autograd.grad(outputs[0].sum() + 2 * outputs[1].sum(), (a, b))
    torch.testing.assert_close(ga, x)
    torch.testing.assert_close(gb, 2 * x)


def test_bank_only_formula_uses_current_arena_context_without_fake_input_gradient():
    kind = m.TensorType(("B", "D"), ("B", 1), dtype="floating", domain="activation")
    w = m.BankBinding("weight", "arti/response-test@1", "weight", kind)
    rate = m.BankBinding("rate", "arti/response-test@1", "rate", kind)
    program = m.FormulaProgram.build(outputs=(m.scale(w, rate),))
    leaf = m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
        "bankonly", program, input_slots={}, output_slots={program.outputs[0]: "y"},
        operands={"weight": torch.full((1, 1), 2.0), "rate": torch.full((1, 1), 3.0)},
        trainable_operands=("rate",), batch_broadcast_operands=("weight", "rate"),
    ), plastic_bank_slot="weight", bank_owner_id="memory")
    model = m.FormulaProgramQueryV7(
        slot_ids=("x", "y"), candidates=(leaf,), terminal_slots={"y": "y"},
        entry_candidates=("bankonly",), continuations={}, max_steps=1,
    )
    x = torch.ones(1, 1, requires_grad=True)
    searched = search(model, x)
    assert searched.dependency_tape.occurrences[0].inputs == ()
    slow = leaf.candidate.operand_store.tensor("rate")
    with torch.no_grad():
        slow.fill_(5)
    fast = torch.full((1, 1), 7.0, requires_grad=True)
    state = replace(model.initial_bank_state(), values=(fast,))
    rebuilt = replay_cooperative_dependencies(model, {"x": x}, tape=searched.dependency_tape,
                                              endpoint=searched.winner.dependency_endpoint,
                                              initial_states=(state,))
    torch.testing.assert_close(rebuilt.outputs["y"], torch.full((1, 1), 35.0))
    dw, dr, dx = torch.autograd.grad(rebuilt.outputs["y"].sum(), (fast, slow, x), allow_unused=True)
    torch.testing.assert_close(dw, torch.full_like(fast, 5))
    torch.testing.assert_close(dr, torch.full_like(slow, 7))
    assert dx is None
