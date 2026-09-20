from dataclasses import replace

import pytest
import torch

from arti import mechanisms as m
from benchmarks._federated_plasticity_participation import (
    compose_participant_proposals, replay_participant_graphs,
)
from benchmarks._federated_recursive_search import (
    replay_recursive_graph, search_recursive_graphs, start_recursive_search,
)
from test_formula_program_query_v4 import _effect, _producer, _type


def _fork_query(device="cpu", *, unused_branches=True):
    first = _producer("first", "x", "made")
    shared = _effect("shared", "made", "shared", writer_scale=0.1, execution_count=2.0)
    left = _effect("left", "shared", "tail", writer_scale=0.2, execution_count=2.0)
    right = _effect("right", "shared", "tail", writer_scale=0.3, execution_count=2.0)
    with torch.no_grad():
        right.operand_store.tensor("gain").fill_(0.1)
    read = _producer("read", "tail", "out")
    unused = _effect("unused", "absent", "tail", execution_count=2.0)
    other = _producer("other", "absent", "unused", owner_id="other-bank")
    query = m.FormulaProgramQueryV5(
        slot_ids=("x", "made", "shared", "tail", "out", "absent", "unused"),
        candidates=(first, shared, left, right, read, *((unused, other) if unused_branches else ())),
        terminal_slots={"answer": "out"}, min_steps=4, max_steps=4, hidden_dim=8,
    ).to(device)
    return query, first, shared, left, right, read, unused, other


def _search(query, x, state):
    with torch.no_grad():
        return search_recursive_graphs(
            (start_recursive_search(query, {"x": x}, bank_state=state),),
            width=2, beam_width=4, preserve_effect_coverage=False, record_query_choices=True,
        )


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_shared_prefix_composes_noncommuting_suffixes_without_changing_output(device):
    query, first, shared, left, right, read, unused, other = _fork_query(device)
    x = torch.tensor([[0.25, -0.4, 0.7]], device=device)
    initial = query.initial_bank_state()
    result = _search(query, x, initial)
    assert len(result.branches) == 2
    paths = [b.execution.proposals for b in result.branches]
    assert paths[0][0] is paths[1][0]
    before = result.winner.execution.outputs["answer"].clone()
    combined = compose_participant_proposals(initial, paths)
    slot = first.bank_slot_ref
    expected = initial.value(slot)
    for proposal in (paths[0][0], paths[0][1], paths[1][1]):
        expected = proposal.transition.apply(expected)
    torch.testing.assert_close(combined.state.value(slot), expected)
    assert combined.state.revision(slot) == 3
    assert combined.shared_references == 1
    assert combined.reused_successors == 2 and combined.recomposed_transitions == 1
    reverse = compose_participant_proposals(initial, tuple(reversed(paths)))
    assert not torch.allclose(reverse.state.value(slot), expected)
    average = sum(b.execution.bank_state.value(slot) for b in result.branches) / 2
    assert not torch.allclose(expected, average)
    torch.testing.assert_close(result.winner.execution.outputs["answer"], before, rtol=0, atol=0)
    assert combined.state.value(other.bank_slot_ref) is initial.value(other.bank_slot_ref)
    assert initial.revision(slot) == query.initial_bank_state().revision(slot) == 0


def test_separate_occurrences_with_identical_candidate_and_operands_are_not_deduplicated():
    query, first, shared, *_ = _fork_query()
    initial = query.initial_bank_state()
    entry = first(query._arena({"x": torch.ones(1, 3)}, bank_state=initial))
    a, b = shared(entry).proposals, shared(entry).proposals
    assert a[0] is not b[0]
    assert a[0].effect_candidate_id == b[0].effect_candidate_id
    combined = compose_participant_proposals(initial, (a, b))
    slot = first.bank_slot_ref
    assert len(combined.occurrences) == 2 and combined.shared_references == 0
    assert combined.state.revision(slot) == 2
    expected = b[0].transition.apply(a[0].successor)
    torch.testing.assert_close(combined.state.value(slot), expected)


def test_single_participant_reuses_successor_and_empty_participation_is_identity():
    query, first, shared, left, *_ = _fork_query()
    initial = query.initial_bank_state()
    arena = left(shared(first(query._arena({"x": torch.ones(1, 3)}, bank_state=initial))))
    combined = compose_participant_proposals(initial, (arena.proposals,))
    assert combined.state.value(first.bank_slot_ref) is arena.proposals[-1].successor
    assert combined.state.revision(first.bank_slot_ref) == 2
    assert combined.reused_successors == 2 and combined.recomposed_transitions == 0
    assert compose_participant_proposals(initial, ()).state is initial


def test_missing_captured_operands_cannot_be_replaced_with_successor_deltas():
    query, first, shared, *_ = _fork_query()
    initial = query.initial_bank_state()
    entry = first(query._arena({"x": torch.ones(1, 3)}, bank_state=initial))
    a, b = shared(entry).proposals[0], shared(entry).proposals[0]
    with pytest.raises(ValueError, match="recorded Formula operands"):
        compose_participant_proposals(initial, ((a,), (replace(b, transition=None),)))


def test_later_effect_operands_keep_branch_local_bank_causality():
    first = _producer("first", "x", "made")
    a = _effect("a", "made", "mid", writer_scale=0.2)
    b = _effect("b", "made", "mid", writer_scale=0.4)
    read = _producer("read", "mid", "seen")
    after = _effect("after", "seen", "out", writer_scale=0.1)
    query = m.FormulaProgramQueryV5(
        slot_ids=("x", "made", "mid", "seen", "out"), candidates=(first, a, b, read, after),
        terminal_slots={"answer": "out"}, max_steps=4, hidden_dim=8,
    )
    initial = query.initial_bank_state()
    x = torch.tensor([[0.4, -0.3, 0.7]])
    entry = first(query._arena({"x": x}, bank_state=initial))
    left, right = after(read(a(entry))), after(read(b(entry)))
    paths = (left.proposals, right.proposals)
    assert not torch.equal(paths[0][1].transition.effect.operands[0], paths[1][1].transition.effect.operands[0])
    expected = initial.value(first.bank_slot_ref)
    for p in (paths[0][0], paths[1][0], paths[0][1], paths[1][1]):
        expected = p.transition.apply(expected)
    combined = compose_participant_proposals(initial, paths)
    torch.testing.assert_close(combined.state.value(first.bank_slot_ref), expected)
    first_writes = compose_participant_proposals(initial, ((paths[0][0],), (paths[1][0],)))
    regenerated = after(read(query._arena({"mid": entry.values.get("made")}, bank_state=first_writes.state)))
    wrong = regenerated.proposals[0].transition.apply(first_writes.state.value(first.bank_slot_ref))
    wrong = regenerated.proposals[0].transition.apply(wrong)
    assert not torch.allclose(wrong, expected)


def test_participant_replay_preserves_query_and_recorded_event_identity():
    query, *_ = _fork_query()
    other, *_ = _fork_query()
    x = torch.ones(1, 3)
    initial = query.initial_bank_state()
    result = _search(query, x, initial)
    with pytest.raises(ValueError, match="supplied Query"):
        replay_participant_graphs(other, {"x": x}, result.branches, bank_state=other.initial_bank_state())
    cloned = m.FormulaProgramBankState(initial.slot_refs, tuple(v.clone() for v in initial.values), initial.revisions)
    second = _search(query, x, cloned)
    with pytest.raises(ValueError, match="one recorded entry Bank snapshot"):
        replay_participant_graphs(query, {"x": x}, (result.branches[0], second.branches[0]), bank_state=initial)
    revised = m.FormulaProgramBankState(initial.slot_refs, initial.values, tuple(r + 1 for r in initial.revisions))
    with pytest.raises(ValueError, match="entry slots and revisions"):
        replay_participant_graphs(query, {"x": x}, result.branches, bank_state=revised)
    # Same-valued replay tensors may carry gradients from the preceding event.
    replayed, _ = replay_participant_graphs(query, {"x": x}, result.branches, bank_state=cloned)
    torch.testing.assert_close(replayed[0].execution.outputs["answer"], result.branches[0].execution.outputs["answer"])


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("recorded_choices", (False, True))
def test_prefix_dag_replay_preserves_outputs_and_future_loss_credit(monkeypatch, device, recorded_choices):
    from arti import _formula_candidate_batch as batch

    query, first, shared, left, right, read, unused, other = _fork_query(device)
    x = torch.tensor([[0.25, -0.4, 0.7]], device=device)
    initial = query.initial_bank_state()
    result = _search(query, x, initial)
    expected = compose_participant_proposals(initial, [b.execution.proposals for b in result.branches])
    calls = []
    original = batch.execute_many

    def track(requests, **kwargs):
        calls.extend(candidate.candidate_id for candidate, _ in requests)
        return original(requests, **kwargs)

    monkeypatch.setattr(batch, "execute_many", track)
    replayed, combined = replay_participant_graphs(
        query, {"x": x}, result.branches, bank_state=initial, use_recorded_choices=recorded_choices,
    )
    assert calls.count("first") == calls.count("shared") == 1
    assert calls.count("left") == calls.count("right") == 1 and calls.count("read") == 2
    assert combined.shared_references == 1 and len(combined.occurrences) == 3
    for branch, actual in zip(result.branches, replayed, strict=True):
        torch.testing.assert_close(branch.execution.outputs["answer"], actual.execution.outputs["answer"])
        torch.testing.assert_close(branch.log_probability, actual.log_probability)
        independent = replay_recursive_graph(query, {"x": x}, branch.route, bank_state=initial)
        torch.testing.assert_close(independent.execution.outputs["answer"], actual.execution.outputs["answer"])
    slot = first.bank_slot_ref
    torch.testing.assert_close(combined.state.value(slot), expected.state.value(slot))

    # Supervision reads the next event's composed Bank, not a local writer target.
    future = read(query._arena({"tail": x * 0.8}, bank_state=combined.state)).values.get("out")
    loss = future.square().mean() + 0.1 * torch.stack([b.log_probability for b in replayed]).sum()
    loss.backward()
    for effect in (shared, left, right):
        for parameter in (effect.operand_store.tensor("writer"), effect.execution_count_tensor()):
            assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            assert bool((parameter.grad != 0).any())
    assert all(parameter.grad is None for parameter in unused.parameters())
    assert any(p.grad is not None for p in query.network.parameters())
    assert not first.bank_owner.value.requires_grad and first.bank_owner.value.grad is None


@pytest.mark.parametrize("nested", (False, True))
def test_shared_dag_replay_matches_native_search_all_parameter_gradients(nested):
    query, first, shared, left, right, read, *_ = _fork_query(unused_branches=not nested)
    if nested:
        call = m.FormulaProgramCallCandidateV1(
            "call", query, input_slots={"x": "x"}, output_slots={"answer": "answer"},
        )
        query = m.FormulaProgramQueryV5(
            slot_ids=("x", "answer"), candidates=(call,),
            terminal_slots={"answer": "answer"}, max_steps=1, hidden_dim=8,
        )
    x = torch.tensor([[0.25, -0.4, 0.7]])
    initial = query.initial_bank_state()
    native = search_recursive_graphs(
        (start_recursive_search(query, {"x": x}, bank_state=initial),),
        width=2, beam_width=4, preserve_effect_coverage=False, record_query_choices=True,
    )
    reference = compose_participant_proposals(initial, [b.execution.proposals for b in native.branches])
    replayed, combined = replay_participant_graphs(query, {"x": x}, native.branches, bank_state=initial)
    assert len(reference.occurrences) == len(combined.occurrences) == 3
    assert reference.shared_references == combined.shared_references == 1
    slot = next(ref for ref in initial.slot_refs if ref.producer_id.endswith("shared-producer"))

    def objective(state, branches):
        return (state.value(slot) * x).square().mean() + sum(
            0.1 * branch.log_probability + 0.01 * branch.execution.outputs["answer"].square().mean()
            for branch in branches
        )

    parameters = tuple(query.parameters())
    expected = torch.autograd.grad(objective(reference.state, native.branches), parameters, allow_unused=True)
    actual = torch.autograd.grad(objective(combined.state, replayed), parameters, allow_unused=True)
    for a, b in zip(actual, expected, strict=True):
        if b is None:
            assert a is None
        else:
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def _family_candidate(family):
    value = m.InputBinding("value", _type())
    operands = {}

    def binding(name, tensor, axes=("B", "D")):
        operands[name] = tensor
        sizes = tuple("B" if axis == "B" else size for axis, size in zip(axes, tensor.shape, strict=True))
        kind = m.TensorType(axes, sizes, dtype="float32", domain="activation")
        return m.BankBinding(name, "arti/plasticity-participation-test@1", name, kind)

    bias = m.scale(value, binding("bias", torch.full((1, 3), 0.1)))
    if family == "affine":
        expression = m.neural_plasticity(value, bias, binding("gain", torch.full((1, 3), 0.02)))
    elif family == "blend":
        expression = m.neural_plasticity_blend(value, bias, binding("amount", torch.full((1, 3), -0.8)))
    elif family == "proximal":
        expression = m.neural_plasticity_proximal(value, bias, binding("strength", torch.full((1, 3), -4.0)))
    elif family.startswith("outer"):
        left = m.reduce_sum(value, axis="D")
        right = binding("right", torch.tensor([0.2, -0.1, 0.4]), ("D",))
        rate = binding("rate", torch.tensor(0.05), ())
        count = binding("count", torch.tensor(2.0), ()) if family == "outer2" else None
        expression = m.neural_plasticity_outer(value, left, right, rate, count, max_executions=4)
    else:
        factor = torch.tensor([[0.1, 0.2], [-0.2, 0.3], [0.4, -0.1]])
        output = binding("output", factor, ("D", "R"))
        left = binding("left", factor.flip(0), ("D", "R"))
        rate = binding("rate", torch.tensor(0.1), ())
        if family == "transport":
            expression = m.neural_plasticity_transport(value, bias, output, left, rate, state_axis="D")
        else:
            right = binding("right", factor.flip(1), ("D", "R"))
            expression = m.neural_plasticity_polynomial(value, bias, output, left, right, rate, state_axis="D")
    used = {b.name for b in m.FormulaProgram.build(outputs=(expression,)).bindings}
    operands = {name: tensor for name, tensor in operands.items() if name in used}
    return m.FormulaProgramEffectCandidateV3(
        family, m.FormulaEffectProgramV2(m.FormulaProgram.build(outputs=(expression,)),
                                       data_input_name="value", state_type=_type()),
        input_slot="made", output_slot="out", operands=operands, trainable_operands=tuple(operands),
        execution_count=None if family == "outer2" else torch.tensor(2.0),
        trainable_execution_count=family != "outer2", max_executions=4,
    )


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("family", ("affine", "blend", "outer", "outer2", "transport", "polynomial", "proximal"))
@pytest.mark.parametrize("mode", ("ordered-operators", "parallel-delta"))
def test_all_effect_families_reapply_recorded_operands_with_count_gradients(device, family, mode):
    first = _producer("first", "x", "made")
    effect = _family_candidate(family)
    query = m.FormulaProgramQueryV5(
        slot_ids=("x", "made", "out"), candidates=(first, effect),
        terminal_slots={"answer": "out"}, max_steps=2, hidden_dim=8,
    ).to(device)
    initial = query.initial_bank_state()
    x = torch.tensor([[0.4, -0.3, 0.7]], device=device)

    def run():
        entry = first(query._arena({"x": x}, bank_state=initial))
        a, b = effect(entry).proposals[0], effect(entry).proposals[0]
        assert a.previous is b.previous
        combined = compose_participant_proposals(initial, ((a,), (b,)), mode=mode)
        manual = (initial.value(first.bank_slot_ref) + torch.stack((a.successor - a.previous,
                                                                   b.successor - b.previous)).sum(0)
                  if mode == "parallel-delta" else
                  b.transition.apply(a.transition.apply(initial.value(first.bank_slot_ref))))
        return combined.state.value(first.bank_slot_ref), manual

    with torch.no_grad():
        frozen, _ = run()
    actual, manual = run()
    torch.testing.assert_close(actual, frozen, rtol=0, atol=0)
    torch.testing.assert_close(actual, manual, rtol=0, atol=0)
    parameters = tuple(effect.parameters())
    gradients = torch.autograd.grad(actual.square().mean(), parameters, retain_graph=True)
    expected = torch.autograd.grad(manual.square().mean(), parameters)
    for a, b in zip(gradients, expected, strict=True):
        torch.testing.assert_close(a, b)
        assert bool(torch.isfinite(a).all()) and bool((a != 0).any())
