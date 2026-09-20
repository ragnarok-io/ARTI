import pytest
import torch

from arti import mechanisms as m
from benchmarks._federated_captured_search import captured_search_execution
from benchmarks._federated_plasticity_participation import (
    compose_participant_proposals, replay_participant_graphs,
)
from benchmarks._federated_recursive_search import (
    replay_recursive_graph, search_recursive_graphs, search_recursive_graphs_many, start_recursive_search,
)
from test_formula_program_query_v4 import _producer
from test_federated_plasticity_participation import _family_candidate, _fork_query


def _compare_proposals(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
        assert a.target == b.target and a.effect_atom_ref == b.effect_atom_ref
        assert a.previous_revision == b.previous_revision and a.successor_revision == b.successor_revision
        assert a.transition is not None and b.transition is not None
        torch.testing.assert_close(a.transition.effect.operands, b.transition.effect.operands)
        assert (a.transition.execution_count is None) == (b.transition.execution_count is None)
        if a.transition.execution_count is not None:
            torch.testing.assert_close(a.transition.execution_count, b.transition.execution_count)
        torch.testing.assert_close(a.successor, b.successor)
        # Reapplying to a different current Bank needs the true operator, not delta replay.
        current = a.previous * 0.7 + 0.25
        torch.testing.assert_close(a.transition.apply(current), b.transition.apply(current))


@pytest.mark.parametrize("family", ("affine", "blend", "outer", "outer2", "transport", "polynomial", "proximal"))
def test_captured_tape_preserves_all_families_and_raw_counts_across_replay(family):
    first = _producer("first", "x", "made")
    effect = _family_candidate(family)
    query = m.FormulaProgramQueryV5(
        slot_ids=("x", "made", "out"), candidates=(first, effect),
        terminal_slots={"answer": "out"}, max_steps=2, hidden_dim=8,
    ).cuda()
    x = torch.tensor([[0.4, -0.3, 0.7]], device="cuda")
    options = dict(width=2, beam_width=2, preserve_effect_coverage=False)
    initial = query.initial_bank_state()
    starts = (start_recursive_search(query, {"x": x}, bank_state=initial),)
    with torch.no_grad():
        if effect.execution_count_tensor() is not None:
            effect.execution_count_tensor().fill_(2.4)
        expected = search_recursive_graphs(starts, **options)
        with captured_search_execution(horizon=8, record_effect_operands=True) as backend:
            result = search_recursive_graphs(starts, **options)
            assert backend.completed == 1 and not backend.fallbacks
            _compare_proposals(result.winner.execution.proposals, expected.winner.execution.proposals)
            proposal = result.winner.execution.proposals[0]
            values = (*proposal.transition.effect.operands,
                      *((proposal.transition.execution_count,) if proposal.transition.execution_count is not None else ()))
            saved = tuple(value.clone() for value in values)
            if proposal.transition.execution_count is not None:
                torch.testing.assert_close(proposal.transition.execution_count, torch.tensor(2.4, device="cuda"))
            for parameter in effect.parameters():
                parameter.add_(0.1)
            again = search_recursive_graphs(
                (start_recursive_search(query, {"x": x * -0.6}, bank_state=initial),), **options,
            )
            assert backend.completed == 2 and not backend.fallbacks
            assert again.winner.execution.proposals[0].transition is not None
            reference = replay_recursive_graph(query, {"x": x * -0.6}, again.winner.route, bank_state=initial)
            _compare_proposals(again.winner.execution.proposals, reference.execution.proposals)
            for actual, snapshot in zip(values, saved, strict=True):
                torch.testing.assert_close(actual, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("batched", (False, True))
def test_captured_participants_compose_and_replay_the_true_shared_dag(batched):
    query, first, *_ = _fork_query("cuda")
    x = torch.tensor([[0.25, -0.4, 0.7]], device="cuda")
    initial = query.initial_bank_state()
    inputs = tuple({"x": x * factor} for factor in (1.0, -0.7, 0.5))
    batches = [(start_recursive_search(query, value, bank_state=initial),) for value in inputs]
    options = dict(width=2, beam_width=4, preserve_effect_coverage=False, record_query_choices=True)
    with torch.no_grad():
        expected = search_recursive_graphs_many(batches, **options)
        with captured_search_execution(horizon=8, record_effect_operands=True) as backend:
            result = (search_recursive_graphs_many(batches, **options) if batched else
                      tuple(search_recursive_graphs(starts, **options) for starts in batches))
            assert backend.completed == 3 and not backend.fallbacks
            assert backend.batched_searches == (3 if batched else 0)
            for actual, reference in zip(result, expected, strict=True):
                assert len(actual.branches) == len(reference.branches) == 2
                assert actual.branches[0].execution.proposals[0] is actual.branches[1].execution.proposals[0]
                for a, b in zip(actual.branches, reference.branches, strict=True):
                    assert a.route == b.route
                    _compare_proposals(a.execution.proposals, b.execution.proposals)
                composed = compose_participant_proposals(initial, [b.execution.proposals for b in actual.branches])
                native = compose_participant_proposals(initial, [b.execution.proposals for b in reference.branches])
                assert composed.shared_references == 1 and composed.state.revision(first.bank_slot_ref) == 3
                torch.testing.assert_close(composed.state.values, native.state.values)
            saved = [tuple(v.clone() for v in b.execution.proposals[-1].transition.effect.operands)
                     for r in result for b in r.branches]
            search_recursive_graphs_many(
                [(start_recursive_search(query, {"x": v["x"] + 0.1}, bank_state=initial),) for v in inputs],
                **options,
            )
            for snapshot, branch in zip(saved, [b for r in result for b in r.branches], strict=True):
                torch.testing.assert_close(snapshot, branch.execution.proposals[-1].transition.effect.operands, rtol=0, atol=0)
    for value, source in zip(inputs, result, strict=True):
        replayed, composed = replay_participant_graphs(query, value, source.branches, bank_state=initial,
                                                      use_recorded_choices=True)
        assert composed.shared_references == 1 and len(composed.occurrences) == 3
        torch.testing.assert_close(replayed[0].execution.outputs, source.branches[0].execution.outputs)
        composed.state.value(first.bank_slot_ref).square().mean().backward()
    assert not first.bank_owner.value.requires_grad


@pytest.mark.parametrize("batched", (False, True))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_typed_shape_changing_child_tape_matches_native(batched, dtype):
    from test_formula_device_typed_search import _query

    with torch.device("cuda"), torch.no_grad():
        query = _query(encoded=True).to(dtype=dtype)
        initial = query.initial_bank_state()
        batches = [(start_recursive_search(query, {"x": torch.randn(1, 2, 3, dtype=dtype)}, bank_state=initial),) for _ in range(2)]
        options = dict(width=4, beam_width=4)
        native = search_recursive_graphs_many(batches, **options)
        with captured_search_execution(horizon=12, record_effect_operands=True,
                                       value_samples=(torch.zeros(1, 3, dtype=dtype), torch.zeros(1, 2, 3, dtype=dtype))) as backend:
            actual = (search_recursive_graphs_many(batches, **options) if batched else
                      tuple(search_recursive_graphs(starts, **options) for starts in batches))
            assert backend.completed == 2 and not backend.fallbacks
            shapes = set()
            for a, b in zip(actual, native, strict=True):
                for branch, reference in zip(a.branches, b.branches, strict=True):
                    _compare_proposals(branch.execution.proposals, reference.execution.proposals)
                    shapes.update(tuple(p.transition.effect.operands[0].shape) for p in branch.execution.proposals)
                composed = compose_participant_proposals(initial, [branch.execution.proposals for branch in a.branches])
                expected = compose_participant_proposals(initial, [branch.execution.proposals for branch in b.branches])
                torch.testing.assert_close(composed.state.values, expected.state.values)
            assert shapes == {(1, 3), (1, 2, 3)}


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_optional_tape_in_direct_episode_batch_matches_independent_waves(device):
    from torch.utils._pytree import tree_map
    from test_formula_device_search import _query, _runtime

    with torch.device(device), torch.no_grad():
        wave, state, data, bank, dc, bc, finite = _runtime(_query())
        tape = wave.execution.dispatch.allocate_effect_tape(8, 2 * wave.width, data, bank)
        state, data, bank, dc, bc, tape = tree_map(lambda value: torch.stack((value, value)),
                                                 (state, data, bank, dc, bc, tape))
        data[1] *= -2.0
        expected = tree_map(torch.clone, (state, data, bank, dc, bc, tape))
        positions = torch.tensor([[0], [1]], dtype=torch.int64)
        for _ in range(5):
            actual = wave.forward_batch(state, data, bank, dc, bc, finite, effect_tape=tape,
                                        effect_tape_position=positions)
            previous, rd, rb, rdc, rbc, rtape = expected
            references = [wave(tree_map(lambda v: v[i], previous), rd[i], rb[i], rdc[i], rbc[i], finite,
                               effect_tape=tree_map(lambda v: v[i], rtape), effect_tape_position=positions[i])
                          for i in range(2)]
            reference = tree_map(lambda *values: torch.stack(values), *references)
            torch.testing.assert_close(actual, reference)
            for a, b in zip(tape, rtape, strict=True):
                torch.testing.assert_close(a[0], b[0])
                valid = a[0] >= 0
                for av, bv in zip(a[1:], b[1:], strict=True):
                    torch.testing.assert_close(av[valid], bv[valid])
            state, dc, bc = actual.state, actual.data_cursor, actual.bank_cursor
            expected = (reference.state, rd, rb, reference.data_cursor, reference.bank_cursor, rtape)
            positions += 1


def test_tape_disabled_is_empty_and_capture_key_separates_enabled_mode():
    query, *_ = _fork_query("cuda")
    starts = (start_recursive_search(query, {"x": torch.ones(1, 3, device="cuda")}),)
    options = dict(width=2, beam_width=4, preserve_effect_coverage=False)
    with torch.no_grad(), captured_search_execution(horizon=8) as backend:
        first = search_recursive_graphs(starts, **options)
        session = next(iter(backend.sessions.values()))
        assert session.effect_tape == ()
        assert all(p.transition is None for b in first.branches for p in b.execution.proposals)
        backend.record_effect_operands = True
        second = search_recursive_graphs(starts, **options)
        assert len(backend.sessions) == 2 and not backend.fallbacks
        assert all(p.transition is not None for b in second.branches for p in b.execution.proposals)
        assert first.winner.route == second.winner.route
        torch.testing.assert_close(first.winner.execution.outputs, second.winner.execution.outputs, rtol=0, atol=0)
