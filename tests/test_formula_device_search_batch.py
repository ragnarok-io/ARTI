import pytest
import torch
from torch.utils._pytree import tree_map

from tests.test_formula_device_search import _query, _runtime
from benchmarks._federated_captured_search import captured_search_execution
from benchmarks._federated_recursive_search import search_recursive_graphs, search_recursive_graphs_many, start_recursive_search


@pytest.fixture(params=["cpu", "cuda"], autouse=True)
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with torch.device(request.param):
        yield request.param


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("noise", [False, True])
def test_episode_batched_wave_matches_independent_searches(nested, noise):
    wave, state, data, bank, dc, bc, finite = _runtime(_query(nested=nested))
    state = tree_map(lambda x: torch.stack((x.clone(), x.clone(), x.clone())), state)
    data = torch.stack((data, data * 2, data * -3))
    bank = torch.stack((bank, bank + 2, bank - 1))
    dc, bc = dc.repeat(3), bc.repeat(3)
    expected = tree_map(torch.clone, (state, data, bank, dc, bc))
    if noise:
        torch.manual_seed(63)
    for _ in range(8):
        exploration = (torch.randn(3, wave.width, wave.kernel.candidate_count,
                                   device=data.device, dtype=torch.float64) if noise else None)
        actual = wave.forward_batch(state, data, bank, dc, bc, finite, exploration)
        old, reference_data, reference_bank, ref_dc, ref_bc = expected
        reference = [wave(tree_map(lambda t: t[i], old), reference_data[i], reference_bank[i],
                          ref_dc[i], ref_bc[i], finite, None if exploration is None else exploration[i])
                     for i in range(3)]
        reference = tree_map(lambda *t: torch.stack(t), *reference)
        torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(data, reference_data)
        torch.testing.assert_close(bank, reference_bank)
        state, dc, bc = actual.state, actual.data_cursor, actual.bank_cursor
        expected = (reference.state, reference_data, reference_bank, reference.data_cursor, reference.bank_cursor)
    assert not state.frames.active.any()
    assert state.frames.completed.any(1).all()


def test_episode_failure_does_not_reject_other_episodes():
    wave, state, data, bank, dc, bc, finite = _runtime(_query())
    state = tree_map(lambda x: torch.stack((x, x)), state)
    data, bank = torch.stack((data, data)), torch.stack((bank, bank))
    dc, bc = dc.repeat(2), bc.repeat(2)
    dc[1] = wave.execution.data_capacity
    result = wave.forward_batch(state, data, bank, dc, bc, finite)
    assert result.requires_fallback.tolist() == [False, True]
    torch.testing.assert_close(result.state.frames.value_handles[1], state.frames.value_handles[1])
    assert result.data_cursor[0] > dc[0]
    assert result.data_cursor[1] == dc[1]


@pytest.mark.parametrize("explore", [False, True])
def test_captured_episode_batch_matches_single_and_refreshes_live_banks(device, explore):
    if device == "cpu":
        return
    query = _query(nested=True)
    starts = [(start_recursive_search(query, {"x": torch.full((1, 3), float(i + 1))}),) for i in range(3)]

    def generators():
        return [torch.Generator(device="cpu").manual_seed(103 + i) if explore else None for i in range(3)]

    with torch.no_grad(), captured_search_execution(horizon=16) as backend:
        single = [search_recursive_graphs(b, width=4, beam_width=4, record_effect_metrics=True,
                                          record_query_choices=True, exploration_generator=g)
                  for b, g in zip(starts, generators(), strict=True)]
        actual = search_recursive_graphs_many(starts, width=4, beam_width=4, record_effect_metrics=True,
                                              record_query_choices=True, exploration_generators=generators())
        assert backend.batched_searches == 3
        assert backend.batch_invocations == 1
        assert not backend.fallbacks
        for expected, result in zip(single, actual, strict=True):
            assert len(expected.branches) == len(result.branches)
            for left, right in zip(expected.branches, result.branches, strict=True):
                assert left.route == right.route
                torch.testing.assert_close(left.log_probability, right.log_probability)
                torch.testing.assert_close(left.execution.outputs, right.execution.outputs)
                assert left.execution.bank_state.revisions == right.execution.bank_state.revisions
                torch.testing.assert_close(left.execution.bank_state.values, right.execution.bank_state.values)
        # A second event reads the previous hard winner's own Bank, not a cache
        # of the initial model state or the neighboring episode's history.
        continuation = [(start_recursive_search(query, {"x": torch.full((1, 3), float(7 - i))},
                                                 bank_state=r.winner.execution.bank_state),)
                        for i, r in enumerate(actual)]
        again = search_recursive_graphs_many(continuation, width=4, beam_width=4)
        expected = [search_recursive_graphs(b, width=4, beam_width=4) for b in continuation]
        for a, b in zip(again, expected, strict=True):
            assert a.winner.route == b.winner.route
            torch.testing.assert_close(a.winner.execution.bank_state.values, b.winner.execution.bank_state.values)
        for expected, result in zip(single, actual, strict=True):
            torch.testing.assert_close(expected.winner.execution.outputs, result.winner.execution.outputs)
    assert not backend.sessions


def test_episode_batch_partitions_shapes_and_keeps_native_shape_changes(device):
    if device == "cpu":
        return
    from tests.test_formula_device_search import _shape_query

    query, changing = _shape_query(), _shape_query(changing=True)
    batches = [(start_recursive_search(query, {"x": torch.ones(1, n, 3)}),) for n in (1, 2, 1, 2)]
    batches += [(start_recursive_search(changing, {"x": torch.ones(1, 2, 3)}),)] * 2
    with torch.no_grad(), captured_search_execution(horizon=8) as backend:
        result = search_recursive_graphs_many(batches, width=4, beam_width=4)
        assert [len(row.branches) for row in result] == [1, 2, 1, 2, 2, 2]
        assert backend.batched_searches == 4
        assert backend.fallbacks == ["heterogeneous Formula binding bucket"]
        assert result[-1].winner.execution.outputs["answer"].shape == (1, 3)


def test_episode_batch_rejects_shared_exploration_generator():
    query = _query()
    starts = (start_recursive_search(query, {"x": torch.ones(1, 3)}),)
    generator = torch.Generator(device="cpu").manual_seed(34)
    with pytest.raises(ValueError, match="independent exploration generators"):
        search_recursive_graphs_many((starts, starts), exploration_generators=(generator, generator))
