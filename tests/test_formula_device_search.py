import pytest
import torch

from arti import mechanisms as m
from arti._formula_device_dispatch import FormulaDeviceNumericalDispatch, _candidate_records
from arti._formula_device_execution import FormulaDeviceExecutionWave
from arti._formula_device_frames import FormulaDeviceFrameKernel
from arti._formula_device_search import FormulaDeviceSearchState, FormulaDeviceSearchWave
from benchmarks._federated_recursive_search import search_recursive_graphs, start_recursive_search


@pytest.fixture(params=["cpu", "cuda"], autouse=True)
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with torch.device(request.param):
        yield request.param


def _query(*, nested=False):
    type_ = m.TensorType(("B", "D"), ("B", 3), dtype="floating", domain="activation")
    x = m.InputBinding("x", type_)
    value = m.BankBinding("value", "arti/device-search-test@1", "value", type_)
    candidates = []
    for name, weight in (("a", 2.0), ("b", 3.0)):
        candidates.append(m.FormulaProgramTensorCandidateV3(
            m.FormulaProgramCandidateV2(
                name, m.FormulaProgram.build(outputs=(m.scale(x, value),)),
                input_slots={"x": "x"}, output_slot="owned",
                operands={"value": torch.full((1, 3), weight)},
            ), plastic_bank_slot="value", bank_owner_id=name,
        ))
    rate = m.BankBinding("rate", "arti/device-search-test@1", "rate", type_)
    zero = m.BankBinding("zero", "arti/device-search-test@1", "zero", type_)
    candidates.append(m.FormulaProgramEffectCandidateV3(
        "write", m.FormulaEffectProgramV2(
            m.FormulaProgram.build(outputs=(m.neural_plasticity(x, m.scale(x, rate), zero),)),
            data_input_name="x", state_type=type_,
        ), input_slot="owned", output_slot="tail",
        operands={"rate": torch.full((1, 3), 0.1), "zero": torch.zeros(1, 3)},
    ))
    query = m.FormulaProgramQueryV5(
        slot_ids=("x", "owned", "tail"), candidates=tuple(candidates),
        terminal_slots={"answer": "owned"}, max_steps=2, hidden_dim=8,
    )
    if nested:
        query = m.FormulaProgramQueryV5(
            slot_ids=("x", "out"),
            candidates=tuple(m.FormulaProgramCallCandidateV1(
                name, query, input_slots={"x": "x"}, output_slots={"answer": "out"},
            ) for name in ("call-a", "call-b")),
            terminal_slots={"answer": "out"}, max_steps=1, hidden_dim=8,
        )
    with torch.no_grad():
        # Keep Formula operands intact; equal Query scores exercise route ties.
        for module in query.modules():
            if isinstance(module, m.FormulaProgramQueryV5):
                for parameter in module.network.parameters():
                    parameter.zero_()
    return query


def _runtime(query, *, width=4, local_width=4, preserve_coverage=True):
    kernel = FormulaDeviceFrameKernel.from_query(query)
    dispatch = FormulaDeviceNumericalDispatch.from_query(query, frame_kernel=kernel)
    execution = FormulaDeviceExecutionWave(dispatch, kernel, data_capacity=128, bank_capacity=128)
    families = [2] * kernel.candidate_count
    members = [[False] for _ in families]
    for global_id, _, candidate in _candidate_records(query):
        effect = isinstance(candidate, m.FormulaProgramEffectCandidateV3)
        families[global_id] = 1 if effect else 0
        members[global_id] = [effect]
    wave = FormulaDeviceSearchWave(
        query, execution, width=width, local_width=local_width,
        candidate_family_ids=families, candidate_membership=members,
        preserve_coverage=preserve_coverage,
    ).to(next(query.parameters()).device)
    handles = torch.full((2 * width, len(query.slot_ids)), -1, dtype=torch.int64)
    handles[:, 0] = 0
    frames = kernel.initial_state(2 * width, handles, bank_value_handles=torch.arange(2))
    frames.active[1:] = False
    state = FormulaDeviceSearchState(
        frames, torch.zeros(2 * width), torch.zeros(2 * width),
        torch.full((2 * width, 12), -1, dtype=torch.int64),
        torch.zeros(2 * width, dtype=torch.int64), torch.zeros((2 * width, 1), dtype=torch.bool),
    )
    data = torch.zeros((129, 1, 3))
    data[0] = 1
    bank = torch.zeros_like(data)
    bank[0] = 2
    bank[1] = 3
    return wave, state, data, bank, torch.tensor(1), torch.tensor(2), torch.ones(len(families), dtype=torch.bool)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("coverage", [False, True])
def test_complete_device_beam_matches_recursive_search(nested, coverage):
    query = _query(nested=nested)
    wave, state, data, bank, dc, bc, finite = _runtime(query, preserve_coverage=coverage)
    with torch.no_grad():
        native = search_recursive_graphs(
            (start_recursive_search(query, {"x": data[0]}),), width=4, beam_width=4,
            preserve_effect_coverage=coverage,
        )
        for _ in range(8):
            result = wave(state, data, bank, dc, bc, finite)
            assert not bool(result.requires_fallback)
            state, dc, bc = result.state, result.data_cursor, result.bank_cursor
    assert not bool(state.frames.active.any())
    rows = torch.nonzero(state.frames.completed).flatten().tolist()
    assert len(rows) == len(native.branches)
    for row, branch in zip(rows, native.branches, strict=True):
        route = tuple(wave.execution_names[index] for index in state.routes[row, :state.lengths[row]].tolist())
        assert route == tuple(step["candidate_id"] for step in branch.route)
        torch.testing.assert_close(state.scores[row], branch.log_probability)
        output_slot = query.slot_ids.index(query.terminal_slots["answer"])
        output = data[state.frames.value_handles[row, 0, output_slot]]
        torch.testing.assert_close(output, branch.execution.outputs["answer"])
        for bank_index in range(2):
            actual = bank[state.frames.bank_value_handles[row, bank_index]]
            native_value = branch.execution.bank_state.values[bank_index]
            torch.testing.assert_close(actual, native_value)
            assert state.frames.bank_revisions[row, bank_index].item() == branch.execution.bank_state.revisions[bank_index]


def test_noise_changes_selection_not_model_scores():
    query = _query()
    wave, state, data, bank, dc, bc, finite = _runtime(
        query, width=1, local_width=1, preserve_coverage=False,
    )
    ordinary = wave(state, data.clone(), bank.clone(), dc, bc, finite)
    noise = torch.zeros((1, wave.kernel.candidate_count))
    noise[0, 1] = 10
    explored = wave(state, data.clone(), bank.clone(), dc, bc, finite, noise)
    assert ordinary.selected_candidates[0].item() == 0
    assert explored.selected_candidates[0].item() == 1
    torch.testing.assert_close(ordinary.state.scores[0], explored.state.scores[0])
    assert not bool(explored.requires_fallback)


def test_search_signals_overflow_instead_of_publishing_partial_beam():
    wave, state, data, bank, _dc, bc, finite = _runtime(_query())
    result = wave(state, data, bank, torch.tensor(128), bc, finite)
    assert bool(result.requires_fallback)
    assert result.data_cursor.item() == 128
    assert torch.equal(state.frames.bank_value_handles, result.state.frames.bank_value_handles)


def test_no_legal_survivor_requests_native_fallback():
    wave, state, data, bank, dc, bc, finite = _runtime(_query())
    state.frames.value_handles.fill_(-1)
    result = wave(state, data, bank, dc, bc, finite)
    assert bool(result.requires_fallback)
    assert torch.equal(result.state.frames.active, state.frames.active)


def test_numerical_failure_keeps_entry_handles_and_cursors():
    wave, state, data, bank, dc, bc, finite = _runtime(_query())
    # Both inputs are finite, but the selected scale overflows during execution.
    data[0].fill_(torch.finfo(data.dtype).max)
    before = tuple(t.clone() for t in (*state.frames, *state[1:], dc, bc))
    result = wave(state, data, bank, dc, bc, finite)
    assert bool(result.requires_fallback)
    for actual, expected in zip((*result.state.frames, *result.state[1:], result.data_cursor, result.bank_cursor), before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_complete_search_can_replay_with_cuda_feedback(device):
    if device == "cpu":
        return
    wave, state, data, bank, dc, bc, finite = _runtime(_query(nested=True))
    original = tuple(t.clone() for t in (*state.frames, state.scores, state.priorities, state.routes,
                                       state.lengths, state.membership, data, bank, dc, bc))

    def copy_result(result):
        for dst, src in zip(state.frames, result.state.frames, strict=True):
            dst.copy_(src)
        for dst, src in zip(state[1:], result.state[1:], strict=True):
            dst.copy_(src)
        dc.copy_(result.data_cursor)
        bc.copy_(result.bank_cursor)

    def restore():
        for dst, src in zip((*state.frames, *state[1:], data, bank, dc, bc), original, strict=True):
            dst.copy_(src)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        copy_result(wave(state, data, bank, dc, bc, finite))
    torch.cuda.current_stream().wait_stream(stream)
    restore()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = wave(state, data, bank, dc, bc, finite)
        copy_result(result)
    restore()
    for _ in range(8):
        graph.replay()
    torch.cuda.synchronize()
    assert not bool(result.requires_fallback)
    assert not bool(state.frames.active.any())
    assert int(state.frames.completed.sum()) == 4


@pytest.mark.parametrize("exploration_seed", [None, 193])
def test_captured_adapter_preserves_episode_routes_and_assets(device, exploration_seed):
    if device == "cpu":
        return
    from benchmarks._federated_captured_search import captured_search_execution
    from benchmarks._federated_episode_beam import search_episode_beam

    query = _query(nested=True)
    inputs = ({"x": torch.ones(1, 3)}, {"x": torch.full((1, 3), 1.5)})
    expected = search_episode_beam(query, inputs, width=4)
    with captured_search_execution(horizon=16) as backend:
        actual = search_episode_beam(query, inputs, width=4, exploration_seed=exploration_seed)
        saved = actual.result.winner.execution.outputs["answer"].clone()
        repeated = search_episode_beam(query, inputs, width=4, exploration_seed=exploration_seed)
        assert not backend.fallbacks
    assert not backend.sessions
    torch.testing.assert_close(saved, actual.result.winner.execution.outputs["answer"], rtol=0, atol=0)
    for first, second in zip(actual.result.branches, repeated.result.branches, strict=True):
        assert first.route == second.route
        torch.testing.assert_close(first.log_probability, second.log_probability, rtol=0, atol=0)
    if exploration_seed is None:
        assert actual.event_searches == expected.event_searches
        for wanted, got in zip(expected.result.branches, actual.result.branches, strict=True):
            assert got.route == wanted.route
            assert got.execution.trace == wanted.execution.trace
            assert got.execution.bank_state.revisions == wanted.execution.bank_state.revisions
            torch.testing.assert_close(got.log_probability, wanted.log_probability)
            torch.testing.assert_close(got.execution.outputs["answer"], wanted.execution.outputs["answer"])
            for a, b in zip(got.execution.bank_state.values, wanted.execution.bank_state.values, strict=True):
                torch.testing.assert_close(a, b)


def test_adapter_refresh_and_effect_receipts(device):
    if device == "cpu":
        return
    from benchmarks._federated_captured_search import captured_search_execution

    query = _query(nested=True)
    x = torch.ones(1, 3)

    def run():
        return search_recursive_graphs((start_recursive_search(query, {"x": x}),), width=4, beam_width=4,
                                       record_query_choices=True, record_effect_metrics=True)

    with torch.no_grad():
        expected = run()
        with captured_search_execution(horizon=16) as backend:
            first = run()
            query.network[-1].bias[1] = 4
            second = run()
            assert len(backend.sessions) == 1
            assert not backend.fallbacks
        updated = run()
    for got, wanted in ((first, expected), (second, updated)):
        for a, b in zip(got.branches, wanted.branches, strict=True):
            assert a.route == b.route
            torch.testing.assert_close(a.log_probability, b.log_probability)
    assert first.winner.route[0]["candidate_id"] != second.winner.route[0]["candidate_id"]


def test_captured_session_eviction_releases_graph(device):
    if device == "cpu":
        return
    from benchmarks._federated_captured_search import captured_search_execution

    with torch.no_grad(), captured_search_execution(horizon=16, max_sessions=1) as backend:
        query = _query()
        first = search_recursive_graphs((start_recursive_search(query, {"x": torch.ones(1, 3)}),),
                                        width=4, beam_width=4)
        session = next(iter(backend.sessions.values()))
        assert session.graph is not None
        query2 = _query(nested=True)
        search_recursive_graphs((start_recursive_search(query2, {"x": torch.ones(1, 3)}),),
                                width=4, beam_width=4)
        assert len(backend.sessions) == 1
        assert session.graph is None
        assert not backend.fallbacks
    assert not backend.sessions
    assert torch.isfinite(first.winner.execution.outputs["answer"]).all()


def test_captured_start_pruning_keeps_exploration_priority(device):
    if device == "cpu":
        return
    from dataclasses import replace
    from benchmarks._federated_captured_search import captured_search_execution

    query = _query()
    start = start_recursive_search(query, {"x": torch.ones(1, 3)})
    starts = (
        replace(start, log_probability=torch.tensor(0.), selection_priority=torch.tensor(-10.),
                route=({"candidate_id": "history-a", "kind": "stop"},)),
        replace(start, log_probability=torch.tensor(-5.), selection_priority=torch.tensor(10.),
                route=({"candidate_id": "history-b", "kind": "stop"},)),
    )
    with torch.no_grad(), captured_search_execution(horizon=16) as backend:
        result = search_recursive_graphs(starts, width=1, beam_width=1, preserve_effect_coverage=False,
                                         exploration_generator=torch.Generator(device="cpu").manual_seed(9))
        assert backend.completed == 1
        assert not backend.fallbacks
    assert result.winner.route[0]["candidate_id"] == "history-b"


def test_captured_native_only_candidate_falls_back(device):
    if device == "cpu":
        return
    from benchmarks._federated_captured_search import captured_search_execution

    class WrappedTensorCandidate(m.FormulaProgramTensorCandidateV3):
        pass

    ordinary = _query().candidates[0]
    wrapped = WrappedTensorCandidate(ordinary.candidate, plastic_bank_slot="value", bank_owner_id="wrapped")
    query = m.FormulaProgramQueryV5(slot_ids=("x", "owned"), candidates=(wrapped,),
                                   terminal_slots={"answer": "owned"}, max_steps=1, hidden_dim=8)
    start = start_recursive_search(query, {"x": torch.ones(1, 3)})
    with torch.no_grad():
        expected = search_recursive_graphs((start,), width=1, beam_width=1)
        with captured_search_execution(horizon=16) as backend:
            actual = search_recursive_graphs((start,), width=1, beam_width=1)
            assert backend.fallbacks == ["native-only candidate graph"]
            assert not backend.sessions
    assert actual.winner.route == expected.winner.route
    torch.testing.assert_close(actual.winner.execution.outputs["answer"], expected.winner.execution.outputs["answer"])


def _shape_query(*, changing=False):
    candidates = []
    for name, length in (("dynamic", "N"), ("two-rows", 2)):
        value_type = m.TensorType(("B", "N", "D"), ("B", length, 3), dtype="floating", domain="activation")
        value = m.InputBinding("x", value_type)
        output = m.reduce_sum(value, axis="N") if changing else m.add(value, value)
        candidates.append(m.FormulaProgramTensorCandidateV3(m.FormulaProgramCandidateV2(
            name, m.FormulaProgram.build(outputs=(output,)), input_slots={"x": "x"}, output_slot="out",
        )))
    query = m.FormulaProgramQueryV5(slot_ids=("x", "out"), candidates=tuple(candidates),
                                   terminal_slots={"answer": "out"}, max_steps=1, hidden_dim=8)
    with torch.no_grad():
        for parameter in query.network.parameters():
            parameter.zero_()
    return query


def test_shape_bucket_keeps_all_legal_candidates(device):
    if device == "cpu":
        return
    from benchmarks._federated_captured_search import captured_search_execution

    query = _shape_query()
    with torch.no_grad(), captured_search_execution(horizon=8) as backend:
        for length in (1, 2, 1):
            start = start_recursive_search(query, {"x": torch.ones(1, length, 3)})
            actual = search_recursive_graphs((start,), width=4, beam_width=4)
            assert len(actual.branches) == (1 if length == 1 else 2)
            session = next(reversed(backend.sessions.values()))
            assert session.wave.kernel.candidate_count == 3
            assert bool(session.wave.execution.dispatch.shape_excluded_groups) == (length == 1)
            # Temporarily leave the context, without changing or rebuilding Query.
            from benchmarks._federated_captured_search import _CAPTURED_SEARCH
            token = _CAPTURED_SEARCH.set(None)
            try:
                expected = search_recursive_graphs((start,), width=4, beam_width=4)
            finally:
                _CAPTURED_SEARCH.reset(token)
            for a, b in zip(actual.branches, expected.branches, strict=True):
                assert a.route == b.route
                torch.testing.assert_close(a.log_probability, b.log_probability, rtol=0, atol=0)
                torch.testing.assert_close(a.execution.outputs["answer"], b.execution.outputs["answer"], rtol=0, atol=0)
        assert backend.completed == 3
        assert not backend.fallbacks
        assert len(backend.sessions) == 2


def test_legal_shape_change_uses_native_without_removing_candidate(device):
    if device == "cpu":
        return
    from benchmarks._federated_captured_search import captured_search_execution

    query = _shape_query(changing=True)
    start = start_recursive_search(query, {"x": torch.ones(1, 2, 3)})
    with torch.no_grad(), captured_search_execution(horizon=8) as backend:
        result = search_recursive_graphs((start,), width=4, beam_width=4)
        assert backend.completed == 0
        assert backend.fallbacks
    assert len(result.branches) == 2
    assert result.winner.execution.outputs["answer"].shape == (1, 3)
    torch.testing.assert_close(result.winner.execution.outputs["answer"], torch.full((1, 3), 2.))
