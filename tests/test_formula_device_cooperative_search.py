import pytest
import torch

from arti import mechanisms as m
from arti._formula_device_cooperative_search import FormulaDeviceCooperativeBeam, FormulaDeviceCooperativeSearchWave
from benchmarks._federated_recursive_search import search_cooperative_graphs, start_recursive_search
from test_formula_completed_products import sharing_graph
from test_formula_device_cooperative import _runtime as round_runtime
from test_formula_program_query_v7 import graph


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with torch.device(request.param):
        yield request.param


def runtime(query, *, width=1, heads=2, product_slots=("shared",), publish_slots=("p",), typed_first=None,
            product_bindings=None):
    local, frames, data, bank, dc, bc = round_runtime(query, typed_first=typed_first)
    prototype = data if typed_first is None else data[1]
    wave = FormulaDeviceCooperativeSearchWave(query, local.execution, width=width, head_width=heads,
        product_slots=product_slots, publish_slots=publish_slots,
        product_bindings=product_bindings).to(prototype.device)
    frames = wave._rows(frames, torch.zeros(2 * width, dtype=torch.int64))
    frames.active[1:] = False
    state = FormulaDeviceCooperativeBeam(frames, prototype.new_zeros(2 * width),
        torch.full((2 * width, 16), -1, dtype=torch.int64), torch.zeros(2 * width, dtype=torch.int64),
        torch.full((2 * width,), prototype.dtype == torch.float64, dtype=torch.bool))
    directory = wave.empty_directory(device=prototype.device)
    return wave, state, directory, torch.tensor(0), torch.tensor(0), data, bank, dc, bc


def advance(wave, state, directory, cursor, occurrence, data, bank, dc, bc):
    return wave(state, directory, cursor, occurrence, data, bank, dc, bc,
                torch.ones(wave.kernel.candidate_count, dtype=torch.bool))


def test_unstopped_donor_survives_pruning_and_feeds_shared_join(device):
    query = sharing_graph().to(device)
    wave, state, directory, cursor, occurrence, data, bank, dc, bc = runtime(query)
    native = search_cooperative_graphs((start_recursive_search(query, {"x": data[0]}),),
        product_slots=("shared",), publish_slots=("p",), width=2, beam_width=1)
    attempted = completed = 0
    donor_handle = None
    for step in range(4):
        result = advance(wave, state, directory, cursor, occurrence, data, bank, dc, bc)
        assert not result.requires_fallback
        attempted += int(result.numeric_attempts)
        completed += int(result.numeric_completed)
        state, directory, cursor, occurrence = (result.state, result.directory,
                                                result.directory_cursor, result.occurrence_cursor)
        dc, bc = result.data_cursor, result.bank_cursor
        assert cursor == 1
        assert directory.producer[0] == query.action_ids.index("z_donor")
        assert state.frames.value_handles[0, 0, query.slot_ids.index("shared")] == -1
        if step == 0:
            donor_handle = int(directory.handles[0])
            assert state.frames.active.tolist() == [True, False]
            assert state.routes[0, 0] == wave.action_ranks[query.action_ids.index("a_receiver")]
            assert result.numeric_completed == 2
        assert directory.handles[0] == donor_handle
    assert attempted == completed == 5
    assert occurrence == 6  # donor and receiver, c/d, final, STOP
    assert state.frames.completed.tolist() == [False, True]
    assert state.frames.frame_tensor_steps[1, 0] == 4
    names = tuple(sorted(query.action_ids)[i] for i in state.routes[1, :state.lengths[1]].tolist())
    assert names == ("a_receiver", "c", "d", "final", "stop")
    output = data[state.frames.value_handles[1, 0, query.slot_ids.index("y")]]
    torch.testing.assert_close(output, native.winner.execution.outputs["y"])
    torch.testing.assert_close(output, 20 * data[0])
    torch.testing.assert_close(state.scores[1], native.winner.log_probability)
    # Carrying the completed answer neither republishes it nor executes STOP.
    again = advance(wave, state, directory, cursor, occurrence, data, bank, dc, bc)
    assert again.occurrence_cursor == occurrence and again.data_cursor == dc
    assert again.numeric_attempts == 0
    assert again.state.frames.completed.tolist() == [False, True]


def test_no_publication_does_not_manufacture_receiver_input(device):
    query = sharing_graph().to(device)
    wave, state, directory, cursor, occurrence, data, bank, dc, bc = runtime(query, publish_slots=())
    for _ in range(4):
        result = advance(wave, state, directory, cursor, occurrence, data, bank, dc, bc)
        state, directory, cursor, occurrence = (result.state, result.directory,
                                                result.directory_cursor, result.occurrence_cursor)
        dc, bc = result.data_cursor, result.bank_cursor
    assert cursor == 0 and not directory.ready.any()
    assert not state.frames.completed.any()
    assert (state.frames.value_handles[:, 0, query.slot_ids.index("y")] == -1).all()


@pytest.mark.parametrize("width", [1, 3])
def test_no_directory_graph_matches_reference_beam(device, width):
    query = graph().to(device)
    wave, state, directory, cursor, occurrence, data, bank, dc, bc = runtime(
        query, width=width, heads=3, product_slots=(), publish_slots=())
    native = search_cooperative_graphs((start_recursive_search(query, {"x": data[0]}),),
                                      product_slots=(), width=3, beam_width=width)
    for _ in range(8):
        result = advance(wave, state, directory, cursor, occurrence, data, bank, dc, bc)
        assert not result.requires_fallback
        state, directory, cursor, occurrence = (result.state, result.directory,
                                                result.directory_cursor, result.occurrence_cursor)
        dc, bc = result.data_cursor, result.bank_cursor
    rows = state.frames.completed.nonzero().flatten().tolist()
    assert len(rows) == len(native.branches)
    for row, branch in zip(rows, native.branches, strict=True):
        names = tuple(sorted(query.action_ids)[i] for i in state.routes[row, :state.lengths[row]].tolist())
        assert names == tuple(item["candidate_id"] for item in branch.route)
        torch.testing.assert_close(state.scores[row], branch.log_probability)
        torch.testing.assert_close(data[state.frames.value_handles[row, 0, query.slot_ids.index("y")]],
                                   branch.execution.outputs["y"])


def test_real_bank_effects_remain_on_receiver_path(device):
    from test_formula_program_query_v6 import federation

    old = federation(write=True, shared=True)
    query = m.FormulaProgramQueryV7(slot_ids=old.slot_ids, candidates=tuple(old.candidates),
        terminal_slots=old.terminal_slots, entry_candidates=old.entry_candidates,
        continuations=old.continuations, cooperation_width=2, max_steps=4).to(device)
    wave, state, directory, cursor, occurrence, data, bank, dc, bc = runtime(
        query, product_slots=(), publish_slots=(), width=2)
    native = search_cooperative_graphs((start_recursive_search(query, {"x": data[0]}),),
                                      product_slots=(), width=2, beam_width=2)
    for _ in range(6):
        result = advance(wave, state, directory, cursor, occurrence, data, bank, dc, bc)
        assert not result.requires_fallback
        state, directory, cursor, occurrence = (result.state, result.directory,
                                                result.directory_cursor, result.occurrence_cursor)
        dc, bc = result.data_cursor, result.bank_cursor
    rows = state.frames.completed.nonzero().flatten().tolist()
    assert len(rows) == len(native.branches)
    for row, branch in zip(rows, native.branches, strict=True):
        torch.testing.assert_close(data[state.frames.value_handles[row, 0, query.slot_ids.index("y")]],
                                   branch.execution.outputs["result"])
        torch.testing.assert_close(bank[state.frames.bank_value_handles[row, 0]],
                                   branch.arena.committed_state().values[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_whole_shared_search_capture_preserves_directory_across_pruning():
    with torch.device("cuda"):
        query = sharing_graph().cuda()
        wave, initial, empty, zero, first, data, bank, initial_dc, initial_bc = runtime(query)

        def run():
            state, directory, cursor, occurrence = initial, empty, zero, first
            dc, bc = initial_dc, initial_bc
            for _ in range(4):
                result = advance(wave, state, directory, cursor, occurrence, data, bank, dc, bc)
                state, directory, cursor, occurrence = (result.state, result.directory,
                                                        result.directory_cursor, result.occurrence_cursor)
                dc, bc = result.data_cursor, result.bank_cursor
            return result

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            result = run()
        pointers = tuple(field.data_ptr() for field in result.directory)
        for value in (1., 3., .25):
            data[0].fill_(value)
            capture.replay()
            assert not result.requires_fallback
            assert result.state.frames.completed.tolist() == [False, True]
            output = data[result.state.frames.value_handles[1, 0, query.slot_ids.index("y")]]
            torch.testing.assert_close(output, data.new_tensor([[20 * value]]))
            torch.testing.assert_close(data[result.directory.handles[0]], data.new_tensor([[4 * value]]))
            assert tuple(field.data_ptr() for field in result.directory) == pointers
            assert result.occurrence_cursor == 6


def test_same_revision_roots_keep_distinct_products_and_bank_state(device):
    from test_formula_program_query_v6 import member

    producer = member("producer", "x", "p", owner="memory")
    left = member("left", "shared0", "y")
    right = left.with_bindings("right", input_slots={"x": "shared1"}, output_slots=left.candidate.output_slots)
    query = m.FormulaProgramQueryV7(
        slot_ids=("x", "p", "p_negative", "shared0", "shared1", "y", "y_negative"),
        candidates=(producer, left, right), terminal_slots={"y": "y"}, entry_candidates=("producer",),
        continuations={"producer": {"left": "p", "right": "p_negative"}}, max_steps=2,
        cooperation_width=2,
    ).to(device)
    wave, state, directory, cursor, occurrence, data, bank, dc, bc = runtime(
        query, width=2, heads=1, product_slots=("shared0", "shared1"), publish_slots=("p",))
    state.frames.active[1] = True
    state.frames.bank_value_handles[1, 0] = 1
    bank[1] = -bank[0]
    bc.fill_(2)
    for _ in range(3):
        result = advance(wave, state, directory, cursor, occurrence, data, bank, dc, bc)
        assert not result.requires_fallback
        state, directory, cursor, occurrence = (result.state, result.directory,
                                                result.directory_cursor, result.occurrence_cursor)
        dc, bc = result.data_cursor, result.bank_cursor
    assert cursor == 2
    assert directory.occurrence.tolist() == [0, 1]
    assert directory.revision.tolist() == [0, 0]
    assert directory.bank_handle.tolist() == [0, 1]
    assert state.frames.completed.tolist() == [False, False, True, True]
    values = []
    for row in (2, 3):
        output = data[state.frames.value_handles[row, 0, query.slot_ids.index("y")]]
        values.append(float(output.item()))
        bank_handle = state.frames.bank_value_handles[row, 0]
        torch.testing.assert_close(output, bank[bank_handle] * data[0])
    assert sorted(values) == [-2., 2.]


def test_directory_append_is_ordered_bounded_and_does_not_copy_payload(device):
    from arti._formula_device_sources import FormulaDeviceSources, append_completed_sources

    query = sharing_graph().to(device)
    wave, _, directory, cursor, *_ = runtime(query)
    ids = torch.tensor([8, 9, 10, 11])
    ready = torch.tensor([False, True, False, True])
    products = FormulaDeviceSources(ids + 20, ready, ready, ids, ids, ids, ids, ids,
                                    torch.zeros_like(ids))
    result, position, dropped = append_completed_sources(directory, cursor, products)
    assert position == 1 and dropped == 1
    assert result.handles.tolist() == [29] and result.occurrence.tolist() == [9]
    repeated, position2, dropped2 = append_completed_sources(result, position, products)
    assert position2 == 1 and dropped2 == 2
    for first, last in zip(result, repeated, strict=True):
        assert torch.equal(first, last)


def test_batched_events_keep_independent_pools_and_directories(device):
    from torch.utils._pytree import tree_map

    query = sharing_graph().to(device)
    wave, *inputs = runtime(query)
    state, directory, cursor, occurrence, data, bank, dc, bc = tree_map(
        lambda tensor: torch.stack((tensor.clone(), tensor.clone())), tuple(inputs))
    data[0, 0] = 1.
    data[1, 0] = -3.
    finite = torch.ones(wave.kernel.candidate_count, dtype=torch.bool)
    for _ in range(4):
        result = wave.forward_batch(state, directory, cursor, occurrence, data, bank, dc, bc, finite)
        assert not result.requires_fallback.any()
        state, directory, cursor, occurrence = (result.state, result.directory,
                                                result.directory_cursor, result.occurrence_cursor)
        dc, bc = result.data_cursor, result.bank_cursor
    assert cursor.tolist() == [1, 1] and occurrence.tolist() == [6, 6]
    for episode, value in enumerate((20., -60.)):
        handle = state.frames.value_handles[episode, 1, 0, query.slot_ids.index("y")]
        torch.testing.assert_close(data[episode, handle], data.new_tensor([[value]]))


@pytest.mark.parametrize("prior_double", [False, True])
def test_unused_double_pool_preserves_path_score_precision_across_rounds(device, prior_double):
    from dataclasses import replace

    query = graph().to(device)
    wave, state, directory, cursor, occurrence, data, bank, dc, bc = runtime(
        query, width=3, heads=3, product_slots=(), publish_slots=(), typed_first=torch.float64)
    native_dtype = torch.float64 if prior_double else torch.float32
    initial_score = torch.tensor(-16777216., dtype=native_dtype)
    state = state._replace(scores=state.scores.double().fill_(initial_score),
                           score_is_double=state.score_is_double.fill_(prior_double))
    native = search_cooperative_graphs(
        (replace(start_recursive_search(query, {"x": data[1][0]}), log_probability=initial_score),),
        product_slots=(), width=3, beam_width=3)
    for _ in range(8):
        result = advance(wave, state, directory, cursor, occurrence, data, bank, dc, bc)
        assert not result.requires_fallback
        state, directory, cursor, occurrence = (result.state, result.directory,
                                                result.directory_cursor, result.occurrence_cursor)
        dc, bc = result.data_cursor, result.bank_cursor
        kept = state.frames.active | state.frames.completed
        assert (state.score_is_double[kept] == prior_double).all()
        if not prior_double:
            # Each real FP32 increment here is below the parent's rounding unit.
            torch.testing.assert_close(state.scores[kept], state.scores[kept].float().double(),
                                       rtol=0, atol=0)
    rows = state.frames.completed.nonzero().flatten().tolist()
    assert len(rows) == len(native.branches)
    for row, branch in zip(rows, native.branches, strict=True):
        names = tuple(sorted(query.action_ids)[i] for i in state.routes[row, :state.lengths[row]].tolist())
        assert names == tuple(item["candidate_id"] for item in branch.route)
        torch.testing.assert_close(state.scores[row], branch.log_probability.double(), rtol=0, atol=0)
