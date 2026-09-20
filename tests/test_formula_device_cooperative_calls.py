import pytest
import torch

from arti import mechanisms as m
from benchmarks._federated_recursive_search import search_cooperative_graphs, start_recursive_search
from test_formula_device_cooperative_search import runtime, advance
from test_formula_program_query_v6 import federation, member
from test_formula_program_query_v7 import cooperative, graph, wrap_child


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with torch.device(request.param):
        yield request.param


def execute(query, *, rounds=3, **kwargs):
    wave, state, directory, cursor, occurrence, data, bank, dc, bc = runtime(query, **kwargs)
    records = []
    for _ in range(rounds):
        result = advance(wave, state, directory, cursor, occurrence, data, bank, dc, bc)
        assert not result.requires_fallback
        records.append(result)
        state, directory, cursor, occurrence = (result.state, result.directory,
                                                result.directory_cursor, result.occurrence_cursor)
        dc, bc = result.data_cursor, result.bank_cursor
    return wave, records, data, bank


@pytest.mark.parametrize("nested", [False, True])
def test_complete_call_preserves_child_cooperation_and_root_score(device, nested):
    query = wrap_child(wrap_child(graph())) if nested else wrap_child(graph())
    query = query.to(device)
    wave, records, data, bank = execute(query, product_slots=(), publish_slots=(), heads=2)
    expected = query({"x": data[0]})
    first, last = records[0], records[-1]
    assert first.state.frames.depth[0] == 0
    assert first.state.frames.frame_tensor_steps[0, 0] == 1
    assert first.call_dispatches == first.call_returns == (2 if nested else 1)
    assert first.numeric_completed == 5
    assert first.occurrence_cursor == 1  # Root CALL, not its child decisions.
    assert last.occurrence_cursor == 2  # Root STOP only once.
    assert last.state.frames.completed.tolist() == [False, True]
    for name, slot in query.terminal_slots.items():
        actual = data[last.state.frames.value_handles[1, 0, query.slot_ids.index(slot)]]
        torch.testing.assert_close(actual, expected.outputs[name])
    torch.testing.assert_close(last.state.scores[1], sum(f.selection_log_score for f in expected.frontiers))
    assert any(step.accepted.any() and (step.candidates >= 0).sum(-1).max() == 2 for step in first.call_steps)
    assert sum(int(step.accepted.sum()) for step in first.call_steps) == (6 if nested else 4)
    assert all(record.numeric_attempts == 0 for record in records[1:])


@pytest.mark.parametrize("child_v7", [False, True])
def test_child_effect_updates_real_bank_before_return(device, child_v7):
    child = federation(write=True, shared=True)
    child = cooperative(child) if child_v7 else child
    query = wrap_child(child).to(device)
    wave, records, data, bank = execute(query, product_slots=(), publish_slots=())
    native = query({"x": data[0]})
    state = records[-1].state
    assert state.frames.completed[1]
    for index, expected in enumerate(native.bank_state.values):
        torch.testing.assert_close(bank[state.frames.bank_value_handles[1, index]], expected)
        assert state.frames.bank_revisions[1, index] == native.bank_state.revisions[index]
    for name, slot in query.terminal_slots.items():
        torch.testing.assert_close(data[state.frames.value_handles[1, 0, query.slot_ids.index(slot)]], native.outputs[name])
    assert records[0].call_returns == 1
    assert records[0].numeric_completed == 4


def test_two_call_contexts_share_bank_but_execute_separately(device):
    child = cooperative(federation(write=True, shared=True))
    calls = tuple(m.FormulaProgramCallCandidateV1(name, child, input_slots={"x": "x"},
                  output_slots={"result": name + "_out"}) for name in ("first", "second"))
    query = m.FormulaProgramQueryV7(slot_ids=("x", "first_out", "second_out"), candidates=calls,
        terminal_slots={"y": "second_out"}, entry_candidates=("first",),
        continuations={"first": {"second": "first_out"}}, max_steps=2, cooperation_width=4).to(device)
    wave, records, data, bank = execute(query, product_slots=(), publish_slots=(), rounds=4)
    native = query({"x": data[0]})
    state = records[-1].state
    assert sum(int(r.call_dispatches) for r in records) == 2
    assert sum(int(r.call_returns) for r in records) == 2
    assert state.frames.frame_tensor_steps[1, 0] == 2
    assert state.frames.bank_revisions[1, 0] == 2
    torch.testing.assert_close(bank[state.frames.bank_value_handles[1, 0]], native.bank_state.values[0])
    torch.testing.assert_close(data[state.frames.value_handles[1, 0, query.slot_ids.index("second_out")]], native.outputs["y"])


def test_pruned_call_donor_publishes_only_its_returned_outputs(device):
    child = wrap_child(graph())
    donor = m.FormulaProgramCallCandidateV1("z_donor", child, input_slots={"x": "x"},
        output_slots={"y": "p"}, requires_empty_slots=("h",))
    receiver = member("a_receiver", "x", "h")
    receiver.candidate.requires_empty_slots = ("p",)
    consume = member("consume", "shared", "y", weight=3.)
    query = m.FormulaProgramQueryV7(slot_ids=("x", "shared", "h", "h_negative", "p", "y", "y_negative"),
        candidates=(receiver, donor, consume), terminal_slots={"y": "y"},
        entry_candidates=("a_receiver", "z_donor"), continuations={"a_receiver": {"consume": "h"}},
        cooperation_width=2, max_steps=2).to(device)
    wave, records, data, bank = execute(query, rounds=3)
    native = search_cooperative_graphs((start_recursive_search(query, {"x": data[0]}),),
        product_slots=("shared",), publish_slots=("p",), width=2, beam_width=1)
    first, last = records[0], records[-1]
    assert first.directory_cursor == 1 and first.directory.occurrence.tolist() == [1]
    assert first.directory.producer[0] != query.action_ids.index("z_donor")
    assert first.state.routes[0, 0] == wave.action_ranks[0]
    torch.testing.assert_close(data[first.directory.handles[0]], 13 * data[0])
    torch.testing.assert_close(data[last.state.frames.value_handles[1, 0, query.slot_ids.index("y")]],
                               native.winner.execution.outputs["y"])
    assert sum(int(r.numeric_completed) for r in records) == 7


@pytest.mark.parametrize("dtype,logits", [
    (torch.float64, (1., 1. + 1e-10)),
    (torch.float32, (-3e38, 3e38)),
    (torch.float16, (-65504., 65504.)),
])
def test_native_v5_child_keeps_raw_query_precision(device, dtype, logits):
    child = m.FormulaProgramQueryV5(
        slot_ids=("x", "y", "y_negative"),
        candidates=(member("a", "x", "y", weight=2.), member("z", "x", "y", weight=3.)),
        terminal_slots={"y": "y"}, max_steps=1, hidden_dim=8,
    ).to(device)
    child.network.to(dtype=dtype)
    with torch.no_grad():
        child.network[-1].weight.zero_()
        child.network[-1].bias.copy_(torch.tensor((*logits, 0.), dtype=dtype))
    query = wrap_child(child)
    wave, state, directory, cursor, occurrence, data, bank, dc, bc = runtime(
        query, product_slots=(), publish_slots=())
    # Query parameters deliberately differ from the FP32 data/Bank pools.
    data, bank = data.float(), bank.float()
    expected = query({"x": data[0]})
    assert expected.trace.steps[0].child_trace.steps[0].candidate_id == "z"
    first = advance(wave, state, directory, cursor, occurrence, data, bank, dc, bc)
    assert not first.requires_fallback and first.call_returns == 1
    output = data[first.state.frames.value_handles[0, 0, query.slot_ids.index("out_y")]]
    torch.testing.assert_close(output, expected.outputs["y"])


@pytest.mark.parametrize("nonfinite", [False, True])
def test_v6_child_uses_raw_response_validity(device, nonfinite):
    source = member("source", "x", "h", weight=1.8e38 if nonfinite else 1.5e38)
    left, right = member("left", "x", "y"), member("right", "x", "y", weight=3.)
    child = m.FormulaProgramQueryV6(
        slot_ids=("x", "h", "h_negative", "y", "y_negative"),
        candidates=(source, left, right), terminal_slots={"y": "y"},
        entry_candidates=("source",),
        continuations={"source": {"left": "h_negative", "right": "h"}}, max_steps=2,
    ).to(device)
    query = wrap_child(child)
    wave, state, directory, cursor, occurrence, data, bank, dc, bc = runtime(
        query, product_slots=(), publish_slots=())
    result = advance(wave, state, directory, cursor, occurrence, data, bank, dc, bc)
    if nonfinite:
        assert result.requires_fallback and result.call_returns == 0
        assert not result.state.frames.completed.any()
    else:
        expected = query({"x": data[0]})
        assert not result.requires_fallback and result.call_returns == 1
        output = data[result.state.frames.value_handles[0, 0, query.slot_ids.index("out_y")]]
        torch.testing.assert_close(output, expected.outputs["y"])


def test_nested_calls_batch_independent_events(device):
    from torch.utils._pytree import tree_map

    query = wrap_child(wrap_child(graph())).to(device)
    wave, *inputs = runtime(query, product_slots=(), publish_slots=())
    state, directory, cursor, occurrence, data, bank, dc, bc = tree_map(
        lambda tensor: torch.stack((tensor.clone(), tensor.clone())), tuple(inputs))
    data[0, 0] = 1.
    data[1, 0] = -3.
    finite = torch.ones(wave.kernel.candidate_count, dtype=torch.bool)
    for step in range(3):
        result = wave.forward_batch(state, directory, cursor, occurrence, data, bank, dc, bc, finite)
        assert not result.requires_fallback.any()
        assert result.call_returns.tolist() == ([2, 2] if step == 0 else [0, 0])
        state, directory, cursor, occurrence = (result.state, result.directory,
                                                result.directory_cursor, result.occurrence_cursor)
        dc, bc = result.data_cursor, result.bank_cursor
    for episode in range(2):
        native = query({"x": data[episode, 0]})
        slot = query.slot_ids.index(query.terminal_slots["y"])
        output = data[episode, state.frames.value_handles[episode, 1, 0, slot]]
        torch.testing.assert_close(output, native.outputs["y"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_recursive_call_search_cuda_graph_replays_with_new_input():
    with torch.device("cuda"):
        query = wrap_child(cooperative(federation(write=True, shared=True))).cuda()
        wave, initial, empty, zero, first, data, bank, initial_dc, initial_bc = runtime(
            query, product_slots=(), publish_slots=())

        def run():
            state, directory, cursor, occurrence = initial, empty, zero, first
            dc, bc = initial_dc, initial_bc
            for _ in range(3):
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
        for value in (1., -3., .25):
            data[0].fill_(value)
            expected = query({"x": data[0]})
            capture.replay()
            assert not result.requires_fallback
            assert result.state.frames.completed.tolist() == [False, True]
            slot = query.slot_ids.index(query.terminal_slots["result"])
            torch.testing.assert_close(data[result.state.frames.value_handles[1, 0, slot]], expected.outputs["result"])
            for index, expected_value in enumerate(expected.bank_state.values):
                torch.testing.assert_close(bank[result.state.frames.bank_value_handles[1, index]], expected_value)


def test_unfinished_child_never_publishes_as_completed_call(device):
    base = wrap_child(graph())
    query = m.FormulaProgramQueryV7(
        slot_ids=(*base.slot_ids, "shared"), candidates=tuple(base.candidates),
        terminal_slots=base.terminal_slots, entry_candidates=base.entry_candidates,
        continuations=base.continuations, max_steps=base.max_steps,
    ).to(device)
    wave, state, directory, cursor, occurrence, data, bank, dc, bc = runtime(
        query, product_slots=("shared",), publish_slots=("out_y",))
    wave.call_round_limit = 1
    result = advance(wave, state, directory, cursor, occurrence, data, bank, dc, bc)
    assert result.requires_fallback and result.call_returns == 0
    assert result.directory_cursor == 0 and result.occurrence_cursor == 0
    assert not result.state.frames.completed.any()
