import pytest
import torch

from arti import mechanisms as m
from arti._formula_device_cooperative import (
    FormulaDeviceCooperativeSelection, FormulaDeviceCooperativeWave, cooperative_sequence_score,
)
from arti._formula_device_dispatch import FormulaDeviceNumericalDispatch
from arti._formula_device_execution import FormulaDeviceExecutionWave
from arti._formula_device_frames import FormulaDeviceFrameKernel
from test_formula_program_query_v7 import graph


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with torch.device(request.param):
        yield request.param


def _runtime(query, *, heads=1, typed_first=None):
    parameter = next(query.parameters())
    device, dtype = parameter.device, parameter.dtype
    kernel = FormulaDeviceFrameKernel.from_query(query).to(device)
    dispatch = FormulaDeviceNumericalDispatch(query, kernel).to(device)
    if typed_first is not None:
        from arti._formula_device_pools import FormulaDevicePoolLayout
        dl = FormulaDevicePoolLayout.from_samples((torch.zeros(1, 1, dtype=typed_first),
                                                  torch.zeros(1, 1, dtype=dtype)), 128)
        bl = FormulaDevicePoolLayout.from_samples((torch.zeros(1, 1, dtype=dtype),), 64)
        dispatch.prepare_typed_pools_(dl, bl)
    execution = FormulaDeviceExecutionWave(dispatch, kernel,
        data_capacity=128 if typed_first is None else dl.capacities,
        bank_capacity=64 if typed_first is None else bl.capacities)
    wave = FormulaDeviceCooperativeWave(query, execution, head_width=heads).to(device)
    handles = torch.full((1, kernel.spec.max_slots), -1)
    handles[0, query.slot_ids.index("x")] = 0 if typed_first is None else dl.offsets[1]
    values = query.initial_bank_state().values
    frames = kernel.initial_state(1, handles, bank_value_handles=torch.arange(len(values)))
    data, bank = torch.zeros(129, 1, 1, dtype=dtype), torch.zeros(65, 1, 1, dtype=dtype)
    if typed_first is not None:
        data, bank = dl.allocate(device), bl.allocate(device)
    (data if typed_first is None else data[1])[0] = 2.
    for index, value in enumerate(values):
        (bank if typed_first is None else bank[0])[index] = value.detach()
    return (wave, frames, data, bank,
            torch.tensor(1 if typed_first is None else [0, 1]),
            torch.tensor(len(values) if typed_first is None else [len(values)]))


def test_cooperative_device_graph_matches_native_frontiers_and_outputs(device):
    query = graph().to(device)
    wave, frames, data, bank, dc, bc = _runtime(query)
    native = query({"x": data[0]})
    occurrence = torch.tensor(0)
    names, scores = [], []
    for expected in native.frontiers:
        result = wave(frames, data, bank, dc, bc,
                      torch.ones(wave.kernel.candidate_count, dtype=torch.bool), occurrence)
        assert not result.requires_fallback
        assert result.accepted.tolist() == [[True]]
        selected = result.frontiers.candidates[0, 0].tolist()
        names.append(tuple(query.action_ids[i] for i in selected if i >= 0))
        scores.append(result.frontiers.selection_log_score[0, 0])
        torch.testing.assert_close(scores[-1], expected.selection_log_score)
        frames, dc, bc, occurrence = result.frames, result.data_cursor, result.bank_cursor, result.next_occurrence
    assert names == [("a", "b"), ("c", "d"), ("final",), ("stop",)]
    assert frames.completed.all()
    assert frames.frame_steps[0, 0] == 5
    assert occurrence == 6
    torch.testing.assert_close(data[frames.value_handles[0, 0, query.slot_ids.index("y")]], native.outputs["y"])


def test_equivalent_heads_do_not_execute_or_publish_siblings_twice(device):
    query = graph().to(device)
    wave, frames, data, bank, dc, bc = _runtime(query, heads=4)
    result = wave(frames, data, bank, dc, bc,
                  torch.ones(wave.kernel.candidate_count, dtype=torch.bool), torch.tensor(7))
    assert result.frontiers.valid.sum() == 1
    assert result.accepted.sum() == 1
    assert result.next_occurrence == 9
    assert result.data_cursor == dc + 4  # two actual multi-output ordinary nodes
    assert result.products.occurrence[result.products.ready].tolist() == [7, 7, 8, 8]
    slots = [query.slot_ids.index(s) for s in ("a_out", "b_out")]
    assert (result.frames.value_handles[0, 0, slots] >= 0).all()
    assert result.frames.frame_steps[0, 0] == 2
    assert result.frames.frame_tensor_steps[0, 0] == 2


def test_selector_control_priority_ends_frontier_instead_of_skipping(device):
    from test_formula_program_query_v6 import federation

    old = federation(write=True, shared=True)
    query = m.FormulaProgramQueryV7(
        slot_ids=old.slot_ids, candidates=tuple(old.candidates), terminal_slots=old.terminal_slots,
        entry_candidates=old.entry_candidates, continuations=old.continuations,
        max_steps=4, cooperation_width=4,
    ).to(device)
    kernel = FormulaDeviceFrameKernel.from_query(query).to(device)
    selector = FormulaDeviceCooperativeSelection(query, kernel, head_width=1).to(device)
    # The structural mask here isolates the scheduler's control ordering rule.
    scores = torch.zeros(1, kernel.candidate_count)
    scores[0, :3] = torch.tensor([3., 2., 1.])
    eligible = torch.zeros_like(scores, dtype=torch.bool)
    eligible[0, :3] = True
    result = selector(scores, eligible,
                      torch.tensor([0]), torch.tensor([0]), torch.tensor([0]))
    assert result.candidates[0, 0].tolist() == [0, -1, -1, -1]


def test_selector_respects_total_and_tensor_budget(device):
    query = graph(width=4).to(device)
    kernel = FormulaDeviceFrameKernel.from_query(query).to(device)
    selector = FormulaDeviceCooperativeSelection(query, kernel, head_width=1).to(device)
    scores = torch.zeros(1, kernel.candidate_count)
    eligible = torch.zeros_like(scores, dtype=torch.bool)
    eligible[0, :2] = True
    result = selector(scores, eligible, torch.tensor([0]), torch.tensor([4]), torch.tensor([0]))
    assert result.candidates[0, 0].tolist() == [0, -1, -1, -1]
    selector.max_tensor_steps.fill_(1)
    result = selector(scores, eligible, torch.tensor([0]), torch.tensor([0]), torch.tensor([0]))
    assert result.candidates[0, 0].tolist() == [0, -1, -1, -1]


def test_selection_bias_changes_support_but_not_recorded_energy(device):
    query = graph(width=1).to(device)
    kernel = FormulaDeviceFrameKernel.from_query(query).to(device)
    selector = FormulaDeviceCooperativeSelection(query, kernel, head_width=1).to(device)
    scores = torch.zeros(1, kernel.candidate_count)
    scores[0, 0] = 2.
    eligible = torch.zeros_like(scores, dtype=torch.bool)
    eligible[0, :2] = True
    metadata = (torch.tensor([0]), torch.tensor([0]), torch.tensor([0]))
    original = selector(scores, eligible, *metadata)
    zero = selector(scores, eligible, *metadata, selection_bias=torch.zeros_like(scores))
    for actual, expected in zip(zero, original, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    bias = torch.full_like(scores, 1000.)
    bias[0, 0], bias[0, 1] = 0., 3.
    changed = selector(scores, eligible, *metadata, selection_bias=bias)
    assert original.candidates[0, 0, 0] == 0 and changed.candidates[0, 0, 0] == 1
    expected = cooperative_sequence_score(scores, changed.candidates, changed.remaining)
    torch.testing.assert_close(changed.selection_log_score, expected, rtol=0, atol=0)
    torch.testing.assert_close(changed.selection_log_score[0, 0], scores[0, :2].log_softmax(0)[1])


def test_old_one_action_search_does_not_silently_accept_v7(device):
    from arti._formula_device_search import FormulaDeviceSearchWave

    query = graph().to(device)
    wave, *_ = _runtime(query)
    with pytest.raises(TypeError, match="cooperative frontier"):
        FormulaDeviceSearchWave(query, wave.execution, width=1, local_width=1,
                                candidate_family_ids=[0] * wave.kernel.candidate_count,
                                candidate_membership=[[False]] * wave.kernel.candidate_count)


@pytest.mark.parametrize("dtype,typed_first", [(torch.float16, None), (torch.bfloat16, None),
                                              (torch.float64, torch.float16)])
def test_close_response_ranking_keeps_native_precision(device, dtype, typed_first):
    from test_formula_program_query_v6 import member

    p = member("p", "x", "h", weight=0.0001)
    a, b = member("a", "x", "y"), member("b", "x", "y", weight=2.)
    query = m.FormulaProgramQueryV7(
        slot_ids=("x", "h", "h_negative", "y", "y_negative"), candidates=(p, a, b),
        terminal_slots={"y": "y"}, entry_candidates=("p",),
        continuations={"p": {"a": "h_negative", "b": "h"}},
        cooperation_width=2, max_steps=2,
    ).to(device=device, dtype=dtype)
    wave, frames, data, bank, dc, bc = _runtime(query, typed_first=typed_first)
    native = query({"x": (data if typed_first is None else data[1])[0]})
    occurrence = torch.tensor(0)
    for expected in native.frontiers:
        result = wave(frames, data, bank, dc, bc,
                      torch.ones(wave.kernel.candidate_count, dtype=torch.bool), occurrence)
        names = tuple(query.action_ids[i] for i in result.frontiers.candidates[0, 0].tolist() if i >= 0)
        assert names == tuple(n.candidate_id for n in expected.nodes)
        assert result.accepted.all() and not result.requires_fallback
        expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        assert result.frontiers.selection_log_score.dtype == expected_dtype
        torch.testing.assert_close(result.frontiers.selection_log_score[0, 0], expected.selection_log_score)
        frames, dc, bc, occurrence = result.frames, result.data_cursor, result.bank_cursor, result.next_occurrence
    assert native.frontiers[1].nodes[0].candidate_id == "b"


def test_effect_singleton_changes_actual_bank_and_keeps_data_alias(device):
    from test_formula_program_query_v6 import federation

    old = federation(write=True, shared=True)
    query = m.FormulaProgramQueryV7(
        slot_ids=old.slot_ids, candidates=tuple(old.candidates), terminal_slots=old.terminal_slots,
        entry_candidates=old.entry_candidates, continuations=old.continuations,
        max_steps=4, cooperation_width=4,
    ).to(device)
    wave, frames, data, bank, dc, bc = _runtime(query)
    native = query({"x": data[0]})
    occurrence = torch.tensor(0)
    for expected in native.frontiers:
        result = wave(frames, data, bank, dc, bc,
                      torch.ones(wave.kernel.candidate_count, dtype=torch.bool), occurrence)
        names = tuple(query.action_ids[i] for i in result.frontiers.candidates[0, 0].tolist() if i >= 0)
        assert names == tuple(n.candidate_id for n in expected.nodes)
        assert result.accepted.all() and not result.requires_fallback
        frames, dc, bc, occurrence = result.frames, result.data_cursor, result.bank_cursor, result.next_occurrence
        if names == ("write",):
            left, right = (frames.value_handles[0, 0, query.slot_ids.index(s)] for s in ("h", "tail"))
            torch.testing.assert_close(data[left], data[right], rtol=0, atol=0)
            assert frames.bank_revisions[0, 0] == 1
    torch.testing.assert_close(data[frames.value_handles[0, 0, query.slot_ids.index("y")]],
                               native.outputs["result"])


def test_failed_sibling_does_not_publish_partial_frontier(device):
    from test_formula_program_query_v6 import member

    bad = member("a_bad", "x", "y", weight=40000.)
    good = member("b_good", "x", "y", weight=3.)
    c = member("c", "x", "z", weight=4.)
    query = m.FormulaProgramQueryV7(
        slot_ids=("x", "y", "y_negative", "z", "z_negative"), candidates=(bad, good, c),
        terminal_slots={"y": "y"}, entry_candidates=("a_bad", "b_good", "c"),
        continuations={}, cooperation_width=2, max_steps=2,
    ).to(device=device, dtype=torch.float16)
    wave, frames, data, bank, dc, bc = _runtime(query, heads=3)
    result = wave(frames, data, bank, dc, bc,
                  torch.ones(wave.kernel.candidate_count, dtype=torch.bool), torch.tensor(0))
    assert result.frontiers.valid.tolist() == [[True, True, False]]
    assert result.accepted.tolist() == [[False, True, False]]
    assert result.next_occurrence == 2
    assert not result.frames.active[0] and result.frames.active[1]
    assert (result.frames.value_handles[0] == frames.value_handles[0]).all()
    assert result.products.occurrence[result.products.ready].tolist() == [0, 0, 1, 1]
    assert result.data_cursor > dc + 4  # attempted failed frontier consumes real pool space
    assert not result.requires_fallback
    for slot, value in (("y", 6.), ("z", 8.)):
        handle = result.frames.value_handles[1, 0, query.slot_ids.index(slot)]
        torch.testing.assert_close(data[handle], data.new_tensor([[value]]))


def test_selector_conflicts_match_native_in_both_directions(device):
    query = graph(width=4).to(device)
    kernel = FormulaDeviceFrameKernel.from_query(query).to(device)
    selector = FormulaDeviceCooperativeSelection(query, kernel, head_width=4).to(device)
    spec = kernel.spec
    output = (spec.candidate_output_slots[:, :, None] ==
              torch.arange(spec.max_slots, device=spec.candidate_output_slots.device)[None, None]).any(1)
    expected_table = (output[:, None] & (output | spec.candidate_empty)[None]).any(-1)
    expected_table |= (spec.candidate_empty[:, None] & output[None]).any(-1)
    expected_table |= torch.eye(kernel.candidate_count, dtype=torch.bool, device=expected_table.device)
    assert torch.equal(selector.conflict.cpu(), expected_table.cpu())
    # Test every admissible pair against the same declared output/empty sets.
    for i, candidate in enumerate(query.candidates):
        for j, other in enumerate(query.candidates):
            output, other_output = set(candidate.output_slot_ids), set(other.output_slot_ids)
            guards, other_guards = set(candidate.requires_empty_slots), set(other.requires_empty_slots)
            expected = i == j or bool(output & (other_output | other_guards) or guards & other_output)
            assert bool(selector.conflict[i, j]) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_entire_cooperative_round_cuda_capture_reuses_new_input():
    with torch.device("cuda"):
        query = graph().cuda()
        wave, frames, data, bank, dc, bc = _runtime(query, heads=3)
        finite = torch.ones(wave.kernel.candidate_count, dtype=torch.bool)
        occurrence = torch.tensor(0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                wave(frames, data, bank, dc, bc, finite, occurrence)
        torch.cuda.current_stream().wait_stream(stream)
        graph_capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_capture):
            result = wave(frames, data, bank, dc, bc, finite, occurrence)
        graph_capture.replay()
        assert result.accepted.sum() == 1
        pointers = tuple(field.data_ptr() for field in result.frames)
        for value in (3., -2.):
            data[0].fill_(value)
            graph_capture.replay()
            for slot, gain in (("a_out", 2.), ("b_out", 3.)):
                handle = result.frames.value_handles[0, 0, query.slot_ids.index(slot)]
                torch.testing.assert_close(data[handle], data.new_tensor([[value * gain]]))
            assert tuple(field.data_ptr() for field in result.frames) == pointers
            assert result.next_occurrence == 2


def _response_graph(weights, *, control=False):
    from test_formula_program_query_v6 import member

    producers = tuple(member(f"p{i}", "x", f"h{i}", weight=weight) for i, weight in enumerate(weights))
    if len(weights) == 3:
        consumers = tuple(member(name, "x", f"out_{name}") for name in ("a", "b", "c"))
        continuations = {f"p{i}": {("stop" if control and i == 2 else name): f"h{i}"}
                         for i, name in enumerate(("a", "b", "c"))}
        maximum = 5
    else:
        consumers = tuple(member(name, "x", f"out_{name}") for name in ("a", "b"))
        continuations = {f"p{i}": {"a" if i < 3 else "b": f"h{i}"} for i in range(4)}
        maximum = 5
    candidates = producers + consumers
    return m.FormulaProgramQueryV7(
        slot_ids=("x", *(s for c in candidates for s in c.output_slot_ids)), candidates=candidates,
        terminal_slots={"y": "x" if control else "out_a"},
        entry_candidates=tuple(p.candidate_id for p in producers), continuations=continuations,
        cooperation_width=len(producers), max_steps=maximum,
    )


@pytest.mark.parametrize("control", [False, True])
def test_response_frontier_uses_raw_logits_after_dominant_head(device, control):
    query = _response_graph((5e7, 0., .5), control=control).to(device)
    wave, frames, data, bank, dc, bc = _runtime(query)
    native_arena = query._arena({"x": data[0]})
    occurrence = torch.tensor(0)
    steps = 0
    for _ in range(2):
        expected = query.advance_frontier(native_arena, steps=steps)
        result = wave(frames, data, bank, dc, bc,
                      torch.ones(wave.kernel.candidate_count, dtype=torch.bool), occurrence)
        names = tuple(query.action_ids[i] for i in result.frontiers.candidates[0, 0].tolist() if i >= 0)
        assert names == tuple(n.candidate_id for n in expected.frontier.nodes)
        torch.testing.assert_close(result.frontiers.selection_log_score[0, 0],
                                   expected.frontier.selection_log_score)
        frames, dc, bc, occurrence = result.frames, result.data_cursor, result.bank_cursor, result.next_occurrence
        native_arena, steps = expected.arena, steps + len(expected.trace_steps)
    assert names == (("a",) if control else ("a", "c"))


@pytest.mark.parametrize("typed_first", [None, torch.float64])
def test_unused_double_pool_does_not_change_response_cancellation(device, typed_first):
    query = _response_graph((5e7, .5, -5e7, .25)).to(device)
    wave, frames, data, bank, dc, bc = _runtime(query, typed_first=typed_first)
    value = (data if typed_first is None else data[1])[0]
    arena = query._arena({"x": value})
    native_first = query.advance_frontier(arena, steps=0)
    first = wave(frames, data, bank, dc, bc,
                 torch.ones(wave.kernel.candidate_count, dtype=torch.bool), torch.tensor(0))
    second = wave(first.frames, data, bank, first.data_cursor, first.bank_cursor,
                  torch.ones(wave.kernel.candidate_count, dtype=torch.bool), first.next_occurrence)
    expected = query.advance_frontier(native_first.arena, steps=4)
    assert expected.frontier.nodes[0].candidate_id == "b"
    assert query.action_ids[int(second.frontiers.candidates[0, 0, 0])] == "b"
    torch.testing.assert_close(second.frontiers.selection_log_score[0, 0].float(),
                               expected.frontier.selection_log_score)


def test_one_double_action_does_not_promote_other_actions_partial_sum(device):
    from dataclasses import replace

    query = _response_graph((5e7, .5, -5e7, .25)).to(device)
    wave, frames, data, bank, dc, bc = _runtime(query, typed_first=torch.float64)
    initial = query._arena({"x": data[1][0]})
    native = query.advance_frontier(initial, steps=0).arena
    first = wave(frames, data, bank, dc, bc,
                 torch.ones(wave.kernel.candidate_count, dtype=torch.bool), torch.tensor(0))
    slot = query.slot_ids.index("h3")
    # A real FP64 response for b promotes the final stack, not a's FP32 sum.
    mixed = tuple(value.double() if index == slot else value
                  for index, value in enumerate(native.values.values))
    native = replace(native, values=replace(native.values, values=mixed))
    data[0][0] = mixed[slot]
    first.frames.value_handles[0, 0, slot] = 0
    cursor = first.data_cursor.clone()
    cursor[0] = 1
    result = wave(first.frames, data, bank, cursor, first.bank_cursor,
                  torch.ones(wave.kernel.candidate_count, dtype=torch.bool), first.next_occurrence)
    expected = query.advance_frontier(native, steps=4)
    assert query.query_logits(native)[0, query.action_ids.index("a")] == 0
    assert query.query_logits(native).dtype == torch.float64
    assert query.action_ids[int(result.frontiers.candidates[0, 0, 0])] == "b"
    torch.testing.assert_close(result.frontiers.selection_log_score[0, 0],
                               expected.frontier.selection_log_score)
