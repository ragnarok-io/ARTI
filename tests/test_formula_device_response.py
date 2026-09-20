from types import SimpleNamespace

import pytest
import torch

from arti._formula_device_query import FormulaDeviceQuery
from arti._formula_device_response import FormulaDeviceResponse


def _response(device="cpu", *, coverage=False):
    return FormulaDeviceResponse(
        action_ids=("a", "b", "finish", "stop"), entry_candidates=("a",),
        continuations={"a": {"b": "s-a", "finish": "s-a"}, "b": {"finish": "s-b"}},
        width=2, action_priority=(2, 1, 0, 3), candidate_family_ids=(0, 0, 1, 2),
        preserve_family_coverage=coverage,
    ).to(device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_action_sum_preserves_native_order_and_gradient(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    module = FormulaDeviceResponse(
        action_ids=("a", "b", "c", "d", "stop"), entry_candidates=("a",), width=2,
        continuations={"a": {"b": "ab", "c": "ac", "stop": "ac"},
                       "b": {"c": "bc", "stop": "bc"}, "c": {"stop": "cs"}},
    ).to(device)
    values = torch.tensor([[1e8, 1., -1e8, 3.], [7., -4., 5., 2.]],
                          dtype=dtype, device=device, requires_grad=True)
    present = torch.tensor([[True, True, True, True], [True, False, True, False]], device=device)
    source_double = torch.tensor([[False, False, True, True], [False, False, False, False]], device=device)
    reference_double = torch.tensor([False, False], device=device)
    parents = torch.zeros(2, dtype=dtype, device=device)
    actual, available = module.response_logits(values, present, parents, reference_double, source_double)
    columns = []
    for sources in module.action_sources:
        value = torch.zeros_like(parents)
        double = reference_double
        for source in sources:
            part = torch.where(present[:, source], values[:, source], torch.zeros_like(parents))
            if dtype == torch.float64:
                double = double | (present[:, source] & source_double[:, source])
                value = torch.where(double, value + part, (value.float() + part.float()).double())
            else:
                value = value + part
        columns.append(value)
    expected = torch.stack(columns, -1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.arange(1, actual.numel() + 1, device=device).reshape_as(actual)
    actual_grad, = torch.autograd.grad((actual * weights).sum(), values, retain_graph=True)
    expected_grad, = torch.autograd.grad((expected * weights).sum(), values)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert not available[:, 0].any() and not available[:, 3].any()
    assert module.ordered_sources.shape == (3, 5)


def test_batched_response_has_finite_large_double_backward():
    module = _response()
    values = torch.tensor([[1e100, 2.]], dtype=torch.float64, requires_grad=True)
    present = torch.ones_like(values, dtype=torch.bool)
    result, _ = module.response_logits(values, present, torch.zeros(1, dtype=torch.float64),
                                       torch.tensor([False]), torch.tensor([[True, False]]))
    gradient, = torch.autograd.grad(result.sum(), values)
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, torch.tensor([[2., 1.]], dtype=torch.float64))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_response_changes_without_reexecuting_or_recompiling_producers(device, backend):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    module = _response(device)
    values = torch.tensor([[2., 3.], [4., 1.], [float("nan"), float("nan")]], device=device)
    present = torch.tensor([[True, True], [True, False], [False, False]], device=device)
    eligible = torch.tensor([[True, True, True, False]] * 3, device=device)
    parents = torch.zeros(3, device=device)
    steps = torch.tensor([2, 1, 0], device=device)
    inputs = (values, present, eligible, parents, steps)
    assert not list(module.parameters())
    assert not hasattr(module, "network")
    assert not hasattr(module, "tensor_encoder")
    builds = []
    kernel = module
    if backend != "eager":
        graph = torch.export.export(module, inputs, strict=True).module()
        from torch._dynamo.backends.registry import lookup_backend
        def counting_backend(gm, example_inputs):
            builds.append(1)
            return lookup_backend(backend)(gm, example_inputs)
        kernel = torch.compile(graph, backend=counting_backend, fullgraph=True)
    result = kernel(*inputs)
    assert result.eligible.tolist() == [[False, True, True, False], [False, True, True, False], [True, False, False, False]]
    torch.testing.assert_close(result.masked_logits[0, 1:3], torch.tensor([2., 5.], device=device))
    assert result.order[1, :2].tolist() == [2, 1]
    before = len(builds)
    values[0, 1] = -10
    changed = kernel(*inputs)
    assert changed.order[0, 0] == 1
    assert len(builds) == before
    assert torch.isfinite(changed.scores[2, 0])
    assert not torch.isnan(changed.scores[2]).any()


def test_sum_gradient_reaches_each_producer_but_not_absent_scores():
    weights = torch.tensor([2., 3.], requires_grad=True)
    emitted = (weights.square() * torch.tensor([2., 4.]))[None]
    module = _response()
    logits, _ = module.response_logits(emitted, torch.tensor([[True, True]]), torch.zeros(1))
    logits[0, 2].backward()
    torch.testing.assert_close(weights.grad, torch.tensor([8., 24.]))
    values = torch.tensor([[2., float("nan")]], requires_grad=True)
    logits, _ = module.response_logits(values, torch.tensor([[True, False]]), torch.zeros(1))
    logits[0, 2].backward()
    torch.testing.assert_close(values.grad, torch.tensor([[1., 0.]]))


def test_response_accumulation_order_precision_and_overflow_match_native():
    from test_formula_program_query_v6 import response_sum
    for weights, dtype in (((1e8, -1e8, 1.0), torch.float32), ((40000.0, 40000.0), torch.float16),
                           ((3e38, 3e38), torch.float32)):
        query, arena = response_sum(weights, reverse=True, dtype=dtype)
        device = FormulaDeviceResponse.from_query(query, width=2)
        values = torch.stack([arena.values.get(slot).reshape(()) for _, slot in device.score_sources])[None]
        present = torch.ones_like(values, dtype=torch.bool)
        actual = device(values, present, query.eligible(arena, steps=len(weights))[None],
                        torch.zeros(1, dtype=dtype), torch.tensor(len(weights)))
        expected = query.query_logits(arena)
        logits, _ = device.response_logits(values, present, torch.zeros(1, dtype=dtype))
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        if weights[0] == 3e38:
            assert not actual.finite.any() and not actual.local_selected.any()
        else:
            assert actual.finite.all() and logits.dtype == torch.float32


def test_legal_only_finiteness_dead_rows_stop_and_parent_minus_infinity():
    module = _response()
    values = torch.tensor([[1., float("nan")], [1., float("nan")], [1., 5.], [1., 1.]])
    present = torch.ones(4, 2, dtype=torch.bool)
    eligible = torch.tensor([[False, True, False, False], [False, True, True, False],
                             [False, True, True, True], [False, False, False, False]])
    result = module(values, present, eligible, torch.tensor([0., 0., -float("inf"), float("nan")]), torch.tensor(2))
    assert result.finite.tolist() == [True, False, True, True]
    assert result.order[2, :3].tolist() == [2, 1, 3]
    assert not result.local_selected[1].any()
    assert not result.local_selected[3].any()
    assert result.eligible[2, -1]


@pytest.mark.parametrize("coverage", [False, True])
def test_response_ranking_matches_existing_query_packet(coverage):
    module = _response(coverage=coverage)
    class Fixed(torch.nn.Module):
        def forward(self, value):
            return torch.tensor([[0., 2., 5., 0.]]).expand(value.shape[0], -1)
    native = FormulaDeviceQuery(Fixed(), slot_count=1, width=2,
        candidate_family_ids=(0, 0, 1, 2), action_priority=(2, 1, 0, 3), preserve_family_coverage=coverage)
    values, present = torch.tensor([[2., 3.]]), torch.ones(1, 2, dtype=torch.bool)
    eligible, parents = torch.tensor([[False, True, True, True]]), torch.zeros(1)
    actual = module(values, present, eligible, parents, torch.tensor(2))
    expected = native.score_summary(torch.zeros(1, 1), torch.ones(1, dtype=torch.bool), eligible, parents)
    for name in actual._fields:
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))


def _query(device="cpu"):
    from arti.formula_program_query_v6 import FormulaProgramQueryV6 as v6
    from arti import mechanisms as m
    value_type = m.TensorType(("B", "D"), ("B", 2), dtype="floating", domain="activation")
    x = m.InputBinding("x", value_type)
    weight = m.BankBinding("weight", "arti/device-response-test@1", "weight", value_type)
    candidates = []
    for name, gain in (("a", 2.), ("b", 3.)):
        value = m.scale(x, weight)
        program = m.FormulaProgram.build(outputs=(value, m.reduce_sum(value, axis="D")))
        candidates.append(m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
            name, program, input_slots={"x": "x"},
            output_slots=dict(zip(program.outputs, (name, f"s-{name}"), strict=True)),
            operands={"weight": torch.full((1, 2), gain)}, trainable_operands=("weight",),
        )))
    program = m.FormulaProgram.build(outputs=(m.add(x, x),))
    candidates.append(m.FormulaProgramTensorCandidateV3(m.FormulaProgramCandidateV2(
        "finish", program, input_slots={"x": "x"}, output_slot="out", operands={},
    )))
    return v6(slot_ids=("x", "a", "s-a", "b", "s-b", "out"), candidates=candidates,
              terminal_slots={"answer": "out"}, entry_candidates=("a",),
              continuations={"a": {"b": "s-a", "finish": "s-a"}, "b": {"finish": "s-b"}},
              max_steps=3).to(device)


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_v6_benchmark_reads_actual_scores_and_keeps_native_packet(backend, monkeypatch):
    from benchmarks._federated_device_query import device_query_execution, device_ranked_candidates
    query = _query()
    arena = query._arena({"x": torch.tensor([[1., 2.]])})
    first = query.candidates[0](arena)
    both = query.candidates[1](first)
    rows = tuple(SimpleNamespace(arena=a, route=()) for a in (arena, first, both))
    def forbidden(*args, **kwargs):
        raise AssertionError("scoring executed a producer")
    for candidate in query.candidates:
        monkeypatch.setattr(candidate, "forward", forbidden)
    with torch.no_grad(), device_query_execution(backend=backend):
        for step, row in enumerate(rows):
            receipt = {}
            actual = device_ranked_candidates(query, (row,), tuple(query.candidates), steps=step,
                width=4, include_stop=True, numerical_rejections=[], eligibility_records=receipt)
            expected = query.query(row.arena, steps=step)
            allowed = expected.eligible.nonzero().flatten().tolist()
            assert receipt[0] == tuple(allowed)
            scores = expected.masked_logits.log_softmax(-1)[0]
            for action, score in actual[0]:
                index = len(query.candidates) if action is None else tuple(query.candidates).index(action)
                torch.testing.assert_close(score, scores[index])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_v6_typed_search_wave_matches_native_execution(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from arti._formula_device_dispatch import FormulaDeviceNumericalDispatch
    from arti._formula_device_execution import FormulaDeviceExecutionWave
    from arti._formula_device_frames import FormulaDeviceFrameKernel
    from arti._formula_device_pools import FormulaDevicePoolLayout
    from arti._formula_device_search import FormulaDeviceSearchState, FormulaDeviceSearchWave
    query = _query(device)
    sample = torch.tensor([[1., 2.]], device=device)
    kernel = FormulaDeviceFrameKernel.from_query(query).to(device)
    dispatch = FormulaDeviceNumericalDispatch.from_query(query, frame_kernel=kernel).to(device)
    data_layout = FormulaDevicePoolLayout.from_samples((sample, sample.new_zeros(1)), 64)
    bank_layout = FormulaDevicePoolLayout.from_samples((sample,), 64)
    dispatch.prepare_typed_pools_(data_layout, bank_layout)
    execution = FormulaDeviceExecutionWave(dispatch, kernel, data_capacity=data_layout.capacities,
                                          bank_capacity=bank_layout.capacities)
    wave = FormulaDeviceSearchWave(query, execution, width=2, local_width=4,
        candidate_family_ids=[0] * 4, candidate_membership=[[False]] * 4, preserve_coverage=False).to(device)
    handles = torch.full((4, len(query.slot_ids)), -1, dtype=torch.int64, device=device)
    handles[0, 0] = 0
    frames = kernel.initial_state(4, handles)
    frames.active[1:] = False
    state = FormulaDeviceSearchState(frames, sample.new_zeros(4), sample.new_zeros(4),
        torch.full((4, 12), -1, dtype=torch.int64, device=device),
        torch.zeros(4, dtype=torch.int64, device=device), torch.zeros(4, 1, dtype=torch.bool, device=device))
    data, bank = data_layout.allocate(device), bank_layout.allocate(device)
    data[0][0].copy_(sample)
    dc = torch.tensor([1, 0], device=device)
    bc = torch.zeros(1, dtype=torch.int64, device=device)
    finite = torch.ones(4, dtype=torch.bool, device=device)
    for _ in range(5):
        result = wave(state, data, bank, dc, bc, finite)
        assert not result.requires_fallback.any()
        state, dc, bc = result.state, result.data_cursor, result.bank_cursor
    assert state.frames.completed.sum() == 2
    assert not state.frames.active.any()
    output_slot = query.slot_ids.index("out")
    for row in range(2, 4):
        handle = state.frames.value_handles[row, 0, output_slot]
        torch.testing.assert_close(data[0][handle], 2 * sample)
    assert not any(isinstance(child, FormulaDeviceQuery) for child in wave.decision.query_waves)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_response_raw_cuda_graph_reuses_changed_values_and_presence():
    module = _response("cuda")
    values = torch.tensor([[2., 3.]], device="cuda")
    present = torch.ones(1, 2, dtype=torch.bool, device="cuda")
    mask = torch.tensor([[False, True, True, True]], device="cuda")
    parents, steps = torch.zeros(1, device="cuda"), torch.tensor(2, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            module(values, present, mask, parents, steps)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = module(values, present, mask, parents, steps)
    graph.replay()
    first = result.scores.clone()
    values[0, 1] = -20
    graph.replay()
    assert not torch.equal(first, result.scores)
    present.zero_()
    graph.replay()
    assert result.eligible.tolist() == [[False, False, False, True]]
