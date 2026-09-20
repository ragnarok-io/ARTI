import pytest
import torch

from arti import mechanisms as m
from arti._formula_device_dispatch import FormulaDeviceNumericalDispatch, _candidate_records
from arti._formula_device_execution import FormulaDeviceExecutionWave
from arti._formula_device_frames import FormulaDeviceFrameKernel
from arti._formula_device_pools import FormulaDevicePoolLayout
from arti._formula_device_search import FormulaDeviceSearchState, FormulaDeviceSearchWave
from benchmarks._federated_recursive_search import search_recursive_graphs, search_recursive_graphs_many, start_recursive_search


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with torch.device(request.param):
        yield torch.device(request.param)


def _query(encoded=False):
    def encoder():
        return m.FormulaProgramQueryTensorEncoderV1(3, 4) if encoded else None

    narrow = m.TensorType(("B", "D"), ("B", 3), dtype="floating", domain="activation")
    broad = m.TensorType(("B", "N", "D"), ("B", 2, 3), dtype="floating", domain="activation")
    candidates = []
    for name, kind, shape, weight in (("narrow", narrow, (1, 3), 2.), ("broad", broad, (1, 2, 3), 3.)):
        x = m.InputBinding("x", kind)
        bank = m.BankBinding("bank", f"arti/typed-search-{name}@1", "bank", kind)
        candidates.append(m.FormulaProgramTensorCandidateV3(m.FormulaProgramCandidateV2(
            f"read-{name}", m.FormulaProgram.build(outputs=(m.scale(x, bank),)),
            input_slots={"x": "y"}, output_slot="owned", operands={"bank": torch.full(shape, weight)},
        ), plastic_bank_slot="bank", bank_owner_id=name))
        rate = m.BankBinding("rate", f"arti/typed-search-{name}@1", "rate", kind)
        zero = m.BankBinding("zero", f"arti/typed-search-{name}@1", "zero", kind)
        candidates.append(m.FormulaProgramEffectCandidateV3(
            f"write-{name}", m.FormulaEffectProgramV2(
                m.FormulaProgram.build(outputs=(m.neural_plasticity(x, m.scale(x, rate), zero),)),
                data_input_name="x", state_type=kind,
            ), input_slot="owned", output_slot="tail",
            operands={"rate": torch.full(shape, .1), "zero": torch.zeros(shape)},
        ))
        output = m.add(x, x) if name == "narrow" else m.reduce_sum(x, axis="N")
        candidates.append(m.FormulaProgramTensorCandidateV3(m.FormulaProgramCandidateV2(
            f"finish-{name}", m.FormulaProgram.build(outputs=(output,)),
            input_slots={"x": "tail"}, output_slot="out",
        )))
    child = m.FormulaProgramQueryV5(
        slot_ids=("y", "owned", "tail", "out"), candidates=tuple(candidates),
        terminal_slots={"answer": "out"}, max_steps=3, hidden_dim=8, tensor_encoder=encoder(),
    )
    x = m.InputBinding("x", broad)
    root_candidates = tuple(m.FormulaProgramTensorCandidateV3(m.FormulaProgramCandidateV2(
        name, m.FormulaProgram.build(outputs=(value,)), input_slots={"x": "x"}, output_slot="mid",
    )) for name, value in (("compress", m.reduce_sum(x, axis="N")), ("keep", m.add(x, x))))
    root_candidates += (m.FormulaProgramCallCandidateV1(
        "child", child, input_slots={"y": "mid"}, output_slots={"answer": "out"},
    ),)
    return m.FormulaProgramQueryV5(
        slot_ids=("x", "mid", "out"), candidates=root_candidates,
        terminal_slots={"answer": "out"}, max_steps=2, hidden_dim=8, tensor_encoder=encoder(),
    )


def _runtime(query, device, width=4):
    kernel = FormulaDeviceFrameKernel.from_query(query).to(device)
    dl = FormulaDevicePoolLayout.from_samples((torch.zeros(1, 3), torch.zeros(1, 2, 3)), 128)
    bl = FormulaDevicePoolLayout.from_samples((torch.zeros(1, 3), torch.zeros(1, 2, 3)), 64)
    dispatch = FormulaDeviceNumericalDispatch(query, kernel).to(device)
    dispatch.prepare_typed_pools_(dl, bl)
    execution = FormulaDeviceExecutionWave(dispatch, kernel, data_capacity=dl.capacities, bank_capacity=bl.capacities)
    families = [2] * kernel.candidate_count
    members = [[False] for _ in families]
    for global_id, _, candidate in _candidate_records(query):
        effect = isinstance(candidate, m.FormulaProgramEffectCandidateV3)
        families[global_id] = 1 if effect else 0
        members[global_id] = [effect]
    wave = FormulaDeviceSearchWave(query, execution, width=width, local_width=width,
                                  candidate_family_ids=families, candidate_membership=members).to(device)
    handles = torch.full((2 * width, 3), -1, dtype=torch.int64, device=device)
    handles[:, 0] = dl.offsets[1]
    initial_banks = query.initial_bank_state().values
    bank_handles = [bl.offsets[bl.index(value)] for value in initial_banks]
    frames = kernel.initial_state(2 * width, handles, bank_value_handles=torch.tensor(bank_handles, device=device))
    frames.active[1:] = False
    state = FormulaDeviceSearchState(
        frames, torch.zeros(2 * width, device=device), torch.zeros(2 * width, device=device),
        torch.full((2 * width, 12), -1, dtype=torch.int64, device=device),
        torch.zeros(2 * width, dtype=torch.int64, device=device),
        torch.zeros((2 * width, 1), dtype=torch.bool, device=device),
    )
    data, banks = dl.allocate(device), bl.allocate(device)
    data[1][0] = torch.tensor([[[1., -2., 3.], [4., .5, -1.]]], device=device)
    banks[0][0] = 2
    banks[1][0] = 3
    return wave, state, data, banks, torch.tensor([0, 1], device=device), torch.tensor([1, 1], device=device), torch.ones(len(families), dtype=torch.bool, device=device)


def _assert_native(query, wave, state, data, banks, native):
    assert not bool(state.frames.active.any())
    rows = torch.nonzero(state.frames.completed).flatten().tolist()
    assert len(rows) == len(native.branches) == 2
    dl, bl = wave.execution.dispatch.data_layout, wave.execution.dispatch.bank_layout
    for row, branch in zip(rows, native.branches, strict=True):
        route = tuple(wave.execution_names[i] for i in state.routes[row, :state.lengths[row]].tolist())
        assert route == tuple(step["candidate_id"] for step in branch.route)
        torch.testing.assert_close(state.scores[row], branch.log_probability)
        actual = dl.gather(data, state.frames.value_handles[row:row + 1, 0, 2], 0)[0]
        torch.testing.assert_close(actual, branch.execution.outputs["answer"])
        for index, value in enumerate(branch.execution.bank_state.values):
            bucket = bl.index(value)
            actual = bl.gather(banks, state.frames.bank_value_handles[row:row + 1, index], bucket)[0]
            torch.testing.assert_close(actual, value)
            assert state.frames.bank_revisions[row, index].item() == branch.execution.bank_state.revisions[index]


@pytest.mark.parametrize("encoded", [False, True])
def test_shape_changing_k_wide_search_matches_native(device, encoded):
    torch.manual_seed(151)
    query = _query(encoded).to(device)
    wave, state, data, banks, dc, bc, finite = _runtime(query, device)
    with torch.no_grad():
        native = search_recursive_graphs((start_recursive_search(query, {"x": data[1][0]}),), width=4, beam_width=4)
        for iteration in range(10):
            result = wave(state, data, banks, dc, bc, finite)
            assert not bool(result.requires_fallback)
            state, dc, bc = result.state, result.data_cursor, result.bank_cursor
            if iteration == 0:
                # Both true physical shapes coexist in one unfinished K frontier.
                handles = state.frames.value_handles[:4, 0, 1]
                assert bool(wave.execution.dispatch.data_layout.contains(handles, 0).any())
                assert bool(wave.execution.dispatch.data_layout.contains(handles, 1).any())
        _assert_native(query, wave, state, data, banks, native)


def test_typed_child_entry_checks_shapes_before_call(device):
    query = _query().to(device)
    wave, state, data, banks, _dc, _bc, finite = _runtime(query, device)
    # Existing but shape-incompatible value: a root Call may not enter a dead child.
    scalar_layout = FormulaDevicePoolLayout.from_samples((torch.zeros(1, 3), torch.zeros(1, 2, 3), torch.zeros(1)), 128)
    wave.execution.dispatch.prepare_typed_pools_(scalar_layout, wave.execution.dispatch.bank_layout)
    wave.decision.prepare_typed_pools_(wave.execution.dispatch)
    data = (*data, torch.zeros(129, 1, device=device))
    state.frames.value_handles[0, 0, 1] = scalar_layout.offsets[2]
    current = wave._rows(state.frames, torch.arange(4, device=device))
    result = wave.decision(current, data, banks, finite, state.scores[:4])
    assert not bool(result.eligible[0, 2])


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("encoded", [False, True])
def test_existing_captured_backend_runs_heterogeneous_search(batched, encoded):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from benchmarks._federated_captured_search import captured_search_execution

    with torch.device("cuda"), torch.no_grad():
        torch.manual_seed(991)
        query = _query(encoded)
        values = (torch.randn(1, 2, 3), torch.randn(1, 2, 3))
        batches = [[(start_recursive_search(query, {"x": value + shift}),) for value in values]
                   for shift in (0., 2.)]
        options = dict(width=4, beam_width=4, record_query_choices=True, record_effect_metrics=True)
        expected_batches = [search_recursive_graphs_many(starts, **options) for starts in batches]
        # A global pool unused by this Query must not invalidate its encoder.
        samples = (torch.zeros(1, 3), torch.zeros(1, 2, 3), torch.zeros(1, 5))
        with captured_search_execution(horizon=12, value_samples=samples) as backend:
            retained = []
            for starts, expected in zip(batches, expected_batches, strict=True):
                actual = (search_recursive_graphs_many(starts, **options) if batched else
                          tuple(search_recursive_graphs(start, **options) for start in starts))
                assert not backend.fallbacks
                for result, reference in zip(actual, expected, strict=True):
                    assert len(result.branches) == len(reference.branches)
                    for branch, native in zip(result.branches, reference.branches, strict=True):
                        assert tuple(row["candidate_id"] for row in branch.route) == tuple(row["candidate_id"] for row in native.route)
                        assert tuple(row.get("eligible_action_indices") for row in branch.route) == tuple(row.get("eligible_action_indices") for row in native.route)
                        torch.testing.assert_close(branch.log_probability, native.log_probability)
                        torch.testing.assert_close(branch.execution.outputs["answer"], native.execution.outputs["answer"])
                        assert branch.execution.bank_state.revisions == native.execution.bank_state.revisions
                        for a, b in zip(branch.execution.bank_state.values, native.execution.bank_state.values, strict=True):
                            torch.testing.assert_close(a, b)
                    for field in ("scored_expansions", "executed_expansions", "entered_calls", "returned_calls", "root_stops"):
                        assert getattr(result, field) == getattr(reference, field)
                retained.extend((branch.execution.outputs["answer"], branch.execution.outputs["answer"].clone())
                                for result in actual for branch in result.branches)
            for output, saved in retained:
                torch.testing.assert_close(output, saved, rtol=0, atol=0)
            assert backend.completed == 4
            assert backend.batched_searches == (4 if batched else 0)


def test_captured_typed_search_groups_distinct_input_shapes():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from benchmarks._federated_captured_search import captured_search_execution
    from test_formula_device_search import _shape_query

    with torch.device("cuda"), torch.no_grad():
        query = _shape_query(changing=True)
        starts = [(start_recursive_search(query, {"x": torch.randn(1, length, 3)}),)
                  for length in (1, 2, 1, 2)]
        options = dict(width=4, beam_width=4)
        expected = search_recursive_graphs_many(starts, **options)
        samples = (torch.zeros(1, 3), torch.zeros(1, 1, 3), torch.zeros(1, 2, 3))
        with captured_search_execution(horizon=8, value_samples=samples) as backend:
            actual = search_recursive_graphs_many(starts, **options)
            assert backend.batched_searches == 4
            assert len(backend.sessions) == 2
            assert not backend.fallbacks
        for result, reference in zip(actual, expected, strict=True):
            assert len(result.branches) == len(reference.branches)
            for branch, native in zip(result.branches, reference.branches, strict=True):
                assert branch.route == native.route
                torch.testing.assert_close(branch.execution.outputs["answer"], native.execution.outputs["answer"])
                torch.testing.assert_close(branch.log_probability, native.log_probability)


def test_unprepared_legal_typed_output_keeps_native_candidate():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from benchmarks._federated_captured_search import captured_search_execution
    from test_formula_device_search import _shape_query

    with torch.device("cuda"), torch.no_grad():
        query = _shape_query(changing=True)
        starts = (start_recursive_search(query, {"x": torch.randn(1, 2, 3)}),)
        options = dict(width=4, beam_width=4)
        expected = search_recursive_graphs(starts, **options)
        with captured_search_execution(horizon=8, value_samples=(torch.zeros(1, 2, 3),)) as backend:
            actual = search_recursive_graphs(starts, **options)
            assert backend.completed == 0
            assert backend.fallbacks == ["heterogeneous Formula binding bucket"]
        assert len(actual.branches) == len(expected.branches) == 2
        for branch, native in zip(actual.branches, expected.branches, strict=True):
            assert branch.route == native.route
            torch.testing.assert_close(branch.log_probability, native.log_probability)
            torch.testing.assert_close(branch.execution.outputs["answer"], native.execution.outputs["answer"])


@pytest.mark.parametrize("batched", [False, True])
def test_captured_search_rebuilds_rebound_query_storage(batched, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from benchmarks._federated_captured_search import _CAPTURED_SEARCH, captured_search_execution

    with torch.device("cuda"), torch.no_grad():
        query = _query(encoded=True)
        values = (torch.randn(1, 2, 3), torch.randn(1, 2, 3))
        options = dict(width=4, beam_width=4)

        def run():
            starts = [(start_recursive_search(query, {"x": value}),) for value in values]
            return (search_recursive_graphs_many(starts, **options) if batched else
                    tuple(search_recursive_graphs(start, **options) for start in starts))

        samples = (torch.zeros(1, 3), torch.zeros(1, 2, 3))
        with captured_search_execution(horizon=12, value_samples=samples) as backend:
            previous = None
            for change in ("initial", "in-place", "assign", "same-object-storage"):
                if change == "in-place":
                    for parameter in query.parameters():
                        parameter.add_(.03)
                elif change == "assign":
                    query.load_state_dict({key: value.clone() for key, value in query.state_dict().items()}, assign=True)
                elif change == "same-object-storage":
                    for parameter in query.parameters():
                        parameter.data = parameter.detach().clone().add_(.01)
                token = _CAPTURED_SEARCH.set(None)
                try:
                    expected = run()
                finally:
                    _CAPTURED_SEARCH.reset(token)
                traversals = []
                with monkeypatch.context() as patch:
                    if change == "in-place":
                        def count(method, name):
                            def wrapped(*args, **kwargs):
                                traversals.append(name)
                                return method(*args, **kwargs)
                            return wrapped
                        for name in ("parameters", "buffers"):
                            patch.setattr(query, name, count(getattr(query, name), name))
                    actual = run()
                if change == "in-place":
                    assert traversals == ["parameters", "buffers"] * (1 if batched else len(values))
                session = next(iter(backend.sessions.values()))
                if change == "in-place":
                    assert session is previous
                elif previous is not None:
                    assert session is not previous
                    assert previous.graph is None
                previous = session
                assert len(backend.sessions) == 1
                assert not backend.fallbacks
                for result, reference in zip(actual, expected, strict=True):
                    assert len(result.branches) == len(reference.branches)
                    for branch, native in zip(result.branches, reference.branches, strict=True):
                        assert branch.route == native.route
                        torch.testing.assert_close(branch.execution.outputs["answer"], native.execution.outputs["answer"])
                        torch.testing.assert_close(branch.log_probability, native.log_probability)
                        for a, b in zip(branch.execution.bank_state.values, native.execution.bank_state.values, strict=True):
                            torch.testing.assert_close(a, b)
