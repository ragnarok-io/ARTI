import pytest
import torch

from arti import mechanisms as m
from arti._formula_device_admission import FormulaDeviceAdmission
from arti._formula_device_dispatch import (
    FormulaDeviceDispatchLayout,
    FormulaDeviceNumericalDispatch,
    formula_device_dispatch_groups,
)
from arti._formula_device_execution import FormulaDeviceExecutionWave
from arti._formula_device_execution import FormulaDeviceCapturedExecutionWave
from arti._formula_device_pools import FormulaDevicePoolLayout
from arti._formula_device_frames import FormulaDeviceFrameKernel
from arti._formula_device_sources import (
    FormulaDeviceSourceBindings,
    FormulaDeviceSources,
    completed_sources,
    select_sources,
)
from test_formula_device_dispatch import _effect, _producer, _type


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with torch.device(request.param):
        yield request.param


def _join(input_slots=None):
    a, b = m.InputBinding("a", _type()), m.InputBinding("b", _type())
    coefficient = m.BankBinding("c", "arti/device-sources-test@1", "c", _type())
    program = m.FormulaProgram.build(outputs=(m.add(a, m.scale(b, coefficient)),))
    return m.FormulaProgramTensorCandidateV3(m.FormulaProgramCandidateV2(
        "join", program, input_slots={"b": "owned", "a": "owned"} if input_slots is None else input_slots,
        output_slot="answer", operands={"c": torch.full((1, 3), -3.0)},
    ))


def _setup(device):
    query = m.FormulaProgramQueryV5(
        slot_ids=("x", "owned", "right", "answer", "tail"),
        candidates=(
            _producer("left", "owned", "left-memory", weight_value=2.0),
            _producer("right", "right", "right-memory", weight_value=5.0),
            _join(), _effect(),
        ), terminal_slots={"y": "answer"}, max_steps=8,
    ).to(device)
    kernel = FormulaDeviceFrameKernel.from_query(query).to(device)
    layout = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(query)[0]).to(device)
    dispatch = FormulaDeviceNumericalDispatch.from_query(query, frame_kernel=kernel).to(device)
    wave = FormulaDeviceExecutionWave(dispatch, kernel, data_capacity=16, bank_capacity=8).to(device)
    state = kernel.initial_state(1, torch.tensor([[0, -1, -1, -1, -1]]),
                                 bank_value_handles=torch.tensor([0, 1]))
    data, bank = torch.zeros(17, 1, 3), torch.zeros(9, 1, 3)
    data[0] = torch.tensor([[1., 2., 3.]])
    bank[0], bank[1] = 2., 5.
    packet = layout(torch.tensor([[0, 1]]))
    produced = wave(state, packet, data, bank, torch.tensor(1), torch.tensor(2))
    directory = completed_sources(kernel, produced, packet, torch.tensor([10, 11]))
    return query, kernel, layout, wave, state, data, bank, produced, directory


def _resolve(kernel, state, directory, *, reverse=False):
    refs = torch.full((1, kernel.candidate_count, kernel.spec.max_ports), -1)
    refs[0, 2] = torch.tensor([1, 0] if not reverse else [0, 1])  # input order b, a
    refs[0, 3, 0] = 0
    occurrences = torch.full_like(refs, -1)
    ports = torch.full_like(refs, -1)
    indices = refs.clamp_min(0)
    occurrences.copy_(directory.occurrence[indices])
    ports.copy_(directory.port[indices])
    resolver = FormulaDeviceSourceBindings(kernel).to(refs.device)
    local_finite = state.value_handles[:, 0] >= 0
    return resolver, (state, local_finite, refs, occurrences, ports, directory)


def _admit(query, kernel, state, sources):
    admission = FormulaDeviceAdmission.from_query(query, kernel).to(state.active.device)
    return admission(state, state.value_handles[:, 0] >= 0,
                     torch.ones_like(state.bank_value_handles, dtype=torch.bool),
                     torch.ones(kernel.candidate_count, dtype=torch.bool),
                     reference_flags=True, input_sources=sources)


def test_published_ports_feed_real_join_without_installing_donor_frame(device):
    query, kernel, layout, wave, state, data, bank, produced, directory = _setup(device)
    resolver, args = _resolve(kernel, state, directory)
    sources = resolver(*args)
    assert _admit(query, kernel, state, sources)[0, 2]
    packet = layout(torch.tensor([[2]]))
    packed = select_sources(sources, packet.source_rows, packet.candidate_ids)
    result = wave(state, packet, data, bank, produced.pool.data_cursor,
                  produced.pool.bank_cursor, input_sources=packed)
    assert result.event.accepted.tolist() == [True]
    torch.testing.assert_close(data[result.pool.output_handles[0, 0]], -13 * data[0])
    assert result.state.value_handles[0, 0, 1] == -1
    assert result.state.value_handles[0, 0, 2] == -1
    torch.testing.assert_close(result.state.bank_value_handles, state.bank_value_handles)
    assert directory.occurrence.tolist() == [10, 11]


def test_same_candidate_distinct_lane_sources_survive_group_packing(device):
    _, kernel, layout, wave, state, data, bank, produced, directory = _setup(device)
    resolver, args = _resolve(kernel, state, directory)
    forward = resolver(*args)
    resolver, args = _resolve(kernel, state, directory, reverse=True)
    reverse = resolver(*args)
    packet = layout(torch.tensor([[2, 0, 2]]))
    # Packet order is left producer followed by both joins, not selection order.
    lane_sources = FormulaDeviceSources(*(torch.stack((a[0, 2], a[0, 0], b[0, 2]))
                                         for a, b in zip(forward, reverse, strict=True)))
    packed = FormulaDeviceSources(*(field.index_select(0, packet.source_lanes) for field in lane_sources))
    result = wave(state, packet, data, bank, produced.pool.data_cursor,
                  produced.pool.bank_cursor, input_sources=packed)
    assert result.event.accepted.all()
    selected_outputs = result.pool.output_handles.index_select(0, packet.inverse_order)[:, 0]
    torch.testing.assert_close(data[selected_outputs[0]], -13 * data[0])
    torch.testing.assert_close(data[selected_outputs[2]], -data[0])


@pytest.mark.parametrize("failure", ["pending", "occurrence", "port", "range", "unallocated"])
def test_unavailable_sources_do_not_fall_back_to_local_inputs(device, failure):
    query, kernel, layout, wave, state, data, bank, produced, directory = _setup(device)
    resolver, args = _resolve(kernel, state, directory)
    state, local, refs, occurrences, ports, directory = args
    if failure == "pending":
        directory.ready[1] = False
    elif failure == "occurrence":
        occurrences[0, 2, 0] += 1
    elif failure == "port":
        ports[0, 2, 0] += 1
    elif failure == "range":
        refs[0, 2, 0] = 100
    else:
        directory.handles[1] = 14
    sources = resolver(state, local, refs, occurrences, ports, directory)
    if failure != "unallocated":
        assert not _admit(query, kernel, state, sources)[0, 2]
    packet = layout(torch.tensor([[2]]))
    packed = select_sources(sources, packet.source_rows, packet.candidate_ids)
    result = wave(state, packet, data, bank, produced.pool.data_cursor,
                  produced.pool.bank_cursor, input_sources=packed)
    assert not result.event.accepted.any()
    assert result.pool.data_cursor == produced.pool.data_cursor


def test_imported_effect_preserves_actual_predecessor_and_old_version(device):
    query, kernel, layout, wave, state, data, bank, produced, directory = _setup(device)
    resolver, args = _resolve(kernel, state, directory)
    sources = resolver(*args)
    assert _admit(query, kernel, state, sources)[0, 3]
    packet = layout(torch.tensor([[3]]))
    packed = select_sources(sources, packet.source_rows, packet.candidate_ids)
    result = wave(state, packet, data, bank, produced.pool.data_cursor,
                  produced.pool.bank_cursor, input_sources=packed)
    assert result.event.accepted.tolist() == [True]
    assert result.event.target_bank.tolist() == [0]
    assert result.pool.output_handles[0, 0] == directory.handles[0]
    torch.testing.assert_close(bank[result.pool.bank_handles[0]], bank[0] + .1 * data[directory.handles[0]])
    changed = state._replace(bank_value_handles=result.state.bank_value_handles,
                             bank_revisions=result.state.bank_revisions)
    assert not _admit(query, kernel, changed, sources)[0, 3]
    stale = wave(changed, packet, data, bank, result.pool.data_cursor,
                 result.pool.bank_cursor, input_sources=packed)
    assert not stale.event.accepted.any()
    assert stale.pool.bank_cursor == result.pool.bank_cursor
    assert _admit(query, kernel, changed, sources)[0, 2]  # old data still usable


def test_source_resolver_strict_export(device):
    _, kernel, _, _, state, _, _, _, directory = _setup(device)
    resolver, args = _resolve(kernel, state, directory)
    args = (tuple(args[0]), *args[1:-1], tuple(args[-1]))
    exported = torch.export.export(resolver, args, strict=True).module()
    for actual, expected in zip(exported(*args), resolver(*args), strict=True):
        torch.testing.assert_close(actual, expected)


def test_sources_enter_selection_before_group_dispatch(device):
    from arti._formula_device_dispatch import FormulaDeviceRoutedDecisionWave

    query, kernel, _, _, state, data, bank, _, directory = _setup(device)
    with torch.no_grad():
        for parameter in query.network.parameters():
            parameter.zero_()
        query.network[-1].bias[2] = 20
    routed = FormulaDeviceRoutedDecisionWave.from_query(
        query, frame_kernel=kernel, candidate_family_ids=[0] * kernel.candidate_count,
        width=1,
    ).to(device)
    resolver, args = _resolve(kernel, state, directory)
    sources = resolver(*args)
    common = (state, data, bank, torch.ones(kernel.candidate_count, dtype=torch.bool), torch.zeros(1))
    selected = routed(*common, input_sources=sources)
    assert selected.selected_candidates.tolist() == [[2]]
    directory.ready[1] = False
    selected = routed(*common, input_sources=resolver(*args))
    assert selected.selected_candidates.tolist() != [[2]]


def test_typed_execution_uses_selected_ports_and_joint_shape_admission(device):
    query, kernel, layout, _, state, data, bank, produced, directory = _setup(device)
    data_layout = FormulaDevicePoolLayout.from_samples((torch.zeros(1, 3), torch.zeros(1, 6)), (16, 4))
    bank_layout = FormulaDevicePoolLayout.from_samples((torch.zeros(1, 3),), (8,))
    dispatch = FormulaDeviceNumericalDispatch(query, kernel).to(device)
    dispatch.prepare_typed_pools_(data_layout, bank_layout)
    wave = FormulaDeviceExecutionWave(dispatch, kernel, data_capacity=(16, 4), bank_capacity=(8,))
    pools = (data, torch.ones(5, 1, 6))
    resolver, args = _resolve(kernel, state, directory)
    sources = resolver(*args)
    admission = FormulaDeviceAdmission.from_query(query, kernel).to(device)
    admission.shape_admission = dispatch.typed_admission
    flags = (state.value_handles[:, 0] >= 0, torch.ones(1, 2, dtype=torch.bool),
             torch.ones(kernel.candidate_count, dtype=torch.bool))
    assert admission(state, *flags, reference_flags=True, input_sources=sources)[0, 2]
    packet = layout(torch.tensor([[2]]))
    result = wave(state, packet, pools, (bank,), torch.stack((produced.pool.data_cursor, torch.tensor(1))),
                  produced.pool.bank_cursor[None], input_sources=select_sources(sources, packet.source_rows, packet.candidate_ids))
    assert result.event.accepted.all()
    torch.testing.assert_close(data[result.pool.output_handles[0, 0]], -13 * data[0])
    directory.handles[1] = data_layout.offsets[1]
    sources = resolver(*args)
    assert not admission(state, *flags, reference_flags=True, input_sources=sources)[0, 2]
    result = wave(state, packet, pools, (bank,), result.pool.data_cursor,
                  result.pool.bank_cursor, input_sources=select_sources(sources, packet.source_rows, packet.candidate_ids))
    assert not result.event.accepted.any()


def test_call_imports_port_values_and_lineage_but_not_donor_bank(device):
    child = m.FormulaProgramQueryV5(
        slot_ids=("x", "owned", "answer"), candidates=(_join(),),
        terminal_slots={"y": "answer"}, max_steps=1,
    ).to(device)
    call = m.FormulaProgramCallCandidateV1(
        "call", child, input_slots={"owned": "x"}, output_slots={"y": "answer"},
    )
    query = m.FormulaProgramQueryV5(slot_ids=("x", "answer"), candidates=(call,),
                                  terminal_slots={"y": "answer"}, max_steps=1).to(device)
    kernel = FormulaDeviceFrameKernel.from_query(query).to(device)
    state = kernel.initial_state(1, torch.tensor([[0, -1, -1]]))
    refs = torch.full((1, kernel.candidate_count, kernel.spec.max_ports), -1)
    refs[0, 0, 0] = 0
    directory = FormulaDeviceSources(
        torch.tensor([7]), torch.tensor([True]), torch.tensor([True]),
        torch.tensor([1]), torch.tensor([-1]), torch.tensor([-1]), torch.tensor([-1]),
        torch.tensor([12]), torch.tensor([0]),
    )
    resolver = FormulaDeviceSourceBindings(kernel).to(device)
    sources = resolver(state, state.value_handles[:, 0] >= 0, refs,
                       torch.full_like(refs, 12), torch.zeros_like(refs), directory)
    assert _admit(query, kernel, state, sources)[0, 0]
    packed = select_sources(sources, torch.tensor([0]), torch.tensor([0]))
    entered, event = kernel(state, torch.tensor([0]), torch.tensor([[-1]]),
                            torch.tensor([True]), torch.tensor([-1]), packed)
    assert event.accepted.all()
    assert entered.value_handles[0, 1, 1] == 7
    assert entered.producer_candidate[0, 1, 1] == 1
    assert entered.response_candidate[0, 1, 1] == -1
    assert entered.value_handles[0, 0, 0] == 0


def test_cuda_capture_changes_sources_without_changing_buffers():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with torch.device("cuda"):
        _, kernel, layout, wave, state, data, bank, produced, directory = _setup("cuda")
        resolver, args = _resolve(kernel, state, directory)
        sources = resolver(*args)
        packet = layout(torch.tensor([[2]]))
        packed = select_sources(sources, packet.source_rows, packet.candidate_ids)
        captured = FormulaDeviceCapturedExecutionWave.capture(
            wave, state, packet, data, bank, produced.pool.data_cursor, produced.pool.bank_cursor,
            input_sources=packed, warmup_steps=1,
        )
        try:
            addresses = tuple(field.data_ptr() for field in captured.input_sources)
            result = captured.replay()
            assert result.event.accepted.all()
            torch.testing.assert_close(captured.data_pool[result.pool.output_handles[0, 0]], -13 * data[0])
            resolver, args = _resolve(kernel, state, directory, reverse=True)
            packed = select_sources(resolver(*args), packet.source_rows, packet.candidate_ids)
            captured.copy_inputs_(input_sources=packed)
            result = captured.replay()
            torch.testing.assert_close(captured.data_pool[result.pool.output_handles[0, 0]], -data[0])
            assert tuple(field.data_ptr() for field in captured.input_sources) == addresses
        finally:
            captured.close()


def test_search_wave_remaps_sources_after_pruning_and_forking(device):
    from arti._formula_device_search import FormulaDeviceSearchState, FormulaDeviceSearchWave

    query, kernel, _, execution, state, data, bank, produced, directory = _setup(device)
    with torch.no_grad():
        for parameter in query.network.parameters():
            parameter.zero_()
        query.network[-1].bias[2] = 20
    wave = FormulaDeviceSearchWave(
        query, execution, width=2, local_width=1,
        candidate_family_ids=[0] * kernel.candidate_count,
        candidate_membership=[[False]] * kernel.candidate_count,
        preserve_coverage=False,
    ).to(device)
    resolver, args = _resolve(kernel, state, directory)
    forward = resolver(*args)
    resolver, args = _resolve(kernel, state, directory, reverse=True)
    reverse = resolver(*args)
    sources = FormulaDeviceSources(*(torch.cat((a, b, a, b))
                                    for a, b in zip(forward, reverse, strict=True)))
    frames = execution._fork_state(state, torch.zeros(4, dtype=torch.int64))
    frames.active[2:] = False
    search = FormulaDeviceSearchState(
        frames, torch.tensor([0., 10., 0., 0.]), torch.tensor([0., 10., 0., 0.]),
        torch.full((4, 4), -1), torch.zeros(4, dtype=torch.int64), torch.zeros(4, 1, dtype=torch.bool),
    )
    result = wave(search, data, bank, produced.pool.data_cursor, produced.pool.bank_cursor,
                  torch.ones(kernel.candidate_count, dtype=torch.bool), input_sources=sources)
    assert not result.requires_fallback
    assert result.parent_rows[:2].tolist() == [1, 0]
    outputs = result.state.frames.value_handles[:2, 0, 3]
    torch.testing.assert_close(data[outputs[0]], -data[0])
    torch.testing.assert_close(data[outputs[1]], -13 * data[0])
    assert result.state.frames.value_handles[:2, 0, 1].tolist() == [-1, -1]


def test_joint_symbolic_shapes_reject_only_mixed_port_tuple(device):
    shape = m.TensorType(("B", "D"), ("B", "D"), dtype="floating", domain="activation")
    a, b = m.InputBinding("a", shape), m.InputBinding("b", shape)
    candidate = m.FormulaProgramTensorCandidateV3(m.FormulaProgramCandidateV2(
        "sum", m.FormulaProgram.build(outputs=(m.add(a, b),)),
        input_slots={"a": "x", "b": "x"}, output_slot="answer",
    ))
    query = m.FormulaProgramQueryV5(slot_ids=("x", "answer"), candidates=(candidate,),
                                  terminal_slots={"y": "answer"}, max_steps=1).to(device)
    kernel = FormulaDeviceFrameKernel.from_query(query).to(device)
    layout = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(query)[0]).to(device)
    data_layout = FormulaDevicePoolLayout.from_samples((torch.zeros(1, 3), torch.zeros(1, 6)), 8)
    bank_layout = FormulaDevicePoolLayout.from_samples((torch.zeros(1, 3),), 1)
    dispatch = FormulaDeviceNumericalDispatch(query, kernel).to(device)
    dispatch.prepare_typed_pools_(data_layout, bank_layout)
    wave = FormulaDeviceExecutionWave(dispatch, kernel, data_capacity=(8, 8), bank_capacity=(1,))
    state = kernel.initial_state(3, torch.tensor([[0, -1]]).expand(3, -1))
    handles = torch.tensor([[[0, 0], [-1, -1]], [[9, 9], [-1, -1]], [[0, 9], [-1, -1]]])
    sources = FormulaDeviceSources(handles, torch.ones_like(handles, dtype=torch.bool),
                                   torch.ones_like(handles, dtype=torch.bool),
                                   *(torch.full_like(handles, -1) for _ in range(6)))
    mask = dispatch.typed_admission(state.value_handles[:, 0], state.producer_bank_handle[:, 0],
                                    state.bank_value_handles, sources)
    assert mask[:, 0].tolist() == [True, True, False]
    data, banks = data_layout.allocate(device), bank_layout.allocate(device)
    data[0][0], data[1][0] = 2, 7
    packet = layout(torch.zeros(3, 1, dtype=torch.int64))
    result = wave(state, packet, data, banks, torch.tensor([1, 1]), torch.tensor([0]),
                  input_sources=select_sources(sources, packet.source_rows, packet.candidate_ids))
    assert result.event.accepted.tolist() == [True, True, False]
    torch.testing.assert_close(data[0][1], torch.full((1, 3), 4.0))
    torch.testing.assert_close(data[1][1], torch.full((1, 6), 14.0))


def test_real_two_port_call_round_trip_uses_imported_payloads(device):
    child_join = _join(input_slots={"a": "lhs", "b": "rhs"})
    child = m.FormulaProgramQueryV5(slot_ids=("lhs", "rhs", "answer"),
                                  candidates=(child_join,), terminal_slots={"y": "answer"}, max_steps=1)
    call = m.FormulaProgramCallCandidateV1("call", child,
                                         input_slots={"rhs": "right", "lhs": "owned"},
                                         output_slots={"y": "answer"})
    query = m.FormulaProgramQueryV5(
        slot_ids=("x", "owned", "right", "answer"),
        candidates=(_producer("left", "owned", "a", weight_value=2.0),
                    _producer("right", "right", "b", weight_value=5.0), call),
        terminal_slots={"y": "answer"}, max_steps=3,
    ).to(device)
    kernel = FormulaDeviceFrameKernel.from_query(query).to(device)
    layout = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(query)[0]).to(device)
    dispatch = FormulaDeviceNumericalDispatch(query, kernel).to(device)
    wave = FormulaDeviceExecutionWave(dispatch, kernel, data_capacity=16, bank_capacity=4)
    state = kernel.initial_state(1, torch.tensor([[0, -1, -1, -1]]), bank_value_handles=torch.tensor([0, 1]))
    data, banks = torch.zeros(17, 1, 3), torch.zeros(5, 1, 3)
    data[0], banks[0], banks[1] = 1, 2, 5
    packet = layout(torch.tensor([[0, 1]]))
    result = wave(state, packet, data, banks, torch.tensor(1), torch.tensor(2))
    directory = completed_sources(kernel, result, packet, torch.tensor([30, 31]))
    refs = torch.full((1, kernel.candidate_count, kernel.spec.max_ports), -1)
    refs[0, 2] = torch.tensor([1, 0])
    resolver = FormulaDeviceSourceBindings(kernel).to(device)
    sources = resolver(state, state.value_handles[:, 0] >= 0, refs,
                       directory.occurrence[refs.clamp_min(0)], torch.zeros_like(refs), directory)
    packet = layout(torch.tensor([[2]]))
    result = wave(state, packet, data, banks, result.pool.data_cursor, result.pool.bank_cursor,
                  input_sources=select_sources(sources, packet.source_rows, packet.candidate_ids))
    assert result.event.accepted.all()
    assert result.state.producer_bank[0, 1, :2].tolist() == [0, 1]
    for action in (kernel.candidate_id(1, 0), kernel.candidate_id(1, 1)):
        result = wave(result.state, layout(torch.tensor([[action]])), data, banks,
                      result.pool.data_cursor, result.pool.bank_cursor)
        assert result.event.accepted.all()
    assert result.state.depth.tolist() == [0]
    torch.testing.assert_close(data[result.state.value_handles[0, 0, 3]], torch.full((1, 3), -13.0))
    assert result.state.value_handles[0, 0, 1:3].tolist() == [-1, -1]
