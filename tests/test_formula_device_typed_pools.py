import pytest
import torch

from arti import mechanisms as m
from arti._formula_device_dispatch import (
    FormulaDeviceDispatchLayout,
    FormulaDeviceNumericalDispatch,
    formula_device_dispatch_groups,
)
from arti._formula_device_execution import (
    FormulaDeviceCapturedExecutionWave,
    FormulaDeviceExecutionWave,
)
from arti._formula_device_frames import FormulaDeviceFrameKernel
from arti._formula_device_pools import FormulaDevicePoolLayout


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with torch.device(request.param):
        yield torch.device(request.param)


def _type(width):
    return m.TensorType(("B", "D"), ("B", width), dtype="floating", domain="activation")


def _query():
    x = m.InputBinding("x", _type(3))
    wide_type = m.concat(x, x, axis="D").value_type
    weight = m.BankBinding("weight", "arti/typed-pool-test@1", "weight", wide_type)
    wide = m.scale(m.concat(x, x, axis="D"), weight)
    program = m.FormulaProgram.build(outputs=(m.add(x, x), wide))
    producer = m.FormulaProgramTensorCandidateV4(
        m.FormulaProgramCandidateV3(
            "heads",
            program,
            input_slots={"x": "x"},
            output_slots=dict(zip(program.outputs, ("plain", "owned"), strict=True)),
            operands={"weight": torch.full((1, 6), 2.0)},
        ),
        plastic_bank_slot="weight",
        bank_owner_id="memory",
    )
    reread = m.FormulaProgramTensorCandidateV4(
        m.FormulaProgramCandidateV3(
            "reread",
            program,
            input_slots={"x": "x"},
            output_slots=dict(zip(program.outputs, ("plain_read", "read"), strict=True)),
            operands={"weight": torch.full((1, 6), 2.0)},
        ),
        plastic_bank_slot="weight",
        bank_owner_id="memory",
    )
    y = m.InputBinding("y", wide_type)
    rate = m.BankBinding("rate", "arti/typed-pool-test@1", "rate", wide_type)
    zero = m.BankBinding("zero", "arti/typed-pool-test@1", "zero", wide_type)
    effect = m.FormulaProgramEffectCandidateV3(
        "write",
        m.FormulaEffectProgramV2(
            m.FormulaProgram.build(outputs=(m.neural_plasticity(y, m.scale(y, rate), zero),)),
            data_input_name="y",
            state_type=wide_type,
        ),
        input_slot="owned",
        output_slot="tail",
        operands={"rate": torch.full((1, 6), 0.1), "zero": torch.zeros(1, 6)},
    )
    total = m.FormulaProgramTensorCandidateV3(
        m.FormulaProgramCandidateV2(
            "sum",
            m.FormulaProgram.build(outputs=(m.reduce_sum(y, axis="D"),)),
            input_slots={"y": "read"},
            output_slot="sum",
        )
    )
    return m.FormulaProgramQueryV5(
        slot_ids=("x", "plain", "owned", "tail", "read", "sum", "plain_read"),
        candidates=(producer, effect, reread, total),
        terminal_slots={"answer": "sum"},
        max_steps=4,
        hidden_dim=8,
    )


def _runtime(device, capacities=(8, 8, 8)):
    query = _query().to(device)
    kernel = FormulaDeviceFrameKernel.from_query(query).to(device)
    data_layout = FormulaDevicePoolLayout.from_samples(
        (torch.zeros(1, 3), torch.zeros(1, 6), torch.zeros(1)),
        capacities,
    )
    bank_layout = FormulaDevicePoolLayout.from_samples(
        (torch.zeros(1, 3), torch.zeros(1, 6)), (4, 4)
    )
    dispatch = FormulaDeviceNumericalDispatch(query, kernel).to(device)
    dispatch.prepare_typed_pools_(data_layout, bank_layout)
    wave = FormulaDeviceExecutionWave(
        dispatch, kernel, data_capacity=data_layout.capacities, bank_capacity=bank_layout.capacities
    )
    packet_layout = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(query)[0]).to(device)
    data, banks = data_layout.allocate(device), bank_layout.allocate(device)
    data[0][0] = torch.tensor([[1.0, 2.0, 3.0]])
    banks[1][0] = 2
    state = kernel.initial_state(
        1,
        torch.tensor([[0, -1, -1, -1, -1, -1, -1]], device=device),
        bank_value_handles=torch.tensor([bank_layout.offsets[1]], device=device),
    )
    return (
        query,
        packet_layout,
        wave,
        state,
        data,
        banks,
        torch.tensor([1, 0, 0], device=device),
        torch.tensor([0, 1], device=device),
    )


def test_typed_multihead_write_and_reread_matches_native(device):
    query, layout, wave, state, data, banks, dc, bc = _runtime(device)
    arena = query._arena({"x": data[0][0].clone()})
    with torch.no_grad():
        for index, slots in enumerate(
            (("plain", "owned"), ("tail",), ("plain_read", "read"), ("sum",))
        ):
            arena = query.candidates[index](arena)
            result = wave(
                state, layout(torch.tensor([[index]], device=device)), data, banks, dc, bc
            )
            assert result.event.accepted.tolist() == [True]
            for head, slot in enumerate(slots):
                expected = arena.values.get(slot)
                bucket = wave.dispatch.data_layout.index(expected)
                actual = wave.dispatch.data_layout.gather(
                    data, result.pool.output_handles[:, head], bucket
                )[0]
                torch.testing.assert_close(actual, expected)
            if index == 1:
                assert result.pool.output_handles[0, 0] == state.value_handles[0, 0, 2]
                assert torch.equal(result.pool.data_cursor, dc)
                assert result.state.bank_revisions.tolist() == [[1]]
                torch.testing.assert_close(
                    banks[1][1],
                    2 + torch.tensor([[2.0, 4.0, 6.0, 2.0, 4.0, 6.0]], device=device) * 0.1,
                )
            state, dc, bc = result.state, result.pool.data_cursor, result.pool.bank_cursor
    assert dc.tolist() == [3, 2, 1]
    assert bc.tolist() == [0, 2]
    assert [tuple(pool.shape[1:]) for pool in data] == [(1, 3), (1, 6), (1,)]
    torch.testing.assert_close(banks[1][0], torch.full((1, 6), 2.0, device=device))


def test_typed_overflow_does_not_publish_other_head(device):
    _q, layout, wave, state, data, banks, dc, bc = _runtime(device, (1, 8, 8))
    before = tuple(pool.clone() for pool in (*data, *banks))
    result = wave(state, layout(torch.tensor([[0]], device=device)), data, banks, dc, bc)
    assert bool(result.pool.overflow)
    assert result.event.accepted.tolist() == [False]
    assert torch.equal(result.pool.data_cursor, dc) and torch.equal(result.pool.bank_cursor, bc)
    for actual, expected in zip((*data, *banks), before, strict=True):
        assert torch.equal(actual, expected)


def test_legal_missing_output_pool_is_not_dropped(device):
    query = _query().to(device)
    dispatch = FormulaDeviceNumericalDispatch.from_query(query).to(device)
    with pytest.raises(m.FormulaBindingError, match="no prepared shape/dtype pool"):
        dispatch.prepare_typed_pools_(
            FormulaDevicePoolLayout.from_samples((torch.zeros(1, 3),), 8),
            FormulaDevicePoolLayout.from_samples((torch.zeros(1, 6),), 4),
        )


def test_one_k_wide_wave_keeps_shape_and_dtype_variants(device):
    kind = m.TensorType(("B", "D"), ("B", "D"), dtype="floating", domain="activation")
    x = m.InputBinding("x", kind)
    candidates = tuple(
        m.FormulaProgramTensorCandidateV3(
            m.FormulaProgramCandidateV2(
                name,
                m.FormulaProgram.build(outputs=(expr,)),
                input_slots={"x": "x"},
                output_slot=name,
            )
        )
        for name, expr in (("double", m.add(x, x)), ("triple", m.add(x, m.add(x, x))))
    )
    query = m.FormulaProgramQueryV5(
        slot_ids=("x", "double", "triple"),
        candidates=candidates,
        terminal_slots={"a": "double"},
        max_steps=1,
    ).to(device)
    kernel = FormulaDeviceFrameKernel.from_query(query).to(device)
    samples = (
        torch.zeros(1, 3),
        torch.zeros(1, 6),
        torch.zeros(1, 3, dtype=torch.bfloat16),
        torch.zeros(()),
    )
    dl = FormulaDevicePoolLayout.from_samples(samples, 4)
    bl = FormulaDevicePoolLayout.from_samples((torch.zeros(1),), 1)
    dispatch = FormulaDeviceNumericalDispatch(query, kernel).to(device)
    dispatch.prepare_typed_pools_(dl, bl)
    wave = FormulaDeviceExecutionWave(
        dispatch, kernel, data_capacity=dl.capacities, bank_capacity=bl.capacities
    )
    data, banks = dl.allocate(device), bl.allocate(device)
    for pool in data:
        pool[0] = 2
    state = kernel.initial_state(
        3, torch.tensor([[offset, -1, -1] for offset in dl.offsets[:3]], device=device)
    )
    layout = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(query)[0]).to(device)
    packet = layout(torch.tensor([[0, 1], [0, 1], [0, 1]], device=device))
    result = wave(
        state,
        packet,
        data,
        banks,
        torch.tensor([1, 1, 1, 1], device=device),
        torch.tensor([0], device=device),
    )
    assert result.event.accepted.tolist() == [True] * 6
    assert result.pool.data_cursor.tolist() == [3, 3, 3, 1]
    for lane in range(6):
        bucket = int(packet.source_rows[lane])
        expected = 4 if int(packet.candidate_ids[lane]) == 0 else 6
        actual = dl.gather(data, result.pool.output_handles[lane : lane + 1, 0], bucket)[0]
        assert actual.shape == samples[bucket].shape and actual.dtype == samples[bucket].dtype
        torch.testing.assert_close(actual, torch.full_like(actual, expected))
    # A scalar belongs to another real pool but does not satisfy this Formula's rank.
    state = kernel.initial_state(1, torch.tensor([[dl.offsets[3], -1, -1]], device=device))
    result = wave(
        state,
        layout(torch.tensor([[0]], device=device)),
        data,
        banks,
        result.pool.data_cursor,
        result.pool.bank_cursor,
    )
    assert result.event.accepted.tolist() == [False]
    assert result.pool.data_cursor.tolist() == [3, 3, 3, 1]


def test_child_call_returns_typed_result_and_changed_bank(device):
    child, _layout, prepared, _state, data, banks, dc, bc = _runtime(device)
    parent = m.FormulaProgramQueryV5(
        slot_ids=("x", "answer"),
        candidates=(
            m.FormulaProgramCallCandidateV1(
                "call",
                child,
                input_slots={"x": "x"},
                output_slots={"answer": "answer"},
            ),
        ),
        terminal_slots={"answer": "answer"},
        max_steps=1,
    ).to(device)
    kernel = FormulaDeviceFrameKernel.from_query(parent).to(device)
    dispatch = FormulaDeviceNumericalDispatch(parent, kernel).to(device)
    dispatch.prepare_typed_pools_(prepared.dispatch.data_layout, prepared.dispatch.bank_layout)
    wave = FormulaDeviceExecutionWave(
        dispatch, kernel, data_capacity=prepared.data_capacity, bank_capacity=prepared.bank_capacity
    )
    state = kernel.initial_state(
        1,
        torch.tensor([[0, -1]], device=device),
        bank_value_handles=torch.tensor([dispatch.bank_layout.offsets[1]], device=device),
    )
    layout = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(parent)[0]).to(device)
    # Root Call/Stop precede the child's four candidates and Stop in DFS order.
    for action in (0, 2, 3, 4, 5, 6, 1):
        result = wave(state, layout(torch.tensor([[action]], device=device)), data, banks, dc, bc)
        assert result.event.accepted.tolist() == [True]
        state, dc, bc = result.state, result.pool.data_cursor, result.pool.bank_cursor
    assert state.completed.tolist() == [True]
    assert state.bank_revisions.tolist() == [[1]]
    actual = dispatch.data_layout.gather(data, state.value_handles[:, 0, 1], 2)[0]
    with torch.no_grad():
        arena = child._arena({"x": data[0][0]})
        for candidate in child.candidates:
            arena = candidate(arena)
    torch.testing.assert_close(actual, arena.values.get("sum"))


def test_typed_capture_reuses_real_shapes_and_refreshes_inputs():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device("cuda")
    with torch.device(device):
        _q, layout, wave, state, data, banks, dc, bc = _runtime(device)
        packet = layout(torch.tensor([[0]], device=device))
        captured = FormulaDeviceCapturedExecutionWave.capture(
            wave, state, packet, data, banks, dc, bc
        )
        try:
            pointers = tuple(pool.data_ptr() for pool in captured.data_pool)
            for scale in (1.0, 3.0):
                data[0][0] = scale
                captured.copy_inputs_(data_pool=data, bank_pool=banks)
                result = captured.replay()
                assert result.event.accepted.tolist() == [True]
                torch.testing.assert_close(
                    captured.data_pool[0][1], torch.full((1, 3), scale * 2, device=device)
                )
                torch.testing.assert_close(
                    captured.data_pool[1][0], torch.full((1, 6), scale * 2, device=device)
                )
                assert pointers == tuple(pool.data_ptr() for pool in captured.data_pool)
        finally:
            captured.close()


def test_capture_complete_shape_change_write_reread_chain():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device("cuda")
    with torch.device(device):
        _q, layout, wave, state, data, banks, dc, bc = _runtime(device)
        packets = tuple(layout(torch.tensor([[action]], device=device)) for action in range(4))

        class Chain(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.wave = wave
                self.dispatch = wave.dispatch

            def forward(self, state, packet, data, banks, dc, bc):
                for current in (packet, *packets[1:]):
                    result = self.wave(state, current, data, banks, dc, bc)
                    state, dc, bc = result.state, result.pool.data_cursor, result.pool.bank_cursor
                return result

        captured = FormulaDeviceCapturedExecutionWave.capture(
            Chain(), state, packets[0], data, banks, dc, bc
        )
        try:
            for value in (1.0, 2.0):
                data[0][0] = value
                captured.copy_inputs_(data_pool=data, bank_pool=banks)
                result = captured.replay()
                assert result.event.accepted.tolist() == [True]
                assert result.state.bank_revisions.tolist() == [[1]]
                torch.testing.assert_close(
                    captured.bank_pool[1][1], torch.full((1, 6), 2 + 0.2 * value, device=device)
                )
                torch.testing.assert_close(
                    captured.data_pool[2][0],
                    torch.tensor([6 * value * (2 + 0.2 * value)], device=device),
                )
                assert result.pool.data_cursor.tolist() == [3, 2, 1]
        finally:
            captured.close()
