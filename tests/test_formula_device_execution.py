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


@pytest.fixture(params=["cpu", "cuda"], autouse=True)
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with torch.device(request.param):
        yield request.param


def _type():
    return m.TensorType(("B", "D"), ("B", 3), dtype="floating", domain="activation")


def _producer(name, output, owner, weight):
    x = m.InputBinding("x", _type())
    value = m.BankBinding("value", "arti/device-execution-test@1", "value", _type())
    return m.FormulaProgramTensorCandidateV3(
        m.FormulaProgramCandidateV2(
            name,
            m.FormulaProgram.build(outputs=(m.scale(x, value),)),
            input_slots={"x": "x"},
            output_slot=output,
            operands={"value": torch.full((1, 3), weight)},
        ),
        plastic_bank_slot="value",
        bank_owner_id=owner,
    )


def _effect():
    x = m.InputBinding("x", _type())
    rate = m.BankBinding("rate", "arti/device-execution-test@1", "rate", _type())
    zero = m.BankBinding("zero", "arti/device-execution-test@1", "zero", _type())
    effect = m.neural_plasticity(x, m.scale(x, rate), zero)
    return m.FormulaProgramEffectCandidateV3(
        "write",
        m.FormulaEffectProgramV2(
            m.FormulaProgram.build(outputs=(effect,)),
            data_input_name="x",
            state_type=_type(),
        ),
        input_slot="owned",
        output_slot="tail",
        operands={"rate": torch.full((1, 3), 0.1), "zero": torch.zeros(1, 3)},
    )


def _zero_network(query, bias):
    with torch.no_grad():
        for parameter in query.network.parameters():
            parameter.zero_()
        query.network[-1].bias.copy_(torch.tensor(bias, dtype=query.network[-1].bias.dtype))
    return query


def _runtime(*, data_capacity=8, bank_capacity=4):
    query = _zero_network(
        m.FormulaProgramQueryV5(
            slot_ids=("x", "owned", "tail"),
            candidates=(
                _producer("producer", "owned", "memory", 2.0),
                _effect(),
            ),
            terminal_slots={"answer": "tail"},
            max_steps=2,
            hidden_dim=8,
        ),
        (3.0, 2.0, 1.0),
    )
    kernel = FormulaDeviceFrameKernel.from_query(query)
    group_ids, _keys = formula_device_dispatch_groups(query)
    layout = FormulaDeviceDispatchLayout(group_ids).to(next(query.parameters()).device)
    dispatch = FormulaDeviceNumericalDispatch.from_query(
        query, frame_kernel=kernel,
    ).to(next(query.parameters()).device)
    wave = FormulaDeviceExecutionWave(
        dispatch,
        kernel,
        data_capacity=data_capacity,
        bank_capacity=bank_capacity,
    ).to(next(query.parameters()).device)
    state = kernel.initial_state(
        1,
        torch.tensor([[0, -1, -1]], dtype=torch.int64),
        bank_value_handles=torch.tensor([0], dtype=torch.int64),
    )
    data_pool = torch.zeros((data_capacity + 1, 1, 3))
    data_pool[0] = 1
    bank_pool = torch.zeros((bank_capacity + 1, 1, 3))
    bank_pool[0] = 2
    return query, kernel, layout, wave, state, data_pool, bank_pool


def test_execution_wave_allocates_values_and_advances_real_bank():
    _query, _kernel, layout, wave, state, data_pool, bank_pool = _runtime()
    first = wave(
        tuple(state),
        tuple(layout(torch.tensor([[0]], dtype=torch.int64))),
        data_pool,
        bank_pool,
        torch.tensor(1, dtype=torch.int64),
        torch.tensor(1, dtype=torch.int64),
    )
    assert not bool(first.pool.overflow)
    assert first.pool.data_cursor.item() == 2
    assert first.pool.bank_cursor.item() == 1
    assert first.pool.output_handles.tolist() == [[1]]
    assert first.event.accepted.tolist() == [True]
    torch.testing.assert_close(data_pool[0], torch.ones_like(data_pool[0]), rtol=0, atol=0)
    torch.testing.assert_close(data_pool[1], torch.full_like(data_pool[1], 2), rtol=0, atol=0)

    second = wave(
        tuple(first.state),
        tuple(layout(torch.tensor([[1]], dtype=torch.int64))),
        data_pool,
        bank_pool,
        first.pool.data_cursor,
        first.pool.bank_cursor,
    )
    assert not bool(second.pool.overflow)
    assert second.pool.data_cursor.item() == 2
    assert second.pool.bank_cursor.item() == 2
    assert second.pool.output_handles.tolist() == [[1]]
    assert second.pool.bank_handles.tolist() == [1]
    assert second.event.accepted.tolist() == [True]
    assert second.state.bank_value_handles.tolist() == [[1]]
    assert second.state.bank_revisions.tolist() == [[1]]
    torch.testing.assert_close(bank_pool[0], torch.full_like(bank_pool[0], 2), rtol=0, atol=0)
    torch.testing.assert_close(bank_pool[1], torch.full_like(bank_pool[1], 2.2))


def test_effect_alias_must_reference_an_allocated_value():
    _query, _kernel, layout, wave, state, data_pool, bank_pool = _runtime()
    first = wave(state, layout(torch.tensor([[0]])), data_pool, bank_pool, torch.tensor(1), torch.tensor(1))
    # The producer is valid, but the caller supplied a cursor before its value.
    result = wave(first.state, layout(torch.tensor([[1]])), data_pool, bank_pool,
                  torch.tensor(1), first.pool.bank_cursor)
    assert bool(result.pool.overflow)
    assert result.event.accepted.tolist() == [False]
    assert result.pool.bank_cursor.item() == first.pool.bank_cursor.item()


def test_execution_wave_forks_every_k_wide_lane_without_dropping_candidates():
    query = _zero_network(
        m.FormulaProgramQueryV5(
            slot_ids=("x", "left", "right"),
            candidates=(
                _producer("left", "left", "left-bank", 2.0),
                _producer("right", "right", "right-bank", 3.0),
            ),
            terminal_slots={"left": "left", "right": "right"},
            max_steps=2,
            hidden_dim=8,
        ),
        (3.0, 2.0, 1.0),
    )
    kernel = FormulaDeviceFrameKernel.from_query(query)
    group_ids, _keys = formula_device_dispatch_groups(query)
    layout = FormulaDeviceDispatchLayout(group_ids).to(next(query.parameters()).device)
    dispatch = FormulaDeviceNumericalDispatch.from_query(
        query, frame_kernel=kernel,
    ).to(next(query.parameters()).device)
    wave = FormulaDeviceExecutionWave(
        dispatch, kernel, data_capacity=8, bank_capacity=4,
    ).to(next(query.parameters()).device)
    state = kernel.initial_state(
        1,
        torch.tensor([[0, -1, -1]], dtype=torch.int64),
        bank_value_handles=torch.tensor([0, 1], dtype=torch.int64),
    )
    data_pool = torch.zeros((9, 1, 3))
    data_pool[0] = 1
    bank_pool = torch.zeros((5, 1, 3))
    bank_pool[0] = 2
    bank_pool[1] = 3
    result = wave(
        tuple(state),
        tuple(layout(torch.tensor([[0, 1]], dtype=torch.int64))),
        data_pool,
        bank_pool,
        torch.tensor(1, dtype=torch.int64),
        torch.tensor(2, dtype=torch.int64),
    )
    assert result.state.active.shape == (2,)
    assert result.event.accepted.tolist() == [True, True]
    assert result.pool.output_handles.tolist() == [[1, -1], [2, -1]]
    assert result.state.value_handles[:, 0].tolist() == [[0, 1, -1], [0, -1, 2]]
    torch.testing.assert_close(data_pool[1], torch.full_like(data_pool[1], 2), rtol=0, atol=0)
    torch.testing.assert_close(data_pool[2], torch.full_like(data_pool[2], 3), rtol=0, atol=0)


def test_execution_wave_overflow_is_atomic_and_rejected():
    _query, _kernel, layout, wave, state, data_pool, bank_pool = _runtime(
        data_capacity=1,
    )
    before_data = data_pool.clone()
    before_bank = bank_pool.clone()
    result = wave(
        tuple(state),
        tuple(layout(torch.tensor([[0]], dtype=torch.int64))),
        data_pool,
        bank_pool,
        torch.tensor(1, dtype=torch.int64),
        torch.tensor(1, dtype=torch.int64),
    )
    assert bool(result.pool.overflow)
    assert result.pool.data_cursor.item() == 1
    assert result.pool.bank_cursor.item() == 1
    assert result.event.accepted.tolist() == [False]
    assert torch.equal(result.state.value_handles, state.value_handles)
    assert torch.equal(data_pool, before_data)
    assert torch.equal(bank_pool, before_bank)


@pytest.mark.parametrize("control", ["call", "stop"])
def test_overflow_does_not_commit_control_lane(control):
    child = m.FormulaProgramQueryV5(
        slot_ids=("x", "owned"), candidates=(_producer("child", "owned", "child-bank", 3.0),),
        terminal_slots={"answer": "owned"}, max_steps=1, hidden_dim=8,
    )
    query = m.FormulaProgramQueryV5(
        slot_ids=("x", "owned"), candidates=(
            _producer("parent", "owned", "parent-bank", 2.0),
            m.FormulaProgramCallCandidateV1(
                "call", child, input_slots={"x": "x"}, output_slots={"answer": "owned"},
            ),
        ), terminal_slots={"answer": "x"}, max_steps=1, hidden_dim=8,
    )
    kernel = FormulaDeviceFrameKernel.from_query(query)
    device = next(query.parameters()).device
    layout = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(query)[0]).to(device)
    dispatch = FormulaDeviceNumericalDispatch.from_query(query, frame_kernel=kernel).to(device)
    wave = FormulaDeviceExecutionWave(dispatch, kernel, data_capacity=1, bank_capacity=2).to(device)
    state = kernel.initial_state(1, torch.tensor([[0, -1]]), bank_value_handles=torch.tensor([0, 1]))
    data = torch.ones((2, 1, 3))
    bank = torch.ones((3, 1, 3))
    result = wave(state, layout(torch.tensor([[0, 1 if control == "call" else 2]])),
                  data, bank, torch.tensor(1), torch.tensor(2))
    assert bool(result.pool.overflow)
    assert not bool(result.event.accepted.any())
    assert not bool(result.state.completed.any())
    assert result.state.depth.tolist() == [0, 0]


def test_execution_wave_is_cuda_graph_capturable(device):
    if device == "cpu":
        return
    _query, _kernel, layout, wave, state, data_pool, bank_pool = _runtime()
    packet = layout(torch.tensor([[0]], dtype=torch.int64))
    data_cursor = torch.tensor(1, dtype=torch.int64)
    bank_cursor = torch.tensor(1, dtype=torch.int64)
    captured = FormulaDeviceCapturedExecutionWave.capture(
        wave,
        state,
        packet,
        data_pool,
        bank_pool,
        data_cursor,
        bank_cursor,
        warmup_steps=1,
    )
    first = captured.replay()
    torch.cuda.synchronize()
    output_pointer = first.numeric.output_values.data_ptr()
    assert first.event.accepted.tolist() == [True]
    assert first.pool.output_handles.tolist() == [[1]]
    torch.testing.assert_close(
        captured.data_pool[1], torch.full_like(captured.data_pool[1], 2), rtol=0, atol=0,
    )

    changed_pool = data_pool.clone()
    changed_pool[0] = 3
    captured.copy_inputs_(data_pool=changed_pool)
    second = captured.replay()
    torch.cuda.synchronize()
    assert second.numeric.output_values.data_ptr() == output_pointer
    torch.testing.assert_close(
        captured.data_pool[1], torch.full_like(captured.data_pool[1], 6), rtol=0, atol=0,
    )
    assert captured.close()
    assert not captured.close()
    with pytest.raises(RuntimeError, match="closed"):
        captured.replay()
