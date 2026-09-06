import pytest
import torch

from arti import mechanisms as m
from arti._formula_device_dispatch import (
    FormulaDeviceCapturedNumericalDispatch,
    FormulaDeviceDispatchLayout,
    FormulaDeviceNumericalDispatch,
    FormulaDeviceRoutedDecisionWave,
    formula_device_dispatch_groups,
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


def _producer(name, output, owner, *, weight_value=1.0):
    x = m.InputBinding("x", _type())
    weight = m.BankBinding("weight", "arti/device-dispatch-test@1", "weight", _type())
    program = m.FormulaProgram.build(outputs=(m.scale(x, weight),))
    return m.FormulaProgramTensorCandidateV3(
        m.FormulaProgramCandidateV2(
            name,
            program,
            input_slots={"x": "x"},
            output_slot=output,
            operands={"weight": torch.full((1, 3), weight_value)},
        ),
        plastic_bank_slot="weight",
        bank_owner_id=owner,
    )


def _effect(*, direct_zero=False, count=None, reordered=False):
    x = m.InputBinding("x", _type())
    rate = m.BankBinding("rate", "arti/device-dispatch-test@1", "rate", _type())
    zero = m.BankBinding("zero", "arti/device-dispatch-test@1", "zero", _type())
    multiplicative = zero if direct_zero else m.scale(x, zero)
    effect = m.neural_plasticity(x, m.scale(x, rate), multiplicative)
    program = m.FormulaProgram.build(outputs=(effect,))
    if reordered:
        from dataclasses import replace
        program = replace(program, bindings=(*program.bindings[1:], program.bindings[0]))
    return m.FormulaProgramEffectCandidateV3(
        "write",
        m.FormulaEffectProgramV2(
            program,
            data_input_name="x",
            state_type=_type(),
        ),
        input_slot="owned",
        output_slot="tail",
        operands={"rate": torch.full((1, 3), 0.1), "zero": torch.zeros(1, 3)},
        execution_count=None if count is None else torch.tensor(float(count)),
        max_executions=4,
    )


def _zero_network(query, bias):
    with torch.no_grad():
        for parameter in query.network.parameters():
            parameter.zero_()
        query.network[-1].bias.copy_(torch.tensor(bias, dtype=query.network[-1].bias.dtype))
    return query


def _query():
    return _zero_network(
        m.FormulaProgramQueryV5(
            slot_ids=("x", "left", "right"),
            candidates=(
                _producer("left-producer", "left", "left-memory"),
                _producer("right-producer", "right", "right-memory"),
            ),
            terminal_slots={"left": "left", "right": "right"},
            max_steps=2,
            hidden_dim=8,
        ),
        (3.0, 2.0, 1.0),
    )


def test_multi_output_bank_binding_is_not_tied_to_first_head():
    from arti._formula_device_execution import FormulaDeviceExecutionWave

    x = m.InputBinding("x", _type())
    weight = m.BankBinding("weight", "arti/device-dispatch-test@1", "weight", _type())
    program = m.FormulaProgram.build(outputs=(m.add(x, x), m.scale(x, weight)))
    candidate = m.FormulaProgramTensorCandidateV4(
        m.FormulaProgramCandidateV3(
            "heads", program, input_slots={"x": "x"},
            output_slots=dict(zip(program.outputs, ("plain", "owned"), strict=True)),
            operands={"weight": torch.full((1, 3), 2.0)},
        ), plastic_bank_slot="weight", bank_owner_id="second-head",
    )
    query = m.FormulaProgramQueryV5(
        slot_ids=("x", "plain", "owned"), candidates=(candidate,),
        terminal_slots={"plain": "plain", "answer": "owned"}, max_steps=1,
    )
    device = next(query.parameters()).device
    kernel = FormulaDeviceFrameKernel.from_query(query).to(device)
    layout = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(query)[0]).to(device)
    dispatch = FormulaDeviceNumericalDispatch.from_query(query, frame_kernel=kernel).to(device)
    execution = FormulaDeviceExecutionWave(dispatch, kernel, data_capacity=4, bank_capacity=1).to(device)
    state = kernel.initial_state(1, torch.tensor([[0, -1, -1]]), bank_value_handles=torch.tensor([0]))
    data, bank = torch.ones((5, 1, 3)), torch.full((2, 1, 3), 4.0)
    result = execution(state, layout(torch.tensor([[0]])), data, bank, torch.tensor(1), torch.tensor(1))
    assert result.event.accepted.tolist() == [True]
    assert result.state.producer_bank[0, 0].tolist() == [-1, -1, 0]
    torch.testing.assert_close(data[1], torch.full_like(data[1], 2.0))
    torch.testing.assert_close(data[2], torch.full_like(data[2], 4.0))


def test_dispatch_layout_is_dropless_and_invertible():
    selected = torch.tensor([[2, 1, -1], [0, 3, 1]], dtype=torch.int64)
    layout = FormulaDeviceDispatchLayout((1, 0, 1, 2)).to(selected.device)
    packet = layout(selected)
    assert packet.group_counts.tolist() == [2, 2, 1]
    assert packet.group_offsets.tolist() == [0, 2, 4, 5]
    assert packet.valid.sum().item() == 5
    restored = packet.candidate_ids.index_select(0, packet.inverse_order)
    assert torch.equal(restored, selected.flatten())
    coordinates = torch.stack((packet.source_rows, packet.source_lanes), dim=1)
    restored_coordinates = coordinates.index_select(0, packet.inverse_order)
    expected = torch.tensor(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]],
        dtype=torch.int64,
    )
    assert torch.equal(restored_coordinates, expected)
    assert bool(packet.well_formed)


def test_dispatch_layout_reports_non_sentinel_invalid_ids_without_dropping_them():
    selected = torch.tensor([[0, 7, -1]], dtype=torch.int64)
    layout = FormulaDeviceDispatchLayout((0, 1)).to(selected.device)
    packet = layout(selected)
    assert not bool(packet.well_formed)
    assert packet.candidate_ids.numel() == 3
    assert packet.valid.sum().item() == 1
    assert packet.group_counts.sum().item() == 1


def test_group_table_uses_program_fingerprint_and_control_groups():
    query = _query()
    group_ids, keys = formula_device_dispatch_groups(query)
    assert group_ids[0] == group_ids[1]
    assert keys[group_ids[0]][0] == "ordinary"
    assert keys[group_ids[2]] == "stop"


def test_routed_decision_wave_keeps_complete_k_packet_on_device(device):
    query = _query()
    kernel = FormulaDeviceFrameKernel.from_query(query)
    wave = FormulaDeviceRoutedDecisionWave.from_query(
        query,
        frame_kernel=kernel,
        candidate_family_ids=(0, 0, 1),
        width=3,
    ).to(next(query.parameters()).device)
    state = kernel.initial_state(
        1,
        torch.tensor([[0, -1, -1]], dtype=torch.int64),
        bank_value_handles=torch.tensor([0, 1], dtype=torch.int64),
    )
    args = (
        tuple(state),
        torch.ones((1, 3)),
        torch.ones((2, 1, 3)),
        torch.ones(3, dtype=torch.bool),
        torch.zeros(1),
    )
    expected = wave(*args)
    assert expected.selected_candidates.tolist() == [[0, 1, -1]]
    assert expected.valid.sum().item() == 2
    assert expected.group_counts.tolist() == [2, 0]
    assert bool(expected.well_formed)

    graph = torch.export.export(wave, args, strict=True).module()
    assert not any("_local_scalar_dense" in str(node.target) for node in graph.graph.nodes)
    compiled = torch.compile(
        graph,
        backend="inductor" if device == "cuda" else "aot_eager",
        fullgraph=True,
    )
    actual = compiled(*args)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def _numeric_fixture(*, count=None, reordered=False):
    query = _zero_network(
        m.FormulaProgramQueryV5(
            slot_ids=("x", "owned", "tail"),
            candidates=(
                _producer("producer", "owned", "memory", weight_value=2.0),
                _effect(count=count, reordered=reordered),
            ),
            terminal_slots={"answer": "tail"},
            max_steps=2,
            hidden_dim=8,
        ),
        (3.0, 2.0, 1.0),
    )
    kernel = FormulaDeviceFrameKernel.from_query(query)
    groups, _keys = formula_device_dispatch_groups(query)
    layout = FormulaDeviceDispatchLayout(groups).to(next(query.parameters()).device)
    dispatch = FormulaDeviceNumericalDispatch.from_query(
        query, frame_kernel=kernel,
    ).to(next(query.parameters()).device)
    state = kernel.initial_state(
        1,
        torch.tensor([[0, -1, -1]], dtype=torch.int64),
        bank_value_handles=torch.tensor([0], dtype=torch.int64),
    )
    data_pool = torch.ones((1, 1, 3))
    bank_pool = torch.full((1, 1, 3), 2.0)
    return kernel, layout, dispatch, state, data_pool, bank_pool


@pytest.mark.parametrize("reordered", [False, True])
def test_numerical_dispatch_runs_ordinary_and_real_predecessor_effect(reordered):
    kernel, layout, dispatch, state, data_pool, bank_pool = _numeric_fixture(reordered=reordered)
    producer_packet = layout(torch.tensor([[0]], dtype=torch.int64))
    produced = dispatch(tuple(state), tuple(producer_packet), data_pool, bank_pool)
    assert produced.numeric_valid.tolist() == [True]
    assert produced.output_present.tolist() == [[True]]
    assert produced.output_alias_handles.tolist() == [[-1]]
    torch.testing.assert_close(
        produced.output_values[0, 0], torch.full((1, 3), 2.0), rtol=0, atol=0,
    )
    assert not bool(produced.bank_successor_present[0])

    data_pool = torch.cat((data_pool, produced.output_values[:, 0]), dim=0)
    state, event = kernel(
        state,
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([[1]], dtype=torch.int64),
        produced.numeric_valid,
        torch.tensor([-1], dtype=torch.int64),
    )
    assert event.accepted.tolist() == [True]

    effect_packet = layout(torch.tensor([[1]], dtype=torch.int64))
    updated = dispatch(tuple(state), tuple(effect_packet), data_pool, bank_pool)
    assert updated.numeric_valid.tolist() == [True]
    assert updated.output_present.tolist() == [[True]]
    assert updated.output_alias_handles.tolist() == [[1]]
    assert updated.bank_successor_present.tolist() == [True]
    torch.testing.assert_close(
        updated.bank_successors[0], torch.full((1, 3), 2.2), rtol=1e-6, atol=1e-6,
    )
    torch.testing.assert_close(bank_pool, torch.full_like(bank_pool, 2.0), rtol=0, atol=0)


def test_numerical_dispatch_preserves_direct_effect_bank_bindings():
    query = _zero_network(
        m.FormulaProgramQueryV5(
            slot_ids=("x", "owned", "tail"),
            candidates=(
                _producer("producer", "owned", "memory", weight_value=2.0),
                _effect(direct_zero=True),
            ),
            terminal_slots={"answer": "tail"},
            max_steps=2,
            hidden_dim=8,
        ),
        (3.0, 2.0, 1.0),
    )
    kernel = FormulaDeviceFrameKernel.from_query(query)
    groups, _keys = formula_device_dispatch_groups(query)
    layout = FormulaDeviceDispatchLayout(groups).to(next(query.parameters()).device)
    dispatch = FormulaDeviceNumericalDispatch.from_query(
        query, frame_kernel=kernel,
    ).to(next(query.parameters()).device)
    state = kernel.initial_state(
        1,
        torch.tensor([[0, -1, -1]], dtype=torch.int64),
        bank_value_handles=torch.tensor([0], dtype=torch.int64),
    )
    data_pool = torch.cat((torch.ones((1, 1, 3)), torch.full((1, 1, 3), 2.0)))
    bank_pool = torch.full((1, 1, 3), 2.0)
    state, _event = kernel(
        state,
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([[1]], dtype=torch.int64),
        torch.ones(1, dtype=torch.bool),
        torch.tensor([-1], dtype=torch.int64),
    )
    updated = dispatch(
        tuple(state),
        tuple(layout(torch.tensor([[1]], dtype=torch.int64))),
        data_pool,
        bank_pool,
    )
    assert updated.numeric_valid.tolist() == [True]
    torch.testing.assert_close(
        updated.bank_successors[0], torch.full((1, 3), 2.2), rtol=1e-6, atol=1e-6,
    )


def test_numerical_dispatch_exports_without_runtime_scalar_reads(device):
    _kernel, layout, dispatch, state, data_pool, bank_pool = _numeric_fixture()
    packet = layout(torch.tensor([[0]], dtype=torch.int64))
    args = (tuple(state), tuple(packet), data_pool, bank_pool)
    expected = dispatch(*args)
    graph = torch.export.export(dispatch, args, strict=True).module()
    assert not any("_local_scalar_dense" in str(node.target) for node in graph.graph.nodes)
    compiled = torch.compile(
        graph,
        backend="inductor" if device == "cuda" else "aot_eager",
        fullgraph=True,
    )
    actual = compiled(*args)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_operand_snapshot_refresh_keeps_addresses_and_live_predecessor_bank(device, monkeypatch):
    import arti._formula_device_dispatch as implementation

    kernel, layout, dispatch, state, data_pool, bank_pool = _numeric_fixture(count=2)
    state, _ = kernel(
        state, torch.tensor([0]), torch.tensor([[1]]), torch.ones(1, dtype=torch.bool),
        torch.tensor([-1]),
    )
    data_pool = torch.cat((data_pool, torch.full_like(data_pool, 2)))
    packet = layout(torch.tensor([[1]]))
    group = next(group for group in dispatch.groups if group.is_effect)
    candidate = group.candidates[0]
    rate = candidate.operand_store.tensor("rate")
    parameter_ids = tuple(id(p) for p in dispatch.parameters())
    pointers = tuple(value.data_ptr() for name, value in group.named_buffers()
                     if name.startswith("operand_table_") or name == "execution_counts")
    assert not any("operand_table_" in name or name.endswith("execution_counts")
                   for name in dispatch.state_dict())

    captured = None
    if device == "cuda":
        captured = FormulaDeviceCapturedNumericalDispatch.capture(
            dispatch, state, packet, data_pool, bank_pool,
        )

    def run():
        return (captured.replay() if captured is not None
                else dispatch(tuple(state), tuple(packet), data_pool, bank_pool))

    try:
        torch.testing.assert_close(run().bank_successors, torch.full_like(bank_pool, 2.4))
        with torch.no_grad():
            rate.fill_(0.3)
            candidate.execution_count.fill_(3)
        # A search keeps one parameter snapshot until the optimizer boundary.
        torch.testing.assert_close(run().bank_successors, torch.full_like(bank_pool, 2.4))
        dispatch.refresh_operands_()
        assert pointers == tuple(value.data_ptr() for name, value in group.named_buffers()
                                 if name.startswith("operand_table_") or name == "execution_counts")
        assert parameter_ids == tuple(id(p) for p in dispatch.parameters())

        def no_parameter_lookup(*args):
            raise AssertionError("search wave must only index prepared operand tensors")

        monkeypatch.setattr(implementation, "_operand", no_parameter_lookup)
        torch.testing.assert_close(run().bank_successors, torch.full_like(bank_pool, 3.8))
        bank_pool.fill_(5)
        if captured is not None:
            captured.copy_inputs_(bank_pool=bank_pool)
        torch.testing.assert_close(run().bank_successors, torch.full_like(bank_pool, 6.8))
    finally:
        if captured is not None:
            captured.close()


def test_captured_numerical_dispatch_replays_and_refreshes_fixed_buffers(device):
    kernel, layout, dispatch, state, data_pool, bank_pool = _numeric_fixture()
    packet = layout(torch.tensor([[0]], dtype=torch.int64))
    if device == "cpu":
        with pytest.raises(ValueError, match="one CUDA device"):
            FormulaDeviceCapturedNumericalDispatch.capture(
                dispatch, state, packet, data_pool, bank_pool,
            )
        return

    padded_pool = torch.cat((data_pool, torch.zeros_like(data_pool)))
    captured = FormulaDeviceCapturedNumericalDispatch.capture(
        dispatch,
        state,
        packet,
        padded_pool,
        bank_pool,
        warmup_steps=1,
    )
    first = captured.replay()
    torch.cuda.synchronize()
    output_pointer = first.output_values.data_ptr()
    torch.testing.assert_close(
        first.output_values[0, 0], torch.full((1, 3), 2.0), rtol=0, atol=0,
    )

    padded_pool[1].copy_(first.output_values[0, 0])
    next_state, event = kernel(
        state,
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([[1]], dtype=torch.int64),
        first.numeric_valid,
        torch.tensor([-1], dtype=torch.int64),
    )
    assert event.accepted.tolist() == [True]
    captured.copy_inputs_(
        state=next_state,
        packet=layout(torch.tensor([[1]], dtype=torch.int64)),
        data_pool=padded_pool,
    )
    second = captured.replay()
    torch.cuda.synchronize()
    assert second.output_values.data_ptr() == output_pointer
    assert second.output_alias_handles.tolist() == [[1]]
    assert second.bank_successor_present.tolist() == [True]
    torch.testing.assert_close(
        second.bank_successors[0], torch.full((1, 3), 2.2), rtol=1e-6, atol=1e-6,
    )
    assert captured.close()
    assert not captured.close()
    with pytest.raises(RuntimeError, match="closed"):
        captured.replay()
