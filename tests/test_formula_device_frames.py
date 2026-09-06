import torch
import pytest

from arti import mechanisms as m
from arti._formula_device_frames import (
    EVENT_CALL,
    EVENT_EFFECT,
    EVENT_RETURN,
    EVENT_STOP,
    FormulaDeviceFrameKernel,
)


@pytest.fixture(params=["cpu", "cuda"], autouse=True)
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with torch.device(request.param):
        yield request.param


def _type():
    return m.TensorType(("B", "D"), ("B", 3), dtype="floating", domain="activation")


def _producer(name="producer", *, owner="memory", source="x", output="owned"):
    x = m.InputBinding("x", _type())
    weight = m.BankBinding("weight", "arti/device-frame-test@1", "weight", _type())
    program = m.FormulaProgram.build(outputs=(m.scale(x, weight),))
    candidate = m.FormulaProgramCandidateV2(
        name, program, input_slots={"x": source}, output_slot=output,
        operands={"weight": torch.full((1, 3), 2.0)},
    )
    return m.FormulaProgramTensorCandidateV3(
        candidate, plastic_bank_slot="weight", bank_owner_id=owner,
    )


def _multi_producer(name="producer", *, owner="memory", source="x"):
    x = m.InputBinding("x", _type())
    weight = m.BankBinding("weight", "arti/device-frame-test@1", "weight", _type())
    gain = m.BankBinding("gain", "arti/device-frame-test@1", "gain", _type())
    program = m.FormulaProgram.build(
        outputs=(m.add(x, x), m.scale(m.scale(x, weight), gain)),
    )
    candidate = m.FormulaProgramCandidateV3(
        name,
        program,
        input_slots={"x": source},
        output_slots=dict(zip(program.outputs, ("raw", "owned"), strict=True)),
        operands={"weight": torch.full((1, 3), 2.0), "gain": torch.ones(1, 3)},
    )
    return m.FormulaProgramTensorCandidateV4(
        candidate, plastic_bank_slot="weight", bank_owner_id=owner,
    )


def _effect(name="write", source="owned", output="tail", *, count=None):
    x = m.InputBinding("x", _type())
    rate = m.BankBinding("rate", "arti/device-frame-test@1", "rate", _type())
    zero = m.BankBinding("zero", "arti/device-frame-test@1", "zero", _type())
    effect = m.neural_plasticity(x, m.scale(x, rate), m.scale(x, zero))
    return m.FormulaProgramEffectCandidateV3(
        name,
        m.FormulaEffectProgramV2(
            m.FormulaProgram.build(outputs=(effect,)), data_input_name="x", state_type=_type(),
        ),
        input_slot=source, output_slot=output,
        operands={"rate": torch.full((1, 3), 0.1), "zero": torch.zeros(1, 3)},
        execution_count=None if count is None else torch.tensor(float(count)),
    )


def _query_with_effect():
    producer = _producer()
    effect = _effect()
    query = m.FormulaProgramQueryV5(
        slot_ids=("x", "owned", "tail"), candidates=(producer, effect),
        terminal_slots={"answer": "tail"}, max_steps=2, hidden_dim=8,
    )
    return query, producer, effect


def test_ordinary_effect_stop_updates_handles_and_real_bank_revision():
    query, producer, effect = _query_with_effect()
    kernel = FormulaDeviceFrameKernel.from_query(query)
    state = kernel.initial_state(
        1, torch.tensor([[7, -1, -1]], dtype=torch.int64),
        bank_value_handles=torch.tensor([41], dtype=torch.int64),
    )
    producer_id = 0
    effect_id = 1
    stop_id = 2
    result_handle = torch.tensor([[8]], dtype=torch.int64)
    state, event = kernel(
        state, torch.tensor([producer_id]), result_handle, torch.tensor([True]), torch.tensor([-1]),
    )
    assert event.accepted.tolist() == [True]
    assert state.value_handles[0, 0, 1].item() == 8
    assert state.producer_bank[0, 0, 1].item() == 0
    assert state.producer_revision[0, 0, 1].item() == 0

    state, event = kernel(
        state, torch.tensor([effect_id]), torch.tensor([[8]], dtype=torch.int64),
        torch.tensor([True]), torch.tensor([42]),
    )
    assert event.kind.tolist() == [EVENT_EFFECT]
    assert event.previous_revision.tolist() == [0]
    assert event.successor_revision.tolist() == [1]
    assert state.bank_revisions.tolist() == [[1]]
    assert state.bank_value_handles.tolist() == [[42]]
    assert state.value_handles[0, 0, 2].item() == 8
    assert state.producer_bank[0, 0, 2].item() == 0
    assert state.producer_revision[0, 0, 2].item() == 1

    state, event = kernel(
        state, torch.tensor([stop_id]), torch.full((1, 1), -1, dtype=torch.int64),
        torch.tensor([True]), torch.tensor([-1]),
    )
    assert event.kind.tolist() == [EVENT_STOP]
    assert state.completed.tolist() == [True]
    assert state.active.tolist() == [False]


def test_call_push_return_preserves_multihead_producer_and_bank_revision():
    child_producer = _multi_producer()
    child = m.FormulaProgramQueryV5(
        slot_ids=("x", "raw", "owned"), candidates=(child_producer,),
        terminal_slots={"memory": "owned", "data": "raw"}, max_steps=1, hidden_dim=8,
    )
    call = m.FormulaProgramCallCandidateV1(
        "call", child, input_slots={"x": "x"},
        output_slots={"data": "out-data", "memory": "out-memory"},
    )
    parent = m.FormulaProgramQueryV5(
        slot_ids=("x", "out-data", "out-memory"), candidates=(call,),
        terminal_slots={"answer": "out-memory"}, max_steps=1, hidden_dim=8,
    )
    kernel = FormulaDeviceFrameKernel.from_query(parent)
    state = kernel.initial_state(
        1, torch.tensor([[7, -1, -1]], dtype=torch.int64),
        bank_value_handles=torch.tensor([41], dtype=torch.int64),
    )
    call_id = kernel.candidate_id(0, 0)
    state.response_candidate[0, 0, 0] = call_id
    state, event = kernel(
        state, torch.tensor([call_id]), torch.full((1, 2), -1, dtype=torch.int64),
        torch.tensor([True]), torch.tensor([-1]),
    )
    assert event.kind.tolist() == [EVENT_CALL]
    assert state.depth.tolist() == [1]
    assert state.value_handles[0, 1, 0].item() == 7
    assert state.response_candidate[0, 1].eq(-1).all()

    child_producer_id = kernel.candidate_id(1, 0)
    state, event = kernel(
        state, torch.tensor([child_producer_id]), torch.tensor([[8, 9]], dtype=torch.int64),
        torch.tensor([True]), torch.tensor([-1]),
    )
    assert event.accepted.tolist() == [True]

    child_stop_id = kernel.candidate_id(1, 1)
    state, event = kernel(
        state, torch.tensor([child_stop_id]), torch.full((1, 2), -1, dtype=torch.int64),
        torch.tensor([True]), torch.tensor([-1]),
    )
    assert event.kind.tolist() == [EVENT_RETURN]
    assert state.depth.tolist() == [0]
    assert event.returned_handles.tolist() == [[8, 9]]
    assert state.value_handles[0, 0, 1].item() == 8
    assert state.value_handles[0, 0, 2].item() == 9
    assert state.producer_bank[0, 0, 1].item() == -1
    assert state.producer_bank[0, 0, 2].item() == 0
    assert state.producer_revision[0, 0, 2].item() == 0
    assert state.response_candidate[0, 0, 1:].tolist() == [call_id, call_id]
    assert state.producer_candidate[0, 0, 2].item() == child_producer_id
    assert state.response_candidate[0, 1].eq(-1).all()
    assert state.frame_tensor_steps[0, 0].item() == 1


def test_two_branches_mix_ordinary_effect_call_and_stop_without_cross_row_broadcast():
    child = m.FormulaProgramQueryV5(
        slot_ids=("x", "tail"), candidates=(_producer(owner="child-memory", output="tail"),),
        terminal_slots={"answer": "tail"}, max_steps=1, hidden_dim=8,
    )
    call = m.FormulaProgramCallCandidateV1(
        "call", child, input_slots={"x": "x"}, output_slots={"answer": "tail"},
    )
    root = m.FormulaProgramQueryV5(
        slot_ids=("x", "owned", "tail"), candidates=(_producer(), _effect(), call),
        terminal_slots={"answer": "tail"}, max_steps=2, hidden_dim=8,
    )
    kernel = FormulaDeviceFrameKernel.from_query(root)
    state = kernel.initial_state(
        2, torch.tensor([[7, -1, -1], [7, -1, -1]], dtype=torch.int64),
        bank_value_handles=torch.tensor([41, 51], dtype=torch.int64),
    )
    root_producer = kernel.candidate_id(0, 0)
    root_effect = kernel.candidate_id(0, 1)
    root_call = kernel.candidate_id(0, 2)
    root_stop = kernel.candidate_id(0, 3)
    child_producer = kernel.candidate_id(1, 0)
    child_stop = kernel.candidate_id(1, 1)

    state, event = kernel(
        state, torch.tensor([root_producer, root_call]),
        torch.tensor([[8], [-1]], dtype=torch.int64), torch.tensor([True, True]),
        torch.tensor([-1, -1]),
    )
    assert event.kind.tolist() == [1, EVENT_CALL]
    assert state.depth.tolist() == [0, 1]

    state, event = kernel(
        state, torch.tensor([root_effect, child_producer]),
        torch.tensor([[8], [9]], dtype=torch.int64), torch.tensor([True, True]),
        torch.tensor([42, -1]),
    )
    assert event.kind.tolist() == [2, 1]
    assert state.bank_revisions.tolist() == [[0, 1], [0, 0]]

    state, event = kernel(
        state, torch.tensor([root_stop, child_stop]),
        torch.full((2, 1), -1, dtype=torch.int64), torch.tensor([True, True]),
        torch.tensor([-1, -1]),
    )
    assert event.kind.tolist() == [EVENT_STOP, EVENT_RETURN]
    assert state.completed.tolist() == [True, False]
    assert state.active.tolist() == [False, True]
    assert state.value_handles[0, 0, 2].item() == 8
    assert state.value_handles[1, 0, 2].item() == 9

    state, event = kernel(
        state, torch.tensor([root_stop, root_stop]),
        torch.full((2, 1), -1, dtype=torch.int64), torch.tensor([True, True]),
        torch.tensor([-1, -1]),
    )
    assert event.kind.tolist() == [0, EVENT_STOP]
    assert state.completed.tolist() == [True, True]


def test_unsupported_candidate_type_requests_native_fallback_instead_of_ordinary_mapping():
    class WrappedTensorCandidate(m.FormulaProgramTensorCandidateV3):
        pass

    ordinary = _producer()
    wrapped = WrappedTensorCandidate(
        ordinary.candidate, plastic_bank_slot="weight", bank_owner_id="memory",
    )
    query = m.FormulaProgramQueryV5(
        slot_ids=("x", "owned"), candidates=(wrapped,),
        terminal_slots={"answer": "owned"}, max_steps=1, hidden_dim=8,
    )
    with pytest.raises(TypeError, match="native fallback"):
        FormulaDeviceFrameKernel.from_query(query)


def test_invalid_effect_identity_does_not_advance_state():
    query, _producer_candidate, effect = _query_with_effect()
    kernel = FormulaDeviceFrameKernel.from_query(query)
    state = kernel.initial_state(
        1, torch.tensor([[7, -1, -1]], dtype=torch.int64),
        bank_value_handles=torch.tensor([41], dtype=torch.int64),
    )
    state, _ = kernel(
        state, torch.tensor([0]), torch.tensor([[8]], dtype=torch.int64),
        torch.tensor([True]), torch.tensor([-1]),
    )
    before = state
    state, event = kernel(
        state, torch.tensor([1]), torch.tensor([[99]], dtype=torch.int64),
        torch.tensor([True]), torch.tensor([42]),
    )
    assert event.accepted.tolist() == [False]
    assert event.kind.tolist() != [EVENT_EFFECT]
    assert state.bank_revisions.equal(before.bank_revisions)
    assert state.value_handles.equal(before.value_handles)


def test_compiled_frames_follow_runtime_actions_without_scalar_reads(device):
    query, _, _ = _query_with_effect()
    kernel = FormulaDeviceFrameKernel.from_query(query)
    state = kernel.initial_state(1, torch.tensor([[7, -1, -1]]), bank_value_handles=torch.tensor([41]))
    example = (tuple(state), torch.tensor([0]), torch.tensor([[8]]), torch.tensor([True]), torch.tensor([-1]))
    graph = torch.export.export(kernel, example, strict=True).module()
    assert not any("_local_scalar_dense" in str(node.target) for node in graph.graph.nodes)
    compiled = torch.compile(graph, backend="inductor" if device == "cuda" else "aot_eager", fullgraph=True)
    expected = state
    actual = state
    for candidate, handle, bank_handle in [(0, 8, -1), (1, 8, 42), (2, -1, -1)]:
        arguments = (torch.tensor([candidate]), torch.tensor([[handle]]), torch.tensor([True]), torch.tensor([bank_handle]))
        expected, expected_event = kernel(expected, *arguments)
        actual, actual_event = compiled(tuple(actual), *arguments)
        actual = type(actual)(*(tensor.clone() for tensor in actual))
        for left, right in zip((*actual, *actual_event), (*expected, *expected_event), strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert actual.completed.tolist() == [True]


@pytest.mark.parametrize("count", [0, 3])
def test_frame_advance_matches_real_native_bank_effect(count):
    producer, effect = _producer(), _effect(count=count)
    query = m.FormulaProgramQueryV5(slot_ids=("x", "owned", "tail"),
        candidates=(producer, effect), terminal_slots={"answer": "tail"}, max_steps=2, hidden_dim=8)
    arena = query._arena({"x": torch.ones(1, 3)})
    handles = {}
    retained = []

    def handle(value):
        if id(value) not in handles:
            handles[id(value)] = len(handles)
            retained.append(value)
        return handles[id(value)]

    previous, _ = arena.effect_state(producer.bank_slot_ref)
    kernel = FormulaDeviceFrameKernel.from_query(query)
    state = kernel.initial_state(1, torch.tensor([[handle(arena.values.get("x")), -1, -1]]),
        bank_value_handles=torch.tensor([handle(previous)]))
    for index, candidate in enumerate((producer, effect)):
        with torch.no_grad():
            arena = candidate(arena)
        current, revision = arena.effect_state(producer.bank_slot_ref)
        value = arena.values.get(candidate.output_slot)
        state, event = kernel(state, torch.tensor([index]), torch.tensor([[handle(value)]]),
            torch.tensor([True]), torch.tensor([handle(current)]))
        assert event.accepted.tolist() == [True]
        assert state.bank_revisions.tolist() == [[revision]]
        assert state.bank_value_handles.tolist() == [[handle(current)]]
        assert state.value_handles[0, 0].tolist() == [
            -1 if item is None else handle(item) for item in arena.values.values
        ]
    assert revision == 1
    if count == 0:
        torch.testing.assert_close(current, previous, rtol=0, atol=0)
