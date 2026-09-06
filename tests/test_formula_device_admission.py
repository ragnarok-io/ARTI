import pytest
import torch

from arti import mechanisms as m
from arti._formula_device_admission import FormulaDeviceAdmission
from arti._formula_device_frames import FormulaDeviceFrameKernel


@pytest.fixture(params=["cpu", "cuda"], autouse=True)
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with torch.device(request.param):
        yield request.param


def _type():
    return m.TensorType(("B", "D"), ("B", 3), dtype="floating", domain="activation")


def _producer(name="producer", *, owner="memory", output="owned"):
    x = m.InputBinding("x", _type())
    weight = m.BankBinding("weight", "arti/device-admission-test@1", "weight", _type())
    program = m.FormulaProgram.build(outputs=(m.scale(x, weight),))
    candidate = m.FormulaProgramCandidateV2(
        name,
        program,
        input_slots={"x": "x"},
        output_slot=output,
        operands={"weight": torch.full((1, 3), 2.0)},
    )
    return m.FormulaProgramTensorCandidateV3(
        candidate,
        plastic_bank_slot="weight",
        bank_owner_id=owner,
    )


def _effect():
    x = m.InputBinding("x", _type())
    rate = m.BankBinding("rate", "arti/device-admission-test@1", "rate", _type())
    zero = m.BankBinding("zero", "arti/device-admission-test@1", "zero", _type())
    effect = m.neural_plasticity(x, m.scale(x, rate), m.scale(x, zero))
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


def _query_with_effect():
    query = m.FormulaProgramQueryV5(
        slot_ids=("x", "owned", "tail"),
        candidates=(_producer(), _effect()),
        terminal_slots={"answer": "tail"},
        max_steps=2,
        hidden_dim=8,
    )
    kernel = FormulaDeviceFrameKernel.from_query(query)
    admission = FormulaDeviceAdmission.from_query(query, kernel)
    state = kernel.initial_state(
        1,
        torch.tensor([[0, -1, -1]], dtype=torch.int64),
        bank_value_handles=torch.tensor([0], dtype=torch.int64),
    )
    return query, kernel, admission, state


def _flags(admission, state, *, data=(True, True), bank=(True, True), operands=None):
    if operands is None:
        operands = torch.ones(admission.frame_kernel.candidate_count, dtype=torch.bool)
    return admission(
        tuple(state),
        torch.tensor([data], dtype=torch.bool),
        torch.tensor([bank], dtype=torch.bool),
        operands,
    )


def test_full_mask_tracks_ordinary_effect_stop_and_predecessor_revision():
    _query, kernel, admission, state = _query_with_effect()
    assert _flags(admission, state).tolist() == [[True, False, False]]

    state, _ = kernel(
        state,
        torch.tensor([0]),
        torch.tensor([[1]], dtype=torch.int64),
        torch.tensor([True]),
        torch.tensor([-1]),
    )
    assert _flags(admission, state).tolist() == [[False, True, False]]

    stale = state._replace(producer_bank_handle=state.producer_bank_handle.clone())
    stale.producer_bank_handle[0, 0, 1] = 1
    assert _flags(admission, stale).tolist() == [[False, False, False]]

    state, _ = kernel(
        state,
        torch.tensor([1]),
        torch.tensor([[1]], dtype=torch.int64),
        torch.tensor([True]),
        torch.tensor([1]),
    )
    assert _flags(admission, state).tolist() == [[False, False, True]]


def test_dynamic_finite_flags_reject_only_the_dependent_action():
    _query, kernel, admission, state = _query_with_effect()
    assert _flags(admission, state, data=(False, True)).tolist() == [[False, False, False]]
    operands = torch.tensor([False, True, True])
    assert _flags(admission, state, operands=operands).tolist() == [[False, False, False]]

    state, _ = kernel(
        state,
        torch.tensor([0]),
        torch.tensor([[1]], dtype=torch.int64),
        torch.tensor([True]),
        torch.tensor([-1]),
    )
    assert _flags(admission, state, data=(True, False)).tolist() == [[False, False, False]]
    assert _flags(admission, state, bank=(False, True)).tolist() == [[False, False, False]]


def test_call_admission_uses_child_entry_and_then_child_stop():
    child = m.FormulaProgramQueryV5(
        slot_ids=("x", "out"),
        candidates=(_producer(owner="child-memory", output="out"),),
        terminal_slots={"answer": "out"},
        max_steps=1,
        hidden_dim=8,
    )
    call = m.FormulaProgramCallCandidateV1(
        "call",
        child,
        input_slots={"x": "x"},
        output_slots={"answer": "out"},
    )
    root = m.FormulaProgramQueryV5(
        slot_ids=("x", "out"),
        candidates=(call,),
        terminal_slots={"answer": "out"},
        max_steps=1,
        hidden_dim=8,
    )
    kernel = FormulaDeviceFrameKernel.from_query(root)
    admission = FormulaDeviceAdmission.from_query(root, kernel)
    state = kernel.initial_state(
        1,
        torch.tensor([[0, -1]], dtype=torch.int64),
        bank_value_handles=torch.tensor([0], dtype=torch.int64),
    )
    finite = torch.tensor([[True, True]], dtype=torch.bool)
    bank_finite = torch.tensor([[True, True]], dtype=torch.bool)
    operands = torch.ones(kernel.candidate_count, dtype=torch.bool)
    assert admission(tuple(state), finite, bank_finite, operands).tolist() == [
        [True, False, False, False]
    ]

    state, _ = kernel(
        state,
        torch.tensor([kernel.candidate_id(0, 0)]),
        torch.tensor([[-1]], dtype=torch.int64),
        torch.tensor([True]),
        torch.tensor([-1]),
    )
    assert admission(tuple(state), finite, bank_finite, operands).tolist() == [
        [False, False, True, False]
    ]
    state, _ = kernel(
        state,
        torch.tensor([kernel.candidate_id(1, 0)]),
        torch.tensor([[1]], dtype=torch.int64),
        torch.tensor([True]),
        torch.tensor([-1]),
    )
    assert admission(tuple(state), finite, bank_finite, operands).tolist() == [
        [False, False, False, True]
    ]


def test_exported_mask_has_no_device_scalar_reads(device):
    _query, _kernel, admission, state = _query_with_effect()
    example = (
        tuple(state),
        torch.tensor([[True, True]], dtype=torch.bool),
        torch.tensor([[True, True]], dtype=torch.bool),
        torch.ones(admission.frame_kernel.candidate_count, dtype=torch.bool),
    )
    graph = torch.export.export(admission, example, strict=True).module()
    assert not any("_local_scalar_dense" in str(node.target) for node in graph.graph.nodes)
    compiled = torch.compile(
        graph,
        backend="inductor" if device == "cuda" else "aot_eager",
        fullgraph=True,
    )
    torch.testing.assert_close(compiled(*example), admission(*example), rtol=0, atol=0)


def test_reference_flags_follow_nonidentical_call_port_order():
    child = m.FormulaProgramQueryV5(
        slot_ids=("x", "out"), candidates=(_producer(owner="child", output="out"),),
        terminal_slots={"answer": "out"}, max_steps=1, hidden_dim=8,
    )
    root = m.FormulaProgramQueryV5(
        slot_ids=("unused", "source", "out"),
        candidates=(m.FormulaProgramCallCandidateV1(
            "call", child, input_slots={"x": "source"}, output_slots={"answer": "out"},
        ),), terminal_slots={"answer": "out"}, max_steps=1, hidden_dim=8,
    )
    kernel = FormulaDeviceFrameKernel.from_query(root)
    admission = FormulaDeviceAdmission.from_query(root, kernel)
    state = kernel.initial_state(1, torch.tensor([[0, 1, -1]]), bank_value_handles=torch.tensor([0]))
    operands = torch.ones(kernel.candidate_count, dtype=torch.bool)
    for flags, legal in (((False, True), True), ((True, False), False)):
        pool = torch.tensor([flags], dtype=torch.bool)
        banks = torch.ones(1, 1, dtype=torch.bool)
        expected = admission(state, pool, banks, operands)
        direct = torch.tensor([[*flags, False]], dtype=torch.bool)
        actual = admission(state, direct, banks, operands, reference_flags=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual[0, 0].item() == legal
    assert not admission(state, torch.ones(1, 3, dtype=torch.bool), torch.zeros_like(banks),
                         operands, reference_flags=True).any()
