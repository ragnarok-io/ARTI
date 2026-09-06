import pytest
import torch

from arti import mechanisms as m
from arti._formula_device_decision import FormulaDeviceDecisionWave
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
    weight = m.BankBinding("weight", "arti/device-decision-test@1", "weight", _type())
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
    rate = m.BankBinding("rate", "arti/device-decision-test@1", "rate", _type())
    zero = m.BankBinding("zero", "arti/device-decision-test@1", "zero", _type())
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


def _zero_network(query, bias):
    with torch.no_grad():
        for parameter in query.network.parameters():
            parameter.zero_()
        query.network[-1].bias.copy_(torch.tensor(bias, dtype=query.network[-1].bias.dtype))
    return query


def _fixture():
    query = _zero_network(
        m.FormulaProgramQueryV5(
            slot_ids=("x", "owned", "tail"),
            candidates=(_producer(), _effect()),
            terminal_slots={"answer": "tail"},
            max_steps=2,
            hidden_dim=8,
        ),
        (3.0, 2.0, 1.0),
    )
    kernel = FormulaDeviceFrameKernel.from_query(query)
    wave = FormulaDeviceDecisionWave.from_query(
        query,
        frame_kernel=kernel,
        candidate_family_ids=(0, 1, 2),
        width=3,
    ).to(next(query.parameters()).device)
    state = kernel.initial_state(
        1,
        torch.tensor([[0, -1, -1]], dtype=torch.int64),
        bank_value_handles=torch.tensor([0], dtype=torch.int64),
    )
    data_pool = torch.ones((2, 3))
    bank_pool = torch.full((2, 1, 3), 2.0)
    operands = torch.ones(3, dtype=torch.bool)
    scores = torch.zeros(1)
    return query, kernel, wave, state, data_pool, bank_pool, operands, scores


def test_decision_wave_keeps_admission_scores_and_selection_on_device():
    _query, kernel, wave, state, data_pool, bank_pool, operands, scores = _fixture()
    result = wave(tuple(state), data_pool, bank_pool, operands, scores)
    assert result.eligible.tolist() == [[True, False, False]]
    assert result.selected_candidates.tolist() == [[0, -1, -1]]
    torch.testing.assert_close(result.scores[0, 0], torch.tensor(0.0), rtol=0, atol=0)

    state, _ = kernel(
        state,
        torch.tensor([0]),
        torch.tensor([[1]], dtype=torch.int64),
        torch.tensor([True]),
        torch.tensor([-1]),
    )
    result = wave(tuple(state), data_pool, bank_pool, operands, scores)
    assert result.eligible.tolist() == [[False, True, False]]
    assert result.selected_candidates.tolist() == [[1, -1, -1]]


def test_decision_wave_routes_root_and_child_without_host_packet():
    child = _zero_network(
        m.FormulaProgramQueryV5(
            slot_ids=("x", "out"),
            candidates=(_producer(owner="child-memory", output="out"),),
            terminal_slots={"answer": "out"},
            max_steps=1,
            hidden_dim=8,
        ),
        (1.0, 0.0),
    )
    call = m.FormulaProgramCallCandidateV1(
        "call",
        child,
        input_slots={"x": "x"},
        output_slots={"answer": "out"},
    )
    root = _zero_network(
        m.FormulaProgramQueryV5(
            slot_ids=("x", "out"),
            candidates=(call,),
            terminal_slots={"answer": "out"},
            max_steps=1,
            hidden_dim=8,
        ),
        (1.0, 0.0),
    )
    kernel = FormulaDeviceFrameKernel.from_query(root)
    wave = FormulaDeviceDecisionWave.from_query(
        root,
        frame_kernel=kernel,
        candidate_family_ids=(0, 1, 0, 1),
        width=2,
    ).to(next(root.parameters()).device)
    state = kernel.initial_state(
        1,
        torch.tensor([[0, -1]], dtype=torch.int64),
        bank_value_handles=torch.tensor([0], dtype=torch.int64),
    )
    arguments = (torch.ones((2, 3)), torch.ones((2, 1, 3)), torch.ones(4, dtype=torch.bool), torch.zeros(1))
    result = wave(tuple(state), *arguments)
    assert result.selected_candidates.tolist() == [[0, -1]]

    state, _ = kernel(
        state,
        torch.tensor([0]),
        torch.tensor([[-1]], dtype=torch.int64),
        torch.tensor([True]),
        torch.tensor([-1]),
    )
    result = wave(tuple(state), *arguments)
    assert result.selected_candidates.tolist() == [[2, -1]]


def test_local_ranking_survives_negative_infinite_parent_score():
    query = _zero_network(
        m.FormulaProgramQueryV5(
            slot_ids=("x", "left", "right"),
            candidates=(
                _producer("left-producer", owner="left-memory", output="left"),
                _producer("right-producer", owner="right-memory", output="right"),
            ),
            terminal_slots={"left": "left", "right": "right"},
            max_steps=2,
            hidden_dim=8,
        ),
        (1.0, 3.0, 0.0),
    )
    kernel = FormulaDeviceFrameKernel.from_query(query)
    wave = FormulaDeviceDecisionWave.from_query(
        query,
        frame_kernel=kernel,
        candidate_family_ids=(0, 0, 1),
        width=1,
    ).to(next(query.parameters()).device)
    state = kernel.initial_state(
        1,
        torch.tensor([[0, -1, -1]], dtype=torch.int64),
        bank_value_handles=torch.tensor([0, 1], dtype=torch.int64),
    )
    result = wave(
        tuple(state),
        torch.ones((1, 3)),
        torch.ones((2, 1, 3)),
        torch.ones(3, dtype=torch.bool),
        torch.tensor([-float("inf")]),
    )
    assert result.eligible.tolist() == [[True, True, False]]
    assert result.scores[0, :2].tolist() == [-float("inf"), -float("inf")]
    assert result.selected_candidates.tolist() == [[1]]


def test_compiled_decision_wave_accepts_changed_runtime_state(device):
    _query, kernel, wave, state, data_pool, bank_pool, operands, scores = _fixture()
    example = (tuple(state), data_pool, bank_pool, operands, scores)
    graph = torch.export.export(wave, example, strict=True).module()
    assert not any("_local_scalar_dense" in str(node.target) for node in graph.graph.nodes)
    compiled = torch.compile(
        graph,
        backend="inductor" if device == "cuda" else "aot_eager",
        fullgraph=True,
    )
    expected = wave(*example)
    actual = compiled(*example)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)

    state, _ = kernel(
        state,
        torch.tensor([0]),
        torch.tensor([[1]], dtype=torch.int64),
        torch.tensor([True]),
        torch.tensor([-1]),
    )
    changed = (tuple(state), data_pool, bank_pool, operands, scores)
    expected = wave(*changed)
    actual = compiled(*changed)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_finite_work_follows_current_references_not_pool_capacity(monkeypatch):
    _query, _kernel, wave, state, _data, _bank, operands, scores = _fixture()
    data = torch.ones(4096, 3)
    bank = torch.ones(4096, 1, 3)
    sizes = []
    original = wave._pool_finite

    def record(value):
        sizes.append(value.shape[0])
        return original(value)

    monkeypatch.setattr(wave, "_pool_finite", record)
    assert wave(state, data, bank, operands, scores).eligible.tolist() == [[True, False, False]]
    assert sizes == [3, 1]
    data[-1].fill_(float("nan"))
    bank[-1].fill_(float("nan"))
    assert wave(state, data, bank, operands, scores).eligible.tolist() == [[True, False, False]]
    bank[0].fill_(float("nan"))
    assert not wave(state, data, bank, operands, scores).eligible.any()
    bank[0].fill_(1)
    data[0].fill_(float("nan"))
    assert not wave(state, data, bank, operands, scores).eligible.any()


def test_gather_masks_invalid_handles_without_copying_pool():
    pool = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    pool[0].fill_(float("nan"))
    actual = FormulaDeviceDecisionWave._gather_pool(pool, torch.tensor([-1, 1, 4, 99]))
    torch.testing.assert_close(actual, torch.tensor([[0., 0., 0.], [3., 4., 5.], [0., 0., 0.], [0., 0., 0.]]))


def test_typed_reference_flags_keep_global_handles_and_sinks():
    from arti._formula_device_pools import FormulaDevicePoolLayout

    _query, _kernel, wave, _state, data, _bank, _operands, _scores = _fixture()
    layout = FormulaDevicePoolLayout.from_samples((torch.zeros(1, 3), torch.zeros(1, 2, 3)), 128)
    pools = layout.allocate(data.device)
    handles = torch.tensor([[3, layout.offsets[1] + 5, -1, 128],
                            [layout.offsets[1] + 5, 3, layout.offsets[1] + 128, 999]])
    pools[0][3].fill_(float("nan"))
    pools[0][-1].fill_(float("nan"))
    pools[1][-1].fill_(float("nan"))
    expected = torch.tensor([[False, True, False, False], [True, False, False, False]])
    torch.testing.assert_close(wave._reference_finite(pools, handles, layout), expected)
    pools[0][3].zero_()
    pools[1][5].fill_(float("inf"))
    expected = torch.tensor([[True, False, False, False], [False, True, False, False]])
    torch.testing.assert_close(wave._reference_finite(pools, handles, layout), expected)
    assert wave._reference_finite(pools, handles[:, :0], layout).shape == (2, 0)
