from __future__ import annotations

import json

import pytest
import torch

import arti
from arti import mechanisms as m


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _bindings(mode="affine", dtype="floating", steps="T"):
    state_dim = 3 if mode == "polar" else 2
    types = (
        m.TensorType(("B", "N", "D"), ("B", 6, 3), dtype=dtype),
        m.TensorType(("B", "T", "S"), ("B", steps, state_dim), dtype=dtype),
        m.TensorType(("B", "N"), ("B", 6), dtype="boolean"),
        m.TensorType(("B", "T"), ("B", steps), dtype=dtype),
    )
    inputs = tuple(
        m.InputBinding(name, kind) for name, kind in zip(("x", "states", "mask", "activity"), types)
    )
    banks = (
        (
            m.BankBinding(
                "weight",
                "arti/observation-test@1",
                "projection",
                m.TensorType(("P", "S"), (6, state_dim), dtype=dtype),
            ),
            m.BankBinding(
                "bias",
                "arti/observation-test@1",
                "projection",
                m.TensorType(("P",), (6,), dtype=dtype),
            ),
        )
        if mode == "affine"
        else ()
    )
    return inputs, banks


def _expression(mode, inputs, banks, **options):
    if mode == "identity":
        return m.observe_identity(*inputs)
    if mode == "affine":
        return m.observe_affine(*inputs, *banks)
    return m.observe_fourier(*inputs, spatial_shape=(2, 3), state_mode=mode, **options)


def _values(mode, device="cpu", dtype=torch.float32, batch=2, steps=4):
    torch.manual_seed(173)
    state_dim = 3 if mode == "polar" else 2
    values = {
        "x": torch.randn(batch, 6, 3, device=device, dtype=dtype, requires_grad=True),
        "states": torch.randn(
            batch, steps, state_dim, device=device, dtype=dtype, requires_grad=True
        ),
        "mask": torch.rand(batch, 6, device=device) > 0.3,
        "activity": torch.ones(batch, steps, device=device, dtype=dtype, requires_grad=True),
    }
    values["mask"][-1] = False
    return values


def _envelope(x, mask):
    domain = m.SupportDomain.for_tensor(
        mask,
        domain_id="observe-test",
        owner_ref="arti/adaptive-observation@1",
        partition_id="world",
        transition_id="test",
    )
    return m.TensorEnvelope(m.EnvelopeRef.WORLD, x, mask, domain)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ("identity", "affine", "cartesian", "polar"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16, torch.float64))
def test_observation_program_matches_existing_module_and_gradients(device, mode, dtype):
    inputs, bank_bindings = _bindings(mode)
    expression = _expression(mode, inputs, bank_bindings)
    mask_input = m.InputBinding(
        "trajectory_mask", m.TensorType(("B", "T"), ("B", "T"), dtype="boolean")
    )
    mask_expression = m.observation_mask(inputs[2], mask_input)
    program = m.FormulaProgram.build(outputs=(expression, mask_expression))
    restored = m.FormulaProgram.from_dict(json.loads(json.dumps(program.to_dict())))
    assert restored.fingerprint == program.fingerprint
    fabric = m.FormulaFabricV2(restored)
    assert not list(fabric.parameters())
    values = _values(mode, device, dtype)
    hard = torch.ones_like(values["activity"], dtype=torch.bool)
    hard[:, -1] = False
    soft = values["activity"]
    activity = hard.to(dtype) + soft - soft.detach()
    values["activity"] = activity
    plan = m.ObservationPlan(values["states"], hard, activity)
    operator = (
        m.IdentityObservationOperator()
        if mode == "identity"
        else m.StateAffineObservationOperator(3, 2)
        if mode == "affine"
        else m.FourierShiftObservationOperator((2, 3), state_mode=mode)
    ).to(device=device, dtype=dtype)
    bank_values = (operator.projection.weight, operator.projection.bias) if mode == "affine" else ()
    banks = {
        binding.name: binding.bind(value) for binding, value in zip(bank_bindings, bank_values)
    }
    result = fabric(inputs={**values, "trajectory_mask": hard}, banks=banks, return_trace=True)
    reference = m.AdaptiveObservation(
        m.FixedObservationPolicy(torch.zeros(4, values["states"].shape[-1])), operator=operator
    )(_envelope(values["x"], values["mask"]), plan=plan)
    torch.testing.assert_close(result.values[0], reference.value, rtol=0, atol=0)
    assert torch.equal(result.values[1], reference.mask)
    assert result.values[0].shape == (2, 4, 6, 3)
    variables = (values["x"], values["states"], soft, *bank_values)
    target = torch.randn_like(reference.value)
    actual_grad = torch.autograd.grad(
        (result.values[0] * target).sum(), variables, retain_graph=True, allow_unused=True
    )
    expected_grad = torch.autograd.grad(
        (reference.value * target).sum(), variables, allow_unused=True
    )
    for actual, expected in zip(actual_grad, expected_grad):
        if expected is None:
            assert actual is None
        else:
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual_grad[2][:, -1].abs().sum() > 0  # hard-off activity retains its ST path
    assert arti.validate_component_provenance(arti.component_provenance(fabric))


@pytest.mark.parametrize("mode", ("identity", "affine", "cartesian", "polar"))
def test_atoms_are_parameter_free_versioned_components(mode):
    inputs, banks = _bindings(mode)
    expression = _expression(mode, inputs, banks)
    types = tuple(x.value_type for x in (*inputs, *banks))
    atom = (
        m.IdentityObservationAtom(types)
        if mode == "identity"
        else m.StateAffineObservationAtom(types)
        if mode == "affine"
        else m.FourierObservationAtom(types, spatial_shape=(2, 3), state_mode=mode)
    )
    assert atom.output_type == expression.value_type
    assert not list(atom.parameters()) and not atom.state_dict()
    assert arti.component_ref(atom) == expression.atom_ref
    assert arti.validate_component_provenance(arti.component_provenance(atom))


def test_affine_observes_original_substrate_not_previous_frame():
    inputs, banks = _bindings("affine")
    values = _values("affine")
    weight = torch.randn(6, 2)
    bias = torch.randn(6)
    result = m.FormulaFabricV2(
        m.FormulaProgram.build(outputs=(m.observe_affine(*inputs, *banks),))
    )(inputs=values, banks={b.name: b.bind(v) for b, v in zip(banks, (weight, bias))}).values[0]
    frames = []
    for state in values["states"].unbind(1):
        gain, shift = torch.nn.functional.linear(state, weight, bias).chunk(2, -1)
        frame = values["x"] * (1 + 0.1 * gain.tanh()).unsqueeze(1) + 0.1 * shift.tanh().unsqueeze(1)
        frames.append(torch.where(values["mask"].unsqueeze(-1), frame, 0))
    torch.testing.assert_close(result, torch.stack(frames, 1))


@pytest.mark.parametrize("mode", ("identity", "affine", "cartesian"))
def test_observation_allocation_is_admitted_before_execution(monkeypatch, mode):
    import arti.formula_observation as observation

    inputs, banks = _bindings(mode)
    expression = _expression(mode, inputs, banks)
    program = m.FormulaProgram.build(
        outputs=(expression,), limits=m.FormulaLimits(max_tensor_elements=100)
    )
    values = _values(mode)
    bound = {b.name: b.bind(torch.ones(b.value_type.sizes)) for b in banks}
    calls = []
    monkeypatch.setattr(observation, "execute_observation", lambda *a: calls.append(a))
    with pytest.raises(m.FormulaBindingError, match="limit"):
        m.FormulaFabricV2(program)(inputs=values, banks=bound)
    assert not calls


def test_invalid_shapes_and_runtime_projection_dtypes_are_clear():
    inputs, banks = _bindings("affine")
    with pytest.raises(m.FormulaTypeError, match="mask"):
        m.observe_affine(inputs[0], inputs[1], inputs[3], inputs[3], *banks)
    with pytest.raises(m.FormulaTypeError, match="H\\*W"):
        m.observe_fourier(*inputs, spatial_shape=(4, 4))
    with pytest.raises(m.FormulaTypeError, match="positive"):
        m.observe_affine(*inputs, *banks, scale=float("nan"))
    values = _values("affine")
    program = m.FormulaProgram.build(outputs=(m.observe_affine(*inputs, *banks),))
    bound = {b.name: b.bind(torch.ones(b.value_type.sizes, dtype=torch.float64)) for b in banks}
    with pytest.raises(m.FormulaBindingError, match="dtypes must match"):
        m.FormulaFabricV2(program)(inputs=values, banks=bound)


def test_fourier_complex_temporary_respects_tensor_byte_limit(monkeypatch):
    import arti.formula_observation as observation

    inputs, _ = _bindings("cartesian")
    expression = m.observe_fourier(*inputs, spatial_shape=(2, 3))
    values = _values("cartesian", dtype=torch.bfloat16)
    # Real bf16 output is 288 bytes; the expanded complex64 temporary is 1152.
    program = m.FormulaProgram.build(
        outputs=(expression,), limits=m.FormulaLimits(max_tensor_bytes=600)
    )
    calls = []
    monkeypatch.setattr(observation, "execute_observation", lambda *args: calls.append(args))
    with pytest.raises(m.FormulaBindingError, match="byte limit"):
        m.FormulaFabricV2(program)(inputs=values, banks={})
    assert not calls


@pytest.mark.parametrize("batch,steps", ((1, 1), (3, 7)))
def test_dynamic_trajectory_extent(batch, steps):
    inputs, _ = _bindings("identity")
    program = m.FormulaProgram.build(outputs=(m.observe_identity(*inputs),))
    values = _values("identity", batch=batch, steps=steps)
    assert m.FormulaFabricV2(program)(inputs=values, banks={}).values[0].shape == (
        batch,
        steps,
        6,
        3,
    )


@pytest.mark.parametrize("policy_kind", ("learned", "bank"))
def test_learned_trajectory_and_bank_policy_gradients_reach_existing_parameters(policy_kind):
    policy = (
        m.LearnedObservationPolicy(3, 2, 4)
        if policy_kind == "learned"
        else m.BankConditionedObservationPolicy(
            input_dim=3,
            state_dim=2,
            max_observations=4,
            banks=[m.ObservationOperandBank(8, 4, 3, seed=3, bank_id="obs")],
            key_dim=4,
        )
    )
    values = _values("affine")
    plan = policy(values["x"], values["mask"])
    values.update(states=plan.states, activity=plan.activity_weights())
    inputs, banks = _bindings("affine")
    operator = m.StateAffineObservationOperator(3, 2)
    parameters = (operator.projection.weight, operator.projection.bias)
    actual = m.FormulaFabricV2(
        m.FormulaProgram.build(outputs=(m.observe_affine(*inputs, *banks),))
    )(inputs=values, banks={b.name: b.bind(v) for b, v in zip(banks, parameters)}).values[0]
    expected = m.AdaptiveObservation(policy, operator=operator)(
        _envelope(values["x"], values["mask"]), plan=plan
    ).value
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ("identity", "affine", "cartesian", "polar"))
def test_prepared_plan_and_batched_candidate_execution(device, mode):
    inputs, banks = _bindings(mode, steps=4)
    expression = _expression(mode, inputs, banks)
    program = m.FormulaProgram.build(outputs=(expression,))
    values = _values(mode, device)
    parameters = (torch.randn(6, 2, device=device), torch.randn(6, device=device)) if banks else ()
    bound = {b.name: b.bind(v) for b, v in zip(banks, parameters)}
    fabric = m.FormulaFabricV2(program)
    prepared = fabric.bind_tensors(inputs=values, banks=bound)
    plan = fabric.execution_plan()
    expected = fabric(inputs=values, banks=bound).values[0]
    torch.testing.assert_close(plan(prepared)[0], expected)
    # Candidate lanes are a separate batch, not the Observation trajectory axis.
    tensors = tuple(v.detach() for v in prepared.values)
    stacked = tuple(torch.stack((v, v)) for v in tensors)

    def execute(*args):
        return plan(
            m.PreparedFormulaBindings(prepared.program_fingerprint, prepared.binding_names, args)
        )[0]

    actual = torch.vmap(execute)(*stacked)
    torch.testing.assert_close(actual, expected.detach().unsqueeze(0).expand(2, -1, -1, -1, -1))


@pytest.mark.parametrize("mode", ("identity", "affine", "cartesian", "polar"))
def test_observation_plan_compile_contract(mode):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs, banks = _bindings(mode, steps=4)
    expression = _expression(mode, inputs, banks)
    program = m.FormulaProgram.build(outputs=(expression,))
    values = _values(mode, device)
    parameters = (
        (
            torch.randn(6, 2, device=device, requires_grad=True),
            torch.randn(6, device=device, requires_grad=True),
        )
        if banks
        else ()
    )
    fabric = m.FormulaFabricV2(program)
    prepared = fabric.bind_tensors(
        inputs=values, banks={b.name: b.bind(v) for b, v in zip(banks, parameters)}
    )
    plan = fabric.execution_plan()
    # safe_training retains the existing eager FFT boundary inside a compiled graph.
    compiled = torch.compile(plan, fullgraph=mode in {"identity", "affine"})
    expected = plan(prepared)[0]
    actual = compiled(prepared)[0]
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert values["x"].grad is not None and torch.isfinite(values["x"].grad).all()
    if mode != "identity":
        assert torch.isfinite(values["states"].grad).all()


def test_zero_polar_direction_has_finite_backward():
    inputs, _ = _bindings("polar")
    values = _values("polar")
    values["states"] = torch.zeros(2, 4, 3, requires_grad=True)
    output = m.FormulaFabricV2(
        m.FormulaProgram.build(
            outputs=(m.observe_fourier(*inputs, spatial_shape=(2, 3), state_mode="polar"),)
        )
    )(inputs=values, banks={}).values[0]
    output.square().mean().backward()
    assert torch.isfinite(values["states"].grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("mode", ("identity", "affine", "cartesian"))
def test_observation_candidates_execute_in_captured_k_wide_search(mode):
    from benchmarks._federated_captured_search import captured_search_execution
    from benchmarks._federated_recursive_search import (
        search_recursive_graphs,
        start_recursive_search,
    )

    with torch.device("cuda"), torch.no_grad():
        inputs, banks = _bindings(mode, steps=4)
        expression = _expression(mode, inputs, banks)
        program = m.FormulaProgram.build(outputs=(expression,))
        operands = {b.name: torch.randn(b.value_type.sizes) for b in banks}
        candidate = m.FormulaProgramTensorCandidateV3(
            m.FormulaProgramCandidateV2(
                "observe",
                program,
                input_slots={b.name: b.name for b in inputs},
                output_slot="observed",
                operands=operands,
            )
        )
        observed = m.InputBinding("observed", expression.value_type)
        finish = m.FormulaProgramTensorCandidateV3(
            m.FormulaProgramCandidateV2(
                "reduce-observations",
                m.FormulaProgram.build(outputs=(m.reduce_sum(observed, axis="T"),)),
                input_slots={"observed": "observed"},
                output_slot="out",
            )
        )
        query = m.FormulaProgramQueryV5(
            slot_ids=tuple(b.name for b in inputs) + ("observed", "out"),
            candidates=(candidate, finish),
            terminal_slots={"answer": "out"},
            max_steps=2,
            hidden_dim=8,
        )
        values = {k: v.detach() for k, v in _values(mode, "cuda", batch=1).items()}
        values["mask"].fill_(True)
        starts = (start_recursive_search(query, values),)
        options = dict(width=2, beam_width=2)
        expected = search_recursive_graphs(starts, **options)
        samples = (*values.values(), torch.zeros(1, 4, 6, 3), values["x"])
        with captured_search_execution(horizon=5, value_samples=samples) as backend:
            actual = search_recursive_graphs(starts, **options)
            assert not backend.fallbacks
            assert backend.completed == 1
        torch.testing.assert_close(
            actual.winner.execution.outputs["answer"], expected.winner.execution.outputs["answer"]
        )
        assert len(actual.winner.route) == len(expected.winner.route)


@pytest.mark.parametrize("mode", ("identity", "affine", "cartesian", "polar"))
@pytest.mark.parametrize("backend", ("eager", "captured", "aot_eager", "inductor"))
def test_grouped_training_preserves_unused_gradients_and_fft_boundary(mode, backend):
    from arti._formula_candidate_batch import _checked_plan
    from arti._formula_grouped_training import execute_grouped_training, grouped_formula_training

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs, banks = _bindings(mode, steps=4)
    program = m.FormulaProgram.build(outputs=(_expression(mode, inputs, banks),))
    values = _values(mode, device)
    parameters = (
        (
            torch.randn(6, 2, device=device, requires_grad=True),
            torch.randn(6, device=device, requires_grad=True),
        )
        if banks
        else ()
    )
    fabric = m.FormulaFabricV2(program)
    prepared = fabric.bind_tensors(
        inputs=values, banks={b.name: b.bind(v) for b, v in zip(banks, parameters)}
    )
    variables = (values["x"], values["states"], values["activity"], *parameters)
    plan = _checked_plan(program)
    expected = plan(prepared)[0]
    left = torch.autograd.grad(expected.square().mean(), variables, allow_unused=True)
    with grouped_formula_training(backend=backend):
        results, finite = execute_grouped_training(plan, (prepared, prepared))
        assert finite.all()
        actual = results[0][0]
        right = torch.autograd.grad(actual.square().mean(), variables, allow_unused=True)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(left, right):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b)


def test_boolean_activity_and_mask_outputs_in_grouped_execution():
    from arti._formula_candidate_batch import _checked_plan
    from arti._formula_grouped_training import execute_grouped_training, grouped_formula_training

    inputs, _ = _bindings("identity", steps=4)
    activity = m.InputBinding("activity", m.TensorType(("B", "T"), ("B", 4), dtype="boolean"))
    inputs = (*inputs[:3], activity)
    program = m.FormulaProgram.build(outputs=(m.observe_identity(*inputs), m.observation_mask(inputs[2], activity)))
    values = _values("identity")
    values["activity"] = torch.tensor([[True, False, True, False], [False, False, False, False]])
    fabric = m.FormulaFabricV2(program)
    prepared = fabric.bind_tensors(inputs=values, banks={})
    expected = fabric(inputs=values, banks={}).values
    with grouped_formula_training(backend="aot_eager"):
        result, _ = execute_grouped_training(_checked_plan(program), (prepared, prepared))
        for actual, reference in zip(result[0], expected):
            torch.testing.assert_close(actual, reference)
        assert result[0][1].dtype == torch.bool and not result[0][1].requires_grad
        grad = torch.autograd.grad(result[0][0].sum(), (values["x"], values["states"]), allow_unused=True)
        assert grad[1] is None


@pytest.mark.parametrize("mode", ("cartesian", "polar"))
def test_fourier_fabric_fullgraph_forward_opt_in(mode):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs, _ = _bindings(mode, steps=4)
    expression = _expression(mode, inputs, (), compile_policy="fullgraph_forward")
    fabric = m.FormulaFabricV2(m.FormulaProgram.build(outputs=(expression,)))
    with torch.no_grad():
        prepared = fabric.bind_tensors(inputs=_values(mode, device), banks={})
        execution = fabric.execution_plan()
        actual = torch.compile(execution, fullgraph=True)(prepared)[0]
        torch.testing.assert_close(actual, execution(prepared)[0])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_polar_matches_pre_extraction_formula_away_from_zero(device, dtype):
    from arti.observation import _native_fourier_shift

    x = torch.randn(2, 6, 3, device=device, dtype=dtype, requires_grad=True)
    state = torch.randn(2, 4, 3, device=device, dtype=dtype, requires_grad=True)
    operator = m.FourierShiftObservationOperator((2, 3), state_mode="polar")
    radius, direction = state[..., 0], state[..., 1:3]
    norm = direction.square().sum(-1).sqrt()
    unit = direction / norm.clamp_min(1e-6).unsqueeze(-1)
    unit = torch.where((norm > 1e-6).unsqueeze(-1), unit, torch.zeros_like(unit))
    expected = _native_fourier_shift(x.reshape(2, 2, 3, 3), radius * unit[..., 0], radius * unit[..., 1]).reshape(2, 4, 6, 3)
    actual = operator.forward_trajectory(x, state)
    torch.testing.assert_close(actual, expected)
    a = torch.autograd.grad(actual.square().mean(), (x, state), retain_graph=True)
    b = torch.autograd.grad(expected.square().mean(), (x, state))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right)
