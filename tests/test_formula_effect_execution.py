from contextlib import nullcontext

import pytest
import torch

from arti._formula_effect_execution import _TensorEffectPlan, tensor_effect_execution, _EFFECT_BACKEND
from arti import formula_v3 as f
from arti.formula_v2 import TensorType


KINDS = tuple(sorted(f.NEURAL_PLASTICITY_EFFECT_REFS))


def _fixture(kind, device, dtype):
    generator = torch.Generator(device=device).manual_seed(84)

    def random(*shape):
        return torch.randn(*shape, generator=generator, device=device, dtype=dtype) * 0.1

    state = random(2, 3)
    attributes = ()
    if kind in {f.NEURAL_PLASTICITY_ATOM_REF, f.NEURAL_PLASTICITY_BLEND_ATOM_REF, f.NEURAL_PLASTICITY_PROXIMAL_ATOM_REF}:
        operands = (random(2, 3), random(2, 3))
    elif kind in {f.NEURAL_PLASTICITY_OUTER_ATOM_REF, f.NEURAL_PLASTICITY_OUTER_V2_ATOM_REF}:
        operands = (random(2), random(3), state.new_tensor(0.25))
        if kind == f.NEURAL_PLASTICITY_OUTER_V2_ATOM_REF:
            operands += (state.new_tensor(2.4),)
            attributes = (("max_executions", 4),)
    else:
        factors = 3 if kind == f.NEURAL_PLASTICITY_POLYNOMIAL_ATOM_REF else 2
        operands = (random(2, 3), *(random(3, 2) for _ in range(factors)), state.new_tensor(0.2))
        attributes = (("state_axis", "D"),)
    return state, f.NeuralPlasticityEffectV2("effect", kind, operands, attributes)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", KINDS)
def test_tensor_effect_plan_matches_each_native_family_and_repeat(device, dtype, kind):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    state, effect = _fixture(kind, device, dtype)
    state_type = TensorType(("N", "D"), (2, 3), dtype="floating")
    counts = state.new_tensor([-1., 0., 1., 1.5, 2.5, 3., 9.])
    counted = kind != f.NEURAL_PLASTICITY_OUTER_V2_ATOM_REF
    axis, maximum = f._validate_neural_plasticity_step(effect, state, state_type)
    plan = _TensorEffectPlan(kind, axis, maximum, 4, counted)
    states = torch.stack(tuple(state + i * 0.01 for i in range(len(counts))))
    operands = tuple(torch.stack(tuple(value for _ in counts)) for value in effect.operands)
    with torch.no_grad():
        expected = torch.stack(tuple(f.apply_neural_plasticity_effect(
            effect, value, state_type=state_type,
            execution_count=count if counted else None, max_executions=4,
        ) for value, count in zip(states, counts, strict=True)))
        actual, finite = plan(states, counts, *operands)
    assert finite.all()
    torch.testing.assert_close(actual, expected)
    if counted:
        assert torch.equal(actual[:2], states[:2])


@pytest.mark.parametrize("kind", KINDS)
def test_effect_plan_is_one_full_graph_without_tensor_scalar_reads(kind):
    torch.compiler.reset()
    state, effect = _fixture(kind, "cpu", torch.float32)
    axis, maximum = f._validate_neural_plasticity_step(effect, state, TensorType(("N", "D"), (2, 3)))
    plan = _TensorEffectPlan(kind, axis, maximum, 4, kind != f.NEURAL_PLASTICITY_OUTER_V2_ATOM_REF)
    graphs = []

    def capture(graph, inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(plan, backend=capture, fullgraph=True)
    states = state.expand(3, -1, -1).clone()
    operands = tuple(value.expand(3, *value.shape).clone() for value in effect.operands)
    with torch.no_grad():
        for counts in (torch.tensor([0., 1., 2.]), torch.tensor([4., 0., 3.])):
            expected = plan(states, counts, *operands)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                actual = compiled(states, counts, *operands)
            torch.testing.assert_close(actual, expected)
            assert not any(event.key in {"aten::item", "aten::_local_scalar_dense"} for event in profile.key_averages())
    assert len(graphs) == 1


def test_unselected_overflow_and_invalid_intermediate_are_separate():
    plan = _TensorEffectPlan(f.NEURAL_PLASTICITY_ATOM_REF, 0, 1, 4, True)
    state = torch.full((3, 2, 3), 1e20)
    additive = torch.zeros_like(state)
    multiplier = torch.full_like(state, 1e20)
    with torch.no_grad():
        actual, finite = plan(state, torch.tensor([0., 1., 3.]), additive, multiplier)
    assert finite.tolist() == [True, False, False]
    assert torch.equal(actual[0], state[0])


def test_effect_scope_is_explicit_and_restored_after_failure():
    assert _EFFECT_BACKEND.get() is None
    with pytest.raises(RuntimeError), tensor_effect_execution(backend="eager"):
        assert _EFFECT_BACKEND.get() == "eager"
        with tensor_effect_execution(backend="aot_eager"):
            assert _EFFECT_BACKEND.get() == "aot_eager"
        raise RuntimeError("test")
    assert _EFFECT_BACKEND.get() is None


@pytest.mark.parametrize("backend", [None, "eager", "aot_eager"])
def test_candidate_batch_uses_predecessor_state_not_operands_as_state(backend):
    from test_formula_candidate_batch import _ordinary, _effect, _query

    first = _ordinary("first", "x", "made", owner="owner")
    effects = tuple(_effect(f"effect-{i}", "made", "changed", count=True) for i in range(3))
    reread = _ordinary("reread", "changed", "terminal", owner="owner")
    query = _query((first, *effects, reread), slots=("x", "made", "changed", "terminal"))
    with torch.no_grad():
        for count, effect in zip((0., 1., 3.), effects, strict=True):
            effect.execution_count.fill_(count)
        roots = tuple(first(query._arena({"x": torch.full((1, 3), value)})) for value in (0.2, 0.4, 0.8))
        requests = tuple(zip(effects, roots, strict=True))
        expected = query.execute_many(requests, serial=True)
        with tensor_effect_execution(backend=backend) if backend else nullcontext():
            actual = query.execute_many(requests)
        for root, reference, result in zip(roots, expected, actual, strict=True):
            assert result.values.get("changed") is root.values.get("made")
            assert result.proposals[-1].target == first.bank_slot_ref
            assert result.proposals[-1].previous is root.effect_state(first.bank_slot_ref)[0]
            assert result.proposals[-1].successor_revision == 1
            torch.testing.assert_close(result.proposals[-1].successor, reference.proposals[-1].successor)
            torch.testing.assert_close(reread(result).values.get("terminal"), reread(reference).values.get("terminal"))
        assert first.initial_revision() == 0


def test_gradient_replay_keeps_independent_parameters_and_count_surrogate():
    from test_formula_candidate_batch import _ordinary, _effect, _query

    first = _ordinary("first", "x", "made", owner="owner")
    selected, unused = _effect("selected", count=True), _effect("unused", count=True)
    reread = _ordinary("reread", "changed", "terminal", owner="owner")
    query = _query((first, selected, unused, reread), slots=("x", "made", "changed", "terminal"))
    x = torch.ones(1, 3, requires_grad=True)
    root = first(query._arena({"x": x}))
    with tensor_effect_execution(backend="eager"):
        results = query.execute_many(((selected, root), (unused, root)))
    reread(results[0]).values.get("terminal").sum().backward()
    assert selected.execution_count.grad is not None
    assert selected.operand_store.tensor("writer").grad is not None
    assert unused.execution_count.grad is None
    assert unused.operand_store.tensor("writer").grad is None


@pytest.mark.parametrize("kind", [f.NEURAL_PLASTICITY_ATOM_REF, f.NEURAL_PLASTICITY_TRANSPORT_ATOM_REF])
def test_cuda_inductor_replays_dynamic_counts_and_current_state(kind):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.compiler.reset()
    state, effect = _fixture(kind, "cuda", torch.float32)
    axis, maximum = f._validate_neural_plasticity_step(effect, state, TensorType(("N", "D"), (2, 3)))
    plan = _TensorEffectPlan(kind, axis, maximum, 4, True)
    compiled = torch.compile(plan, backend="inductor", mode="reduce-overhead", fullgraph=True)
    states = state.expand(4, -1, -1).clone()
    operands = tuple(value.expand(4, *value.shape).clone() for value in effect.operands)
    with torch.no_grad():
        for step in range(3):
            states.add_(0.02)
            counts = states.new_tensor([step, 0., 2., 4.])
            expected = plan(states, counts, *operands)
            actual = compiled(states, counts, *operands)
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("count", [float("inf"), -float("inf"), float("nan")])
def test_count_boundary_keeps_native_clamping_and_nan_error(count):
    from test_formula_candidate_batch import _ordinary, _effect, _query
    from arti._formula_candidate_batch import execute_many

    first = _ordinary("first", "x", "made", owner="owner")
    effect = _effect("effect", "made", "terminal", count=True)
    query = _query((first, effect), slots=("x", "made", "terminal"))
    with torch.no_grad():
        effect.execution_count.fill_(count)
        root = first(query._arena({"x": torch.ones(1, 3)}))
        for backend in (None, "eager"):
            with tensor_effect_execution(backend=backend) if backend else nullcontext():
                if count != count:
                    with pytest.raises(ValueError):
                        execute_many(((effect, root), (effect, root)), reject_nonfinite=True)
                else:
                    actual = query.execute_many(((effect, root), (effect, root)))
                    expected = effect(root)
                    for result in actual:
                        torch.testing.assert_close(result.proposals[-1].successor, expected.proposals[-1].successor)


def test_lowered_plan_does_not_silently_replace_gradient_count_semantics():
    plan = _TensorEffectPlan(f.NEURAL_PLASTICITY_ATOM_REF, 0, 1, 4, True)
    with pytest.raises(RuntimeError, match="no_grad"):
        plan(torch.ones(2, 3), torch.ones(2), torch.zeros(2, 3), torch.zeros(2, 3))


@pytest.mark.parametrize("count", [0., 1.])
def test_zero_count_retains_native_law_admission_boundary(count):
    from test_formula_candidate_batch import _ordinary, _effect, _query, _prepared_effect_rows
    from arti import _formula_candidate_batch as batch

    first = _ordinary("first", "x", "made", owner="owner")
    effect = _effect("effect", "made", "terminal", count=True)
    query = _query((first, effect), slots=("x", "made", "terminal"))
    with torch.no_grad():
        effect.execution_count.fill_(count)
        root = first(query._arena({"x": torch.ones(1, 3)}))
        request, prepared, numeric = _prepared_effect_rows((effect,), root)[0]
        # A zero-count effect does not apply this otherwise incompatible law.
        numeric = {name: value.expand(2, 3) for name, value in numeric.items()}
        row = ((request, prepared, numeric),)
        for backend in (None, "eager"):
            with tensor_effect_execution(backend=backend) if backend else nullcontext():
                if count:
                    with pytest.raises(f.FormulaProgramError, match="exactly match"):
                        batch._finish_many_checked(row, reject_nonfinite=False)
                else:
                    result = batch._finish_many_checked(row, reject_nonfinite=False)[0]
                    assert result.proposals[-1].successor is root.effect_state(first.bank_slot_ref)[0]
