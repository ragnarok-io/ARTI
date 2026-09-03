from __future__ import annotations

import pytest
import torch

import arti
import arti.formula_v3 as formula_v3
from arti import mechanisms
from arti.formula_v3 import apply_neural_plasticity_effect


def _schema(width: int = 3) -> mechanisms.TensorSchema:
    return mechanisms.TensorSchema(
        dtype="float32",
        device_class="any",
        dimensions=("B", width),
        semantic_axes=("batch", "feature"),
        mask_semantics="none",
    )


def _type(width: int = 3) -> mechanisms.TensorType:
    return mechanisms.TensorType(
        ("B", "D"),
        ("B", width),
        dtype="float32",
        domain="activation",
    )


def _state_type(*shape: int) -> mechanisms.TensorType:
    axes = () if not shape else ("D",)
    return mechanisms.TensorType(axes, shape, dtype="float32", domain="activation")


def _binding(name: str, value_type: mechanisms.TensorType) -> mechanisms.BankBinding:
    return mechanisms.BankBinding(
        name,
        source_ref="arti/formula-operand-bank@1",
        partition_id=name,
        value_type=value_type,
    )


def _simple_action() -> mechanisms.BankLocalFormulaEffectAction:
    value = mechanisms.InputBinding("value", _type())
    state = _state_type(3)
    writer = _binding("writer", state)
    gain = _binding("gain", state)
    effect = mechanisms.neural_plasticity(
        value,
        mechanisms.reduce_sum(mechanisms.scale(value, writer), axis="B"),
        mechanisms.reduce_sum(mechanisms.scale(value, gain), axis="B"),
    )
    return mechanisms.BankLocalFormulaEffectAction(
        "remember",
        mechanisms.FormulaEffectProgramV2(
            mechanisms.FormulaProgram.build(outputs=(effect,)),
            data_input_name="value",
            state_type=state,
        ),
        input_schema=_schema(),
        operands={
            "writer": torch.tensor([0.1, 0.2, 0.3]),
            "gain": torch.tensor([-0.1, 0.05, 0.2]),
        },
        trainable_operands=("writer", "gain"),
    )


def _blend_action() -> mechanisms.BankLocalFormulaEffectAction:
    value = mechanisms.InputBinding("value", _type())
    state = _state_type(3)
    target_weight = _binding("target-weight", state)
    amount_weight = _binding("amount-weight", state)
    target = mechanisms.reduce_sum(mechanisms.scale(value, target_weight), axis="B")
    amount = mechanisms.scalar_map(
        mechanisms.reduce_sum(mechanisms.scale(value, amount_weight), axis="B"),
        mode="sigmoid",
    )
    effect = mechanisms.neural_plasticity_blend(value, target, amount)
    return mechanisms.BankLocalFormulaEffectAction(
        "blend",
        mechanisms.FormulaEffectProgramV2(
            mechanisms.FormulaProgram.build(outputs=(effect,)),
            data_input_name="value",
            state_type=state,
        ),
        input_schema=_schema(),
        operands={
            "target-weight": torch.tensor([0.5, -0.25, 0.75]),
            "amount-weight": torch.tensor([0.1, 0.2, -0.3]),
        },
        trainable_operands=("target-weight", "amount-weight"),
    )


def _outer_action() -> mechanisms.BankLocalFormulaEffectAction:
    value_type = _type(2)
    state = mechanisms.TensorType(
        ("D", "I"),
        (2, 3),
        dtype="float32",
        domain="activation",
    )
    value = mechanisms.InputBinding("value", value_type)
    right = _binding(
        "right",
        mechanisms.TensorType(
            ("I",),
            (3,),
            dtype="float32",
            domain="activation",
        ),
    )
    rate = _binding("rate", _state_type())
    effect = mechanisms.neural_plasticity_outer(
        value,
        mechanisms.reduce_sum(value, axis="B"),
        right,
        rate,
    )
    return mechanisms.BankLocalFormulaEffectAction(
        "outer",
        mechanisms.FormulaEffectProgramV2(
            mechanisms.FormulaProgram.build(outputs=(effect,)),
            data_input_name="value",
            state_type=state,
        ),
        input_schema=_schema(2),
        operands={"right": torch.tensor([0.25, -0.5, 1.0]), "rate": torch.tensor(0.2)},
        trainable_operands=("right", "rate"),
    )


def _repeated_outer_action(
    execution_count: float = 3.0,
) -> mechanisms.BankLocalFormulaEffectAction:
    value_type = _type(2)
    state = mechanisms.TensorType(
        ("D", "I"),
        (2, 3),
        dtype="float32",
        domain="activation",
    )
    value = mechanisms.InputBinding("value", value_type)
    right = _binding(
        "right",
        mechanisms.TensorType(
            ("I",),
            (3,),
            dtype="float32",
            domain="activation",
        ),
    )
    rate = _binding("rate", _state_type())
    count = _binding("execution-count", _state_type())
    effect = mechanisms.neural_plasticity_outer(
        value,
        mechanisms.reduce_sum(value, axis="B"),
        right,
        rate,
        count,
        max_executions=8,
    )
    return mechanisms.BankLocalFormulaEffectAction(
        "repeated-outer",
        mechanisms.FormulaEffectProgramV2(
            mechanisms.FormulaProgram.build(outputs=(effect,)),
            data_input_name="value",
            state_type=state,
        ),
        input_schema=_schema(2),
        operands={
            "right": torch.tensor([0.25, -0.5, 1.0]),
            "rate": torch.tensor(0.2),
            "execution-count": torch.tensor(execution_count),
        },
        trainable_operands=("execution-count",),
    )


def _coupled_action(kind: str) -> mechanisms.BankLocalFormulaEffectAction:
    value = mechanisms.InputBinding("value", _type())
    state = _state_type(3)
    factor_type = mechanisms.TensorType(
        ("D", "R"),
        (3, 2),
        dtype="float32",
        domain="activation",
    )
    bias = mechanisms.reduce_sum(value, axis="B")
    rate = _binding("rate", _state_type())
    operands: dict[str, torch.Tensor]
    if kind == "transport":
        output_factor = _binding("output-factor", factor_type)
        input_factor = _binding("input-factor", factor_type)
        effect = mechanisms.neural_plasticity_transport(
            value,
            bias,
            output_factor,
            input_factor,
            rate,
            state_axis="D",
        )
        operands = {
            "output-factor": torch.tensor([[1.0, 0.0], [0.5, 1.0], [0.0, -0.5]]),
            "input-factor": torch.tensor([[0.0, 1.0], [1.0, 0.5], [-0.5, 0.0]]),
            "rate": torch.tensor(0.2),
        }
    elif kind == "polynomial":
        output_factor = _binding("output-factor", factor_type)
        left_factor = _binding("left-factor", factor_type)
        right_factor = _binding("right-factor", factor_type)
        effect = mechanisms.neural_plasticity_polynomial(
            value,
            bias,
            output_factor,
            left_factor,
            right_factor,
            rate,
            state_axis="D",
        )
        operands = {
            "output-factor": torch.tensor([[1.0, 0.5], [0.0, 1.0], [-0.5, 0.25]]),
            "left-factor": torch.tensor([[1.0, 0.0], [0.5, 1.0], [0.0, -0.5]]),
            "right-factor": torch.tensor([[0.0, 1.0], [1.0, 0.5], [-0.5, 0.0]]),
            "rate": torch.tensor(0.15),
        }
    elif kind == "proximal":
        strength = _binding("raw-strength", state)
        effect = mechanisms.neural_plasticity_proximal(value, bias, strength)
        operands = {"raw-strength": torch.tensor([-1.0, -0.5, 0.25])}
    else:
        raise AssertionError(kind)
    return mechanisms.BankLocalFormulaEffectAction(
        kind,
        mechanisms.FormulaEffectProgramV2(
            mechanisms.FormulaProgram.build(outputs=(effect,)),
            data_input_name="value",
            state_type=state,
        ),
        input_schema=_schema(),
        operands=operands,
        trainable_operands=tuple(operands),
    )


@pytest.mark.parametrize(
    "factory,state,value",
    [
        (_simple_action, torch.tensor([0.25, -0.5, 2.0]), torch.randn(2, 3)),
        (_blend_action, torch.tensor([1.0, -1.0, 0.5]), torch.randn(2, 3)),
        (_outer_action, torch.zeros(2, 3), torch.randn(2, 2)),
        (lambda: _coupled_action("transport"), torch.tensor([0.75, -1.25, 0.5]), torch.randn(2, 3)),
        (lambda: _coupled_action("polynomial"), torch.tensor([0.75, -1.25, 0.5]), torch.randn(2, 3)),
        (lambda: _coupled_action("proximal"), torch.tensor([0.75, -1.25, 0.5]), torch.randn(2, 3)),
    ],
)
def test_effect_atoms_preserve_data_and_differentiate_successor(
    factory,
    state: torch.Tensor,
    value: torch.Tensor,
) -> None:
    action = factory()
    value = value.requires_grad_()
    state = state.requires_grad_()

    result = action._execute_against(value, state, previous_revision=4)

    assert result.value is value
    assert result.successor.shape == state.shape
    assert result.previous_revision == 4
    assert result.successor_revision == 5
    assert torch.isfinite(result.successor).all()
    result.successor.square().sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert state.grad is not None and torch.isfinite(state.grad).all()
    assert all(parameter.grad is not None for parameter in action.parameters())


def test_additive_multiplicative_effect_matches_formula() -> None:
    action = _simple_action()
    value = torch.tensor([[1.0, 2.0, -1.0], [3.0, -2.0, 0.5]])
    predecessor = torch.tensor([0.25, -0.5, 2.0])
    operands = action.operand_store.tensors()

    result = action._execute_against(value, predecessor, previous_revision=0)

    additive = (value * operands["writer"]).sum(dim=0)
    multiplicative = (value * operands["gain"]).sum(dim=0)
    torch.testing.assert_close(
        result.successor,
        predecessor + additive + predecessor * multiplicative,
    )


def test_outer_v2_uses_direct_trainable_execution_count_in_one_effect() -> None:
    action = _repeated_outer_action()
    value = torch.tensor([[1.0, -2.0], [0.5, 1.0]])
    predecessor = torch.zeros(2, 3)
    operands = action.operand_store.tensors()

    result = action._execute_against(value, predecessor, previous_revision=7)

    one_update = (
        operands["rate"]
        * value.sum(dim=0).unsqueeze(-1)
        * operands["right"].unsqueeze(0)
    )
    assert result.value is value
    assert result.previous_revision == 7
    assert result.successor_revision == 8
    torch.testing.assert_close(result.successor, predecessor + 3.0 * one_update)
    result.successor.square().sum().backward()
    count = action.operand_store.tensor("execution-count")
    assert count.grad is not None
    assert torch.isfinite(count.grad)
    assert count.grad.abs() > 0
    assert (
        action.effect_program.effect_instruction.atom_ref
        == "arti/formula-atom-neural-plasticity-outer@2"
    )
    assert dict(action.effect_program.effect_instruction.attributes) == {
        "max_executions": 8
    }


def test_generic_execution_count_repeats_nonlinear_effect_on_latest_state() -> None:
    state_type = _state_type(2)
    state = torch.tensor([0.0, 2.0])
    target = torch.tensor([4.0, -2.0])
    amount = torch.tensor([0.5, 0.25])
    count = torch.tensor(2.0, requires_grad=True)
    effect = mechanisms.NeuralPlasticityEffectV2(
        "repeat-blend",
        mechanisms.NEURAL_PLASTICITY_BLEND_ATOM_REF,
        (target, amount),
    )

    result = apply_neural_plasticity_effect(
        effect,
        state,
        state_type=state_type,
        execution_count=count,
        max_executions=4,
    )
    once = state + amount * (target - state)
    expected = once + amount * (target - once)

    torch.testing.assert_close(result, expected)
    result.square().sum().backward()
    assert count.grad is not None
    assert bool(torch.isfinite(count.grad))
    assert float(count.grad.abs()) > 0.0


def test_outer_v2_clamps_execution_count_to_declared_atom_budget() -> None:
    action = _repeated_outer_action(execution_count=50.0)
    value = torch.tensor([[1.0, -2.0], [0.5, 1.0]])
    predecessor = torch.zeros(2, 3)
    operands = action.operand_store.tensors()

    result = action._execute_against(value, predecessor, previous_revision=0)

    one_update = (
        operands["rate"]
        * value.sum(dim=0).unsqueeze(-1)
        * operands["right"].unsqueeze(0)
    )
    torch.testing.assert_close(result.successor, predecessor + 8.0 * one_update)


@pytest.mark.parametrize("count_value", (0.0, 1.0))
@pytest.mark.parametrize("maximum", (2, 4))
def test_generic_count_excludes_unselected_overflow(
    count_value: float, maximum: int,
) -> None:
    state = torch.tensor([1.0], requires_grad=True)
    additive = torch.tensor([0.0], requires_grad=True)
    multiplier = torch.tensor([1e20], requires_grad=True)
    count = torch.tensor(count_value, requires_grad=True)
    effect = mechanisms.NeuralPlasticityEffectV2(
        "overflow-after-first", mechanisms.NEURAL_PLASTICITY_ATOM_REF,
        (additive, multiplier),
    )
    result = apply_neural_plasticity_effect(
        effect, state, state_type=_state_type(1),
        execution_count=count, max_executions=maximum,
    )
    with torch.no_grad():
        inference = apply_neural_plasticity_effect(
            effect, state, state_type=_state_type(1),
            execution_count=count, max_executions=maximum,
        )
    assert torch.equal(result, inference)
    assert torch.isfinite(result).all()
    fixed = apply_neural_plasticity_effect(
        effect, state, state_type=_state_type(1),
        execution_count=count.detach(), max_executions=maximum,
    )
    actual = torch.autograd.grad(result.sum(), (state, additive, multiplier, count), allow_unused=True)
    expected = torch.autograd.grad(fixed.sum(), (state, additive, multiplier), allow_unused=True)
    for gradient, hard_gradient in zip(actual[:3], expected, strict=True):
        if hard_gradient is None:
            assert gradient is None
        else:
            assert gradient is not None and torch.isfinite(gradient).all()
            torch.testing.assert_close(gradient, hard_gradient)
    assert actual[3] is not None and torch.isfinite(actual[3])
    assert actual[3].abs() > 0


def test_generic_count_zero_has_no_credit_past_invalid_first_trial() -> None:
    state = torch.tensor([3e38], requires_grad=True)
    count = torch.tensor(0.0, requires_grad=True)
    effect = mechanisms.NeuralPlasticityEffectV2(
        "invalid-first-trial", mechanisms.NEURAL_PLASTICITY_ATOM_REF,
        (torch.zeros(1), torch.tensor([2.0], requires_grad=True)),
    )
    result = apply_neural_plasticity_effect(
        effect, state, state_type=_state_type(1), execution_count=count, max_executions=4,
    )
    assert torch.equal(result, state)
    state_gradient, count_gradient, operand_gradient = torch.autograd.grad(
        result.sum(), (state, count, effect.operands[1]), allow_unused=True,
    )
    torch.testing.assert_close(state_gradient, torch.ones_like(state))
    torch.testing.assert_close(count_gradient, torch.zeros_like(count))
    assert operand_gradient is None


def test_generic_count_does_not_clip_selected_overflow() -> None:
    state = torch.ones(1)
    count = torch.tensor(2.0, requires_grad=True)
    effect = mechanisms.NeuralPlasticityEffectV2(
        "selected-overflow", mechanisms.NEURAL_PLASTICITY_ATOM_REF,
        (torch.zeros(1), torch.tensor([1e20])),
    )
    result = apply_neural_plasticity_effect(
        effect, state, state_type=_state_type(1), execution_count=count, max_executions=4,
    )
    with torch.no_grad():
        inference = apply_neural_plasticity_effect(
            effect, state, state_type=_state_type(1), execution_count=count, max_executions=4,
        )
    assert torch.isinf(result).all()
    assert torch.equal(result, inference)


@pytest.mark.parametrize("count_value", (-1.0, 0.0, 0.49, 0.5, 0.51, 1.5, 2.0, 2.5, 4.0, 9.0))
def test_generic_count_preserves_finite_blend_straight_through_gradients(count_value: float) -> None:
    state = torch.tensor([0.0, 2.0], requires_grad=True)
    target = torch.tensor([4.0, -2.0], requires_grad=True)
    amount = torch.tensor([0.5, 0.25], requires_grad=True)
    count = torch.tensor(count_value, requires_grad=True)
    effect = mechanisms.NeuralPlasticityEffectV2(
        "finite-blend", mechanisms.NEURAL_PLASTICITY_BLEND_ATOM_REF, (target, amount),
    )
    result = apply_neural_plasticity_effect(
        effect, state, state_type=_state_type(2), execution_count=count, max_executions=4,
    )
    states = [state]
    for _ in range(4):
        states.append(states[-1] + amount * (target - states[-1]))
    bounded = count.clamp(0, 4)
    soft = torch.softmax(-4 * (torch.arange(5) - bounded).square(), dim=0)
    hard = torch.nn.functional.one_hot(bounded.detach().round().long(), num_classes=5)
    reference = torch.einsum("k,kd->d", soft + (hard - soft).detach(), torch.stack(states))
    with torch.no_grad():
        inference = apply_neural_plasticity_effect(
            effect, state, state_type=_state_type(2), execution_count=count, max_executions=4,
        )
    assert torch.equal(result, inference)
    parameters = (state, target, amount, count)
    actual = torch.autograd.grad(result.square().sum(), parameters, allow_unused=True)
    expected = torch.autograd.grad(reference.square().sum(), parameters)
    for parameter, gradient, previous_gradient in zip(parameters, actual, expected, strict=True):
        gradient = torch.zeros_like(parameter) if gradient is None else gradient
        assert torch.isfinite(gradient).all()
        torch.testing.assert_close(gradient, previous_gradient)


def test_generic_count_only_records_selected_repetitions_for_backward(monkeypatch) -> None:
    grad_modes = []
    original = apply_neural_plasticity_effect

    def record(effect, state, **kwargs):
        grad_modes.append(torch.is_grad_enabled())
        return original(effect, state, **kwargs)

    monkeypatch.setattr(formula_v3, "apply_neural_plasticity_effect", record)
    effect = mechanisms.NeuralPlasticityEffectV2(
        "count-work", mechanisms.NEURAL_PLASTICITY_BLEND_ATOM_REF,
        (torch.ones(1), torch.tensor([0.5])),
    )
    original(
        effect, torch.zeros(1), state_type=_state_type(1),
        execution_count=torch.tensor(2.0, requires_grad=True), max_executions=4,
    )
    assert grad_modes == [True, True, False, False]


@pytest.mark.parametrize(
    "program_type,fabric_type,intermediate",
    (
        (formula_v3.FormulaEffectProgram, formula_v3.FormulaFabricV3, False),
        (formula_v3.FormulaEffectProgramV2, formula_v3.FormulaFabricV4, False),
        (formula_v3.FormulaEffectProgramV3, formula_v3.FormulaFabricV5, True),
    ),
)
def test_effect_fabric_contract_describes_runtime_predecessor(program_type, fabric_type, intermediate):
    value = mechanisms.InputBinding("value", _type())
    current = mechanisms.add(value, value) if intermediate else value
    drive = mechanisms.reduce_sum(current, axis="B")
    effect = mechanisms.neural_plasticity(current, drive, drive)
    output = mechanisms.add(effect, value) if intermediate else effect
    program = program_type(
        mechanisms.FormulaProgram.build(outputs=(output,)),
        data_input_name="value", state_type=_state_type(3),
    )
    config = fabric_type(program).contract_config()
    assert config["target_binding"] == "runtime-predecessor-bank"
    assert config["state_access"] == "effect-operands-only"


def test_effect_has_no_owned_state_or_caller_target() -> None:
    action = _simple_action()
    config = action.contract_config()

    assert arti.component_ref(action) == "arti/bank-local-formula-effect-action@1"
    assert config["target_resolution"] == "dynamic-immediate-predecessor-bank-slot"
    assert config["data_lane"] == "identity"
    assert "self_state" not in dict(action.named_buffers())
    assert "state_revision" not in dict(action.named_buffers())
    assert "target" not in config
    with pytest.raises(RuntimeError, match="predecessor lineage"):
        action(torch.ones(1, 3))


@pytest.mark.parametrize(
    "factory",
    [
        _simple_action,
        _blend_action,
        _outer_action,
        _repeated_outer_action,
        lambda: _coupled_action("transport"),
        lambda: _coupled_action("polynomial"),
        lambda: _coupled_action("proximal"),
    ],
)
def test_effect_program_round_trip_preserves_atom(factory) -> None:
    action = factory()
    restored = mechanisms.FormulaEffectProgramV2.from_dict(
        action.effect_program.to_dict()
    )

    assert restored.fingerprint == action.effect_program.fingerprint
    assert restored.effect_instruction.atom_ref == action.effect_program.effect_instruction.atom_ref
    provenance = arti.component_provenance(action)
    assert arti.validate_component_provenance(provenance) == provenance


def test_effect_rejects_incompatible_predecessor_shape() -> None:
    action = _simple_action()

    with pytest.raises((ValueError, mechanisms.FormulaV2Error), match="predecessor"):
        action._execute_against(
            torch.ones(1, 3),
            torch.zeros(2, 3),
            previous_revision=0,
        )
