from __future__ import annotations

import pytest
import torch

import arti
from arti import mechanisms
from arti.bank_local_program import (
    BankLocalNeuralPlasticityAction,
    BankLocalNeuralPlasticityActionV2,
)


def _schema() -> mechanisms.TensorSchema:
    return mechanisms.TensorSchema(
        dtype="float32",
        device_class="any",
        dimensions=("B", 3),
        semantic_axes=("batch", "feature"),
        mask_semantics="none",
    )


def _effect_program() -> mechanisms.FormulaEffectProgram:
    data_type = mechanisms.TensorType(
        ("B", "D"),
        ("B", 3),
        dtype="float32",
        domain="activation",
    )
    state_type = mechanisms.TensorType(
        ("D",),
        (3,),
        dtype="float32",
        domain="activation",
    )
    value = mechanisms.InputBinding("value", data_type)
    writer = mechanisms.BankBinding(
        "writer",
        source_ref="arti/formula-operand-bank@1",
        partition_id="writer",
        value_type=state_type,
    )
    gain = mechanisms.BankBinding(
        "gain",
        source_ref="arti/formula-operand-bank@1",
        partition_id="gain",
        value_type=state_type,
    )
    additive = mechanisms.reduce_sum(
        mechanisms.scale(value, writer),
        axis="B",
    )
    multiplicative = mechanisms.reduce_sum(
        mechanisms.scale(value, gain),
        axis="B",
    )
    identity = mechanisms.neural_plasticity(value, additive, multiplicative)
    return mechanisms.FormulaEffectProgram(
        mechanisms.FormulaProgram.build(outputs=(identity,)),
        data_input_name="value",
        state_type=state_type,
    )


def _action(state: torch.Tensor | None = None) -> BankLocalNeuralPlasticityAction:
    return BankLocalNeuralPlasticityAction(
        "remember",
        _effect_program(),
        input_schema=_schema(),
        state=torch.tensor([0.25, -0.5, 2.0]) if state is None else state,
        operands={
            "writer": torch.tensor([0.1, 0.2, 0.3]),
            "gain": torch.tensor([-0.1, 0.05, 0.2]),
        },
    )


def _blend_action() -> BankLocalNeuralPlasticityActionV2:
    data_type = mechanisms.TensorType(
        ("B", "D"),
        ("B", 3),
        dtype="float32",
        domain="activation",
    )
    state_type = mechanisms.TensorType(
        ("D",),
        (3,),
        dtype="float32",
        domain="activation",
    )
    value = mechanisms.InputBinding("value", data_type)
    target_weight = mechanisms.BankBinding(
        "target-weight",
        source_ref="arti/formula-operand-bank@1",
        partition_id="blend-target",
        value_type=state_type,
    )
    amount_weight = mechanisms.BankBinding(
        "amount-weight",
        source_ref="arti/formula-operand-bank@1",
        partition_id="blend-amount",
        value_type=state_type,
    )
    target = mechanisms.reduce_sum(mechanisms.scale(value, target_weight), axis="B")
    amount = mechanisms.scalar_map(
        mechanisms.reduce_sum(mechanisms.scale(value, amount_weight), axis="B"),
        mode="sigmoid",
    )
    identity = mechanisms.neural_plasticity_blend(value, target, amount)
    program = mechanisms.FormulaEffectProgramV2(
        mechanisms.FormulaProgram.build(outputs=(identity,)),
        data_input_name="value",
        state_type=state_type,
    )
    return BankLocalNeuralPlasticityActionV2(
        "blend-memory",
        program,
        input_schema=_schema(),
        state=torch.tensor([1.0, -1.0, 0.5]),
        operands={
            "target-weight": torch.tensor([0.5, -0.25, 0.75]),
            "amount-weight": torch.tensor([0.1, 0.2, -0.3]),
        },
        trainable_operands=("target-weight", "amount-weight"),
    )


def _outer_action() -> BankLocalNeuralPlasticityActionV2:
    data_type = mechanisms.TensorType(
        ("B", "O"),
        ("B", 2),
        dtype="float32",
        domain="activation",
    )
    left_type = mechanisms.TensorType(
        ("O",),
        (2,),
        dtype="float32",
        domain="activation",
    )
    right_type = mechanisms.TensorType(
        ("I",),
        (3,),
        dtype="float32",
        domain="activation",
    )
    rate_type = mechanisms.TensorType((), (), dtype="float32", domain="activation")
    state_type = mechanisms.TensorType(
        ("O", "I"),
        (2, 3),
        dtype="float32",
        domain="activation",
    )
    value = mechanisms.InputBinding("value", data_type)
    right = mechanisms.BankBinding(
        "right",
        source_ref="arti/formula-operand-bank@1",
        partition_id="outer-right",
        value_type=right_type,
    )
    rate = mechanisms.BankBinding(
        "rate",
        source_ref="arti/formula-operand-bank@1",
        partition_id="outer-rate",
        value_type=rate_type,
    )
    left = mechanisms.reduce_sum(value, axis="B")
    assert left.value_type == left_type
    identity = mechanisms.neural_plasticity_outer(value, left, right, rate)
    program = mechanisms.FormulaEffectProgramV2(
        mechanisms.FormulaProgram.build(outputs=(identity,)),
        data_input_name="value",
        state_type=state_type,
    )
    return BankLocalNeuralPlasticityActionV2(
        "outer-memory",
        program,
        input_schema=mechanisms.TensorSchema(
            dtype="float32",
            device_class="any",
            dimensions=("B", 2),
            semantic_axes=("batch", "feature"),
            mask_semantics="none",
        ),
        state=torch.zeros(2, 3),
        operands={"right": torch.tensor([0.25, -0.5, 1.0]), "rate": torch.tensor(0.2)},
        trainable_operands=("right", "rate"),
    )


def _state_coupled_action(kind: str) -> BankLocalNeuralPlasticityActionV2:
    data_type = mechanisms.TensorType(
        ("B", "D"),
        ("B", 3),
        dtype="float32",
        domain="activation",
    )
    state_type = mechanisms.TensorType(
        ("D",),
        (3,),
        dtype="float32",
        domain="activation",
    )
    factor_type = mechanisms.TensorType(
        ("D", "R"),
        (3, 2),
        dtype="float32",
        domain="activation",
    )
    scalar_type = mechanisms.TensorType((), (), dtype="float32", domain="activation")
    value = mechanisms.InputBinding("value", data_type)
    bias = mechanisms.reduce_sum(value, axis="B")
    operands: dict[str, torch.Tensor]
    if kind == "transport":
        output_factor = mechanisms.BankBinding(
            "output-factor",
            source_ref="arti/formula-operand-bank@1",
            partition_id="transport-output",
            value_type=factor_type,
        )
        input_factor = mechanisms.BankBinding(
            "input-factor",
            source_ref="arti/formula-operand-bank@1",
            partition_id="transport-input",
            value_type=factor_type,
        )
        rate = mechanisms.BankBinding(
            "rate",
            source_ref="arti/formula-operand-bank@1",
            partition_id="transport-rate",
            value_type=scalar_type,
        )
        identity = mechanisms.neural_plasticity_transport(
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
        bindings = {
            name: mechanisms.BankBinding(
                name,
                source_ref="arti/formula-operand-bank@1",
                partition_id=f"polynomial-{name}",
                value_type=factor_type,
            )
            for name in ("output-factor", "left-factor", "right-factor")
        }
        rate = mechanisms.BankBinding(
            "rate",
            source_ref="arti/formula-operand-bank@1",
            partition_id="polynomial-rate",
            value_type=scalar_type,
        )
        identity = mechanisms.neural_plasticity_polynomial(
            value,
            bias,
            bindings["output-factor"],
            bindings["left-factor"],
            bindings["right-factor"],
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
        raw_strength = mechanisms.BankBinding(
            "raw-strength",
            source_ref="arti/formula-operand-bank@1",
            partition_id="proximal-strength",
            value_type=state_type,
        )
        identity = mechanisms.neural_plasticity_proximal(value, bias, raw_strength)
        operands = {"raw-strength": torch.tensor([-1.0, -0.5, 0.25])}
    else:
        raise AssertionError(f"unknown test effect {kind!r}")
    program = mechanisms.FormulaEffectProgramV2(
        mechanisms.FormulaProgram.build(outputs=(identity,)),
        data_input_name="value",
        state_type=state_type,
    )
    return BankLocalNeuralPlasticityActionV2(
        f"{kind}-memory",
        program,
        input_schema=_schema(),
        state=torch.tensor([0.75, -1.25, 0.5]),
        operands=operands,
        trainable_operands=tuple(operands),
    )


def test_effect_is_one_ssa_instruction_with_an_identity_data_lane() -> None:
    effect_program = _effect_program()
    action = _action()
    value = torch.tensor(
        [[1.0, 2.0, -1.0], [3.0, -2.0, 0.5]],
        requires_grad=True,
    )
    state = action.initial_state().requires_grad_()

    result = action._execute_with_state(value, state, return_trace=True)

    operands = action.operand_store.tensors()
    additive = (value * operands["writer"]).sum(dim=0)
    multiplicative = (value * operands["gain"]).sum(dim=0)
    expected = state + additive + state * multiplicative
    assert result.value is value
    torch.testing.assert_close(result.successor_state, expected)
    assert effect_program.effect_instruction.atom_ref == mechanisms.NEURAL_PLASTICITY_ATOM_REF
    assert effect_program.program.outputs == (effect_program.effect_instruction.output_slot,)

    result.successor_state.square().sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert state.grad is not None and torch.isfinite(state.grad).all()


def test_self_state_parameterizes_effect_without_becoming_a_formula_binding() -> None:
    action = _action()
    value = torch.tensor([[1.0, 2.0, 3.0]])
    left_state = torch.tensor([0.5, 0.5, 0.5])
    right_state = torch.tensor([1.5, 1.5, 1.5])

    left = action._execute_with_state(value, left_state)
    right = action._execute_with_state(value, right_state)

    assert not torch.equal(left.update, right.update)
    assert "self_state" not in action.effect_program.program.input_names
    assert "self_state" not in action.effect_program.program.bank_names


def test_same_site_state_but_different_data_produces_different_effects() -> None:
    action = _action()
    state = action.initial_state()

    left = action._execute_with_state(torch.ones(2, 3), state).successor_state
    right = action._execute_with_state(torch.full((2, 3), 2.0), state).successor_state

    assert not torch.equal(left, right)


def test_effect_execution_is_site_bound_and_caller_cannot_supply_target() -> None:
    effect_program = _effect_program()
    fabric = mechanisms.FormulaFabricV3(effect_program)

    with pytest.raises(RuntimeError, match="execution-site-bound"):
        fabric(
            inputs={"value": torch.ones(2, 3)},
            banks={},
        )
    with pytest.raises(ValueError, match="exactly once"):
        BankLocalNeuralPlasticityAction(
            "remember",
            effect_program,
            input_schema=_schema(),
            state=torch.ones(3),
            operands={
                "writer": torch.ones(3),
                "gain": torch.ones(3),
                "self_state": torch.zeros(3),
            },
        )


def test_formula_effect_program_round_trips_and_has_versioned_dependencies() -> None:
    effect_program = _effect_program()

    restored = mechanisms.FormulaEffectProgram.from_dict(effect_program.to_dict())

    assert restored == effect_program
    assert restored.fingerprint == effect_program.fingerprint
    assert mechanisms.NEURAL_PLASTICITY_ATOM_REF in restored.dependency_refs
    assert arti.component_ref(restored) == "arti/formula-effect-program@1"
    assert arti.component_ref(mechanisms.FormulaFabricV3(restored)) == "arti/formula-fabric@3"
    assert arti.component_ref(mechanisms.NeuralPlasticityAtom()) == (
        "arti/formula-atom-neural-plasticity@1"
    )


def test_formula_fabric_v3_provenance_does_not_register_a_v2_child() -> None:
    fabric = mechanisms.FormulaFabricV3(_effect_program())

    provenance = arti.component_provenance(fabric)
    references = {component["ref"] for component in provenance["components"]}

    assert "arti/formula-fabric@3" in references
    assert "arti/formula-fabric@2" not in references
    assert arti.validate_component_provenance(provenance) == provenance


def test_effect_rejects_explicit_self_state_binding() -> None:
    valid = _effect_program()
    bindings = {binding.name: binding for binding in valid.program.bindings}
    value = bindings["value"]
    self_state = mechanisms.BankBinding(
        "self_state",
        source_ref=mechanisms.NEURAL_PLASTICITY_ATOM_REF,
        partition_id="self",
        value_type=valid.state_type,
    )
    update = mechanisms.add(
        self_state,
        mechanisms.reduce_sum(value, axis="B"),
    )
    program = mechanisms.FormulaProgram.build(
        outputs=(mechanisms.neural_plasticity(value, update, update),)
    )

    with pytest.raises(mechanisms.FormulaProgramError, match="cannot appear"):
        mechanisms.FormulaEffectProgram(
            program,
            data_input_name="value",
            state_type=valid.state_type,
        )


def test_formula_effect_program_requires_data_conditioned_update() -> None:
    valid = _effect_program()
    bindings = {binding.name: binding for binding in valid.program.bindings}
    value = bindings["value"]
    writer = bindings["writer"]
    program = mechanisms.FormulaProgram.build(
        outputs=(mechanisms.neural_plasticity(value, writer, writer),)
    )

    with pytest.raises(mechanisms.FormulaProgramError, match="current data"):
        mechanisms.FormulaEffectProgram(
            program,
            data_input_name="value",
            state_type=valid.state_type,
        )


def test_effect_rejects_a_transformed_public_data_lane() -> None:
    valid = _effect_program()
    bindings = {binding.name: binding for binding in valid.program.bindings}
    value = bindings["value"]
    writer = bindings["writer"]
    gain = bindings["gain"]
    additive = mechanisms.reduce_sum(mechanisms.scale(value, writer), axis="B")
    multiplicative = mechanisms.reduce_sum(mechanisms.scale(value, gain), axis="B")
    transformed = mechanisms.add(value, value)
    program = mechanisms.FormulaProgram.build(
        outputs=(mechanisms.neural_plasticity(transformed, additive, multiplicative),)
    )

    with pytest.raises(mechanisms.FormulaProgramError, match="unmodified current data"):
        mechanisms.FormulaEffectProgram(
            program,
            data_input_name="value",
            state_type=valid.state_type,
        )


def test_pure_formula_executors_reject_effect_programs() -> None:
    program = _effect_program().program

    with pytest.raises(mechanisms.FormulaProgramError, match="cannot execute effect-bearing"):
        mechanisms.FormulaFabricV2(program)
    with pytest.raises(mechanisms.FormulaProgramError, match="cannot lower effect-bearing"):
        mechanisms.FormulaExecutionPlanV2(program)


def test_blend_effect_moves_owned_state_toward_a_data_conditioned_target() -> None:
    action = _blend_action()
    value = torch.tensor([[1.0, 2.0, -1.0], [0.5, -1.0, 2.0]], requires_grad=True)
    state = action.initial_state().requires_grad_()

    result = action._execute_with_state(value, state)

    operands = action.operand_store.tensors()
    target = (value * operands["target-weight"]).sum(dim=0)
    amount = torch.sigmoid((value * operands["amount-weight"]).sum(dim=0))
    expected = state + amount * (target - state)
    assert result.value is value
    torch.testing.assert_close(result.successor_state, expected)
    assert result.successor_state.shape == state.shape
    assert action.effect_program.effect_instruction.atom_ref == (
        mechanisms.NEURAL_PLASTICITY_BLEND_ATOM_REF
    )

    result.successor_state.square().sum().backward()
    assert value.grad is not None and torch.count_nonzero(value.grad) > 0
    assert state.grad is not None and torch.count_nonzero(state.grad) > 0
    assert action.operand_store.tensors()["target-weight"].grad is not None


def test_outer_effect_applies_a_rank_one_update_without_materializing_a_target() -> None:
    action = _outer_action()
    value = torch.tensor([[1.0, -2.0], [0.5, 3.0]], requires_grad=True)
    state = action.initial_state().requires_grad_()

    result = action._execute_with_state(value, state)

    operands = action.operand_store.tensors()
    expected_update = operands["rate"] * torch.outer(value.sum(dim=0), operands["right"])
    assert result.value is value
    torch.testing.assert_close(result.update, expected_update)
    torch.testing.assert_close(result.successor_state, state + expected_update)
    assert action.effect_program.effect_instruction.atom_ref == (
        mechanisms.NEURAL_PLASTICITY_OUTER_ATOM_REF
    )

    result.successor_state.square().sum().backward()
    assert value.grad is not None and torch.count_nonzero(value.grad) > 0
    assert action.operand_store.tensors()["right"].grad is not None
    assert action.operand_store.tensors()["rate"].grad is not None


def test_transport_effect_introduces_cross_coordinate_state_feedback() -> None:
    action = _state_coupled_action("transport")
    value = torch.tensor([[0.1, -0.2, 0.3]], requires_grad=True)
    state = action.initial_state().requires_grad_()
    operands = action.operand_store.tensors()

    result = action._execute_with_state(value, state)
    flat = state.reshape(-1)
    expected = state + value.sum(dim=0) + operands["rate"] * (
        operands["output-factor"]
        @ (operands["input-factor"].transpose(0, 1) @ flat)
    )
    torch.testing.assert_close(result.successor_state, expected)
    assert result.value is value

    jacobian = torch.autograd.functional.jacobian(
        lambda current: action._execute_with_state(value, current).successor_state,
        state,
    )
    off_diagonal = jacobian - torch.diag(torch.diagonal(jacobian))
    assert torch.count_nonzero(off_diagonal) > 0

    result.successor_state.square().sum().backward()
    assert value.grad is not None and torch.count_nonzero(value.grad) > 0
    assert operands["output-factor"].grad is not None
    assert operands["input-factor"].grad is not None


def test_transport_acts_on_one_named_axis_without_flattening_other_axes() -> None:
    state_type = mechanisms.TensorType(
        ("M", "D"),
        ("M", "D"),
        dtype="float32",
        domain="activation",
    )
    factor_type = mechanisms.TensorType(
        ("D", "R"),
        ("D", "R"),
        dtype="float32",
        domain="activation",
    )
    scalar_type = mechanisms.TensorType((), (), dtype="float32", domain="activation")
    value = mechanisms.InputBinding("value", state_type)
    output_factor = mechanisms.BankBinding(
        "output-factor",
        source_ref="arti/formula-operand-bank@1",
        partition_id="matrix-transport-output",
        value_type=factor_type,
    )
    input_factor = mechanisms.BankBinding(
        "input-factor",
        source_ref="arti/formula-operand-bank@1",
        partition_id="matrix-transport-input",
        value_type=factor_type,
    )
    rate = mechanisms.BankBinding(
        "rate",
        source_ref="arti/formula-operand-bank@1",
        partition_id="matrix-transport-rate",
        value_type=scalar_type,
    )
    effect = mechanisms.neural_plasticity_transport(
        value,
        value,
        output_factor,
        input_factor,
        rate,
        state_axis="D",
    )
    action = BankLocalNeuralPlasticityActionV2(
        "matrix-transport",
        mechanisms.FormulaEffectProgramV2(
            mechanisms.FormulaProgram.build(outputs=(effect,)),
            data_input_name="value",
            state_type=state_type,
        ),
        input_schema=mechanisms.TensorSchema(
            dtype="float32",
            device_class="any",
            dimensions=("M", "D"),
            semantic_axes=("item", "feature"),
            mask_semantics="none",
        ),
        state=torch.tensor([[0.5, -0.25, 0.75], [1.0, 0.5, -0.5]]),
        operands={
            "output-factor": torch.tensor([[1.0, 0.0], [0.5, 1.0], [0.0, -0.5]]),
            "input-factor": torch.tensor([[0.0, 1.0], [1.0, 0.5], [-0.5, 0.0]]),
            "rate": torch.tensor(0.2),
        },
    )
    data = torch.tensor([[0.1, 0.2, -0.1], [-0.2, 0.3, 0.4]])
    state = action.initial_state()
    operands = action.operand_store.tensors()

    result = action._execute_with_state(data, state)
    expected = state + data + operands["rate"] * (
        (state @ operands["input-factor"])
        @ operands["output-factor"].transpose(0, 1)
    )

    assert result.value is data
    assert result.successor_state.shape == (2, 3)
    torch.testing.assert_close(result.successor_state, expected)


def test_polynomial_effect_has_nonzero_state_curvature() -> None:
    action = _state_coupled_action("polynomial")
    value = torch.tensor([[0.05, -0.1, 0.2]], requires_grad=True)
    state = action.initial_state().requires_grad_()
    direction = torch.tensor([0.2, -0.1, 0.15])

    center = action._execute_with_state(value, state).successor_state
    plus = action._execute_with_state(value, state + direction).successor_state
    minus = action._execute_with_state(value, state - direction).successor_state
    second_difference = plus - 2.0 * center + minus
    assert torch.linalg.vector_norm(second_difference) > 1e-5

    center.square().sum().backward()
    assert value.grad is not None and torch.count_nonzero(value.grad) > 0
    assert action.operand_store.tensors()["left-factor"].grad is not None
    assert action.operand_store.tensors()["right-factor"].grad is not None


def test_proximal_effect_is_sparse_and_nonexpansive() -> None:
    action = _state_coupled_action("proximal")
    value = torch.tensor([[0.2, 0.8, -0.4]], requires_grad=True)
    state = action.initial_state().requires_grad_()
    shifted = state + torch.tensor([0.1, -0.05, 0.2])

    left = action._execute_with_state(value, state).successor_state
    right = action._execute_with_state(value, shifted).successor_state
    threshold = torch.nn.functional.softplus(
        action.operand_store.tensors()["raw-strength"]
    )
    candidate = state + value.sum(dim=0)
    expected = torch.sign(candidate) * torch.relu(torch.abs(candidate) - threshold)
    torch.testing.assert_close(left, expected)
    assert torch.count_nonzero(left) < left.numel()
    assert torch.linalg.vector_norm(right - left) <= (
        torch.linalg.vector_norm(shifted - state) + 1e-6
    )

    left.square().sum().backward()
    assert value.grad is not None and torch.count_nonzero(value.grad) > 0
    assert action.operand_store.tensors()["raw-strength"].grad is not None


@pytest.mark.parametrize(
    "factory",
    [
        _blend_action,
        _outer_action,
        lambda: _state_coupled_action("transport"),
        lambda: _state_coupled_action("polynomial"),
        lambda: _state_coupled_action("proximal"),
    ],
)
def test_formula_effect_program_v2_round_trips_with_its_effect_atom(factory) -> None:
    action = factory()
    program = action.effect_program

    restored = mechanisms.FormulaEffectProgramV2.from_dict(program.to_dict())
    fabric = mechanisms.FormulaFabricV4(restored)
    provenance = arti.component_provenance(fabric)
    root = next(component for component in provenance["components"] if component["path"] == "$")

    assert restored == program
    assert restored.fingerprint == program.fingerprint
    assert arti.component_ref(restored) == "arti/formula-effect-program@2"
    assert arti.component_ref(fabric) == "arti/formula-fabric@4"
    assert restored.effect_instruction.atom_ref in root["dependencies"]
    assert "arti/formula-effect-program@2" in root["dependencies"]
    assert arti.validate_component_provenance(provenance) == provenance
