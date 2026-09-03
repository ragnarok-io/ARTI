from __future__ import annotations

import torch

import arti
from arti import mechanisms


def _schema() -> mechanisms.TensorSchema:
    return mechanisms.TensorSchema(
        dtype="float32",
        device_class="any",
        dimensions=("B", 3),
        semantic_axes=("batch", "feature"),
        mask_semantics="none",
    )


def _type() -> mechanisms.TensorType:
    return mechanisms.TensorType(
        ("B", "D"),
        ("B", 3),
        dtype="float32",
        domain="activation",
    )


def _state_type() -> mechanisms.TensorType:
    return mechanisms.TensorType(
        ("D",),
        (3,),
        dtype="float32",
        domain="activation",
    )


def _operand(name: str) -> mechanisms.BankBinding:
    return mechanisms.BankBinding(
        name,
        source_ref="arti/formula-operand-bank@1",
        partition_id=name,
        value_type=_state_type(),
    )


def _action() -> mechanisms.BankLocalNeuralPlasticityActionV3:
    value = mechanisms.InputBinding("value", _type())
    pre_weight = _operand("pre-weight")
    writer = _operand("writer")
    gain = _operand("gain")
    middle_weight = _operand("middle-weight")
    target_weight = _operand("target-weight")
    amount_weight = _operand("amount-weight")
    post_weight = _operand("post-weight")

    pre = mechanisms.scale(value, pre_weight)
    additive = mechanisms.reduce_sum(mechanisms.scale(pre, writer), axis="B")
    multiplicative = mechanisms.reduce_sum(mechanisms.scale(pre, gain), axis="B")
    first_effect = mechanisms.neural_plasticity(pre, additive, multiplicative)
    middle = mechanisms.scale(first_effect, middle_weight)
    target = mechanisms.reduce_sum(
        mechanisms.scale(middle, target_weight),
        axis="B",
    )
    amount = mechanisms.scalar_map(
        mechanisms.reduce_sum(
            mechanisms.scale(middle, amount_weight),
            axis="B",
        ),
        mode="sigmoid",
    )
    second_effect = mechanisms.neural_plasticity_blend(middle, target, amount)
    output = mechanisms.scale(second_effect, post_weight)
    effect_program = mechanisms.FormulaEffectProgramV3(
        mechanisms.FormulaProgram.build(outputs=(output,)),
        data_input_name="value",
        state_type=_state_type(),
    )
    return mechanisms.BankLocalNeuralPlasticityActionV3(
        "adapt-in-path",
        effect_program,
        input_schema=_schema(),
        output_schema=_schema(),
        state=torch.zeros(3),
        operands={
            "pre-weight": torch.tensor([2.0, 1.0, 0.5]),
            "writer": torch.tensor([0.1, 0.2, 0.3]),
            "gain": torch.zeros(3),
            "middle-weight": torch.tensor([0.5, 2.0, 1.0]),
            "target-weight": torch.tensor([1.0, 0.5, -1.0]),
            "amount-weight": torch.zeros(3),
            "post-weight": torch.tensor([-1.0, 0.25, 2.0]),
        },
        trainable_operands=("writer", "target-weight"),
    )


def _layout() -> mechanisms.TensorViewLayoutTransition:
    return mechanisms.TensorViewLayoutTransition(
        ("batch", "feature"),
        ("batch", "feature"),
        index_transition="identity",
    )


def test_multiple_self_effects_are_intermediate_forward_nodes() -> None:
    action = _action()
    value = torch.tensor([[1.0, 2.0, 3.0]])

    result = action._execute_with_state(value, action.initial_state(), return_trace=True)

    torch.testing.assert_close(result.value, torch.tensor([[-1.0, 1.0, 3.0]]))
    torch.testing.assert_close(
        result.successor_state,
        torch.tensor([0.6, 1.2, -0.525]),
    )
    assert result.previous_revision == 0
    assert result.successor_revision == 1
    assert len(action.effect_program.effect_instructions) == 2
    assert action.contract_config()["effect_count"] == 2
    assert arti.component_ref(action.effect_program) == "arti/formula-effect-program@3"
    assert arti.component_ref(action.fabric) == "arti/formula-fabric@5"
    assert arti.component_ref(action) == "arti/bank-local-neural-plasticity-action@3"


def test_updated_state_is_available_to_downstream_ordinary_formula() -> None:
    action = _action()
    effect_view = mechanisms.TensorViewNeuralPlasticityActionV3(
        action,
        layout=_layout(),
    )
    path_state = mechanisms.BankBinding(
        "path-state",
        source_ref="arti/formula-operand-bank@1",
        partition_id="path-state",
        value_type=_state_type(),
    )
    value = mechanisms.InputBinding("value", _type())
    reader = mechanisms.TensorViewFormulaAction(
        mechanisms.BankLocalFormulaAction(
            "read-adapted-state",
            mechanisms.FormulaProgram.build(
                outputs=(mechanisms.scale(value, path_state),),
            ),
            input_schema=_schema(),
            output_schema=_schema(),
            operands={"path-state": torch.zeros(3)},
        ),
        layout=_layout(),
        state_operands={"path-state": action},
    )
    view = mechanisms.TensorView.from_tensor(
        torch.tensor([[1.0, 2.0, 3.0]]),
        axis_names=("batch", "feature"),
        axis_roles=("batch", "feature"),
    )

    adapted = effect_view._execute(view, ())
    output = reader._execute(adapted.view, adapted.state_overlays)

    torch.testing.assert_close(
        output.view.value,
        torch.tensor([[-0.6, 1.2, -1.575]]),
    )
    assert len(adapted.state_overlays) == 1
    assert adapted.successor_revision == 1
    assert arti.component_ref(effect_view) == "arti/tensor-view-neural-plasticity-action@3"


def test_effect_program_v3_round_trip_preserves_ordered_chain() -> None:
    action = _action()
    program = action.effect_program
    restored = mechanisms.FormulaEffectProgramV3.from_dict(program.to_dict())

    assert restored.fingerprint == program.fingerprint
    assert [item.instruction_id for item in restored.effect_instructions] == [
        item.instruction_id for item in program.effect_instructions
    ]
    for component in (action.fabric, action):
        provenance = arti.component_provenance(component)
        assert arti.validate_component_provenance(provenance) == provenance
