from __future__ import annotations

import json

import pytest
import torch

import arti
from arti import mechanisms


def _schema(width: int) -> mechanisms.TensorSchema:
    return mechanisms.TensorSchema(
        dtype="float32",
        device_class="any",
        dimensions=("B", width),
        semantic_axes=("batch", "feature"),
        mask_semantics="none",
    )


def _pattern() -> mechanisms.TensorViewPattern:
    return mechanisms.TensorViewPattern(min_rank=2, max_rank=2)


def _terminal_abi(width: int = 3) -> mechanisms.TerminalOutputABI:
    return mechanisms.TerminalOutputABI(
        fields=(
            mechanisms.TerminalField(
                "value",
                _schema(width),
                "terminal-value",
            ),
            mechanisms.TerminalField(
                "validity",
                mechanisms.TensorSchema(
                    dtype="boolean",
                    device_class="any",
                    dimensions=("B",),
                    semantic_axes=("batch",),
                    mask_semantics="boolean-validity",
                ),
                "terminal-validity",
            ),
            mechanisms.TerminalField(
                "score",
                mechanisms.TensorSchema(
                    dtype="float32",
                    device_class="any",
                    dimensions=("B",),
                    semantic_axes=("batch",),
                    mask_semantics="none",
                ),
                "terminal-score",
            ),
        ),
        factor_order=(),
        validity_contract="one validity value per row",
        packing_contract="named terminal tensors",
        score_contract="one route score per row",
        consumer_contract="hard one winner",
        gradient_contract=mechanisms.GradientContract.autograd(),
    )


def _query(*member_ids: str) -> mechanisms.SealedTensorViewBankQuery:
    query = mechanisms.TensorViewBankQuery(
        pattern=_pattern(),
        observer=mechanisms.CoordinateTensorViewObserver(
            max_rank=2,
            query_dim=4,
            hidden_dim=8,
            max_observations=16,
        ),
        matcher=mechanisms.BankMemberMatcher(
            torch.randn(len(member_ids), 4),
            member_ids=member_ids,
        ),
    )
    return mechanisms.seal_tensor_view_bank_query(query)


def _identity_layout() -> mechanisms.TensorViewLayoutTransition:
    return mechanisms.TensorViewLayoutTransition(
        ("batch", "feature"),
        ("batch", "feature"),
        index_transition="identity",
    )


def _effect_action(
    action_id: str,
    child_bank_id: str | None,
    *,
    result_kind: str = "descend",
) -> mechanisms.TensorViewNeuralPlasticityAction:
    value_type = mechanisms.TensorType(
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
    value = mechanisms.InputBinding("value", value_type)
    writer = mechanisms.BankBinding(
        "writer",
        source_ref="arti/formula-operand-bank@1",
        partition_id=f"{action_id}-writer",
        value_type=state_type,
    )
    gain = mechanisms.BankBinding(
        "gain",
        source_ref="arti/formula-operand-bank@1",
        partition_id=f"{action_id}-gain",
        value_type=state_type,
    )
    additive = mechanisms.reduce_sum(mechanisms.scale(value, writer), axis="B")
    multiplicative = mechanisms.reduce_sum(mechanisms.scale(value, gain), axis="B")
    effect = mechanisms.neural_plasticity(value, additive, multiplicative)
    effect_program = mechanisms.FormulaEffectProgram(
        mechanisms.FormulaProgram.build(outputs=(effect,)),
        data_input_name="value",
        state_type=state_type,
    )
    action = mechanisms.BankLocalNeuralPlasticityAction(
        action_id,
        effect_program,
        input_schema=_schema(3),
        state=torch.zeros(3),
        operands={"writer": torch.ones(3), "gain": torch.ones(3)},
        result_kind=result_kind,
        next_bank_id=child_bank_id,
        trainable_operands=("writer",),
    )
    return mechanisms.TensorViewNeuralPlasticityAction(
        action,
        layout=_identity_layout(),
    )


def _blend_effect_action(
    action_id: str,
    child_bank_id: str,
) -> mechanisms.TensorViewNeuralPlasticityActionV2:
    value_type = mechanisms.TensorType(
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
    value = mechanisms.InputBinding("value", value_type)
    amount_weight = mechanisms.BankBinding(
        "amount-weight",
        source_ref="arti/formula-operand-bank@1",
        partition_id=f"{action_id}-amount",
        value_type=state_type,
    )
    target = mechanisms.reduce_sum(value, axis="B")
    amount = mechanisms.scalar_map(
        mechanisms.reduce_sum(mechanisms.scale(value, amount_weight), axis="B"),
        mode="sigmoid",
    )
    effect = mechanisms.neural_plasticity_blend(value, target, amount)
    program = mechanisms.FormulaEffectProgramV2(
        mechanisms.FormulaProgram.build(outputs=(effect,)),
        data_input_name="value",
        state_type=state_type,
    )
    action = mechanisms.BankLocalNeuralPlasticityActionV2(
        action_id,
        program,
        input_schema=_schema(3),
        state=torch.zeros(3),
        operands={"amount-weight": torch.tensor([0.2, -0.1, 0.3])},
        result_kind="descend",
        next_bank_id=child_bank_id,
        trainable_operands=("amount-weight",),
    )
    return mechanisms.TensorViewNeuralPlasticityActionV2(
        action,
        layout=_identity_layout(),
    )


def _outer_effect_action(
    action_id: str,
    child_bank_id: str,
) -> mechanisms.TensorViewNeuralPlasticityActionV2:
    value_type = mechanisms.TensorType(
        ("B", "D"),
        ("B", 3),
        dtype="float32",
        domain="activation",
    )
    vector_type = mechanisms.TensorType(
        ("D",),
        (3,),
        dtype="float32",
        domain="activation",
    )
    output_type = mechanisms.TensorType(
        ("O",),
        (3,),
        dtype="float32",
        domain="activation",
    )
    rate_type = mechanisms.TensorType((), (), dtype="float32", domain="activation")
    state_type = mechanisms.TensorType(
        ("D", "O"),
        (3, 3),
        dtype="float32",
        domain="activation",
    )
    value = mechanisms.InputBinding("value", value_type)
    right = mechanisms.BankBinding(
        "right",
        source_ref="arti/formula-operand-bank@1",
        partition_id=f"{action_id}-right",
        value_type=output_type,
    )
    rate = mechanisms.BankBinding(
        "rate",
        source_ref="arti/formula-operand-bank@1",
        partition_id=f"{action_id}-rate",
        value_type=rate_type,
    )
    left = mechanisms.reduce_sum(value, axis="B")
    assert left.value_type == vector_type
    effect = mechanisms.neural_plasticity_outer(value, left, right, rate)
    program = mechanisms.FormulaEffectProgramV2(
        mechanisms.FormulaProgram.build(outputs=(effect,)),
        data_input_name="value",
        state_type=state_type,
    )
    action = mechanisms.BankLocalNeuralPlasticityActionV2(
        action_id,
        program,
        input_schema=_schema(3),
        state=torch.zeros(3, 3),
        operands={"right": torch.tensor([0.5, -0.25, 0.75]), "rate": torch.tensor(0.2)},
        result_kind="descend",
        next_bank_id=child_bank_id,
        trainable_operands=("right", "rate"),
    )
    return mechanisms.TensorViewNeuralPlasticityActionV2(
        action,
        layout=_identity_layout(),
    )


def _state_coupled_effect_action(
    action_id: str,
    child_bank_id: str,
    kind: str,
) -> mechanisms.TensorViewNeuralPlasticityActionV2:
    value_type = mechanisms.TensorType(
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
    value = mechanisms.InputBinding("value", value_type)
    bias = mechanisms.reduce_sum(value, axis="B")
    operands: dict[str, torch.Tensor]
    if kind in {"transport", "polynomial"}:
        factor_names = (
            ("output-factor", "input-factor")
            if kind == "transport"
            else ("output-factor", "left-factor", "right-factor")
        )
        factors = {
            name: mechanisms.BankBinding(
                name,
                source_ref="arti/formula-operand-bank@1",
                partition_id=f"{action_id}-{name}",
                value_type=factor_type,
            )
            for name in factor_names
        }
        rate = mechanisms.BankBinding(
            "rate",
            source_ref="arti/formula-operand-bank@1",
            partition_id=f"{action_id}-rate",
            value_type=scalar_type,
        )
        if kind == "transport":
            effect = mechanisms.neural_plasticity_transport(
                value,
                bias,
                factors["output-factor"],
                factors["input-factor"],
                rate,
                state_axis="D",
            )
        else:
            effect = mechanisms.neural_plasticity_polynomial(
                value,
                bias,
                factors["output-factor"],
                factors["left-factor"],
                factors["right-factor"],
                rate,
                state_axis="D",
            )
        operands = {
            name: torch.tensor([[1.0, 0.25], [-0.5, 0.75], [0.4, -0.2]])
            for name in factor_names
        }
        operands["rate"] = torch.tensor(0.1)
    elif kind == "proximal":
        raw_strength = mechanisms.BankBinding(
            "raw-strength",
            source_ref="arti/formula-operand-bank@1",
            partition_id=f"{action_id}-raw-strength",
            value_type=state_type,
        )
        effect = mechanisms.neural_plasticity_proximal(value, bias, raw_strength)
        operands = {"raw-strength": torch.full((3,), -2.0)}
    else:
        raise AssertionError(f"unknown test effect {kind!r}")
    program = mechanisms.FormulaEffectProgramV2(
        mechanisms.FormulaProgram.build(outputs=(effect,)),
        data_input_name="value",
        state_type=state_type,
    )
    action = mechanisms.BankLocalNeuralPlasticityActionV2(
        action_id,
        program,
        input_schema=_schema(3),
        state=torch.tensor([0.5, -0.25, 0.75]),
        operands=operands,
        result_kind="descend",
        next_bank_id=child_bank_id,
        trainable_operands=tuple(operands),
    )
    return mechanisms.TensorViewNeuralPlasticityActionV2(
        action,
        layout=_identity_layout(),
    )


def _transport_effect_action(
    action_id: str,
    child_bank_id: str,
) -> mechanisms.TensorViewNeuralPlasticityActionV2:
    return _state_coupled_effect_action(action_id, child_bank_id, "transport")


def _polynomial_effect_action(
    action_id: str,
    child_bank_id: str,
) -> mechanisms.TensorViewNeuralPlasticityActionV2:
    return _state_coupled_effect_action(action_id, child_bank_id, "polynomial")


def _proximal_effect_action(
    action_id: str,
    child_bank_id: str,
) -> mechanisms.TensorViewNeuralPlasticityActionV2:
    return _state_coupled_effect_action(action_id, child_bank_id, "proximal")


def _state_reader_action(
    action_id: str,
    source: mechanisms.TensorViewNeuralPlasticityAction,
) -> mechanisms.TensorViewFormulaAction:
    value_type = mechanisms.TensorType(
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
    value = mechanisms.InputBinding("value", value_type)
    path_state = mechanisms.BankBinding(
        "path_state",
        source_ref="arti/formula-operand-bank@1",
        partition_id=f"{action_id}-path-state",
        value_type=state_type,
    )
    action = mechanisms.BankLocalFormulaAction(
        action_id,
        mechanisms.FormulaProgram.build(
            outputs=(mechanisms.scale(value, path_state),),
        ),
        input_schema=_schema(3),
        output_schema=_schema(3),
        operands={"path_state": torch.zeros(3)},
    )
    return mechanisms.TensorViewFormulaAction(
        action,
        layout=_identity_layout(),
        state_operands={"path_state": source.action},
    )


def _outer_state_reader_action(
    action_id: str,
    source: mechanisms.TensorViewNeuralPlasticityActionV2,
) -> mechanisms.TensorViewFormulaAction:
    value_type = mechanisms.TensorType(
        ("B", "D"),
        ("B", 3),
        dtype="float32",
        domain="activation",
    )
    state_type = mechanisms.TensorType(
        ("D", "O"),
        (3, 3),
        dtype="float32",
        domain="activation",
    )
    value = mechanisms.InputBinding("value", value_type)
    path_state = mechanisms.BankBinding(
        "path_state",
        source_ref="arti/formula-operand-bank@1",
        partition_id=f"{action_id}-path-state",
        value_type=state_type,
    )
    projected = mechanisms.contract(
        value,
        path_state,
        reduce_axes=(("D", "D"),),
        output_axes=("B", "O"),
    )
    action = mechanisms.BankLocalFormulaAction(
        action_id,
        mechanisms.FormulaProgram.build(outputs=(projected,)),
        input_schema=_schema(3),
        output_schema=_schema(3),
        operands={"path_state": torch.zeros(3, 3)},
    )
    return mechanisms.TensorViewFormulaAction(
        action,
        layout=_identity_layout(),
        state_operands={"path_state": source.action},
    )


def _reader_bank(
    bank_id: str,
    source: mechanisms.TensorViewNeuralPlasticityAction,
    *,
    operand_source: mechanisms.TensorViewNeuralPlasticityAction | None = None,
) -> mechanisms.TensorViewFormulaProgram:
    state_source = source if operand_source is None else operand_source
    reader = _state_reader_action("read-state", state_source)
    return mechanisms.TensorViewFormulaProgram(
        bank_id=bank_id,
        query=_query("read-state", "exit"),
        actions=(reader,),
        terminal_action=mechanisms.BankLocalTerminalAction(
            "exit",
            input_schema=_schema(3),
        ),
        local_refine=mechanisms.BankLocalRefinePolicy(min_steps=2, max_steps=2),
        input_pattern=_pattern(),
        exit_pattern=_pattern(),
        terminal_abi=_terminal_abi(),
    )


def _outer_reader_bank(
    bank_id: str,
    source: mechanisms.TensorViewNeuralPlasticityActionV2,
) -> mechanisms.TensorViewFormulaProgram:
    reader = _outer_state_reader_action("read-state", source)
    return mechanisms.TensorViewFormulaProgram(
        bank_id=bank_id,
        query=_query("read-state", "exit"),
        actions=(reader,),
        terminal_action=mechanisms.BankLocalTerminalAction(
            "exit",
            input_schema=_schema(3),
        ),
        local_refine=mechanisms.BankLocalRefinePolicy(min_steps=2, max_steps=2),
        input_pattern=_pattern(),
        exit_pattern=_pattern(),
        terminal_abi=_terminal_abi(),
    )


def _root_bank(
    bank_id: str,
    effects: tuple[mechanisms.TensorViewNeuralPlasticityAction, ...],
) -> mechanisms.TensorViewFormulaProgram:
    return mechanisms.TensorViewFormulaProgram(
        bank_id=bank_id,
        query=_query(*(item.action_id for item in effects), "exit"),
        actions=effects,
        terminal_action=mechanisms.BankLocalTerminalAction(
            "exit",
            input_schema=_schema(4),
        ),
        local_refine=mechanisms.BankLocalRefinePolicy(min_steps=1, max_steps=1),
        input_pattern=_pattern(),
        exit_pattern=_pattern(),
        terminal_abi=_terminal_abi(),
    )


def _view(value: torch.Tensor) -> mechanisms.TensorView:
    return mechanisms.TensorView.from_tensor(
        value,
        axis_names=("batch", "feature"),
        axis_roles=("batch", "feature"),
    )


def test_neural_plasticity_action_rejects_direct_forward() -> None:
    effect = _effect_action("remember", "reader")
    state_before = effect.action.self_state.detach().clone()

    with pytest.raises(RuntimeError, match="FederalRecall"):
        effect(_view(torch.tensor([[1.0, 2.0, 3.0]])))

    torch.testing.assert_close(effect.action.self_state, state_before)
    assert int(effect.action.state_revision) == 0


def test_effect_trace_site_ref_is_qualified_by_owning_bank() -> None:
    left = _effect_action("remember", "reader-left")
    right = _effect_action("remember", "reader-right")
    runtime = mechanisms.FederalRecallV3(
        {
            "root-left": _root_bank("root-left", (left,)),
            "root-right": _root_bank("root-right", (right,)),
            "reader-left": _reader_bank("reader-left", left),
            "reader-right": _reader_bank("reader-right", right),
        },
        terminal_abi=_terminal_abi(),
        root_bank_ids=("root-left", "root-right"),
        max_levels=2,
        max_k=2,
    )
    value = _view(torch.tensor([[1.0, 2.0, 3.0]]))

    _left_output, left_trace = runtime(
        value,
        root_bank_id="root-left",
        return_trace=True,
    )
    _right_output, right_trace = runtime(
        value,
        root_bank_id="root-right",
        return_trace=True,
    )

    left_effect = next(
        local
        for step in left_trace.steps
        for local in step.local_refine
        if getattr(local, "effect_site_ref", None) is not None
    )
    right_effect = next(
        local
        for step in right_trace.steps
        for local in step.local_refine
        if getattr(local, "effect_site_ref", None) is not None
    )
    assert left_effect.effect_site_id == right_effect.effect_site_id == "remember"
    assert left_effect.effect_site_ref == "root-left/remember"
    assert right_effect.effect_site_ref == "root-right/remember"
    assert left_effect.effect_site_ref != right_effect.effect_site_ref


def test_k_wide_siblings_read_entry_overlay_until_next_dispatch() -> None:
    effect = _effect_action("remember", None, result_kind="continue")
    reader = _state_reader_action("read-state", effect)
    bank = mechanisms.TensorViewFormulaProgram(
        bank_id="root",
        query=_query("remember", "read-state", "exit"),
        actions=(effect, reader),
        terminal_action=mechanisms.BankLocalTerminalAction(
            "exit",
            input_schema=_schema(3),
        ),
        local_refine=mechanisms.BankLocalRefinePolicy(min_steps=1, max_steps=2),
        input_pattern=_pattern(),
        exit_pattern=_pattern(),
        terminal_abi=_terminal_abi(),
    )
    view = _view(torch.tensor([[1.0, 2.0, 3.0]]))
    observed = bank.query(view)
    sibling_query = type(observed)(
        torch.tensor([[2.0, 1.0, -100.0]]),
        observed.member_ids,
        observed.observation,
    )

    siblings = bank._execute_once(
        view,
        sibling_query,
        max_candidates=2,
        state_overlays=(),
    )
    by_id = {candidate.candidate_id: candidate for candidate in siblings}

    torch.testing.assert_close(by_id["remember"].next_view.value, view.value)
    torch.testing.assert_close(
        by_id["read-state"].next_view.value,
        torch.zeros_like(view.value),
    )
    assert by_id["read-state"].state_overlays == ()

    next_dispatch = bank._execute_once(
        view,
        sibling_query,
        max_candidates=2,
        state_overlays=by_id["remember"].state_overlays,
    )
    next_by_id = {candidate.candidate_id: candidate for candidate in next_dispatch}
    expected_state = view.value.sum(dim=0)
    torch.testing.assert_close(
        next_by_id["read-state"].next_view.value,
        view.value * expected_state,
    )


def test_effect_successor_is_visible_to_next_federation_dispatch_and_final_loss() -> None:
    effect = _effect_action("remember", "reader")
    root = _root_bank("root", (effect,))
    reader = _reader_bank("reader", effect)
    runtime = mechanisms.FederalRecallV3(
        {"root": root, "reader": reader},
        terminal_abi=_terminal_abi(),
        root_bank_ids=("root",),
        max_levels=2,
        max_k=2,
    )
    value = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    output, trace = runtime(_view(value), return_trace=True)

    expected_state = value.detach().sum(dim=0)
    torch.testing.assert_close(effect.action.self_state, expected_state)
    torch.testing.assert_close(output["value"], value * expected_state)
    effect_steps = [
        local
        for step in trace.steps
        for local in step.local_refine
        if getattr(local, "effect_site_id", None) is not None
    ]
    assert len(effect_steps) == 1
    assert effect_steps[0].effect_site_id == "remember"
    assert effect_steps[0].effect_state_change_norm is not None
    assert effect_steps[0].effect_previous_revision == 0
    assert effect_steps[0].effect_successor_revision == 1
    assert int(effect.action.state_revision) == 1

    output["value"].square().mean().backward()
    writer = effect.action.operand_store.tensors()["writer"]
    assert writer.grad is not None and torch.count_nonzero(writer.grad) > 0


@pytest.mark.parametrize(
    ("effect_factory", "reader_factory", "expected_atom_ref"),
    [
        (
            _blend_effect_action,
            _reader_bank,
            mechanisms.NEURAL_PLASTICITY_BLEND_ATOM_REF,
        ),
        (
            _outer_effect_action,
            _outer_reader_bank,
            mechanisms.NEURAL_PLASTICITY_OUTER_ATOM_REF,
        ),
        (
            _transport_effect_action,
            _reader_bank,
            mechanisms.NEURAL_PLASTICITY_TRANSPORT_ATOM_REF,
        ),
        (
            _polynomial_effect_action,
            _reader_bank,
            mechanisms.NEURAL_PLASTICITY_POLYNOMIAL_ATOM_REF,
        ),
        (
            _proximal_effect_action,
            _reader_bank,
            mechanisms.NEURAL_PLASTICITY_PROXIMAL_ATOM_REF,
        ),
    ],
)
def test_extensible_effect_algebra_updates_the_winning_federation_path(
    effect_factory,
    reader_factory,
    expected_atom_ref: str,
) -> None:
    torch.manual_seed(941)
    effect = effect_factory("remember", "reader")
    runtime = mechanisms.FederalRecallV3(
        {
            "root": _root_bank("root", (effect,)),
            "reader": reader_factory("reader", effect),
        },
        terminal_abi=_terminal_abi(),
        root_bank_ids=("root",),
        max_levels=2,
        max_k=2,
    )
    value = torch.tensor([[1.0, 2.0, -1.0]], requires_grad=True)

    output, trace = runtime(_view(value), return_trace=True)

    assert torch.count_nonzero(effect.action.self_state) > 0
    assert torch.count_nonzero(output["value"]) > 0
    assert len(trace.committed_effects) == 1
    assert trace.committed_effects[0].effect_atom_ref == expected_atom_ref
    effect_steps = [
        local
        for step in trace.steps
        for local in step.local_refine
        if getattr(local, "effect_atom_ref", None) is not None
    ]
    assert len(effect_steps) == 1
    assert effect_steps[0].effect_atom_ref == expected_atom_ref

    output["value"].square().mean().backward()
    assert value.grad is not None and torch.count_nonzero(value.grad) > 0
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in effect.action.operand_store.parameters()
    )


def test_effect_state_persists_and_changes_the_next_invocation() -> None:
    effect = _effect_action("remember", "reader")
    runtime = mechanisms.FederalRecallV3(
        {
            "root": _root_bank("root", (effect,)),
            "reader": _reader_bank("reader", effect),
        },
        terminal_abi=_terminal_abi(),
        root_bank_ids=("root",),
        max_levels=2,
        max_k=2,
    )
    value = torch.tensor([[1.0, 2.0, 3.0]])

    first = runtime(_view(value))["value"]
    second = runtime(_view(value))["value"]

    torch.testing.assert_close(second, 2.0 * first + value.pow(3))


def test_committed_network_state_round_trips_through_state_dict() -> None:
    torch.manual_seed(1701)
    effect = _effect_action("remember", "reader")
    runtime = mechanisms.FederalRecallV3(
        {
            "root": _root_bank("root", (effect,)),
            "reader": _reader_bank("reader", effect),
        },
        terminal_abi=_terminal_abi(),
        root_bank_ids=("root",),
        max_levels=2,
        max_k=2,
    )
    value = torch.tensor([[1.0, 2.0, 3.0]])
    runtime(_view(value))
    checkpoint = {name: tensor.detach().clone() for name, tensor in runtime.state_dict().items()}

    torch.manual_seed(1701)
    restored_effect_view = _effect_action("remember", "reader")
    restored = mechanisms.FederalRecallV3(
        {
            "root": _root_bank("root", (restored_effect_view,)),
            "reader": _reader_bank("reader", restored_effect_view),
        },
        terminal_abi=_terminal_abi(),
        root_bank_ids=("root",),
        max_levels=2,
        max_k=2,
    )
    restored.load_state_dict(checkpoint)

    expected = runtime(_view(value))["value"]
    actual = restored(_view(value))["value"]
    torch.testing.assert_close(actual, expected)


def test_federation_effect_state_has_a_valid_component_state_contract() -> None:
    effect = _effect_action("remember", "reader")
    runtime = mechanisms.FederalRecallV3(
        {
            "root": _root_bank("root", (effect,)),
            "reader": _reader_bank("reader", effect),
        },
        terminal_abi=_terminal_abi(),
        root_bank_ids=("root",),
        max_levels=2,
        max_k=2,
    )

    contract = arti.component_state_contract(runtime, runtime.state_dict(), scope="all")

    assert arti.validate_component_state_contract(
        contract,
        state_dict=runtime.state_dict(),
        model=runtime,
    ) == contract


def test_k_wide_losing_effect_path_does_not_commit() -> None:
    left = _effect_action("remember-left", "reader-left")
    right = _effect_action("remember-right", "reader-right")
    runtime = mechanisms.FederalRecallV3(
        {
            "root": _root_bank("root", (left, right)),
            "reader-left": _reader_bank("reader-left", left),
            "reader-right": _reader_bank("reader-right", right),
        },
        terminal_abi=_terminal_abi(),
        root_bank_ids=("root",),
        max_levels=2,
        max_k=2,
    )

    _output, trace = runtime(_view(torch.tensor([[1.0, 2.0, 3.0]])), return_trace=True)

    changed = {
        left.action_id: bool(torch.count_nonzero(left.action.self_state)),
        right.action_id: bool(torch.count_nonzero(right.action.self_state)),
    }
    assert sum(changed.values()) == 1
    winner_path = trace.winner_paths[0]
    assert next(action_id for action_id, was_changed in changed.items() if was_changed) in winner_path


def test_trace_reports_one_json_safe_commit_for_only_the_winner_effect() -> None:
    left = _effect_action("remember-left", "reader-left")
    right = _effect_action("remember-right", "reader-right")
    runtime = mechanisms.FederalRecallV3(
        {
            "root": _root_bank("root", (left, right)),
            "reader-left": _reader_bank("reader-left", left),
            "reader-right": _reader_bank("reader-right", right),
        },
        terminal_abi=_terminal_abi(),
        root_bank_ids=("root",),
        max_levels=2,
        max_k=2,
    )

    _output, trace = runtime(
        _view(torch.tensor([[1.0, 2.0, 3.0]])),
        return_trace=True,
    )

    payload = trace.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert len(trace.committed_effects) == 1
    assert len(payload["committed_effects"]) == 1

    receipt = trace.committed_effects[0]
    changed_ids = {
        effect.action_id
        for effect in (left, right)
        if int(effect.action.state_revision) == 1
    }
    assert changed_ids == {receipt.action_id}
    assert receipt.site_ref == f"root/{receipt.action_id}"
    assert receipt.winner_path == trace.winner_paths[0]
    assert receipt.previous_revision == 0
    assert receipt.successor_revision == 1
    assert payload["committed_effects"] == [receipt.to_dict()]


def test_wrong_site_control_reads_no_uncommitted_successor() -> None:
    correct = _effect_action("remember-correct", "reader")
    wrong = _effect_action("remember-wrong", "unused-reader")
    runtime = mechanisms.FederalRecallV3(
        {
            "root": _root_bank("root", (correct,)),
            "reader": _reader_bank("reader", correct, operand_source=wrong),
            "unused-root": _root_bank("unused-root", (wrong,)),
            "unused-reader": _reader_bank("unused-reader", wrong),
        },
        terminal_abi=_terminal_abi(),
        root_bank_ids=("root",),
        max_levels=2,
        max_k=2,
    )

    output = runtime(_view(torch.tensor([[1.0, 2.0, 3.0]])))

    torch.testing.assert_close(output["value"], torch.zeros(1, 3))
    assert torch.count_nonzero(correct.action.self_state) > 0
    assert torch.count_nonzero(wrong.action.self_state) == 0
