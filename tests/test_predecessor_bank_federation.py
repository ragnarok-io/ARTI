from __future__ import annotations

import json
from types import MethodType

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


def _type(width: int) -> mechanisms.TensorType:
    return mechanisms.TensorType(
        ("B", "D"),
        ("B", width),
        dtype="float32",
        domain="activation",
    )


def _pattern() -> mechanisms.TensorViewPattern:
    return mechanisms.TensorViewPattern(min_rank=2, max_rank=2)


def _layout() -> mechanisms.TensorViewLayoutTransition:
    return mechanisms.TensorViewLayoutTransition(
        ("batch", "feature"),
        ("batch", "feature"),
        index_transition="identity",
    )


def _terminal_abi() -> mechanisms.TerminalOutputABI:
    return mechanisms.TerminalOutputABI(
        fields=(
            mechanisms.TerminalField("value", _schema(3), "terminal-value"),
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


def _query(member_ids: tuple[str, ...], scores: dict[str, float]):
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
    sealed = mechanisms.seal_tensor_view_bank_query(query)

    def scripted(self, view):
        self.validate_runtime_state()
        observation = self.query.observer(view)
        values = torch.tensor(
            [[scores[name] for name in self.signature.member_ids]],
            dtype=observation.tokens.dtype,
            device=observation.tokens.device,
        )
        return mechanisms.TensorViewQueryResult(
            values,
            self.signature.member_ids,
            observation,
        )

    sealed.forward = MethodType(scripted, sealed)
    return sealed


def _producer(
    action_id: str,
    next_bank_id: str,
    *,
    partition_id: str,
    scalar_bank: bool = False,
) -> mechanisms.TensorViewFormulaAction:
    value = mechanisms.InputBinding("value", _type(3))
    bank_type = (
        mechanisms.TensorType((), (), dtype="float32", domain="activation")
        if scalar_bank
        else _type(3)
    )
    weight = mechanisms.BankBinding(
        "weight",
        source_ref="arti/test-predecessor-bank@1",
        partition_id=partition_id,
        value_type=bank_type,
    )
    action = mechanisms.BankLocalFormulaAction(
        action_id,
        mechanisms.FormulaProgram.build(
            outputs=(mechanisms.scale(value, weight),)
        ),
        input_schema=_schema(3),
        output_schema=_schema(3),
        operands={"weight": torch.ones(()) if scalar_bank else torch.ones(1, 3)},
        result_kind="descend",
        next_bank_id=next_bank_id,
        plastic_bank_slot="weight",
    )
    return mechanisms.TensorViewFormulaAction(action, layout=_layout())


def _effect(
    action_id: str,
    next_bank_id: str,
    *,
    scale: float,
    scalar_bank: bool = False,
) -> mechanisms.TensorViewFormulaEffectAction:
    value = mechanisms.InputBinding("value", _type(3))
    bank_type = (
        mechanisms.TensorType((), (), dtype="float32", domain="activation")
        if scalar_bank
        else _type(3)
    )
    bank_shape = () if scalar_bank else (1, 3)
    writer = mechanisms.BankBinding(
        "writer",
        source_ref="arti/formula-operand-bank@1",
        partition_id=f"{action_id}-writer",
        value_type=bank_type,
    )
    gain = mechanisms.BankBinding(
        "gain",
        source_ref="arti/formula-operand-bank@1",
        partition_id=f"{action_id}-gain",
        value_type=bank_type,
    )
    effect_data = (
        mechanisms.reduce_sum(mechanisms.reduce_sum(value, axis="D"), axis="B")
        if scalar_bank
        else value
    )
    effect = mechanisms.neural_plasticity(
        value,
        mechanisms.scale(effect_data, writer),
        mechanisms.scale(effect_data, gain),
    )
    action = mechanisms.BankLocalFormulaEffectAction(
        action_id,
        mechanisms.FormulaEffectProgramV2(
            mechanisms.FormulaProgram.build(outputs=(effect,)),
            data_input_name="value",
            state_type=bank_type,
        ),
        input_schema=_schema(3),
        operands={
            "writer": torch.full(bank_shape, scale),
            "gain": torch.zeros(bank_shape),
        },
        result_kind="descend",
        next_bank_id=next_bank_id,
        trainable_operands=("writer",),
    )
    return mechanisms.TensorViewFormulaEffectAction(action, layout=_layout())


def _dummy_action() -> mechanisms.TensorViewFormulaAction:
    value = mechanisms.InputBinding("value", _type(4))
    weight = mechanisms.BankBinding(
        "weight",
        source_ref="arti/formula-operand-bank@1",
        partition_id="dummy-weight",
        value_type=_type(4),
    )
    action = mechanisms.BankLocalFormulaAction(
        "dummy",
        mechanisms.FormulaProgram.build(
            outputs=(mechanisms.scale(value, weight),)
        ),
        input_schema=_schema(4),
        output_schema=_schema(4),
        operands={"weight": torch.ones(1, 4)},
    )
    return mechanisms.TensorViewFormulaAction(action, layout=_layout())


def _program(
    bank_id: str,
    actions,
    *,
    query_scores: dict[str, float],
    terminal_width: int,
) -> mechanisms.TensorViewFormulaProgram:
    action_ids = tuple(action.action_id for action in actions)
    return mechanisms.TensorViewFormulaProgram(
        bank_id=bank_id,
        query=_query((*action_ids, "exit"), query_scores),
        actions=tuple(actions),
        terminal_action=mechanisms.BankLocalTerminalAction(
            "exit",
            input_schema=_schema(terminal_width),
        ),
        local_refine=mechanisms.BankLocalRefinePolicy(min_steps=1, max_steps=1),
        input_pattern=_pattern(),
        exit_pattern=_pattern(),
        terminal_abi=_terminal_abi(),
    )


def _runtime(*, reverse_producers: bool = False) -> mechanisms.FederalRecallV3:
    torch.manual_seed(811)
    left = _producer("a-producer", "effect-a", partition_id="a-weight")
    right = _producer("z-producer", "effect-z", partition_id="z-weight")
    producers = (right, left) if reverse_producers else (left, right)
    root = _program(
        "root",
        producers,
        query_scores={"a-producer": 4.0, "z-producer": 3.0, "exit": -100.0},
        terminal_width=4,
    )
    effect_a = _program(
        "effect-a",
        (_effect("imprint-a", "terminal-a", scale=1.0),),
        query_scores={"imprint-a": 10.0, "exit": -10.0},
        terminal_width=4,
    )
    effect_z = _program(
        "effect-z",
        (_effect("imprint-z", "terminal-z", scale=10.0),),
        query_scores={"imprint-z": 10.0, "exit": -10.0},
        terminal_width=4,
    )
    terminal_a = _program(
        "terminal-a",
        (_dummy_action(),),
        query_scores={"dummy": -10.0, "exit": 10.0},
        terminal_width=3,
    )
    terminal_z = _program(
        "terminal-z",
        (_dummy_action(),),
        query_scores={"dummy": -10.0, "exit": 10.0},
        terminal_width=3,
    )
    return mechanisms.FederalRecallV3(
        {
            "root": root,
            "effect-a": effect_a,
            "effect-z": effect_z,
            "terminal-a": terminal_a,
            "terminal-z": terminal_z,
        },
        terminal_abi=_terminal_abi(),
        root_bank_ids=("root",),
        max_levels=3,
        max_k=2,
    )


def _view(value: torch.Tensor) -> mechanisms.TensorView:
    return mechanisms.TensorView.from_tensor(
        value,
        axis_names=("batch", "feature"),
        axis_roles=("batch", "feature"),
    )


def _plastic_action(runtime, action_id: str):
    return next(
        action
        for action in runtime.banks["root"].plastic_actions
        if action.action_id == action_id
    )


def test_k_wide_commits_only_the_winner_predecessor_bank_slot() -> None:
    runtime = _runtime()
    value = torch.tensor([[1.0, 2.0, 3.0]])
    left = _plastic_action(runtime, "a-producer")
    right = _plastic_action(runtime, "z-producer")
    left_before = left.action.initial_bank_value()
    right_before = right.action.initial_bank_value()

    output, trace = runtime(_view(value), max_k=2, return_trace=True)

    torch.testing.assert_close(output["value"], value)
    torch.testing.assert_close(left.action.initial_bank_value(), left_before + value)
    torch.testing.assert_close(right.action.initial_bank_value(), right_before)
    assert left.action.initial_revision() == 1
    assert right.action.initial_revision() == 0
    assert len(trace.committed_effects) == 1
    receipt = trace.committed_effects[0]
    assert receipt.producer_action_id == "a-producer"
    assert receipt.effect_action_id == "imprint-a"
    assert receipt.bank_slot_ref == left.bank_slot_ref
    assert receipt.winner_path == trace.winner_paths[0]
    assert json.loads(json.dumps(trace.to_dict())) == trace.to_dict()


def test_pending_proposal_is_not_visible_to_same_invocation_formula() -> None:
    runtime = _runtime()
    value = _view(torch.tensor([[1.0, 2.0, 3.0]]))
    entry = runtime.initial_bank_state()
    root = runtime.banks["root"]
    root_candidates = root._execute_once(
        value,
        root.query(value),
        max_candidates=2,
        bank_state=entry,
        bank_proposals=(),
        producer=None,
    )
    produced = next(item for item in root_candidates if item.candidate_id == "a-producer")
    effect_bank = runtime.banks["effect-a"]
    effected = effect_bank._execute_once(
        produced.next_view,
        effect_bank.query(produced.next_view),
        max_candidates=1,
        bank_state=entry,
        bank_proposals=produced.bank_proposals,
        producer=produced.producer,
    )[0]
    producer = _plastic_action(runtime, "a-producer")
    repeated = producer._execute(
        value,
        bank_state=entry,
        proposals=effected.bank_proposals,
    )

    assert effected.next_view.value is produced.next_view.value
    torch.testing.assert_close(repeated.view.value, produced.next_view.value)
    assert effected.producer == produced.producer
    assert len(effected.bank_proposals) == 1


def test_next_invocation_reexecutes_the_committed_predecessor() -> None:
    runtime = _runtime()
    value = torch.tensor([[1.0, 2.0, 3.0]])

    first = runtime(_view(value), max_k=1)["value"]
    second = runtime(_view(value), max_k=1)["value"]

    torch.testing.assert_close(first, value)
    torch.testing.assert_close(second, value * (1.0 + value))


def test_k_one_and_k_wide_commit_the_same_hard_winner() -> None:
    narrow = _runtime()
    wide = _runtime()
    value = torch.tensor([[0.5, -1.0, 2.0]])

    narrow_output = narrow(_view(value), max_k=1)["value"]
    wide_output = wide(_view(value), max_k=2)["value"]

    torch.testing.assert_close(narrow_output, wide_output)
    torch.testing.assert_close(
        narrow.initial_bank_state().value(_plastic_action(narrow, "a-producer").bank_slot_ref),
        wide.initial_bank_state().value(_plastic_action(wide, "a-producer").bank_slot_ref),
    )
    assert _plastic_action(narrow, "z-producer").action.initial_revision() == 0
    assert _plastic_action(wide, "z-producer").action.initial_revision() == 0


def test_candidate_permutation_preserves_owner_and_winner() -> None:
    canonical = _runtime()
    permuted = _runtime(reverse_producers=True)
    value = torch.tensor([[0.75, 1.25, -0.5]])

    canonical_output = canonical(_view(value), return_trace=True)
    permuted_output = permuted(_view(value), return_trace=True)

    torch.testing.assert_close(canonical_output[0]["value"], permuted_output[0]["value"])
    assert canonical_output[1].committed_effects[0].producer_action_id == "a-producer"
    assert permuted_output[1].committed_effects[0].producer_action_id == "a-producer"
    assert _plastic_action(canonical, "z-producer").action.initial_revision() == 0
    assert _plastic_action(permuted, "z-producer").action.initial_revision() == 0


def test_committed_predecessor_bank_round_trips_through_state_dict() -> None:
    runtime = _runtime()
    value = torch.tensor([[1.0, -0.5, 2.0]])
    runtime(_view(value), max_k=2)
    checkpoint = {
        name: tensor.detach().clone() for name, tensor in runtime.state_dict().items()
    }
    restored = _runtime()
    restored.load_state_dict(checkpoint)

    expected = runtime(_view(value), max_k=2)["value"]
    actual = restored(_view(value), max_k=2)["value"]

    torch.testing.assert_close(actual, expected)
    restored_left = _plastic_action(restored, "a-producer")
    assert restored_left.action.initial_revision() == 2
    contract = arti.component_state_contract(
        restored,
        restored.state_dict(),
        scope="all",
    )
    assert (
        arti.validate_component_state_contract(
            contract,
            state_dict=restored.state_dict(),
            model=restored,
        )
        == contract
    )


def test_effect_without_dynamic_plastic_predecessor_is_ineligible() -> None:
    runtime = _runtime()
    effect = runtime.banks["effect-a"].effect_actions[0]
    value = _view(torch.ones(1, 3))

    assert not effect.accepts(value, None)
    assert "self_state" not in dict(effect.action.named_buffers())
    assert "state_revision" not in dict(effect.action.named_buffers())
    assert (
        arti.component_ref(effect)
        == "arti/tensor-view-formula-effect-action@1"
    )


@pytest.mark.parametrize("max_k", (1, 2))
@pytest.mark.parametrize("scalar_bank", (False, True), ids=("vector", "scalar"))
def test_effect_eligibility_uses_predecessor_bank_type(
    monkeypatch: pytest.MonkeyPatch, max_k: int, scalar_bank: bool
) -> None:
    # Equal terminal scores use path order, so name the intended winner first.
    vector = _producer(
        "z-vector" if scalar_bank else "a-vector",
        "effects",
        partition_id="vector-weight",
    )
    scalar = _producer(
        "a-scalar" if scalar_bank else "z-scalar",
        "effects",
        partition_id="scalar-weight",
        scalar_bank=True,
    )
    winner = scalar if scalar_bank else vector
    loser = vector if scalar_bank else scalar
    root = _program(
        "root",
        (vector, scalar),
        query_scores={winner.action_id: 10.0, loser.action_id: 0.0, "exit": -100.0},
        terminal_width=4,
    )
    vector_effect = _effect("vector-effect", "terminal", scale=1.0)
    scalar_effect = _effect(
        "scalar-effect", "terminal", scale=2.0, scalar_bank=True
    )
    compatible = scalar_effect if scalar_bank else vector_effect
    incompatible = vector_effect if scalar_bank else scalar_effect
    effects = _program(
        "effects",
        (vector_effect, scalar_effect),
        query_scores={
            incompatible.action_id: 10.0,
            compatible.action_id: 9.0,
            "exit": -100.0,
        },
        terminal_width=4,
    )
    terminal = _program(
        "terminal",
        (_dummy_action(),),
        query_scores={"dummy": -10.0, "exit": 10.0},
        terminal_width=3,
    )
    runtime = mechanisms.FederalRecallV3(
        {"root": root, "effects": effects, "terminal": terminal},
        terminal_abi=_terminal_abi(),
        root_bank_ids=("root",),
        max_levels=3,
        max_k=2,
    )
    value = _view(torch.tensor([[1.0, 2.0, 3.0]]))
    entry = runtime.initial_bank_state()
    produced = root._execute_once(
        value,
        root.query(value),
        max_candidates=2,
        bank_state=entry,
        bank_proposals=(),
        producer=None,
    )
    assert {tuple(item.producer.bound_value.shape) for item in produced} == {
        (),
        (1, 3),
    }
    for item in produced:
        torch.testing.assert_close(item.next_view.value, value.value)

    attempted = []

    def guard(effect, expected_producer):
        execute = effect._execute

        def checked(view, **kwargs):
            producer_id = kwargs["producer"].action_id
            attempted.append((producer_id, effect.action_id))
            assert producer_id == expected_producer, "incompatible effect was attempted"
            return execute(view, **kwargs)

        monkeypatch.setattr(effect, "_execute", checked)

    guard(vector_effect, vector.action_id)
    guard(scalar_effect, scalar.action_id)
    winner_before = winner.action.initial_bank_value()
    loser_before = loser.action.initial_bank_value()

    output, trace = runtime(value, max_k=max_k, return_trace=True)

    torch.testing.assert_close(output["value"], value.value)
    expected_update = value.value.sum() * 2.0 if scalar_bank else value.value
    torch.testing.assert_close(
        winner.action.initial_bank_value(), winner_before + expected_update
    )
    torch.testing.assert_close(loser.action.initial_bank_value(), loser_before)
    assert winner.action.initial_revision() == 1
    assert loser.action.initial_revision() == 0
    expected_attempts = {(winner.action_id, compatible.action_id)}
    if max_k == 2:
        expected_attempts.add((loser.action_id, incompatible.action_id))
    assert set(attempted) == expected_attempts
    assert len(trace.committed_effects) == 1
    assert trace.committed_effects[0].bank_slot_ref == winner.bank_slot_ref
    assert trace.committed_effects[0].effect_action_id == compatible.action_id


def test_federal_plastic_slot_cannot_also_be_optimizer_trainable() -> None:
    value = mechanisms.InputBinding("value", _type(3))
    weight = mechanisms.BankBinding(
        "weight",
        source_ref="arti/test-predecessor-bank@1",
        partition_id="weight",
        value_type=_type(3),
    )

    with pytest.raises(ValueError, match="forward-written state"):
        mechanisms.BankLocalFormulaAction(
            "producer",
            mechanisms.FormulaProgram.build(outputs=(mechanisms.scale(value, weight),)),
            input_schema=_schema(3),
            output_schema=_schema(3),
            operands={"weight": torch.ones(1, 3)},
            trainable_operands=("weight",),
            plastic_bank_slot="weight",
        )
