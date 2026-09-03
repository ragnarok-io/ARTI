from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

import arti
from arti import mechanisms


SOURCE_REF = "arti/program-query-test-bank@1"
SLOTS = ("x", "stage-a", "after-a", "stage-b", "after-b", "output")
EXPECTED_PATH = (
    "stage-a-good",
    "effect-a",
    "stage-b-good",
    "effect-b",
    "read-state",
    "stop",
)


def _type() -> mechanisms.TensorType:
    return mechanisms.TensorType(
        ("B", "D"),
        ("B", 3),
        dtype="float32",
        domain="activation",
    )


def _vector_type() -> mechanisms.TensorType:
    return mechanisms.TensorType(
        ("D",),
        (3,),
        dtype="float32",
        domain="activation",
    )


def _bank(name: str, value_type: mechanisms.TensorType) -> mechanisms.BankBinding:
    return mechanisms.BankBinding(name, SOURCE_REF, name, value_type)


def _scale_candidate(
    candidate_id: str,
    input_slot: str,
    output_slot: str,
    factor: Tensor,
) -> mechanisms.FormulaProgramTensorCandidate:
    value = mechanisms.InputBinding("value", _type())
    weight = _bank("weight", _vector_type())
    candidate = mechanisms.FormulaProgramCandidate(
        candidate_id,
        mechanisms.FormulaProgram.build(outputs=(mechanisms.scale(value, weight),)),
        input_slots={"value": input_slot},
        output_slot=output_slot,
        operands={"weight": factor},
    )
    return mechanisms.FormulaProgramTensorCandidate(candidate)


def _effect_candidate(
    candidate_id: str,
    input_slot: str,
    output_slot: str,
    *,
    requires_empty_slots: tuple[str, ...] = (),
) -> mechanisms.FormulaProgramEffectCandidate:
    value = mechanisms.InputBinding("value", _type())
    writer = _bank("writer", _vector_type())
    gain = _bank("gain", _vector_type())
    additive = mechanisms.scale(value, writer)
    multiplicative = mechanisms.scale(value, gain)
    effect = mechanisms.neural_plasticity(value, additive, multiplicative)
    program = mechanisms.FormulaEffectProgramV2(
        mechanisms.FormulaProgram.build(outputs=(effect,)),
        data_input_name="value",
        state_type=_type(),
    )
    return mechanisms.FormulaProgramEffectCandidate(
        candidate_id,
        program,
        input_slot=input_slot,
        output_slot=output_slot,
        state_id="memory",
        requires_empty_slots=requires_empty_slots,
        operands={
            "writer": torch.ones(3),
            "gain": torch.zeros(3),
        },
    )


def _reader() -> mechanisms.FormulaProgramTensorCandidate:
    value = mechanisms.InputBinding("value", _type())
    memory = _bank("memory", _type())
    candidate = mechanisms.FormulaProgramCandidate(
        "read-state",
        mechanisms.FormulaProgram.build(outputs=(mechanisms.scale(value, memory),)),
        input_slots={"value": "after-b"},
        output_slot="output",
        operands={"memory": torch.zeros(1, 3)},
    )
    return mechanisms.FormulaProgramTensorCandidate(
        candidate,
        state_operands={"memory": "memory"},
    )


@dataclass(frozen=True)
class Search:
    query: mechanisms.FormulaProgramQueryV2
    good_a: Tensor
    good_b: Tensor


def _build(seed: int) -> Search:
    torch.manual_seed(seed)
    good_a = torch.tensor([0.7, -1.1, 0.4])
    good_b = torch.tensor([1.3, 0.6, -0.8])
    candidates = (
        _scale_candidate("stage-a-good", "x", "stage-a", good_a),
        _scale_candidate(
            "stage-a-wrong",
            "x",
            "stage-a",
            torch.tensor([-0.2, 1.5, 0.9]),
        ),
        _effect_candidate("effect-a", "stage-a", "after-a"),
        _effect_candidate(
            "effect-single",
            "stage-a",
            "after-b",
            requires_empty_slots=("after-a",),
        ),
        _scale_candidate("stage-b-good", "after-a", "stage-b", good_b),
        _effect_candidate("effect-b", "stage-b", "after-b"),
        _reader(),
    )
    query = mechanisms.FormulaProgramQueryV2(
        slot_ids=SLOTS,
        state_ids=("memory",),
        candidates=candidates,
        terminal_slot="output",
        min_steps=1,
        max_steps=6,
        hidden_dim=48,
    )
    return Search(query, good_a, good_b)


def _sample(search: Search, count: int, seed: int) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(count, 3, generator=generator)
    stage_a = x * search.good_a
    stage_b = stage_a * search.good_b
    target = stage_b * (stage_a + stage_b)
    return x, target


def _loss(output: Tensor, target: object) -> Tensor:
    assert isinstance(target, Tensor)
    return F.mse_loss(output, target, reduction="none").mean(dim=-1)


def _train(search: Search, *, steps: int = 80) -> None:
    trainer = mechanisms.ExactFormulaProgramQueryTrainingV2(
        invalid_weight=2.0,
        max_states=256,
        exploration_probability=0.25,
    )
    optimizer = torch.optim.AdamW(search.query.network.parameters(), lr=0.03)
    for step in range(steps):
        x, target = _sample(search, 64, 50_000 + step)
        optimizer.zero_grad(set_to_none=True)
        result = trainer.loss(
            search.query,
            initial={"x": x},
            states={"memory": torch.zeros_like(x)},
            target=target,
            task_loss=_loss,
        )
        result.total.backward()
        optimizer.step()


def test_effect_state_is_implicit_to_query_but_changes_program_result() -> None:
    search = _build(17)
    x, _target = _sample(search, 4, 701)
    zero = search.query.arena({"x": x}, states={"memory": torch.zeros_like(x)})
    nonzero = search.query.arena({"x": x}, states={"memory": torch.randn_like(x)})

    zero_query = search.query.query(zero, steps=0)
    nonzero_query = search.query.query(nonzero, steps=0)
    torch.testing.assert_close(zero_query.logits, nonzero_query.logits)
    assert search.query.contract_config()["query_state_access"] is False

    first = search.query.candidates[0](zero)
    effect = search.query.candidates[2](first)
    torch.testing.assert_close(effect.state("memory"), first.values.get("stage-a"))
    assert effect.revision("memory") == 1


def test_final_task_loss_search_discovers_self_effect_topology() -> None:
    search = _build(23)
    _train(search)
    x, target = _sample(search, 12, 80_023)
    outputs: list[Tensor] = []
    paths: list[tuple[str, ...]] = []
    effect_counts: list[int] = []
    with torch.no_grad():
        for row in range(x.shape[0]):
            result = search.query(
                {"x": x[row : row + 1]},
                states={"memory": torch.zeros_like(x[row : row + 1])},
            )
            outputs.append(result.value)
            path = tuple(step.candidate_id for step in result.trace.steps)
            paths.append(path)
            effect_counts.append(
                sum(step.effect_site_id is not None for step in result.trace.steps)
            )

    prediction = torch.cat(outputs)
    assert all(path == EXPECTED_PATH for path in paths)
    assert effect_counts == [2] * len(effect_counts)
    torch.testing.assert_close(prediction, target, atol=1e-6, rtol=1e-5)
    assert arti.component_ref(search.query) == "arti/formula-program-query@2"
    assert (
        arti.component_ref(mechanisms.ExactFormulaProgramQueryTrainingV2())
        == "arti/exact-formula-program-query-training@2"
    )


def test_effect_topology_query_reloads_without_prebuilt_effect_chain() -> None:
    search = _build(29)
    _train(search, steps=60)
    restored = _build(999)
    restored.query.load_state_dict(search.query.state_dict(), strict=True)
    x, _target = _sample(search, 1, 912)

    expected = search.query({"x": x}, states={"memory": torch.zeros_like(x)})
    actual = restored.query({"x": x}, states={"memory": torch.zeros_like(x)})

    torch.testing.assert_close(actual.value, expected.value, atol=0.0, rtol=0.0)
    assert actual.trace == expected.trace
    assert actual.revisions == expected.revisions == (2,)
