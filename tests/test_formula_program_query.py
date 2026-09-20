from __future__ import annotations

import pytest
import torch

import arti
from arti import mechanisms
from benchmarks.train_formula_program_query_circuit import (
    EXPECTED_PATH,
    build_circuit,
    evaluate,
    final_task_loss,
    sample,
    train,
)


def _eligible_ids(
    query: mechanisms.FormulaProgramQuery,
    arena: mechanisms.FormulaProgramArena,
    *,
    steps: int,
) -> set[str]:
    mask = query.eligible(arena, steps=steps)
    return {
        action_id
        for action_id, enabled in zip(query.action_ids, mask.tolist(), strict=True)
        if enabled
    }


def test_program_query_only_exposes_shape_valid_ssa_actions() -> None:
    circuit = build_circuit(seed=17, device=torch.device("cpu"))
    x, _target = sample(circuit, count=2, seed=701)
    arena = circuit.query.arena({"x": x})

    assert _eligible_ids(circuit.query, arena, steps=0) == {
        "project-good",
        "project-wrong",
    }
    projected = circuit.query.candidates[0](arena)
    assert _eligible_ids(circuit.query, projected, steps=1) == {
        "gelu",
        "relu",
        "shortcut",
    }
    activated = circuit.query.candidates[2](projected)
    assert _eligible_ids(circuit.query, activated, steps=2) == {
        "output",
        "shortcut",
    }
    completed = circuit.query.candidates[4](activated)
    assert _eligible_ids(circuit.query, completed, steps=3) == {"stop"}


def test_program_candidate_requires_one_atom_and_write_once_ssa() -> None:
    value_type = mechanisms.TensorType.axes(
        ("B", "D"), sizes=("B", 4), dtype="float32"
    )
    value = mechanisms.InputBinding("value", value_type)
    two_atoms = mechanisms.scalar_map(mechanisms.scalar_map(value, mode="relu"), mode="gelu")
    with pytest.raises(ValueError, match="exactly one atom"):
        mechanisms.FormulaProgramCandidate(
            "two-atoms",
            mechanisms.FormulaProgram.build(outputs=(two_atoms,)),
            input_slots={"value": "x"},
            output_slot="hidden",
        )

    circuit = build_circuit(seed=17, device=torch.device("cpu"))
    x, _target = sample(circuit, count=1, seed=702)
    arena = circuit.query.candidates[0](circuit.query.arena({"x": x}))
    with pytest.raises(ValueError, match="already occupied"):
        circuit.query.candidates[1](arena)


def test_program_candidate_can_close_a_completed_ssa_branch() -> None:
    circuit = build_circuit(seed=17, device=torch.device("cpu"))
    x, _target = sample(circuit, count=2, seed=703)
    guarded = mechanisms.FormulaProgramCandidate(
        "guarded-project",
        circuit.query.candidates[0].program,
        input_slots={"value": "x"},
        output_slot="hidden",
        requires_empty_slots=("output",),
        operands={"weight": circuit.good_weight},
    )
    query = mechanisms.FormulaProgramQuery(
        slot_ids=circuit.query.slot_ids,
        candidates=(guarded, *circuit.query.candidates[2:]),
        terminal_slot="output",
        min_steps=1,
        max_steps=3,
    )
    arena = query.arena({"x": x})

    assert guarded.accepts(arena)
    output_arena = query.candidates[-1](
        query.candidates[1](guarded(arena))
    )
    assert not guarded.accepts(output_arena)
    assert guarded.contract_config()["requires_empty_slots"] == ["output"]


def test_program_query_rejects_unknown_empty_slot_guard() -> None:
    circuit = build_circuit(seed=17, device=torch.device("cpu"))
    guarded = mechanisms.FormulaProgramCandidate(
        "unknown-guard",
        circuit.query.candidates[0].program,
        input_slots={"value": "x"},
        output_slot="hidden",
        requires_empty_slots=("missing",),
        operands={"weight": circuit.good_weight},
    )
    with pytest.raises(ValueError, match="declared SSA slots"):
        mechanisms.FormulaProgramQuery(
                slot_ids=circuit.query.slot_ids,
            candidates=(guarded,),
            terminal_slot="output",
        )


def test_exact_program_query_training_uses_only_final_task_loss() -> None:
    circuit = build_circuit(seed=19, device=torch.device("cpu"))
    x, target = sample(circuit, count=24, seed=1901)
    trainer = mechanisms.ExactFormulaProgramQueryTraining(max_states=96)

    loss = trainer.loss(
        circuit.query,
        initial={"x": x},
        target=target,
        task_loss=final_task_loss,
    )
    loss.total.backward()

    assert loss.visited_states > 1
    assert loss.total.ndim == 0
    assert all(parameter.grad is not None for parameter in circuit.query.network.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in circuit.query.network.parameters()
    )
    assert trainer.contract_config()["route_teacher"] is False
    assert trainer.contract_config()["transition_teacher"] is False


def test_exact_training_merges_equivalent_ssa_execution_orders() -> None:
    value_type = mechanisms.TensorType.axes(
        ("B", "D"), sizes=("B", 4), dtype="float32"
    )

    def relu_candidate(candidate_id: str, output_slot: str):
        value = mechanisms.InputBinding("value", value_type)
        return mechanisms.FormulaProgramCandidate(
            candidate_id,
            mechanisms.FormulaProgram.build(
                outputs=(mechanisms.scalar_map(value, mode="relu"),)
            ),
            input_slots={"value": "x"},
            output_slot=output_slot,
        )

    left = mechanisms.InputBinding("left", value_type)
    right = mechanisms.InputBinding("right", value_type)
    combine = mechanisms.FormulaProgramCandidate(
        "combine",
        mechanisms.FormulaProgram.build(
            outputs=(mechanisms.add(left, right),)
        ),
        input_slots={"left": "left", "right": "right"},
        output_slot="output",
    )
    query = mechanisms.FormulaProgramQuery(
        slot_ids=("x", "left", "right", "output"),
        candidates=(relu_candidate("left", "left"), relu_candidate("right", "right"), combine),
        terminal_slot="output",
        min_steps=3,
        max_steps=3,
    )
    x = torch.randn(7, 4)
    target = 2 * torch.relu(x)
    loss = mechanisms.ExactFormulaProgramQueryTraining(max_states=5).loss(
        query,
        initial={"x": x},
        target=target,
        task_loss=final_task_loss,
    )

    assert loss.visited_states == 5
    assert loss.task < 1e-7
    assert loss.invalid < 1e-7


def test_program_query_learns_atom_wiring_and_stop_from_final_loss() -> None:
    circuit = build_circuit(seed=17, device=torch.device("cpu"))
    train(circuit, seed=17, steps=40, batch_size=64)
    report = evaluate(circuit, seed=80_018, count=32)

    assert report["path_accuracy"] == 1.0
    assert report["mse"] < 1e-10
    assert all(tuple(path) == EXPECTED_PATH for path in report["sample_paths"])


def test_program_query_state_and_component_graph_round_trip() -> None:
    original = build_circuit(seed=23, device=torch.device("cpu"))
    train(original, seed=23, steps=12, batch_size=32)
    restored = build_circuit(seed=999, device=torch.device("cpu"))
    restored.query.load_state_dict(original.query.state_dict(), strict=True)
    x, _target = sample(original, count=1, seed=2301)

    expected, expected_trace = original.query({"x": x}, return_trace=True)
    actual, actual_trace = restored.query({"x": x}, return_trace=True)

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert actual_trace == expected_trace
    assert arti.component_ref(original.query) == "arti/formula-program-query@1"
    provenance = arti.component_provenance(original.query)
    assert arti.validate_component_provenance(provenance) == provenance


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_program_query_hard_execution_matches_cuda() -> None:
    cpu = build_circuit(seed=31, device=torch.device("cpu"))
    train(cpu, seed=31, steps=40, batch_size=64)
    cuda = build_circuit(seed=31, device=torch.device("cuda"))
    cuda.query.load_state_dict(cpu.query.state_dict(), strict=True)
    x, _target = sample(cpu, count=1, seed=3101)

    expected, expected_trace = cpu.query({"x": x}, return_trace=True)
    actual, actual_trace = cuda.query({"x": x.cuda()}, return_trace=True)

    torch.testing.assert_close(actual.cpu(), expected, atol=1e-6, rtol=1e-5)
    assert actual_trace == expected_trace
