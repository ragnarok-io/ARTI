from __future__ import annotations

import torch
from torch.nn import functional as F

import arti
from arti import mechanisms


def _type() -> mechanisms.TensorType:
    return mechanisms.TensorType.axes(
        ("B", "D"),
        sizes=("B", 4),
        dtype="float32",
        domain="program-query-test",
    )


def _candidate(
    candidate_id: str,
    *,
    input_slot: str,
    output_slot: str,
    mode: str,
) -> mechanisms.FormulaProgramCandidate:
    value = mechanisms.InputBinding("value", _type())
    output = mechanisms.scalar_map(value, mode=mode)
    return mechanisms.FormulaProgramCandidate(
        candidate_id,
        mechanisms.FormulaProgram.build(outputs=(output,)),
        input_slots={"value": input_slot},
        output_slot=output_slot,
    )


def _query(seed: int = 17) -> mechanisms.FormulaProgramQuery:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return mechanisms.FormulaProgramQuery(
            slot_ids=("x", "hidden", "output"),
            candidates=(
                _candidate(
                    "relu-hidden",
                    input_slot="x",
                    output_slot="hidden",
                    mode="relu",
                ),
                _candidate(
                    "gelu-hidden",
                    input_slot="x",
                    output_slot="hidden",
                    mode="gelu",
                ),
                _candidate(
                    "tanh-output",
                    input_slot="hidden",
                    output_slot="output",
                    mode="tanh",
                ),
            ),
            terminal_slot="output",
            min_steps=2,
            max_steps=2,
            hidden_dim=24,
        )


def _task_loss(output: torch.Tensor, target: object) -> torch.Tensor:
    assert isinstance(target, torch.Tensor)
    return F.mse_loss(output, target, reduction="none").mean(dim=-1)


def test_program_query_exposes_only_shape_valid_ssa_candidates() -> None:
    query = _query()
    arena = query.arena({"x": torch.randn(3, 4)})

    first = {
        action_id
        for action_id, enabled in zip(
            query.action_ids,
            query.eligible(arena, steps=0).tolist(),
            strict=True,
        )
        if enabled
    }
    assert first == {"relu-hidden", "gelu-hidden"}

    arena = query.candidates[0](arena)
    second = {
        action_id
        for action_id, enabled in zip(
            query.action_ids,
            query.eligible(arena, steps=1).tolist(),
            strict=True,
        )
        if enabled
    }
    assert second == {"tanh-output"}


def test_exact_program_query_training_uses_final_task_loss() -> None:
    query = _query()
    x = torch.randn(8, 4)
    target = torch.tanh(torch.relu(x))
    trainer = mechanisms.ExactFormulaProgramQueryTraining(max_states=16)

    loss = trainer.loss(
        query,
        initial={"x": x},
        target=target,
        task_loss=_task_loss,
    )
    loss.total.backward()

    assert loss.visited_states > 1
    assert loss.total.ndim == 0
    assert all(parameter.grad is not None for parameter in query.network.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in query.network.parameters()
    )
    assert trainer.contract_config()["route_teacher"] is False
    assert trainer.contract_config()["transition_teacher"] is False


def test_program_query_state_and_component_identity_round_trip() -> None:
    original = _query(seed=23)
    restored = _query(seed=999)
    restored.load_state_dict(original.state_dict(), strict=True)
    x = torch.randn(1, 4)

    expected, expected_trace = original({"x": x}, return_trace=True)
    actual, actual_trace = restored({"x": x}, return_trace=True)

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert actual_trace == expected_trace
    assert arti.component_ref(original) == "arti/formula-program-query@1"
    provenance = arti.component_provenance(original)
    assert arti.validate_component_provenance(provenance) == provenance
