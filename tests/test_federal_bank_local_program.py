from __future__ import annotations

import pytest
import torch

import arti
from arti.component_registry import component_provenance, validate_component_provenance
from benchmarks.train_federal_bank_local_program import (
    ACTION_IDS,
    LearnedLocalProgramQuery,
    LocalProgramWorkspace,
    build_runtime,
    final_task_policy_loss,
    sample_task,
    train_query,
)


def test_task_requires_distinct_fold_and_unfold_formula_paths() -> None:
    workspace = LocalProgramWorkspace()
    batch = sample_task(count=32, seed=7, device=torch.device("cpu"))
    path_losses = []
    for fold_action in range(2):
        folded = workspace.run_fold(batch.value, fold_action)
        for unfold_action in range(2):
            value = workspace.run_unfold(folded, unfold_action)
            path_losses.append(
                torch.nn.functional.mse_loss(
                    value, batch.target, reduction="none"
                ).mean(dim=(1, 2))
            )
    losses = torch.stack(path_losses, dim=-1)

    expected_path = batch.fold_action * 2 + batch.unfold_action
    assert torch.equal(losses.argmin(dim=-1), expected_path)
    assert bool((losses.gather(1, expected_path[:, None]).squeeze(1) < 1e-7).all())


def test_final_task_policy_loss_trains_query_without_route_targets() -> None:
    query = LearnedLocalProgramQuery()
    workspace = LocalProgramWorkspace()
    batch = sample_task(count=16, seed=11, device=torch.device("cpu"))

    loss = final_task_policy_loss(query, workspace, batch)
    loss.total.backward()

    assert loss.total.ndim == 0
    assert 0.0 <= float(loss.success_probability.detach()) <= 1.0
    assert all(parameter.grad is not None for parameter in query.parameters())
    assert all(bool(torch.isfinite(parameter.grad).all()) for parameter in query.parameters())


def test_formula_program_exposes_top_k_eligible_actions_without_route_mixing() -> None:
    runtime = build_runtime(LearnedLocalProgramQuery())
    bank = runtime.banks["learned-program"]
    value = sample_task(count=1, seed=701, device=torch.device("cpu")).value
    result = bank.execute(
        value,
        query_result=arti.mechanisms.BankQueryResult(
            torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
        ),
        max_candidates=2,
    )

    assert [candidate.candidate_id for candidate in result.candidates] == [
        "fold-even",
        "fold-odd",
    ]
    assert all(
        candidate.next_value is not None
        and candidate.next_bank_id is None
        and candidate.terminal_outputs is None
        for candidate in result.candidates
    )


def test_detached_on_policy_training_requeries_variable_shape_states() -> None:
    query = train_query(
        seed=13,
        steps=100,
        batch_size=64,
        device=torch.device("cpu"),
    )
    workspace = LocalProgramWorkspace()
    batch = sample_task(count=1, seed=1301, device=torch.device("cpu"))
    training = arti.mechanisms.DetachedBankLocalProgramTraining()

    rollout = training.capture(
        query,
        actions=tuple(workspace.actions),
        terminal_action=workspace.terminal_action,
        initial=batch.value,
        min_steps=3,
        max_steps=3,
    )
    assert rollout.terminated is True
    assert [tuple(state.shape) for state in rollout.states] == [
        (1, 8, 2),
        (1, 4, 2),
        (1, 6, 2),
    ]
    assert all(not state.requires_grad and state.grad_fn is None for state in rollout.states)
    assert not hasattr(rollout, "route")

    query.zero_grad(set_to_none=True)

    def task_loss(value: torch.Tensor, target: object, _depth: int) -> torch.Tensor:
        assert isinstance(target, torch.Tensor)
        prediction = value.square().mean(dim=(1, 2))
        expected = target.square().mean(dim=(1, 2))
        return torch.nn.functional.mse_loss(prediction, expected, reduction="none")

    loss = training.loss(
        query,
        rollout,
        actions=tuple(workspace.actions),
        terminal_action=workspace.terminal_action,
        target=batch.target,
        task_loss=task_loss,
    )
    loss.total.backward()

    assert loss.visited_states == 3
    assert loss.total.ndim == 0
    assert all(parameter.grad is not None for parameter in query.parameters())
    assert all(bool(torch.isfinite(parameter.grad).all()) for parameter in query.parameters())


def test_trained_query_seals_and_runs_three_latest_state_queries() -> None:
    query = train_query(
        seed=19,
        steps=220,
        batch_size=96,
        device=torch.device("cpu"),
    )
    runtime = build_runtime(query)
    batch = sample_task(count=12, seed=1901, device=torch.device("cpu"))

    for index in range(batch.value.shape[0]):
        output, trace = runtime(batch.value[index : index + 1], return_trace=True)
        torch.testing.assert_close(output["value"], batch.target[index : index + 1])
        local = trace.steps[0].local_refine
        assert len(local) == 3
        assert [item.action for item in local] == [
            "continue-local",
            "continue-local",
            "terminal",
        ]
        assert [item.candidate_id for item in local] == [
            ACTION_IDS[int(batch.fold_action[index])],
            ACTION_IDS[2 + int(batch.unfold_action[index])],
            "exit",
        ]
        assert [item.input_shape for item in local] == [
            (1, 8, 2),
            (1, 4, 2),
            (1, 6, 2),
        ]

    sealed = runtime.banks["learned-program"].query
    assert all(not parameter.requires_grad for parameter in sealed.parameters())
    before = sealed.signature.state_fingerprint
    runtime.train()
    assert sealed.training is False
    assert sealed.signature.state_fingerprint == before


def test_query_state_changes_are_detected_after_sealing() -> None:
    query = train_query(
        seed=23,
        steps=80,
        batch_size=64,
        device=torch.device("cpu"),
    )
    runtime = build_runtime(query)
    with torch.no_grad():
        next(iter(runtime.banks["learned-program"].query.query.parameters())).add_(1.0)

    with pytest.raises(arti.mechanisms.BankQueryError, match="modified"):
        runtime(sample_task(count=1, seed=2, device=torch.device("cpu")).value)


def test_learned_query_asset_round_trip_and_runtime_provenance(tmp_path) -> None:
    query = train_query(
        seed=31,
        steps=120,
        batch_size=64,
        device=torch.device("cpu"),
    )
    sealed = arti.mechanisms.seal_bank_query(
        query,
        capabilities=("latest-state-requery",),
    )
    value = sample_task(count=5, seed=4, device=torch.device("cpu")).value
    expected = sealed(value).value

    arti.mechanisms.save_bank_query(sealed, tmp_path / "local-program-query")
    restored = arti.mechanisms.load_bank_query(
        tmp_path / "local-program-query",
        LearnedLocalProgramQuery(),
    )

    torch.testing.assert_close(restored(value).value, expected)
    assert restored.signature == sealed.signature
    runtime = build_runtime(query)
    provenance = component_provenance(runtime)
    assert validate_component_provenance(provenance) == provenance
    bank = next(
        item
        for item in provenance["components"]
        if item["ref"] == "arti/bank-local-formula-program@1"
    )
    assert "arti/bank-local-formula-action@1" in bank["dependencies"]
    assert "arti/bank-local-terminal-action@1" in bank["dependencies"]
    dependency_refs = {
        dependency
        for component in provenance["components"]
        for dependency in component["dependencies"]
    }
    assert "arti/formula-atom-gather@1" in dependency_refs
    assert "arti/formula-atom-scatter@1" in dependency_refs
    assert "arti/formula-atom-refine-exit@1" in dependency_refs
