from __future__ import annotations

import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from benchmarks.federal_bank_asset_compounding import (
    CandidateRollout,
    CompoundingConfig,
    FederalExecutionPlan,
    FederatedTransitionQuery,
    _length_normalized_log_probability,
    _recipes,
    _targets_for_batch,
    execute_plan,
    run,
    route_loss,
    sample_rollout,
)
from benchmarks.federal_refine_architecture_search import (
    ExperimentConfig as SourceConfig,
    FederalArchitectureSearch,
)


def _experts(seed: int = 41) -> FederalArchitectureSearch:
    config = SourceConfig(
        tasks=4,
        depths=(1, 2),
        steps=2,
        batch_size=2,
        candidates=2,
        eval_size_per_task=1,
        max_seconds=30.0,
    )
    experts = FederalArchitectureSearch(config, seed=seed).eval()
    for parameter in experts.parameters():
        parameter.requires_grad_(False)
    return experts


def test_query_only_k_wide_compounding_backpropagates() -> None:
    experts = _experts()
    query = FederatedTransitionQuery(
        tasks=2,
        expert_count=4,
        dim=16,
        query_dim=8,
        seed=43,
    )
    value = torch.randn(2, 16)
    task_id = torch.tensor([0, 1])
    recipes = _recipes(4, 2)
    target = _targets_for_batch(
        experts,
        recipes,
        value,
        task_id,
        inner_depth=1,
        inner_k=1,
    )
    rollout = sample_rollout(
        query,
        experts,
        value,
        task_id,
        refine_steps=2,
        candidates=4,
        inner_depth=1,
        inner_k=1,
        generator=torch.Generator().manual_seed(47),
    )

    assert rollout.value.shape == (8, 16)
    assert rollout.path.shape == (8, 2)
    loss, metrics = route_loss(
        rollout,
        target,
        candidates=4,
        identity_index=4,
    )
    loss.backward()

    assert metrics["best"] <= metrics["mean"]
    assert any(parameter.grad is not None for parameter in query.parameters())
    assert all(not parameter.requires_grad for parameter in experts.parameters())
    assert all(parameter.grad is None for parameter in experts.parameters())


def test_federal_execution_plan_round_trip_is_exact() -> None:
    experts = _experts(seed=53)
    plan = FederalExecutionPlan(
        source_sha256="a" * 64,
        inner_depth=1,
        inner_k=1,
        task_paths=((0, 4), (1, 4)),
    )
    loaded = FederalExecutionPlan.from_dict(plan.to_dict())
    value = torch.randn(4, 16)
    task_id = torch.tensor([0, 0, 1, 1])

    expected = execute_plan(plan, experts, value, task_id)
    actual = execute_plan(loaded, experts, value, task_id)

    assert loaded.fingerprint == plan.fingerprint
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_route_loss_does_not_reward_early_identity_for_fewer_decisions() -> None:
    rollout = CandidateRollout(
        value=torch.stack((torch.ones(2), torch.zeros(2))),
        log_probability=torch.tensor((-1.6, -6.4), requires_grad=True),
        path=torch.tensor(((4, 4, 4, 4), (0, 1, 2, 4))),
    )
    loss, metrics = route_loss(
        rollout,
        torch.zeros(1, 2),
        candidates=2,
        identity_index=4,
    )

    assert metrics["best_nll"] == pytest.approx(1.6)
    loss.backward()
    assert rollout.log_probability.grad is not None


def test_beam_score_does_not_reward_early_identity_for_fewer_decisions() -> None:
    score = _length_normalized_log_probability(
        torch.tensor((-1.6, -6.4)),
        torch.tensor((1, 4)),
    )

    torch.testing.assert_close(score, torch.tensor((-1.6, -1.6)))


def test_tiny_run_writes_reloadable_compounding_artifacts(tmp_path: Path) -> None:
    experts = _experts(seed=59)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    weights = source_dir / "federal-architecture-search.arti.st"
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in experts.state_dict().items()},
        str(weights),
    )
    source_results = source_dir / "results.json"
    source_results.write_text(
        json.dumps(
            {
                "experiment": "arti.federal-refine-architecture-search.v1",
                "config": {
                    **experts.config.__dict__,
                    "depths": list(experts.config.depths),
                },
                "seeds": [59],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "result"
    result = run(
        CompoundingConfig(
            tasks=2,
            query_dim=8,
            depths=(1, 4),
            steps=2,
            batch_size=2,
            candidates=4,
            eval_size_per_task=1,
            inner_depth=1,
            inner_k=1,
            deployment_depth=4,
            max_seconds=30.0,
        ),
        source_weights=weights,
        source_results=source_results,
        output_dir=output_dir,
        device=torch.device("cpu"),
    )

    assert result["route_or_step_teacher"] is False
    assert result["compiled_plan"]["fresh_reload_max_abs"] == 0.0
    assert (output_dir / "federal-execution-plan.json").is_file()
    assert (output_dir / "federal-transition-query.arti.st").is_file()
    assert (output_dir / "results.json").is_file()
