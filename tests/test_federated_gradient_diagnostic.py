from __future__ import annotations

import pytest
import torch

from benchmarks.probe_qwen_federated_gradient_balance import (
    clip_fork_gradients,
    gradient_rows,
    select_query_credit_gradients,
)
from benchmarks.train_qwen_federated_autonomous_federation import (
    _clip_training_gradients,
    _query_credit_objective,
)


def test_gradient_groups_and_clip_scale_do_not_claim_adam_update_ratio() -> None:
    writer = torch.tensor([0.0])
    router = torch.tensor([0.0])
    result = gradient_rows(
        (writer, router),
        ("writer", "router"),
        {id(writer): "plasticity", id(router): "routing"},
        (torch.tensor([3.0]), None),
        (torch.tensor([3.0]), torch.tensor([4.0])),
        clip_norm=1.0,
    )
    assert result["all"]["task_norm"] == 3.0
    assert result["all"]["weighted_route_norm"] == 4.0
    assert result["all"]["combined_norm"] == 5.0
    assert result["all"]["task_route_cosine"] == 0.0
    assert result["combined_global_clip_scale"] == pytest.approx(0.2)
    assert result["groups"]["plasticity"]["weighted_route_norm"] == 0.0
    assert result["groups"]["routing"]["task_norm"] == 0.0
    assert result["top_route_parameters"][0]["name"] == "router"
    assert "not an Adam" in result["interpretation"]


def test_gradient_difference_reports_opposed_objectives() -> None:
    parameter = torch.tensor([0.0])
    result = gradient_rows(
        (parameter,),
        ("law",),
        {id(parameter): "plasticity"},
        (torch.tensor([2.0]),),
        (torch.tensor([-1.0]),),
        clip_norm=1.0,
    )
    assert result["all"]["weighted_route_norm"] == 3.0
    assert result["all"]["task_route_cosine"] == -1.0


def test_query_credit_preserves_task_gradients_and_absent_connections() -> None:
    parameters = tuple(torch.nn.Parameter(torch.ones(2)) for _ in range(4))
    groups = dict(zip(map(id, parameters), ("routing", "ordinary", "plasticity", "ordinary")))
    task = (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), None, None)
    combined = (torch.tensor([5.0, 6.0]), torch.tensor([-3.0, -4.0]), torch.zeros(2), None)
    selected = select_query_credit_gradients(parameters, groups, task, combined)
    assert selected[0] is combined[0]
    assert selected[1] is task[1]
    assert selected[2] is selected[3] is None
    assert all(parameter.grad is None for parameter in parameters)


def test_query_credit_adam_does_not_step_credit_only_non_router_parameter() -> None:
    router, writer = (torch.nn.Parameter(torch.tensor([1.0])) for _ in range(2))
    parameters = (router, writer)
    groups = {id(router): "routing", id(writer): "plasticity"}
    selected = select_query_credit_gradients(
        parameters, groups, (None, None), (torch.tensor([2.0]), torch.tensor([3.0])),
    )
    optimizer = torch.optim.AdamW(parameters, lr=0.01, weight_decay=0.1)
    for parameter, gradient in zip(parameters, selected, strict=True):
        parameter.grad = gradient
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    assert writer.item() == 1.0
    assert writer not in optimizer.state
    assert router.item() < 1.0


def test_task_clip_groups_ordinary_and_plasticity_together() -> None:
    parameters = tuple(torch.nn.Parameter(torch.ones(1)) for _ in range(3))
    groups = dict(zip(map(id, parameters), ("routing", "ordinary", "plasticity")))
    for parameter, value in zip(parameters, (100.0, 3.0, 4.0), strict=True):
        parameter.grad = torch.tensor([value])
    clip_fork_gradients(parameters, groups, mode="query-credit-task-clip", norm=1.0)
    torch.testing.assert_close(parameters[0].grad, torch.tensor([1.0]))
    torch.testing.assert_close(parameters[1].grad, torch.tensor([0.6]))
    torch.testing.assert_close(parameters[2].grad, torch.tensor([0.8]))


def test_query_credit_proxy_matches_first_order_oracle_without_detaching_task() -> None:
    router, writer, credit_only, unused = (
        torch.nn.Parameter(torch.tensor([0.4, -0.2], dtype=torch.float64))
        for _ in range(4)
    )
    parameters = (router, writer, credit_only, unused)
    task = ((router * writer).sin() - 0.3).square().sum()
    credit = (router * (writer + credit_only).cos()).sum()
    task_grads = torch.autograd.grad(task, parameters, retain_graph=True, allow_unused=True)
    all_grads = torch.autograd.grad(task + credit, parameters, retain_graph=True, allow_unused=True)
    objective = _query_credit_objective(task, credit, (router, unused))
    torch.testing.assert_close(objective, task + credit, rtol=0, atol=0)
    objective.backward()
    for parameter, expected in zip(parameters, (all_grads[0], *task_grads[1:]), strict=True):
        if expected is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected, rtol=1e-12, atol=1e-12)


def test_trainer_partition_clip_matches_probe_oracle() -> None:
    parameters = tuple(torch.nn.Parameter(torch.ones(1)) for _ in range(4))
    for parameter, value in zip(parameters[:3], (100.0, 3.0, 4.0), strict=True):
        parameter.grad = torch.tensor([value])
    norms = _clip_training_gradients(parameters, parameters[:1], norm=1.0, mode="routing-task")
    assert norms == {"routing": 100.0, "task": 5.0}
    torch.testing.assert_close(parameters[0].grad, torch.tensor([1.0]))
    torch.testing.assert_close(parameters[1].grad, torch.tensor([0.6]))
    torch.testing.assert_close(parameters[2].grad, torch.tensor([0.8]))
    assert parameters[3].grad is None
