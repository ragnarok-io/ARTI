import copy

import pytest
import torch

from benchmarks.probe_qwen_federated_task_gradients import balance, predicted_adam
from benchmarks import train_qwen_federated_autonomous_federation as training


def test_balance_keeps_direction_and_large_finite_norms():
    main = (torch.tensor([1e25, 0.0]), None)
    exploration = (torch.tensor([-2e25, 1e25]), torch.tensor([3e25]))
    combined = (main[0] + exploration[0], exploration[1])
    row = balance(main, exploration, combined)
    assert row['linearity_relative_error'] == 0.0
    assert row['combined_projection_on_main'] == pytest.approx(-1.0)
    assert row['cosine_main_exploration'] == pytest.approx(-2 / (14 ** 0.5))
    assert 0 < row['clip_coefficient'] < 1e-24


def test_predicted_adam_matches_inherited_optimizer_without_mutating_it():
    parameter = torch.nn.Parameter(torch.tensor([0.3, -1.4, 2.5]))
    optimizer = torch.optim.AdamW([parameter], lr=0.001, weight_decay=0.0001)
    for step in range(3):
        parameter.grad = torch.tensor([0.1, -0.2, 0.3]) * (step + 1)
        optimizer.step()
    before = parameter.detach().clone()
    state = copy.deepcopy(optimizer.state_dict())
    a, b = torch.tensor([1.1, -0.7, 0.2]), torch.tensor([-0.3, 0.2, 0.8])
    combined, coefficient = a + b, 0.4
    predicted = predicted_adam(
        optimizer, (('parameter', parameter),), (a,), (b,), (combined,), coefficient,
    )
    assert torch.equal(parameter, before)
    for key, value in state['state'][0].items():
        assert torch.equal(value, optimizer.state[parameter][key])
    parameter.grad = combined * coefficient
    optimizer.step()
    delta = parameter.detach() - before
    assert predicted['delta_norm'] == pytest.approx(float(delta.norm()), rel=2e-4)
    assert predicted['main_directional_derivative'] == pytest.approx(float(a @ delta), rel=2e-4)
    assert predicted['exploration_directional_derivative'] == pytest.approx(float(b @ delta), rel=2e-4)


def test_count_surrogate_cannot_suppress_ordinary_task_gradient_when_partitioned():
    routing, count, task = (torch.nn.Parameter(torch.ones(2)) for _ in range(3))
    routing.grad, count.grad, task.grad = (
        torch.tensor([3.0, 4.0]), torch.tensor([1e14, 0.0]), torch.tensor([0.0, 2.0]),
    )
    row = training._clip_training_gradients(
        (routing, count, task), (routing,), norm=1.0, mode='routing-count-task',
        execution_counts=(count,),
    )
    assert row == pytest.approx({'routing': 5.0, 'task': 2.0, 'count': 1e14})
    torch.testing.assert_close(routing.grad, torch.tensor([0.6, 0.8]))
    torch.testing.assert_close(count.grad, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(task.grad, torch.tensor([0.0, 1.0]))


def test_count_partition_includes_atom_owned_and_generic_counts():
    from benchmarks._federated_v4_federation import build_autonomous_effect_federation

    federation = build_autonomous_effect_federation(
        hidden_dim=4, rank=4, seed=731, device=torch.device('cpu'), plastic_branches=4,
        min_operations=1, max_operations=3, max_effect_operations=2,
    )
    counts = training._execution_count_parameters(federation)
    assert {id(p) for p in counts} == {id(effect.execution_count_tensor()) for effect in federation.effects}
    assert len(counts) == len(federation.effects)
    assert any(effect.execution_count is None for effect in federation.effects)
    assert any(effect.execution_count is not None for effect in federation.effects)
    assert {id(p) for p in counts}.isdisjoint(id(p) for p in training._routing_parameters(federation))
