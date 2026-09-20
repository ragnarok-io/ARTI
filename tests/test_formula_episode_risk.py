import pytest
import torch

from arti import mechanisms as m
from benchmarks._federated_endpoint_risk import endpoint_risk_coefficients
from benchmarks._federated_episode_risk import (
    RecordedEpisodeStep, backward_cooperative_episode_risk, replay_cooperative_episode_panel,
)
from benchmarks._federated_product_replay import replay_cooperative_dependencies_many
from benchmarks._federated_recursive_search import search_cooperative_graphs, start_recursive_search
from test_formula_program_query_v6 import federation, member
from test_formula_program_query_v7 import cooperative


def recorded_panel(model, inputs, *, first_width=4):
    with torch.no_grad():
        first = search_cooperative_graphs((start_recursive_search(model, inputs[0]),),
                                          product_slots=(), width=first_width, beam_width=first_width)
        paths = []
        for branch in first.branches:
            step = RecordedEpisodeStep(first.dependency_tape, branch.dependency_endpoint)
            run, = replay_cooperative_dependencies_many(model, inputs[0], tape=step.tape,
                endpoints=(step.endpoint,))
            second = search_cooperative_graphs((start_recursive_search(model, inputs[1], bank_state=run.bank_state),),
                                               product_slots=(), width=4, beam_width=4)
            paths.extend((step, RecordedEpisodeStep(second.dependency_tape, tail.dependency_endpoint))
                         for tail in second.branches[:2])
    return tuple(paths)


def loss(run):
    return (run.outputs["result"] - 0.4).square().mean() + 0.6 * (run.bank_state.values[0] - 0.8).square().mean()


def serial_reference(model, inputs, paths):
    energies, losses, work = [], [], 0
    for path in paths:
        state, event_scores = model.initial_bank_state(), []
        for values, step in zip(inputs, path, strict=True):
            run, = replay_cooperative_dependencies_many(model, values, tape=step.tape,
                endpoints=(step.endpoint,), initial_states=(state,), score_decisions=True)
            state = run.bank_state
            event_scores.append(run.decision_energy)
            work += len(run.executed_occurrences)
        energies.append(torch.stack(event_scores).sum())
        losses.append(loss(run))
    return torch.stack(energies), torch.stack(losses), work


@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("gradient_scale", [1.0, 0.25])
def test_complete_episode_risk_matches_independent_path_gradients(temperature, gradient_scale):
    model = cooperative(federation(write=True, shared=True)).double()
    inputs = tuple({"x": torch.full((1, 1), value, dtype=torch.float64, requires_grad=True)} for value in (1., 0.7))
    paths = recorded_panel(model, inputs)
    leaves = (*tuple(p for p in model.parameters() if p.requires_grad), *(values["x"] for values in inputs))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    scores, losses, serial_work = serial_reference(model, inputs, paths)
    direct = (torch.softmax(scores.double() / temperature, 0) * losses.double()).sum()
    expected = torch.autograd.grad(direct, leaves, allow_unused=True)
    report = backward_cooperative_episode_risk(model, inputs, paths, terminal_loss=loss,
                                              temperature=temperature, gradient_scale=gradient_scale)
    torch.testing.assert_close(report.risk, direct)
    torch.testing.assert_close(report.energies, scores)
    torch.testing.assert_close(report.terminal_losses, losses)
    assert report.replayed_occurrences < 2 * serial_work
    for leaf, gradient in zip(leaves, expected, strict=True):
        if gradient is None:
            assert leaf.grad is None
        else:
            torch.testing.assert_close(leaf.grad, gradient * gradient_scale)
    assert inputs[0]["x"].grad.abs().sum() > 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(not value.requires_grad for state in report.detached_return_states for value in state.values)


def test_identical_early_bank_outcomes_do_not_create_spurious_choice_credit():
    model = cooperative(federation(write=True, shared=True)).double()
    inputs = tuple({"x": torch.full((1, 1), value, dtype=torch.float64, requires_grad=True)} for value in (1., 0.7))
    paths = recorded_panel(model, inputs)
    panel = replay_cooperative_episode_panel(model, inputs, paths)
    losses = torch.stack([loss(run) for run in panel.terminals])
    coefficients = endpoint_risk_coefficients(panel.energies, losses)
    early_choice = (coefficients.choice * panel.event_energies[:, 0]).sum()
    gradient, = torch.autograd.grad(early_choice, inputs[0]["x"])
    assert losses.max() > losses.min()
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)


def test_final_losses_assign_credit_to_distinct_early_bank_outcomes():
    original = cooperative(federation(write=True, shared=True))
    left = member("prepare_left", "x", "g", weight=0.7)
    right = member("prepare_right", "x", "g", weight=-0.3)
    scout = member("scout", "x", "r", weight=0.2)
    owner = member("a", "g", "h", owner="a")
    model = m.FormulaProgramQueryV7(
        slot_ids=(*original.slot_ids, "g", "g_negative", "r", "r_negative"),
        candidates=(scout, left, right, owner, *original.candidates[1:]),
        terminal_slots=original.terminal_slots, entry_candidates=("scout",),
        continuations={**original.continuations,
                       "scout": {"prepare_left": "r", "prepare_right": "r_negative"},
                       "prepare_left": {"a": "g"}, "prepare_right": {"a": "g"}},
        max_steps=original.max_steps + 2, cooperation_width=original.cooperation_width).double()
    inputs = tuple({"x": torch.full((1, 1), value, dtype=torch.float64, requires_grad=True)} for value in (1., 0.7))
    paths = recorded_panel(model, inputs, first_width=16)
    early_banks = [replay_cooperative_dependencies_many(model, inputs[0], tape=path[0].tape,
                   endpoints=(path[0].endpoint,))[0].bank_state.values[0].detach().sum().item() for path in paths]
    assert len(set(early_banks)) > 1
    panel = replay_cooperative_episode_panel(model, inputs, paths)
    losses = torch.stack([loss(run) for run in panel.terminals])
    coefficients = endpoint_risk_coefficients(panel.energies, losses)
    early_choice = (coefficients.choice * panel.event_energies[:, 0]).sum()
    assert len({float(run.bank_state.values[0].detach().sum()) for run in panel.terminals}) > 1
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    gradients = torch.autograd.grad(early_choice, parameters, allow_unused=True)
    assert sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None) > 1e-8


def test_shared_prefix_bank_is_not_reapplied_per_descendant():
    model = cooperative(federation(write=True, shared=True)).double()
    inputs = ({"x": torch.ones(1, 1, dtype=torch.float64)}, {"x": torch.full((1, 1), 0.7, dtype=torch.float64)})
    paths = recorded_panel(model, inputs)
    panel = replay_cooperative_episode_panel(model, inputs, paths)
    scores, losses, serial_work = serial_reference(model, inputs, paths)
    torch.testing.assert_close(panel.energies, scores)
    torch.testing.assert_close(torch.stack([loss(run) for run in panel.terminals]), losses)
    assert panel.replayed_occurrences < serial_work
    for row, path in enumerate(paths):
        state = model.initial_bank_state()
        for values, step in zip(inputs, path, strict=True):
            run, = replay_cooperative_dependencies_many(model, values, tape=step.tape,
                endpoints=(step.endpoint,), initial_states=(state,))
            state = run.bank_state
        for expected, actual in zip(state.values, panel.terminals[row].bank_state.values, strict=True):
            torch.testing.assert_close(actual, expected)


def test_single_trajectory_final_loss_is_not_divided_by_episode_length():
    model = cooperative(federation(write=True, shared=True)).double()
    inputs = ({"x": torch.ones(1, 1, dtype=torch.float64)}, {"x": torch.full((1, 1), 0.7, dtype=torch.float64)})
    paths = recorded_panel(model, inputs)[:1]
    _, expected, _ = serial_reference(model, inputs, paths)
    report = backward_cooperative_episode_risk(model, inputs, paths, terminal_loss=loss)
    torch.testing.assert_close(report.risk, expected[0].double())


def test_duplicate_trajectories_cannot_silently_reweight_risk():
    model = cooperative(federation(write=True, shared=True)).double()
    inputs = ({"x": torch.ones(1, 1, dtype=torch.float64)},) * 2
    paths = recorded_panel(model, inputs)
    with pytest.raises(ValueError, match="duplicate"):
        replay_cooperative_episode_panel(model, inputs, (paths[0], paths[0]))
