from dataclasses import replace

import pytest
import torch

from benchmarks._federated_product_replay import replay_cooperative_dependencies_many
from benchmarks._federated_recursive_search import search_cooperative_graphs, start_recursive_search
from benchmarks._federated_window_risk import RecordedWindowEvent, backward_cooperative_window_risk
from test_formula_program_query_v6 import federation
from test_formula_program_query_v7 import cooperative


def record_window(model, inputs, *, initial=None):
    events = []
    state = model.initial_bank_state() if initial is None else initial
    with torch.no_grad():
        for index, x in enumerate(inputs):
            result = search_cooperative_graphs((start_recursive_search(model, {"x": x}, bank_state=state),),
                                               product_slots=(), width=4, beam_width=4)
            endpoints = tuple(branch.dependency_endpoint for branch in result.branches)
            # Test fixture selects a real writing path; production selection is the caller's policy.
            selected = next(i for i, endpoint in enumerate(endpoints) if endpoint.writes)
            loss = (lambda run: run.outputs["result"].sum() * 0) if index == 0 else (
                lambda run: (run.outputs["result"] - 0.4).square().mean()
                + 0.6 * (run.bank_state.values[0] - 0.8).square().mean())
            event = RecordedWindowEvent({"x": x}, result.dependency_tape, endpoints, selected, loss)
            events.append(event)
            run, = replay_cooperative_dependencies_many(model, event.values, tape=event.tape,
                endpoints=(endpoints[selected],), initial_states=(state,))
            state = run.bank_state
    return tuple(events)


def direct_risk(model, events, *, temperature, initial=None, detach_between=False):
    state = model.initial_bank_state() if initial is None else initial
    risks = []
    for event in events:
        runs = replay_cooperative_dependencies_many(model, event.values, tape=event.tape,
            endpoints=event.endpoints, initial_states=(state,), score_decisions=True)
        scores = torch.stack([run.decision_energy for run in runs]).double()
        losses = torch.stack([event.loss(run) for run in runs]).double()
        risks.append((torch.softmax(scores / temperature, 0) * losses).sum())
        state = runs[event.selected_return].bank_state
        if detach_between:
            state = replace(state, values=tuple(value.detach() for value in state.values))
    return torch.stack(risks).mean(), state


@pytest.mark.parametrize("length", [2, 3])
@pytest.mark.parametrize("temperature", [0.7, 1.0])
def test_window_gradient_matches_direct_conditional_risk(length, temperature):
    model = cooperative(federation(write=True, shared=True)).double()
    inputs = tuple(torch.full((1, 1), 1.0 + i * 0.2, dtype=torch.float64, requires_grad=True)
                   for i in range(length))
    events = record_window(model, inputs)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    direct, final_state = direct_risk(model, events, temperature=temperature)
    expected = torch.autograd.grad(direct, (*parameters, *inputs), allow_unused=True)
    assert expected[len(parameters)].abs().sum() > 0  # First event has no own task loss.
    report = backward_cooperative_window_risk(model, events, temperature=temperature)
    torch.testing.assert_close(report.risk, direct)
    for leaf, gradient in zip((*parameters, *inputs), expected, strict=True):
        if gradient is None:
            assert leaf.grad is None
        else:
            torch.testing.assert_close(leaf.grad, gradient)
    for actual, expected_value in zip(report.detached_return_state.values, final_state.values, strict=True):
        torch.testing.assert_close(actual, expected_value)
        assert not actual.requires_grad
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert report.replayed_occurrences > 0 and not report.risk.requires_grad


def test_detaching_between_events_loses_delayed_input_credit():
    model = cooperative(federation(write=True, shared=True)).double()
    inputs = tuple(torch.ones(1, 1, dtype=torch.float64, requires_grad=True) for _ in range(2))
    events = record_window(model, inputs)
    full, _ = direct_risk(model, events, temperature=1)
    delayed, = torch.autograd.grad(full, inputs[0])
    truncated, _ = direct_risk(model, events, temperature=1, detach_between=True)
    lost, = torch.autograd.grad(truncated, inputs[0])
    assert delayed.abs().sum() > 0
    torch.testing.assert_close(lost, torch.zeros_like(lost))


@pytest.mark.parametrize("detach_boundary", [False, True])
def test_explicit_window_boundary_gradient_contract(detach_boundary):
    model = cooperative(federation(write=True, shared=True)).double()
    previous_law = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    inputs = tuple(torch.ones(1, 1, dtype=torch.float64) for _ in range(2))
    initial = model.initial_bank_state()
    values = tuple(value + previous_law.square() for value in initial.values)
    initial = replace(initial, values=tuple(value.detach() for value in values) if detach_boundary else values)
    events = record_window(model, inputs, initial=initial)
    direct, _ = direct_risk(model, events, temperature=1, initial=initial)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    expected = torch.autograd.grad(direct, (*parameters, previous_law), allow_unused=True, retain_graph=True)
    backward_cooperative_window_risk(model, events, initial_state=initial)
    if detach_boundary:
        assert previous_law.grad is None
    else:
        assert previous_law.grad.abs().sum() > 0
    for parameter, gradient in zip((*parameters, previous_law), expected, strict=True):
        if gradient is not None:
            torch.testing.assert_close(parameter.grad, gradient)


def test_empty_window_is_not_a_training_step():
    with pytest.raises(ValueError, match="at least one"):
        backward_cooperative_window_risk(None, ())
