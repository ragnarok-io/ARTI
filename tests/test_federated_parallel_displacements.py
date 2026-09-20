from dataclasses import replace

import pytest
import torch

from benchmarks._federated_plasticity_participation import compose_participant_proposals
from test_federated_plasticity_participation import _fork_query


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_parallel_fork_telescopes_shared_prefix_and_retains_both_gradient_signs(device, monkeypatch):
    query, first, shared, left, right, *_ = _fork_query(device)
    initial = query.initial_bank_state()
    x = torch.tensor([[0.25, -0.4, 0.7]], device=device, requires_grad=True)
    head = shared(first(query._arena({"x": x}, bank_state=initial)))
    a, b = left(head), right(head)
    paths = (a.proposals, b.proposals)
    shared_value = head.proposals[-1].successor
    reference = a.proposals[-1].successor + b.proposals[-1].successor - shared_value

    def no_reapplication(*args, **kwargs):
        raise AssertionError("parallel displacement must not regenerate an operator")

    monkeypatch.setattr(type(a.proposals[-1].transition), "apply", no_reapplication)
    actual = compose_participant_proposals(initial, paths, mode="parallel-delta", participation_probe=True)
    slot = first.bank_slot_ref
    torch.testing.assert_close(actual.state.value(slot), reference)
    assert actual.shared_references == 1 and len(actual.occurrences) == 3
    assert actual.displacement_contributions == 3 and actual.recomposed_transitions == 0
    assert actual.state.revision(slot) == 3 and initial.revision(slot) == 0
    params = (x, *tuple(shared.parameters()), *tuple(left.parameters()), *tuple(right.parameters()))
    expected = torch.autograd.grad(reference.square().mean(), params, retain_graph=True, allow_unused=True)
    value = actual.state.value(slot)
    gradients = torch.autograd.grad(value.square().mean(), params, retain_graph=True, allow_unused=True)
    for a_grad, b_grad in zip(gradients, expected, strict=True):
        assert (a_grad is None) == (b_grad is None)
        if a_grad is not None:
            torch.testing.assert_close(a_grad, b_grad, rtol=1e-5, atol=1e-6)

    # Probe credit is the final-task sensitivity to each executed displacement.
    marginal = torch.autograd.grad(value.square().mean(), actual.participation_probe)[0]
    expected_marginal = torch.stack([
        (2 * value.detach() / value.numel() * (p.successor - p.previous).detach()).sum()
        for p in actual.occurrences
    ])
    torch.testing.assert_close(marginal, expected_marginal)
    single = compose_participant_proposals(initial, (paths[0],), mode="parallel-delta")
    torch.testing.assert_close(single.state.value(slot), paths[0][-1].successor)
    reversed_ = compose_participant_proposals(initial, paths[::-1], mode="parallel-delta")
    torch.testing.assert_close(reversed_.state.value(slot), actual.state.value(slot))
    average = (paths[0][-1].successor + paths[1][-1].successor) / 2
    assert not torch.allclose(value, average)


def test_parallel_empty_and_distinct_equal_occurrences():
    query, first, shared, *_ = _fork_query()
    initial = query.initial_bank_state()
    assert compose_participant_proposals(initial, (), mode="parallel-delta").state is initial
    root = first(query._arena({"x": torch.ones(1, 3)}, bank_state=initial))
    a, b = shared(root).proposals[0], shared(root).proposals[0]
    actual = compose_participant_proposals(initial, ((a,), (b,)), mode="parallel-delta")
    assert len(actual.occurrences) == 2 and actual.shared_references == 0
    torch.testing.assert_close(actual.state.value(first.bank_slot_ref),
                               initial.value(first.bank_slot_ref) + 2 * (a.successor - a.previous))
    untouched = initial.slot_refs[-1]
    if untouched != first.bank_slot_ref:
        assert actual.state.value(untouched) is initial.value(untouched)


def test_parallel_does_not_hide_nonfinite_displacement():
    query, first, shared, *_ = _fork_query()
    initial = query.initial_bank_state()
    root = first(query._arena({"x": torch.ones(1, 3)}, bank_state=initial))
    proposal = shared(root).proposals[0]
    invalid = replace(proposal, successor=torch.full_like(proposal.successor, float("inf")))
    with pytest.raises(ValueError, match="non-finite"):
        compose_participant_proposals(initial, ((invalid,),), mode="parallel-delta")


def test_inactive_parallel_recipe_cannot_be_labelled_as_executed():
    from benchmarks.train_federated_interacting_learners import interact_many, train_minibatch
    from benchmarks.evaluate_federated_interacting_learners import reset_service_outputs_many
    from test_federated_retained_training import _learners, _episode

    learners, episodes = _learners(), (_episode(),)
    for call in (
        lambda: interact_many(learners, episodes, plasticity_composition="parallel-delta"),
        lambda: train_minibatch(learners, episodes, route_credit="episode-beam", plasticity_composition="parallel-delta"),
        lambda: reset_service_outputs_many(learners, episodes, width=2, plasticity_composition="parallel-delta"),
    ):
        with pytest.raises(ValueError, match="requires retained"):
            call()
