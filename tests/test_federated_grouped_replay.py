from contextlib import nullcontext
import copy
import sys

import pytest
import torch

from arti import mechanisms as m
from arti._formula_grouped_training import _GROUPED_TRAINING, grouped_formula_training
from benchmarks._federated_episode_beam import search_episode_beam, replay_episode_paths
from benchmarks._federated_recursive_search import replay_recursive_graphs


@pytest.mark.parametrize("backend", ["eager", "aot_eager", "captured"])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("recorded", [False, True])
def test_grouped_full_episode_preserves_prefixes_effect_credit_and_unused_gradients(backend, device, recorded):
    from test_federated_recursive_search import _parent, _writer_child, _zero_policy

    parent = _parent((_writer_child(),))
    query = _zero_policy(m.FormulaProgramQueryV5(slot_ids=parent.slot_ids, candidates=tuple(parent.candidates),
        terminal_slots={"output": "answer"}, max_steps=1, hidden_dim=8)).to(device)
    inputs = ({"x": torch.ones(1, 2, device=device)}, {"x": torch.full((1, 2), 0.5, device=device)})
    routes = [branch.route for branch in search_episode_beam(query, inputs, width=16).result.branches]
    parameters = tuple(p for p in query.parameters() if p.requires_grad)

    def run(grouped):
        with grouped_formula_training(backend=backend) if grouped else nullcontext():
            paths, count = replay_episode_paths(query, inputs, routes, use_recorded_choices=recorded)
            losses = torch.stack([(row.output - 0.8).square().mean() for row in paths])
            scores = torch.stack([row.log_probability for row in paths])
            gradients = torch.autograd.grad((scores.softmax(0) * losses).sum(), parameters, allow_unused=True)
            return paths, count, gradients

    expected, expected_count, left = run(False)
    actual, actual_count, right = run(True)
    assert actual_count == expected_count == 6
    assert len({id(path.events[0]) for path in actual}) == 2
    for a, b in zip(expected, actual, strict=True):
        torch.testing.assert_close(a.output, b.output)
        torch.testing.assert_close(a.log_probability, b.log_probability)
        for x, y in zip(a.events, b.events, strict=True):
            assert x.route == y.route
            assert x.execution.bank_state.revisions == y.execution.bank_state.revisions
            torch.testing.assert_close(x.execution.bank_state.values, y.execution.bank_state.values)
    for a, b in zip(left, right, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b)
    assert query.initial_bank_state().revisions == (0,)


def test_grouped_replay_checks_complete_routes():
    from test_federated_recursive_search import _model
    query = _model()
    assert replay_recursive_graphs(()) == ()
    with pytest.raises(ValueError, match="stopped root graph"):
        replay_recursive_graphs(((query, {"x": torch.ones(1, 2)}, (), None),))


@pytest.mark.parametrize("backend", ["eager", "aot_eager", "captured"])
def test_complete_two_learner_round_keeps_adam_and_bank_ownership(backend):
    from test_federated_interacting_learners import _tiny_learners, _episode
    from benchmarks.train_federated_interacting_learners import train_round

    reference = _tiny_learners()
    grouped = copy.deepcopy(reference)
    options = dict(width=4, message_credit="input-vjp", route_credit="episode-beam", exploration_seed=17)
    expected, _ = train_round(reference, _episode(), **options)
    with grouped_formula_training(backend=backend):
        actual, _ = train_round(grouped, _episode(), **options)
    assert [[event.route for event in history] for history in actual.events] == [
        [event.route for event in history] for history in expected.events]
    for a, b in zip(reference, grouped, strict=True):
        torch.testing.assert_close(a.query.state_dict(), b.query.state_dict())
        torch.testing.assert_close(a.optimizer.state_dict(), b.optimizer.state_dict())
        for x, y in zip(a.query.parameters(), b.query.parameters(), strict=True):
            assert (x.grad is None) == (y.grad is None)
            if x.grad is not None:
                torch.testing.assert_close(x.grad, y.grad)
        optimized = {id(value) for group in b.optimizer.param_groups for value in group["params"]}
        assert not optimized.intersection(id(owner.value) for owner in b.query.owner_states)


@pytest.mark.parametrize("backend", ["reference", "eager", "captured"])
@pytest.mark.parametrize("query_backend", ["reference", "eager"])
def test_cli_backend_lives_across_run_and_does_not_change_resume_recipe(monkeypatch, tmp_path, backend, query_backend):
    import benchmarks.train_federated_interacting_learners as harness
    from benchmarks._federated_device_query import _QUERY_WAVES

    config = dict(device="cpu", seed=20260904, hidden_dim=8, branches=32, max_operations=3,
        max_effect_operations=8, width=16, message_credit="input-vjp", route_credit="episode-beam",
        candidate_exploration="ranked")
    called = []

    def build(**_kwargs):
        scope = _GROUPED_TRAINING.get()
        assert (scope is None) == (backend == "reference")
        assert (_QUERY_WAVES.get() is None) == (query_backend == "reference")
        called.append(scope)
        return ()

    monkeypatch.setattr(harness, "build_learners", build)
    monkeypatch.setattr(harness, "load_round", lambda *_: {"step": 7, "config": config})
    monkeypatch.setattr(sys, "argv", ["train", "--output-dir", str(tmp_path), "--resume", str(tmp_path),
        "--rounds", "0", "--device", "cpu", "--numeric-backend", backend, "--query-backend", query_backend])
    harness.main()
    assert len(called) == 1
    assert _GROUPED_TRAINING.get() is None
    assert _QUERY_WAVES.get() is None
