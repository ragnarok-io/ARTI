from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from benchmarks import train_federated_interacting_learners as training
from benchmarks._federated_captured_search import captured_search_execution
from benchmarks._federated_transfer_credit import make_transfer_batch, paired_risk_weights
from test_federated_retained_training import _episode, _learners


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_pair_weights_match_positive_risk_derivatives_and_zero_strength(device):
    losses = torch.tensor([0.1, 0.12, 0.2, 0.1], device=device, requires_grad=True)
    r, u = losses.reshape(-1, 2).unbind(-1)
    risk = (0.5 * (r + u) + 0.2 * (r - 0.95 * u).clamp_min(0)).mean()
    gradient, = torch.autograd.grad(risk, losses)
    weights, observed, active = paired_risk_weights(losses)
    torch.testing.assert_close(weights / len(losses), gradient)
    torch.testing.assert_close(observed.mean(), risk)
    assert not weights.requires_grad and torch.all(weights > 0)
    assert active.tolist() == [0, 1]
    zero, _, _ = paired_risk_weights(losses, strength=0)
    torch.testing.assert_close(zero, torch.ones_like(losses), rtol=0, atol=0)


@pytest.mark.parametrize("recipe", ("transfer-paired", "transfer-stream"))
def test_training_recipe_pairs_same_current_data_and_seed(tmp_path, monkeypatch, recipe):
    batches = []
    monkeypatch.setattr(training, "build_learners", lambda **kw: _learners("cpu"))
    monkeypatch.setattr(training, "save_round", lambda *a, **kw: None)

    def train(learners, episodes, **options):
        batches.append((episodes, options))
        return (), {}

    monkeypatch.setattr(training, "train_minibatch", train)
    args = SimpleNamespace(seed=9, hidden_dim=8, branches=2, device="cpu", max_operations=1,
                           max_effect_operations=1, width=2, message_credit="trajectory-input-vjp",
                           route_credit="retained-vjp", candidate_exploration="gumbel", plasticity_composition="parallel-delta",
                           episode_batch_size=4, episode_execution="batched", rounds=4, resume=None,
                           output_dir=tmp_path, numeric_backend="reference", query_backend="reference",
                           search_backend="reference", gradient_scale=None, episode_task=recipe, max_replacements=4)
    training._train(args, 0)
    expected = [2, 3, 4, 1] if recipe == "transfer-stream" else [1, 2, 3, 1]
    assert [len(batch[0].supports) - 6 for batch, _ in batches] == expected
    for episodes, options in batches:
        assert options["paired_transfer"]
        assert [e.updated_key for e in episodes] == [0, 0, 2, 2]
        seeds = options["exploration_seeds"]
        assert seeds[0] == seeds[1] and seeds[2] == seeds[3] and seeds[0] != seeds[2]
        for related, unrelated in zip(episodes[::2], episodes[1::2], strict=True):
            torch.testing.assert_close(related.supports[5:], unrelated.supports[5:], rtol=0, atol=0)
            torch.testing.assert_close(related.targets, unrelated.targets, rtol=0, atol=0)
            torch.testing.assert_close(related.queries, unrelated.queries, rtol=0, atol=0)
            assert not torch.equal(related.supports[3], unrelated.supports[3])
            if recipe == "transfer-stream" and len(related.supports) > 7:
                assert not torch.equal(related.supports[-1], related.supports[-2])


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_real_paired_backward_scales_entire_objective_and_excludes_answer_bank(device, monkeypatch):
    learners = _learners(device)
    first = _episode(device)
    second = replace(first, supports=(first.supports[0] * 0.7, *first.supports[1:]))
    gradients = []
    original = training.learner_objective

    def observed(*args, **kwargs):
        objective, row = original(*args, **kwargs)
        objective.register_hook(lambda gradient: gradients.append(gradient.detach().cpu()))
        return objective, row

    monkeypatch.setattr(training, "learner_objective", observed)
    with captured_search_execution(horizon=8, record_effect_operands=True) if device == "cuda" else nullcontext():
        _, report = training.train_minibatch(
            learners, (first, second), width=2, route_credit="retained-vjp",
            message_credit="trajectory-input-vjp", plasticity_composition="parallel-delta",
            paired_transfer=True, gradient_scale=2**-20,
        )
    credit = report["paired_transfer_credit"]
    assert credit["kind"] == "paired_hard_mse_stopgrad_surrogate"
    expected = torch.tensor(credit["episode_multipliers"]) / 2
    torch.testing.assert_close(torch.stack(gradients), expected.repeat(2))
    for learner in learners:
        assert all(owner.value.grad is None for owner in learner.query.owner_states)
        assert any(p.grad is not None for p in learner.query.parameters())
        assert all(torch.isfinite(p).all() for p in learner.query.parameters())


def test_paired_batch_rejects_unpaired_size():
    with pytest.raises(ValueError, match="even"):
        make_transfer_batch(seed=1, hidden_dim=8, device="cpu", presentations=1, batch_size=3)
