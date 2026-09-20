from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch

from benchmarks._federated_captured_search import captured_search_execution
from benchmarks import evaluate_federated_lifetime as lifetime
from benchmarks.train_federated_branch_visible_federation import _module_digest
from benchmarks.train_federated_interacting_learners import interact, interact_many
from test_federated_retained_training import _episode, _learners


def _stages(device="cpu", factor=1.0):
    first = _episode(device, factor=factor)
    return first, replace(first, episode_id=f"{factor}/replacement", supports=(first.supports[1] * 0.7,), updated_key=0)


def test_lifetime_targets_use_only_latest_observed_values_and_keep_other_pair():
    stages = lifetime.make_lifetime(seed=907001, hidden_dim=8, device="cpu", replacements=4)
    repeated = lifetime.make_lifetime(seed=907001, hidden_dim=8, device="cpu", replacements=4)
    current = [value[:, 1:].clone() for value in stages[0].supports]
    for stage, (item, same) in enumerate(zip(stages, repeated, strict=True)):
        torch.testing.assert_close(item.supports, same.supports, rtol=0, atol=0)
        torch.testing.assert_close(item.queries, stages[0].queries, rtol=0, atol=0)
        if stage:
            changed_task = (stage - 1) % 2
            key = 0 if changed_task == 0 else 2
            assert len(item.supports) == 1
            torch.testing.assert_close(item.supports[0][:, :1], stages[0].supports[key][:, :1], rtol=0, atol=0)
            current[key] = item.supports[0][:, 1:]
            assert not torch.equal(item.targets[changed_task], stages[stage - 1].targets[changed_task])
            assert torch.equal(item.targets[1 - changed_task], stages[stage - 1].targets[1 - changed_task])
        for i, (left, right) in enumerate(((0, 1), (1, 2))):
            expected = item.queries[i] + (current[left] + current[right]) / 2**0.5
            torch.testing.assert_close(item.targets[i], expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("mode", ("ordered-operators", "parallel-delta"))
def test_initial_states_carry_support_but_never_alias_probe_successors(device, mode):
    learners = _learners(device)
    for learner in learners:
        learner.query.requires_grad_(False)
    stages = _stages(device)
    options = dict(width=2, plasticity_participation="retained", plasticity_composition=mode)
    before = tuple(_module_digest(learner.query) for learner in learners)
    first = interact(learners, stages[0], **options)
    carry = lifetime.support_carry((first,))[0]
    snapshots = [tuple(value.clone() for value in state.values) for state in carry]
    later = interact(learners, stages[1], initial_bank_states=carry, **options)
    for i, history in enumerate(later.events):
        assert history[0].before is carry[i]
        assert sum(carry[i].revisions) == 6
        assert sum(history[later.service_index].before.revisions) == 9
        assert sum(first.events[i][-1].after.revisions) == 12
        torch.testing.assert_close(carry[i].values, snapshots[i], rtol=0, atol=0)
    fresh = interact(learners, stages[1], **options)
    assert all(sum(history[0].before.revisions) == 0 for history in fresh.events)
    assert before == tuple(_module_digest(learner.query) for learner in learners)


def test_continuing_stream_discards_both_query_arms_and_uses_same_pre_support_snapshot(monkeypatch):
    learners = _learners()
    for learner in learners:
        learner.query.requires_grad_(False)
    calls = []

    def observed(*args, **kwargs):
        result = interact_many(*args, **kwargs)
        calls.append((kwargs["initial_bank_states"], result))
        return result

    monkeypatch.setattr(lifetime, "interact_many", observed)
    rows, final = lifetime.evaluate_lifetimes(learners, (_stages(),), width=2)
    assert len(calls) == 4
    assert calls[0][0] is calls[1][0]
    assert calls[2][0] is calls[3][0]
    for i in range(2):
        assert calls[2][0][0][i] is calls[0][1][0].events[i][2].before
        assert calls[2][0][0][i] is not calls[0][1][0].events[i][-1].after
        assert final[0][i] is calls[2][1][0].events[i][1].before
        assert final[0][i] is not calls[3][1][0].events[i][-1].after
    assert [r["task_role"] for r in rows[1]["rows"][0]["learners"]] == ["adapt", "retain"]
    assert all(r["before_revision_sum"] == 6 and r["carry_revision_sum"] == 9
               for r in rows[1]["rows"][0]["learners"])


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_lifetime_batch_capture_and_saved_reload_match_serial(device, tmp_path):
    learners = _learners(device)
    for learner in learners:
        learner.query.requires_grad_(False)
    sequences = (_stages(device), _stages(device, -0.7))
    options = dict(width=2, plasticity_composition="parallel-delta")
    expected = [lifetime.evaluate_lifetimes(learners, (sequence,), **options) for sequence in sequences]
    scope = captured_search_execution(horizon=8, record_effect_operands=True) if device == "cuda" else nullcontext()
    with scope as backend:
        actual, states = lifetime.evaluate_lifetimes(learners, sequences, **options)
        if backend is not None:
            assert backend.completed == 44 and not backend.fallbacks
    for stage, row in enumerate(actual):
        for e, episode in enumerate(row["rows"]):
            for i, values in enumerate(episode["learners"]):
                reference = expected[e][0][stage]["rows"][0]["learners"][i]
                for key in ("mse", "without_support_write_mse", "write_gain", "bank_norm", "bank_change_norm"):
                    assert values[key] == pytest.approx(reference[key], rel=1e-5, abs=1e-6)
    for e, pair in enumerate(states):
        for i, state in enumerate(pair):
            torch.testing.assert_close(state.values, expected[e][1][0][i].values)
    lifetime.save_carry(tmp_path / "carry", states)
    restored = lifetime.load_carry(tmp_path / "carry", states)
    for pair, other in zip(states, restored, strict=True):
        for a, b in zip(pair, other, strict=True):
            assert a.slot_refs == b.slot_refs and a.revisions == b.revisions
            torch.testing.assert_close(a.values, b.values, rtol=0, atol=0)
    probes = tuple(replace(sequence[-1], supports=()) for sequence in sequences)
    before = interact_many(learners, probes, initial_bank_states=states, plasticity_participation="retained", **options)
    after = interact_many(learners, probes, initial_bank_states=restored, plasticity_participation="retained", **options)
    for a, b in zip(before, after, strict=True):
        torch.testing.assert_close(tuple(h[-1].output for h in a.events), tuple(h[-1].output for h in b.events),
                                   rtol=0, atol=0)


def test_initial_state_axes_must_match_episode_and_learner_axes():
    learners = _learners()
    states = tuple(learner.query.initial_bank_state() for learner in learners)
    with pytest.raises(ValueError, match="align"):
        interact_many(learners, (_episode(),), initial_bank_states=())
    with pytest.raises(ValueError, match="align"):
        interact_many(learners, (_episode(),), initial_bank_states=(states[:1],))


def test_updated_key_is_per_episode_evaluation_metadata_not_model_input():
    learners = _learners()
    for learner in learners:
        learner.query.eval().requires_grad_(False)
    first = _stages()
    second = (first[0], replace(first[1], updated_key=2))
    rows, _ = lifetime.evaluate_lifetimes(learners, (first, second), width=2)
    changed = rows[1]
    assert changed["changed_keys"] == [0, 2]
    assert [r["task_role"] for r in changed["rows"][0]["learners"]] == ["adapt", "retain"]
    assert [r["task_role"] for r in changed["rows"][1]["learners"]] == ["retain", "adapt"]
    for left, right in zip(changed["rows"][0]["learners"], changed["rows"][1]["learners"], strict=True):
        for field in ("mse", "without_support_write_mse", "bank_norm", "bank_change_norm"):
            assert left[field] == right[field]
