from __future__ import annotations

import pytest
import torch

from benchmarks._federated_v4_federation import build_branch_visible_effect_federation
from benchmarks.train_federated_branch_visible_federation import (
    _EpisodeTrajectory,
    _SearchBranch,
    _ranked_candidates_many,
    advance_support,
    evaluate_frozen_adaptation,
    expected_episode_loss,
)
from benchmarks.train_federated_self_modifying_federation import (
    make_association_episodes,
)


def test_branch_visible_expected_loss_preserves_functional_fast_state() -> None:
    federation = build_branch_visible_effect_federation(
        hidden_dim=4,
        rank=4,
        seed=919,
        device=torch.device("cpu"),
        plastic_branches=3,
        write_stages=2,
        effect_families=("outer", "blend"),
    )
    episode = make_association_episodes(
        split="test",
        seed=111,
        count=1,
        hidden_dim=4,
        support_count=2,
        device=torch.device("cpu"),
    )[0]
    before = tuple(owner.value.clone() for owner in federation.query.owner_states)
    loss, diagnostics = expected_episode_loss(
        federation,
        episode,
        search_width=2,
        effect_width=2,
        beam_width=4,
    )
    loss.backward()

    assert bool(torch.isfinite(loss))
    assert diagnostics["retained_trajectories"] == 4.0
    assert diagnostics["expected_mse"] + 1e-8 >= diagnostics["best_branch_mse"]
    assert diagnostics["hard_alignment_loss"] > 0.0
    assert torch.allclose(
        loss.detach(),
        loss.new_tensor(
            diagnostics["expected_mse"] + 0.1 * diagnostics["hard_alignment_loss"]
        ),
    )
    assert any(parameter.grad is not None for parameter in federation.query.network.parameters())
    assert any(effect.operand_store.tensor("writer").grad is not None for effect in federation.effects)
    assert all(
        torch.equal(previous, owner.value)
        for previous, owner in zip(before, federation.query.owner_states, strict=True)
    )


def test_batched_stage_ranking_matches_full_query_mask() -> None:
    federation = build_branch_visible_effect_federation(
        hidden_dim=4,
        rank=4,
        seed=921,
        device=torch.device("cpu"),
        plastic_branches=4,
        write_stages=2,
        effect_families=("outer", "blend"),
    )
    value = torch.randn(1, 2, 4)
    arena = federation.query._arena({"x": value})
    branch = _SearchBranch(arena, value.new_zeros(()), ())
    ranked = _ranked_candidates_many(
        federation,
        (branch,),
        federation.producer_stages[0],
        steps=0,
        width=len(federation.producer_stages[0]),
    )[0]

    full = federation.query.query(arena, steps=0).masked_logits.log_softmax(dim=-1)[0]
    candidate_indices = {
        candidate.candidate_id: index
        for index, candidate in enumerate(federation.query.candidates)
    }
    expected = sorted(
        federation.producer_stages[0],
        key=lambda candidate: (
            -float(full[candidate_indices[candidate.candidate_id]].detach()),
            candidate.candidate_id,
        ),
    )

    assert [candidate.candidate_id for candidate, _ in ranked] == [
        candidate.candidate_id for candidate in expected
    ]
    assert torch.allclose(
        torch.stack(tuple(probability for _, probability in ranked)),
        torch.stack(
            tuple(full[candidate_indices[candidate.candidate_id]] for candidate in expected)
        ),
    )


@pytest.mark.parametrize("bad_first", (False, True))
def test_query_batch_rebuild_excludes_overflow_rows_from_shared_gradients(monkeypatch, bad_first):
    federation = build_branch_visible_effect_federation(
        hidden_dim=4, rank=4, seed=921, device=torch.device("cpu"),
        plastic_branches=4, write_stages=2, effect_families=("outer", "blend"),
    )
    query = federation.query
    summary_width = query.network[0].in_features

    def summarize(arena):
        value = arena.get("x")
        return value.square().reshape(value.shape[0], -1).mean(-1, keepdim=True).expand(-1, summary_width)

    monkeypatch.setattr(query, "_summarize_values", summarize)
    good = torch.ones(1, 2, 4, requires_grad=True)
    bad = torch.full((1, 2, 4), 1e20, requires_grad=True)
    values = (bad, good) if bad_first else (good, bad)
    branches = tuple(_SearchBranch(query._arena({"x": value}), value.new_zeros(()), ()) for value in values)
    candidates = federation.producer_stages[0]
    rejections = []
    rows = _ranked_candidates_many(
        federation, branches, candidates, steps=0, width=len(candidates),
        numerical_rejections=rejections,
    )
    valid_index = int(bad_first)
    assert rows[1 - valid_index] == ()
    assert rejections == [{"kind": "query", "code": "FF2_NONFINITE", "step": 0, "prefix": []}]
    parameters = (*query.network.parameters(), good, bad)
    gradients = torch.autograd.grad(-rows[valid_index][0][1], parameters, allow_unused=True)
    oracle = _ranked_candidates_many(federation, (branches[valid_index],), candidates,
                                     steps=0, width=len(candidates))[0]
    expected = torch.autograd.grad(-oracle[0][1], parameters, allow_unused=True)
    assert gradients[-1] is None
    assert [candidate.candidate_id for candidate, _ in rows[valid_index]] == [
        candidate.candidate_id for candidate, _ in oracle
    ]
    torch.testing.assert_close(torch.stack([value for _, value in rows[valid_index]]),
                               torch.stack([value for _, value in oracle]), rtol=0, atol=0)
    for actual, reference in zip(gradients, expected, strict=True):
        if reference is None:
            assert actual is None
        else:
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_hard_alignment_weight_must_be_non_negative() -> None:
    federation = build_branch_visible_effect_federation(
        hidden_dim=4,
        rank=4,
        seed=923,
        device=torch.device("cpu"),
        plastic_branches=2,
        write_stages=1,
        effect_families=("outer",),
    )
    episode = make_association_episodes(
        split="test",
        seed=121,
        count=1,
        hidden_dim=4,
        support_count=2,
        device=torch.device("cpu"),
    )[0]

    try:
        expected_episode_loss(
            federation,
            episode,
            search_width=1,
            effect_width=1,
            beam_width=1,
            hard_alignment_weight=-0.1,
        )
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative hard-alignment weight should fail closed")


def test_terminal_wide_search_keeps_every_bank_candidate() -> None:
    federation = build_branch_visible_effect_federation(
        hidden_dim=4,
        rank=4,
        seed=927,
        device=torch.device("cpu"),
        plastic_branches=4,
        write_stages=1,
        effect_families=("outer",),
    )
    episode = make_association_episodes(
        split="test",
        seed=131,
        count=1,
        hidden_dim=4,
        support_count=2,
        device=torch.device("cpu"),
    )[0]

    _loss, diagnostics = expected_episode_loss(
        federation,
        episode,
        search_width=1,
        effect_width=1,
        beam_width=1,
        terminal_width=4,
    )

    assert diagnostics["retained_trajectories"] == 4.0


def test_support_trajectory_commits_multiple_effect_revisions_without_module_mutation() -> None:
    federation = build_branch_visible_effect_federation(
        hidden_dim=4,
        rank=4,
        seed=929,
        device=torch.device("cpu"),
        plastic_branches=2,
        write_stages=2,
        effect_families=("outer",),
    )
    episode = make_association_episodes(
        split="test",
        seed=222,
        count=1,
        hidden_dim=4,
        support_count=2,
        device=torch.device("cpu"),
    )[0]
    initial = federation.query.initial_bank_state()
    trajectories = advance_support(
        federation,
        (_EpisodeTrajectory(initial, episode.query.new_zeros(()), ()),),
        episode.supports[0],
        search_width=1,
        effect_width=1,
        beam_width=1,
    )

    assert len(trajectories) == 1
    assert sum(trajectories[0].bank_state.revisions) == 2
    assert all(owner.revision.item() == 0 for owner in federation.query.owner_states)
    effect_rows = [row for row in trajectories[0].route if row["kind"] == "effect"]
    assert len(effect_rows) == 2
    assert all(row["data_identity"] is True for row in effect_rows)


def test_frozen_forward_controls_are_reported_without_slow_state_mutation() -> None:
    federation = build_branch_visible_effect_federation(
        hidden_dim=4,
        rank=4,
        seed=939,
        device=torch.device("cpu"),
        plastic_branches=2,
        write_stages=2,
        effect_families=("outer",),
    )
    episodes = make_association_episodes(
        split="held-out",
        seed=333,
        count=2,
        hidden_dim=4,
        support_count=2,
        device=torch.device("cpu"),
    )
    report = evaluate_frozen_adaptation(federation, episodes)

    assert report["fresh_replay_exact_fraction"] == 1.0
    assert report["slow_state_digest_unchanged"] is True
    assert set(report["mse"]) == {"correct", "reset", "no_effect", "swap", "scrub"}
