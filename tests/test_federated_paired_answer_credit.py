from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from benchmarks.train_qwen_federated_autonomous_federation import (
    _episode_pairs,
    _paired_answer_separation,
    _paired_batch_indices,
)


def _item(pair, answer, prompt=(1, 2)):
    return SimpleNamespace(
        episode=SimpleNamespace(pair_id=pair),
        student_prompt_ids=torch.tensor([prompt]),
        answer_ids=torch.tensor([answer]),
    )


def test_pair_sampler_uses_ids_and_covers_complete_pairs_per_epoch():
    cached = (_item("a", [1]), _item("b", [2]), _item("a", [3]), _item("b", [4]),
              _item("c", [1]), _item("d", [2]), _item("d", [3]), _item("c", [4]))
    pairs = _episode_pairs(cached)
    assert pairs == ((0, 2), (1, 3), (4, 7), (5, 6))
    epoch = [_paired_batch_indices(pairs, step=step, batch_size=4, seed=7) for step in (1, 2)]
    assert sorted(index for batch in epoch for index in batch) == list(range(8))
    assert epoch == [_paired_batch_indices(pairs, step=step, batch_size=4, seed=7) for step in (1, 2)]
    for batch in epoch:
        assert tuple(batch[:2]) in pairs and tuple(batch[2:]) in pairs


@pytest.mark.parametrize("batch_size", (0, 1, 3))
def test_pair_sampler_rejects_split_batches(batch_size):
    with pytest.raises(ValueError, match="even"):
        _paired_batch_indices(((0, 1),), step=1, batch_size=batch_size, seed=0)


def test_separation_distinguishes_history_from_shared_answer_bias():
    cached = (_item("a", [0, 1, 4]), _item("a", [0, 2, 4]))
    common = torch.tensor([[9., 0., 0., 0., 0.], [0., 8., 1., 0., 0.], [0., 0., 0., 0., 9.]])
    shared = _paired_answer_separation((common, common.clone()), cached)
    assert shared["both_histories_prefer_own_token_fraction"] == 0
    assert shared["mean_correct_token_log_odds"] == 0
    different = common.clone()
    different[1, 1], different[1, 2] = 1, 8
    separated = _paired_answer_separation((common, different), cached)
    assert separated["both_histories_prefer_own_token_fraction"] == 1
    assert separated["mean_correct_token_log_odds"] == 7
    assert separated["rows"][0]["first_distinct_position"] == 1
    assert separated["rows"][0]["target_ranks"] == [1, 1]
    swapped = _paired_answer_separation((different, common), cached)
    assert swapped["mean_correct_token_log_odds"] == -7


def test_pair_separation_rejects_different_prompts():
    cached = (_item("a", [1]), _item("a", [2], prompt=(1, 3)))
    with pytest.raises(ValueError, match="same latest prompt"):
        _episode_pairs(cached)


def test_pair_separation_does_not_silently_drop_prefix_only_answers():
    cached = (_item("a", [0, 1]), _item("a", [0, 1, 2]))
    with pytest.raises(ValueError, match="distinct encoded"):
        _paired_answer_separation((torch.zeros(2, 3), torch.zeros(3, 3)), cached)
