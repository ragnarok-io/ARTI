from __future__ import annotations

import pytest
import torch

from benchmarks.qwen_ttt_validation_contract import (
    CounterfactualDescriptor,
    LatestTensorFixture,
    assert_equal_history_budget,
    assert_same_latest_tensor,
    causal_delta_cosine,
    causal_delta_relative_error,
    causal_margin_loss,
    gather_pair_logits,
    effective_jacobian_rank,
    heldout_reachability_r2,
    history_difference_loss,
    normalized_logit_difference,
    paired_history_deltas,
    paired_answer_margin,
    teacher_pair_quality,
    paired_topk_indices,
    reachability_r2,
    select_counterfactual_pairs,
)


def test_paired_answer_margin_and_teacher_quality() -> None:
    logits = torch.zeros(4, 8)
    answers = torch.tensor([1, 2, 4, 5])
    logits[0, 1] = 3
    logits[0, 2] = 1
    logits[1, 2] = 4
    logits[1, 1] = 0
    logits[2, 4] = 2
    logits[2, 5] = 5
    logits[3, 5] = 1
    logits[3, 4] = 4
    margins = paired_answer_margin(logits, answers)
    assert margins.tolist() == [2.0, 4.0, -3.0, -3.0]
    quality = teacher_pair_quality(logits, answers)
    assert quality["target_margin_positive_rate"].item() == 0.5
    assert quality["pair_top1_difference_rate"].item() == 1.0


def _fixture(*, offset: int = 0) -> LatestTensorFixture:
    ids = torch.tensor([[10, 11, 12]])
    return LatestTensorFixture(
        input_ids=ids,
        attention_mask=torch.ones_like(ids, dtype=torch.bool),
        position_ids=torch.tensor([[offset, offset + 1, offset + 2]]),
        embeddings=torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
    )


def test_latest_fixture_requires_exact_input_and_position_equality() -> None:
    assert_same_latest_tensor(_fixture(), _fixture())
    with pytest.raises(ValueError, match="position_ids"):
        assert_same_latest_tensor(_fixture(), _fixture(offset=1))


def test_history_budget_rejects_length_shortcut() -> None:
    ids = torch.ones(2, 4, dtype=torch.long)
    good = torch.tensor([[True, True, True, False], [True, True, False, False]])
    bad = torch.ones(2, 4, dtype=torch.bool)
    assert_equal_history_budget(ids, good, ids, good.clone())
    with pytest.raises(ValueError, match="equal active lengths"):
        assert_equal_history_budget(ids, good, ids, bad)


def test_causal_margin_and_difference_losses_are_finite() -> None:
    correct = torch.tensor([0.2, 0.3])
    wrong = torch.tensor([1.0, 1.2])
    assert causal_margin_loss(correct, wrong, margin=0.1).item() == pytest.approx(0.0)
    student = torch.tensor([[2.0, 0.0, -1.0], [0.0, 1.0, -2.0]])
    teacher = student * 2.0
    assert torch.isfinite(history_difference_loss(student, teacher))
    assert torch.allclose(
        normalized_logit_difference(student),
        normalized_logit_difference(teacher),
    )


def test_causal_delta_metrics_reject_wrong_direction() -> None:
    teacher = torch.tensor([[3.0, 0.0, -2.0]])
    same = teacher * 2.0
    opposite = -teacher
    assert causal_delta_cosine(same, teacher).item() == pytest.approx(1.0)
    assert causal_delta_cosine(opposite, teacher).item() == pytest.approx(-1.0)
    assert causal_delta_relative_error(same, teacher).item() == pytest.approx(1.0)
    assert causal_delta_relative_error(teacher, teacher).item() == pytest.approx(0.0)
    assert history_difference_loss(opposite, teacher).item() > 1.5


def test_paired_topk_delta_uses_one_coordinate_set_for_both_histories() -> None:
    logits = torch.tensor(
        [
            [9.0, 1.0, 0.0, -1.0, -2.0],
            [8.0, 0.0, 2.0, -1.0, -3.0],
            [0.0, 7.0, 1.0, -2.0, -4.0],
            [0.0, 6.0, 2.0, -1.0, -3.0],
        ]
    )
    indices = paired_topk_indices(logits, topk=2)
    selected = gather_pair_logits(logits, indices)
    deltas = paired_history_deltas(logits, indices)
    assert indices.shape == (2, 4)
    assert selected.shape == (2, 2, 4)
    assert deltas.shape == (4, 4)
    assert torch.allclose(deltas[0], -deltas[1])
    assert torch.allclose(deltas[2], -deltas[3])


def test_paired_delta_loss_propagates_to_both_history_rows() -> None:
    teacher = torch.randn(4, 17)
    student = torch.randn(4, 17, requires_grad=True)
    indices = paired_topk_indices(teacher, topk=4)
    loss = history_difference_loss(
        paired_history_deltas(student, indices),
        paired_history_deltas(teacher, indices),
    )
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert bool(student.grad[0].abs().sum())
    assert bool(student.grad[1].abs().sum())


def test_reachability_probe_reports_full_and_partial_span() -> None:
    jacobian = torch.eye(3)
    assert reachability_r2(
        jacobian, torch.tensor([1.0, 2.0, 3.0])
    ).item() == pytest.approx(1.0)
    partial = torch.tensor([[1.0], [0.0], [0.0]])
    assert reachability_r2(partial, torch.tensor([1.0, 2.0, 3.0])).item() < 1.0


def test_effective_rank_and_heldout_projection_are_reported_separately() -> None:
    jacobian = torch.eye(4)
    rank, singular_values = effective_jacobian_rank(jacobian)
    assert rank == 4
    assert torch.allclose(singular_values, torch.ones(4))
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    heldout = heldout_reachability_r2(
        jacobian,
        target,
        train_rows=torch.tensor([0, 1]),
        eval_rows=torch.tensor([2, 3]),
    )
    assert heldout < 0.0


def test_counterfactual_pair_selector_requires_same_query_and_budget() -> None:
    latest = _fixture()
    descriptors = [
        CounterfactualDescriptor(1, latest, (4, 5), 100),
        CounterfactualDescriptor(2, latest, (4, 5), 101),
        CounterfactualDescriptor(3, latest, (4, 6), 102),
        CounterfactualDescriptor(4, latest, (4, 5), 100),
        CounterfactualDescriptor(5, latest, (4, 5), 102),
    ]
    pairs = select_counterfactual_pairs(descriptors)
    assert [(left.episode_id, right.episode_id) for left, right in pairs] == [
        (1, 2),
        (4, 5),
    ]


def test_explicit_counterfactual_pair_requires_matching_history_multiset() -> None:
    latest = _fixture()
    shared = "shared-history-multiset"
    descriptors = [
        CounterfactualDescriptor(
            1, latest, (8,), 100, pair_id=shared, history_token_multiset_hash="same"
        ),
        CounterfactualDescriptor(
            2, latest, (8,), 101, pair_id=shared, history_token_multiset_hash="same"
        ),
    ]
    assert len(select_counterfactual_pairs(descriptors)) == 1
    descriptors[1] = CounterfactualDescriptor(
        2, latest, (8,), 101, pair_id=shared, history_token_multiset_hash="different"
    )
    with pytest.raises(ValueError, match="token multiset"):
        select_counterfactual_pairs(descriptors)
    descriptors[1] = CounterfactualDescriptor(2, latest, (8,), 101, pair_id=shared)
    with pytest.raises(ValueError, match="declare"):
        select_counterfactual_pairs(descriptors)
