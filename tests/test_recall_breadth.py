from __future__ import annotations

import torch

import arti


def test_recall_defaults_to_k_wide_winner_selection() -> None:
    torch.manual_seed(3109)
    recall = arti.nn.Recall(dim=4, slots=16, activation="none")
    value = torch.randn(2, 3, 4)

    output, result = recall(value, return_branches=True)

    assert recall.breadth == recall.recommended_breadth == 8
    assert recall.breadth_mode == "independent"
    assert recall.breadth_aggregation == "winner"
    assert arti.component_ref(recall) == "arti/recall@4"
    winner = result.candidates.candidate_log_score.mean(dim=1).argmax(dim=-1)
    expected = result.value.gather(
        1,
        winner[:, None, None, None].expand(-1, 1, value.shape[1], value.shape[-1]),
    ).squeeze(1)
    torch.testing.assert_close(output, expected)


def test_recall_route_weighted_aggregation_is_explicit() -> None:
    torch.manual_seed(3110)
    recall = arti.nn.Recall(
        dim=4,
        slots=8,
        activation="none",
        breadth=4,
        breadth_aggregation="route_weighted",
    )
    value = torch.randn(1, 2, 4)

    output, result = recall(value, return_branches=True)
    scores = result.candidates.candidate_log_score
    weights = torch.softmax(scores.mean(dim=1), dim=-1)
    expected = (result.value * weights[:, :, None, None]).sum(dim=1)

    torch.testing.assert_close(output, expected)


def test_composed_formula_defaults_to_k_wide_winner_selection() -> None:
    torch.manual_seed(3111)
    recall = arti.nn.Recall(
        dim=4,
        slots=4,
        formula="arti/affine@1",
        activation="none",
    )
    value = torch.randn(2, 3, 4)

    output, result = recall(value, return_branches=True)

    assert recall.breadth == 2
    assert result.candidates.max_k == 2
    winner = result.candidates.candidate_log_score.mean(dim=1).argmax(dim=-1)
    expected = result.value.gather(
        1,
        winner[:, None, None, None].expand(-1, 1, value.shape[1], value.shape[-1]),
    ).squeeze(1)
    torch.testing.assert_close(output, expected)


def test_recall_breadth_is_configurable_per_call() -> None:
    recall = arti.nn.Recall(dim=4, slots=8, activation="none", breadth=6)
    value = torch.randn(2, 3, 4)

    _output, result = recall(
        value,
        active_k=torch.tensor([2, 5]),
        return_branches=True,
    )

    assert result.candidates.max_k == 6
    assert result.candidates.active_k.tolist() == [2, 5]


def test_legacy_recall_references_keep_mixed_semantics() -> None:
    recall = arti.resolve_component(
        "arti/recall@2",
        dim=4,
        slots=4,
        activation="none",
        routing="dense",
        key_dim=4,
        group_size=1,
        group_topk=1,
        route_exploration=0.0,
        dropout=0.0,
    )

    assert recall.breadth == 1
    assert recall.breadth_mode == "mixed"
    assert arti.component_ref(recall) == "arti/recall@2"


def test_recall_winner_keeps_route_surrogate_gradients() -> None:
    recall = arti.nn.Recall(dim=4, slots=8, activation="none", breadth=4)
    recall.state.recall.set_bank_gradient_enabled(True)
    value = torch.randn(2, 3, 4)

    recall(value).square().mean().backward()

    assert recall.state.recall.group_bank.grad is not None
    assert torch.isfinite(recall.state.recall.group_bank.grad).all()
