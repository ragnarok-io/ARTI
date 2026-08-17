from __future__ import annotations

import pytest
import torch

from arti.layered_recall import LayerRecall
from arti.nn import Fold, Half, UnFold
from arti.recall_workspace import RecallWorkspace


def _workspace(
    dim: int = 4,
    *,
    use_fold: bool = True,
    use_half: bool = True,
    use_unfold: bool = True,
    condition_unfold: bool = False,
) -> RecallWorkspace:
    return RecallWorkspace(
        dim,
        fold=Fold(k=3, dim=dim) if use_fold else None,
        half=Half(threshold=0.5) if use_half else None,
        unfold=(
            UnFold(
                dim=dim,
                exposed=2,
                condition_dim=dim if condition_unfold else None,
            )
            if use_unfold
            else None
        ),
    )


def test_recall_workspace_compacts_expands_and_reads_per_query() -> None:
    torch.manual_seed(4)
    workspace = _workspace()
    queries = torch.randn(2, 5, 4, requires_grad=True)
    candidates = torch.randn(2, 7, 4, requires_grad=True)
    query_mask = torch.tensor([[True, True, True, False, False], [True] * 5])
    candidate_mask = torch.tensor([[True, True, True, True, False, False, False], [True] * 7])

    context, info = workspace(
        queries,
        candidates,
        query_mask=query_mask,
        candidate_mask=candidate_mask,
        return_info=True,
    )

    assert context.shape == (2, 5, 4)
    assert info["folded"].shape == (2, 3, 4)
    assert info["expanded"].shape == (2, 5, 4)
    assert info["read_weights"].shape == (2, 5, 5)
    assert info["workspace_mask"].dtype == torch.bool
    assert torch.equal(context[0, 3:], torch.zeros_like(context[0, 3:]))
    assert torch.allclose(info["read_weights"][1].sum(-1), torch.ones(5))

    context.square().mean().backward()
    assert queries.grad is not None and torch.isfinite(queries.grad).all()
    assert candidates.grad is not None and torch.isfinite(candidates.grad).all()
    assert workspace.read_query.weight.grad is not None
    assert workspace.read_value.weight.grad is not None
    assert workspace.unfold is not None
    assert workspace.unfold.exposed_queries.grad is not None


@pytest.mark.parametrize(
    ("use_fold", "use_half", "use_unfold", "expected_slots"),
    [
        (False, False, False, 6),
        (True, False, False, 3),
        (False, True, True, 8),
        (True, True, True, 5),
    ],
)
def test_recall_workspace_stages_are_independently_optional(
    use_fold: bool,
    use_half: bool,
    use_unfold: bool,
    expected_slots: int,
) -> None:
    workspace = _workspace(use_fold=use_fold, use_half=use_half, use_unfold=use_unfold)
    output, info = workspace(
        torch.randn(2, 4, 4),
        torch.randn(2, 6, 4),
        return_info=True,
    )
    assert output.shape == (2, 4, 4)
    assert info["expanded"].shape == (2, expected_slots, 4)


def test_recall_workspace_masked_candidates_cannot_change_valid_reads() -> None:
    torch.manual_seed(9)
    workspace = _workspace().eval()
    queries = torch.randn(1, 3, 4)
    candidates = torch.randn(1, 6, 4)
    mask = torch.tensor([[True, True, True, False, False, False]])
    changed = candidates.clone()
    changed[:, 3:] = 1e4

    first = workspace(queries, candidates, candidate_mask=mask)
    second = workspace(queries, changed, candidate_mask=mask)
    torch.testing.assert_close(first, second)


def test_layer_recall_default_path_has_no_workspace_parameters() -> None:
    layer = LayerRecall(dim=8, rank=4, slots=3)
    assert layer.workspace is None
    assert not any(name.startswith("workspace.") for name in layer.state_dict())
    output = layer(torch.randn(2, 5, 8))
    assert output.shape == (2, 5, 8)


def test_layer_recall_opt_in_workspace_reports_internal_diagnostics() -> None:
    torch.manual_seed(12)
    workspace = _workspace(dim=4)
    layer = LayerRecall(dim=8, rank=4, slots=3, workspace=workspace)
    hidden = torch.randn(2, 5, 8)
    mask = torch.tensor([[True, True, True, False, False], [True] * 5])

    delta, info = layer(hidden, mask=mask, return_info=True)

    assert delta.shape == hidden.shape
    assert info["workspace_read_weights"].shape == (2, 5, 5)
    assert info["workspace_slot_usage"].shape == (2, 5)
    assert info["workspace_exposed_mask"].shape == (2, 5)
    assert torch.equal(delta[0, 3:], torch.zeros_like(delta[0, 3:]))
    assert torch.isfinite(info["workspace_survival"]).all()


def test_layer_recall_rejects_workspace_rank_mismatch() -> None:
    with pytest.raises(ValueError, match="workspace feature dimension"):
        LayerRecall(dim=8, rank=4, workspace=_workspace(dim=3))


def test_recall_workspace_state_dict_round_trip() -> None:
    torch.manual_seed(21)
    source = _workspace().eval()
    target = _workspace().eval()
    target.load_state_dict(source.state_dict())
    queries = torch.randn(2, 4, 4)
    candidates = torch.randn(2, 7, 4)
    torch.manual_seed(72)
    expected = source(queries, candidates)
    torch.manual_seed(72)
    actual = target(queries, candidates)
    torch.testing.assert_close(expected, actual)


def test_recall_workspace_can_condition_unfold_on_current_queries() -> None:
    torch.manual_seed(27)
    workspace = _workspace(condition_unfold=True)
    candidates = torch.randn(2, 7, 4)
    queries = torch.randn(2, 4, 4, requires_grad=True)

    _, info = workspace(queries, candidates, return_info=True)

    assert info["expand_condition"].shape == (2, 4)
    info["expanded"].square().mean().backward()
    assert workspace.expand_condition is not None
    assert workspace.expand_condition.weight.grad is not None
    assert queries.grad is not None
