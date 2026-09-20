from __future__ import annotations

import pytest
import torch

from arti.component_registry import ComponentRef, component_ref, component_spec
from arti.shape_query import (
    BankMemberMatcher,
    CoordinateTensorViewObserver,
    ShapeQueryError,
    TensorViewBankQuery,
    seal_tensor_view_bank_query,
)
from arti.tensor_view import (
    AxisDescriptor,
    TensorIndexMap,
    TensorView,
    TensorViewError,
    TensorViewPattern,
)


def _view(value: torch.Tensor, names: tuple[str, ...], roles: tuple[str, ...]) -> TensorView:
    return TensorView.from_tensor(value, axis_names=names, axis_roles=roles)


def _query(*, members: int = 3) -> TensorViewBankQuery:
    observer = CoordinateTensorViewObserver(
        max_rank=5,
        query_dim=8,
        hidden_dim=12,
        max_observations=128,
    )
    matcher = BankMemberMatcher(
        torch.randn(members, 8),
        member_ids=tuple(f"member-{index}" for index in range(members)),
    )
    return TensorViewBankQuery(
        pattern=TensorViewPattern(min_rank=1, max_rank=5),
        observer=observer,
        matcher=matcher,
    )


def test_tensor_view_validates_axes_and_explicit_index_map() -> None:
    value = torch.randn(2, 3, 4)
    coordinates = torch.stack(
        torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij"),
        dim=-1,
    ).to(torch.int64)
    index_map = TensorIndexMap(
        source_axes=("row", "column"),
        source_shape=(3, 4),
        target_shape=(3, 4),
        coordinates=coordinates,
    )
    view = _view(
        value,
        ("batch", "row", "column"),
        ("batch", "spatial", "spatial"),
    )
    mapped = TensorView(view.value, view.axes, index_map=index_map)
    assert mapped.index_map is index_map
    assert mapped.logical_shape == (2, 3, 4)

    with pytest.raises(TensorViewError, match="out of bounds"):
        TensorIndexMap(
            source_axes=("row", "column"),
            source_shape=(3, 4),
            target_shape=(3, 4),
            coordinates=coordinates + 4,
        )


@pytest.mark.parametrize("shape", [(2, 7), (2, 3, 5), (2, 2, 3, 4)])
def test_one_query_accepts_multiple_runtime_ranks(shape: tuple[int, ...]) -> None:
    query = _query()
    names = ("batch", *(f"logical{index}" for index in range(len(shape) - 1)))
    roles = ("batch", *("generic" for _ in shape[1:]))
    result = query(_view(torch.randn(*shape), names, roles))
    assert result.scores.shape == (shape[0], 3)
    assert result.observation.tokens.shape[0] == shape[0]
    assert result.observation.tokens.shape[-1] == 8


def test_axis_permutation_with_descriptors_preserves_observation() -> None:
    torch.manual_seed(7)
    observer = CoordinateTensorViewObserver(max_rank=4, query_dim=6, hidden_dim=9)
    value = torch.randn(2, 3, 4)
    original = _view(
        value,
        ("batch", "row", "column"),
        ("batch", "spatial", "spatial"),
    )
    permuted = _view(
        value.permute(0, 2, 1),
        ("batch", "column", "row"),
        ("batch", "spatial", "spatial"),
    )
    left = observer(original)
    right = observer(permuted)
    torch.testing.assert_close(left.tokens, right.tokens)
    torch.testing.assert_close(left.mask, right.mask)


def test_values_and_axis_identity_both_change_query_scores() -> None:
    torch.manual_seed(13)
    query = _query()
    value = torch.randn(2, 3, 4)
    base = query(
        _view(
            value,
            ("batch", "row", "column"),
            ("batch", "spatial", "spatial"),
        )
    ).scores
    changed_value = query(
        _view(
            value + 2.0,
            ("batch", "row", "column"),
            ("batch", "spatial", "spatial"),
        )
    ).scores
    changed_axes = query(
        _view(
            value,
            ("batch", "time", "feature"),
            ("batch", "sequence", "feature"),
        )
    ).scores
    assert not torch.allclose(base, changed_value)
    assert not torch.allclose(base, changed_axes)


def test_member_concat_preserves_existing_scores_exactly() -> None:
    torch.manual_seed(17)
    observer = CoordinateTensorViewObserver(max_rank=3, query_dim=5, hidden_dim=7)
    observation = observer(
        _view(
            torch.randn(2, 3, 4),
            ("batch", "row", "feature"),
            ("batch", "spatial", "feature"),
        )
    )
    left = BankMemberMatcher(
        torch.randn(2, 5), member_ids=("left-a", "left-b")
    )
    right = BankMemberMatcher(torch.randn(1, 5), member_ids=("right-a",))
    combined = BankMemberMatcher.concatenate(left, right)
    torch.testing.assert_close(combined(observation)[:, :2], left(observation))
    assert combined.member_ids == ("left-a", "left-b", "right-a")


def test_sealed_query_freezes_state_but_preserves_input_gradients() -> None:
    torch.manual_seed(19)
    query = _query()
    sealed = seal_tensor_view_bank_query(query)
    value = torch.randn(2, 3, 4, requires_grad=True)
    result = sealed(
        _view(
            value,
            ("batch", "row", "feature"),
            ("batch", "spatial", "feature"),
        )
    )
    result.scores.sum().backward()
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()
    assert not any(parameter.requires_grad for parameter in sealed.parameters())
    ComponentRef.parse(component_ref(sealed))
    assert component_spec(sealed).lifecycle == "alpha"

    with torch.no_grad():
        next(sealed.query.parameters()).add_(1.0)
    with pytest.raises(ShapeQueryError, match="changed after mounting"):
        sealed(
            _view(
                value.detach(),
                ("batch", "row", "feature"),
                ("batch", "spatial", "feature"),
            )
        )


def test_pattern_rejects_only_resource_bounds_not_one_exact_rank() -> None:
    pattern = TensorViewPattern(min_rank=2, max_rank=4)
    pattern.validate(_view(torch.randn(2, 3), ("batch", "token"), ("batch", "sequence")))
    pattern.validate(
        _view(
            torch.randn(2, 3, 4, 5),
            ("batch", "depth", "row", "column"),
            ("batch", "spatial", "spatial", "spatial"),
        )
    )
    with pytest.raises(TensorViewError, match="rank is outside"):
        pattern.validate(
            TensorView(
                torch.randn(2),
                (AxisDescriptor("batch", "batch", 2),),
            )
        )
