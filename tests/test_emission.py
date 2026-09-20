import pytest
import torch

from arti import (
    EmissionRouter,
    EmissionRouterConfig,
    build_stream_visibility,
    stream_emit_mask,
)


def test_emission_router_is_numbered_stream_generic_and_batch_safe() -> None:
    torch.manual_seed(4)
    router = EmissionRouter(
        EmissionRouterConfig(hidden_dim=5, stream_count=3, emit_streams=(1, 2))
    ).eval()
    hidden = torch.randn(2, 4, 5)
    forced = torch.tensor([[0, 1, 2, 0], [2, 0, 1, 1]], dtype=torch.long)
    valid = torch.tensor(
        [[True, True, True, False], [True, False, True, True]],
        dtype=torch.bool,
    )

    output = router(hidden, stream_ids=forced, valid_mask=valid)

    assert output.stream_logits.shape == (2, 4, 3)
    assert output.stream_probs.shape == (2, 4, 3)
    assert output.stream_ids.dtype is torch.int64
    assert torch.equal(output.stream_ids, forced)
    assert output.emit_mask.tolist() == [
        [False, True, True, False],
        [True, False, True, True],
    ]
    assert torch.isfinite(output.stream_probs).all()


def test_emission_router_supports_single_token_batches() -> None:
    router = EmissionRouter(EmissionRouterConfig(hidden_dim=3, stream_count=2))
    output = router(torch.randn(2, 3), valid_mask=torch.tensor([True, False]))

    assert output.stream_ids.shape == (2,)
    assert output.emit_mask.shape == (2,)
    assert not bool(output.emit_mask[1])


def test_emission_router_rejects_invalid_stream_or_mask_contract() -> None:
    router = EmissionRouter(EmissionRouterConfig(hidden_dim=3, stream_count=2))
    hidden = torch.randn(1, 2, 3)

    with pytest.raises(ValueError, match="outside"):
        router(hidden, stream_ids=torch.tensor([[-1, 0]]))
    with pytest.raises(TypeError, match="torch.bool"):
        router(hidden, valid_mask=torch.ones(1, 2))


def test_stream_visibility_uses_policy_and_valid_mask_without_role_semantics() -> None:
    stream_ids = torch.tensor([[0, 1, 2, 1]], dtype=torch.long)
    viewer_ids = torch.tensor([1], dtype=torch.long)
    readable = torch.tensor(
        [
            [True, False, False],
            [True, True, False],
        ],
        dtype=torch.bool,
    )
    valid = torch.tensor([[True, True, False, True]], dtype=torch.bool)

    visibility = build_stream_visibility(
        stream_ids,
        viewer_ids,
        readable,
        valid_mask=valid,
    )

    assert visibility.shape == (1, 4, 4)
    assert visibility[0, 0].tolist() == [True, True, False, True]
    assert not bool(visibility[:, :, 2].any())


def test_stream_emit_mask_is_explicitly_configured() -> None:
    stream_ids = torch.tensor([[0, 1, 2]])
    assert stream_emit_mask(stream_ids, streams=(2,)).tolist() == [[False, False, True]]
