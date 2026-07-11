from __future__ import annotations

import torch

from arti import (
    MEMBRANE_STREAM_ASSISTANT_INNER,
    MEMBRANE_STREAM_ASSISTANT_PUBLIC,
    MembraneRoutingConfig,
    MembraneVisibilityRouter,
    append_membrane_tokens,
    build_membrane_visibility,
    membrane_emit_tokens,
    membrane_public_emit_mask,
)


ASSISTANT = 0
USER = 1
ADMIN = 2
SYSTEM = 3


def stream_readability() -> torch.Tensor:
    readable = torch.zeros(4, 2, dtype=torch.bool)
    readable[:, MEMBRANE_STREAM_ASSISTANT_PUBLIC] = True
    readable[ASSISTANT, MEMBRANE_STREAM_ASSISTANT_INNER] = True
    return readable


def test_public_tokens_emit_and_inner_tokens_stay_model_side() -> None:
    token_ids = torch.tensor([[10, 11, 12, 13]])
    stream_ids = torch.tensor(
        [[
            MEMBRANE_STREAM_ASSISTANT_PUBLIC,
            MEMBRANE_STREAM_ASSISTANT_INNER,
            MEMBRANE_STREAM_ASSISTANT_PUBLIC,
            MEMBRANE_STREAM_ASSISTANT_INNER,
        ]]
    )

    assert membrane_public_emit_mask(stream_ids).tolist() == [[True, False, True, False]]
    assert membrane_emit_tokens(token_ids, stream_ids) == [[10, 12]]
    assert token_ids.tolist() == [[10, 11, 12, 13]]


def test_assistant_can_read_inner_stream_but_non_assistant_cannot() -> None:
    stream_ids = torch.tensor([[MEMBRANE_STREAM_ASSISTANT_PUBLIC, MEMBRANE_STREAM_ASSISTANT_INNER, MEMBRANE_STREAM_ASSISTANT_PUBLIC]])
    readable = stream_readability()

    assistant_visibility = build_membrane_visibility(stream_ids, torch.tensor([ASSISTANT]), readable)
    user_visibility = build_membrane_visibility(stream_ids, torch.tensor([USER]), readable)
    admin_visibility = build_membrane_visibility(stream_ids, torch.tensor([ADMIN]), readable)
    system_visibility = build_membrane_visibility(stream_ids, torch.tensor([SYSTEM]), readable)

    assert assistant_visibility[0, 0].tolist() == [True, True, True]
    assert user_visibility[0, 0].tolist() == [True, False, True]
    assert admin_visibility[0, 0].tolist() == [True, False, True]
    assert system_visibility[0, 0].tolist() == [True, False, True]


def test_append_membrane_tokens_preserves_inner_as_normal_token_stream() -> None:
    token_ids = torch.tensor([[1, 2]])
    stream_ids = torch.tensor([[MEMBRANE_STREAM_ASSISTANT_PUBLIC, MEMBRANE_STREAM_ASSISTANT_PUBLIC]])
    appended = append_membrane_tokens(
        token_ids,
        stream_ids,
        torch.tensor([[3, 4]]),
        torch.tensor([[MEMBRANE_STREAM_ASSISTANT_INNER, MEMBRANE_STREAM_ASSISTANT_PUBLIC]]),
        phase_ids=torch.tensor([[0, 0]]),
        new_phase_ids=torch.tensor([[1, 0]]),
        participant_ids=torch.tensor([[ASSISTANT, ASSISTANT]]),
        new_participant_ids=torch.tensor([[ASSISTANT, ASSISTANT]]),
    )

    assert appended["token_ids"].tolist() == [[1, 2, 3, 4]]
    assert appended["stream_ids"].tolist() == [[0, 0, 1, 0]]
    assert appended["phase_ids"].tolist() == [[0, 0, 1, 0]]
    assert membrane_emit_tokens(appended["token_ids"], appended["stream_ids"]) == [[1, 2, 4]]


def test_same_token_in_different_stream_has_different_visibility_and_phase() -> None:
    token_ids = torch.tensor([[42, 42]])
    stream_ids = torch.tensor([[MEMBRANE_STREAM_ASSISTANT_PUBLIC, MEMBRANE_STREAM_ASSISTANT_INNER]])
    phase_ids = torch.tensor([[0, 1]])
    user_visibility = build_membrane_visibility(stream_ids, torch.tensor([USER]), stream_readability())

    assert token_ids[0, 0].item() == token_ids[0, 1].item()
    assert phase_ids[0, 0].item() != phase_ids[0, 1].item()
    assert user_visibility[0, 0].tolist() == [True, False]


def test_router_outputs_stream_distribution_without_generating_vocab_tokens() -> None:
    torch.manual_seed(3)
    router = MembraneVisibilityRouter(MembraneRoutingConfig(hidden_dim=5))
    hidden = torch.randn(2, 5)
    forced_streams = torch.tensor([MEMBRANE_STREAM_ASSISTANT_INNER, MEMBRANE_STREAM_ASSISTANT_PUBLIC])

    out = router(hidden, stream_ids=forced_streams)

    assert out.stream_logits.shape == (2, 2)
    assert out.stream_probs.shape == (2, 2)
    assert out.stream_ids.tolist() == [MEMBRANE_STREAM_ASSISTANT_INNER, MEMBRANE_STREAM_ASSISTANT_PUBLIC]
    assert out.public_emit_mask.tolist() == [False, True]
    assert "membrane_inner_probability" in out.diagnostics
