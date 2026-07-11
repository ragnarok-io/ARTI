from __future__ import annotations

import torch

from arti import ARTILayer, build_participant_context, last_non_assistant_participant


def test_last_non_assistant_participant_ignores_assistant_and_padding() -> None:
    participant_ids = torch.tensor([[1, 0, 2, 0], [0, 0, 0, 3]])
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])

    active = last_non_assistant_participant(participant_ids, mask, assistant_id=0)

    assert active.tolist() == [2, 0]


def test_build_participant_context_scopes_visible_tokens_by_active_viewer() -> None:
    participant_ids = torch.tensor([[1, 0, 2, 0]])
    participant_coord = torch.eye(3)
    readable_by = torch.tensor(
        [
            [True, True, True],
            [True, True, True],
            [True, False, True],
        ]
    )

    ctx = build_participant_context(
        participant_ids,
        participant_coord,
        readable_by,
        assistant_id=0,
    )

    assert ctx.active_participant.tolist() == [2]
    assert ctx.observer_participant.tolist() == [0]
    assert ctx.mask.tolist() == [[False, True, True, True]]
    assert ctx.visibility[0, 0].tolist() == [False, True, True, True]

    admin = build_participant_context(
        participant_ids,
        participant_coord,
        readable_by,
        assistant_id=0,
        active_participant=torch.tensor([1]),
        observer_participant=torch.tensor([0]),
    )

    assert admin.mask.tolist() == [[True, True, True, True]]
    assert admin.visibility[0, 1].tolist() == [True, True, True, True]
    assert torch.equal(admin.coord[0, 2], participant_coord[2])
    assert torch.equal(admin.observer_coord[0], participant_coord[0])


def test_participant_context_can_drive_arti_layer_observer_frame() -> None:
    torch.manual_seed(3)
    layer = ARTILayer(input_dim=4, coord_dim=3, hidden_dim=4, use_pairwise_context=False, coord_frame_mode="operator_bank")
    frame_operators = torch.stack([torch.eye(4), torch.eye(4), torch.eye(4)])
    participant_ids = torch.tensor([[1, 0, 2, 0]])
    participant_coord = torch.eye(3)
    readable_by = torch.ones(3, 3, dtype=torch.bool)
    x = torch.randn(1, 4, 4)

    ctx = build_participant_context(
        participant_ids,
        participant_coord,
        readable_by,
        assistant_id=0,
    )
    out = layer(
        x,
        coord=ctx.coord,
        mask=ctx.mask,
        visibility=ctx.visibility,
        observer_coord=ctx.observer_coord,
        frame_operators=frame_operators,
    )

    assert out.y.shape == (1, 4, 4)
    assert out.diagnostics["observer_frame_active"].item() == 1.0
