from __future__ import annotations

import torch

from arti import (
    MEMBRANE_STREAM_ASSISTANT_INNER,
    MEMBRANE_STREAM_ASSISTANT_PUBLIC,
    MembraneRoutingConfig,
    MembraneVisibilityRouter,
    build_membrane_visibility,
    membrane_emit_tokens,
)


ASSISTANT = 0
USER = 1


def main() -> None:
    router = MembraneVisibilityRouter(MembraneRoutingConfig(hidden_dim=8))
    hidden = torch.randn(1, 3, 8)
    forced_streams = torch.tensor(
        [[
            MEMBRANE_STREAM_ASSISTANT_PUBLIC,
            MEMBRANE_STREAM_ASSISTANT_INNER,
            MEMBRANE_STREAM_ASSISTANT_PUBLIC,
        ]]
    )
    routed = router(hidden, stream_ids=forced_streams)

    token_ids = torch.tensor([[101, 202, 303]])
    user_visible_ids = membrane_emit_tokens(token_ids, routed.stream_ids)

    readable_by = torch.zeros(2, 2, dtype=torch.bool)
    readable_by[:, MEMBRANE_STREAM_ASSISTANT_PUBLIC] = True
    readable_by[ASSISTANT, MEMBRANE_STREAM_ASSISTANT_INNER] = True

    assistant_visibility = build_membrane_visibility(
        routed.stream_ids,
        torch.tensor([ASSISTANT]),
        readable_by,
    )
    user_visibility = build_membrane_visibility(
        routed.stream_ids,
        torch.tensor([USER]),
        readable_by,
    )

    print("model_context_tokens:", token_ids.tolist())
    print("stream_ids:", routed.stream_ids.tolist())
    print("user_emitted_tokens:", user_visible_ids)
    print("assistant_can_read:", assistant_visibility[0, 0].tolist())
    print("user_can_read:", user_visibility[0, 0].tolist())


if __name__ == "__main__":
    main()
