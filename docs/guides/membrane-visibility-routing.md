# Membrane Visibility Routing

Membrane Visibility Routing is a versioned mechanism for autoregressive models. It
does not create a hidden latent thought channel. It routes normal generated
tokens into visibility domains:

- `assistant_public`: enters the model context and can be emitted to the user.
- `assistant_inner`: enters the model context but is not emitted to the user.

The intended use is a model-side inner speech loop where assistant continuations
can read previous inner tokens, while user/admin/system/non-assistant
continuations cannot read them through the supplied visibility rules.

```python
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

router = MembraneVisibilityRouter(MembraneRoutingConfig(hidden_dim=8))
hidden = torch.randn(1, 3, 8)
streams = torch.tensor([[0, 1, 0]])
routed = router(hidden, stream_ids=streams)

token_ids = torch.tensor([[101, 202, 303]])
public_tokens = membrane_emit_tokens(token_ids, routed.stream_ids)

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
```

The router predicts stream logits from hidden states. During supervised or
curriculum training, pass `stream_ids` to force the target route and train the
router with a normal cross-entropy loss against the intended stream labels.
During inference, omit `stream_ids` and use the predicted route.

The helper `membrane_emit_tokens` is deliberately separate from context update:
inner tokens are filtered out of user-facing decode output, but they remain in
the model-side token stream. Use `append_membrane_tokens` to append token ids
and stream/phase metadata together.

Run the mechanism check:

```bash
uv run --extra dev python benchmarks/run_membrane_visibility_routing.py
uv run --extra dev python benchmarks/verify_membrane_visibility_routing.py
```

Passing this check means the routing and visibility invariants hold locally. It
does not claim that an LLM has learned when to use inner speech; that requires a
task-level training setup.
