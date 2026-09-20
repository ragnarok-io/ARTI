# Participant Context API

ARTI core stays tensor-first and domain-free. For multi-participant dialogue or
other scoped-context systems, use `build_participant_context` to convert external
authorization decisions into ARTI tensors.

The helper does not know about business roles. It only uses participant ids:

```python
import torch
from arti import build_participant_context
from arti.legacy import ARTILayer

assistant_id = 0

# Token owners/speakers. Shape: [B, N]
participant_ids = torch.tensor([[1, 0, 2, 0]])

# Participant coordinates/phases. Shape: [P, C]
participant_coord = torch.eye(3)

# readable_by[viewer, owner] controls which participant can read which tokens.
readable_by = torch.tensor(
    [
        [True, True, True],    # assistant
        [True, True, True],    # participant 1
        [True, False, True],   # participant 2 cannot read participant 1
    ]
)

ctx = build_participant_context(
    participant_ids,
    participant_coord,
    readable_by,
    assistant_id=assistant_id,
)

layer = ARTILayer(
    input_dim=32,
    coord_dim=3,
    hidden_dim=64,
    coord_frame_mode="operator_bank",
)

out = layer(
    x,
    coord=ctx.coord,
    mask=ctx.mask,
    visibility=ctx.visibility,
    observer_coord=ctx.observer_coord,
    frame_operators=inverse_bank,
)
```

## Autoregressive Observer

When `assistant_id` is provided and `observer_participant` is omitted, the helper
uses the assistant participant as the observer frame. This matches
autoregressive generation where the model is continuing the assistant response:
visible context is interpreted from the assistant reference frame.

During training, pass `observer_participant` explicitly when the next token being
continued belongs to a non-assistant participant.

```python
ctx = build_participant_context(
    participant_ids,
    participant_coord,
    readable_by,
    assistant_id=0,
    observer_participant=torch.tensor([2]),
)
```

## Placement Rule

Participant phase should be applied before token representations from different
participants are mixed. In LLM-style systems this usually means placing the
ARTI participant-context layer immediately after token embedding, or at the
earliest adapter point where the hidden tensor still has a one-to-one mapping to
the original tokens/messages.

This is important because external phase identifies the source frame of an
original token. After several attention or pooling layers, a hidden vector may
already contain information from system, assistant, user, and tool tokens. At
that point, assigning a single participant phase can mislabel a mixed state.

Recommended shape:

```text
token ids
-> embedding / tokenizer adapter
-> ARTI participant phase + visibility layer
-> transformer / downstream model
```

Risky shape:

```text
token ids
-> transformer layers that mix participants
-> ARTI participant phase layer
```

Use later ARTI layers for internal latent dynamics, recall, and denoising only
when the downstream model preserves a reliable token-to-source mapping or when
the later `coord` describes the mixed latent state rather than original token
identity.

## Active Participant

If `active_participant` is omitted, the helper selects the most recent
non-assistant participant as the viewer/recipient. This lets a dialogue model
switch visibility according to the latest non-assistant speaker.

Pass `active_participant` explicitly for system-controlled routing or evaluation:

```python
ctx = build_participant_context(
    participant_ids,
    participant_coord,
    readable_by,
    active_participant=torch.tensor([1]),
    assistant_id=0,
)
```

## Contract

`build_participant_context` returns:

```text
coord                [B, N, C]
mask                 [B, N]
visibility           [B, N, N]
observer_coord       [B, C]
active_participant   [B]
observer_participant [B]
```

The API scopes what is visible and which coordinate frame is active. It does not
insert external facts, retrieve hidden context, or bypass downstream model
policy. Recall remains trace reconstruction of what the model has processed, not
injection of new information.
