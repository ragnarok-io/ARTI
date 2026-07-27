# Custom Recall Formulas

Recall has two separate jobs:

- The **Bank** stores tensors and routes a query to candidate factors.
- The **Formula** combines the current state and those factors into the next
  state.

Training choices such as learning rates, clipping, precision, and schedules
belong to **Policy**. A Formula does not create an optimizer, modify gradients,
or decide how the Bank is trained.

This separation keeps the common path short while allowing experiments without
adding another composition switch to ARTI core.

## Preset Formulas

Use a versioned preset when its behavior matches the application:

```python
import arti
import torch

recall = arti.nn.Recall(
    dim=768,
    slots=68,
    formula="state-v1",
)

x = torch.randn(2, 16, 768)
mask = torch.ones(2, 16, dtype=torch.bool)
y = recall(x, mask=mask)
```

Preset IDs include the behavior version. An artifact records `state-v1`, not
an unversioned `default`, so a future formula cannot silently change an existing
model.

The built-in migration mapping is:

| Previous value | Formula ID |
| --- | --- |
| `single` | `delta-v1` |
| `product` | `affine-v1` |
| `state` | `state-v1` |

Existing `recall_value_composition` configurations remain compatibility inputs.
New code should use `formula=`.

## Custom Formulas

A custom Formula is an ordinary `torch.nn.Module`. It receives the current
state and a statically ordered factor tensor, then returns the complete next
state:

```text
state   [..., D]
factors [..., F, D]
result  [..., D]  # next_state
```

For example:

```python
import torch
import arti


class SignedGate(torch.nn.Module):
    factor_names = ("content", "gate")

    def forward(
        self,
        state: torch.Tensor,
        factors: torch.Tensor,
    ) -> torch.Tensor:
        content, gate = factors.unbind(dim=-2)
        return state + torch.tanh(gate) * content


recall = arti.nn.Recall(
    dim=768,
    slots=32,
    formula=SignedGate(),
)

state = torch.randn(2, 16, 768)
next_state = recall(state)
```

Every Formula returns `next_state`. It must not sometimes return a residual
delta and sometimes return a full state. Recall can derive a diagnostic delta
consistently:

```python
delta = next_state - state
```

Recall applies a custom Formula independently to each latent vector. The
Formula receives one `[D]` state and one `[F, D]` factor tensor per mapped call,
so padding tokens and other batch items cannot influence that vector through
Formula-side reductions.

Keep a custom Formula tensor-only and free of hidden side effects. In
particular, it should not:

- access or mutate the Recall Bank;
- create an optimizer or alter parameter gradients;
- inspect global training steps to change its behavior;
- perform file, network, or process I/O;
- depend on undeclared random state.

Formula-specific learnable parameters may belong to the Formula module itself.
Bank parameters and routing remain owned by Recall. Optimizer configuration
remains an explicit training policy outside both modules.

Before using a custom Formula in a longer run, check at least:

- output shape, dtype, and device match the input state;
- forward and backward values are finite;
- every intended factor receives a gradient;
- initialization has the intended identity or bounded-drift behavior;
- eager, compiled, and mixed-precision modes used by the application agree
  within an application-defined tolerance.

## Process-Local Registered Formulas

Passing a module instance is the local research path. It supports normal
autograd and `state_dict()` behavior, but it is not automatically a portable
ARTI artifact:

```text
custom Formula instance -> process-local, portable=false
```

Applications can give a trusted local Formula factory a stable, versioned
process identity:

```python
arti.register_formula(
    "acme/signed-gate@1",
    factory=SignedGate,
)

recall = arti.nn.Recall(
    dim=768,
    slots=32,
    formula="acme/signed-gate@1",
)
```

Registration is explicit and process-local. ARTI does not scan Python entry
points, import packages named by an artifact, or install Formula providers.
Third-party registered Formulas remain `portable=false` in Formula API v1.

`formula_manifest()` exposes passive identity and factor-layout metadata for
inspection:

```python
metadata = recall.formula_manifest()
print(metadata.factor_names)
print(metadata.layout_fingerprint)
```

This metadata is not yet an automatic save/load authorization protocol.
Portable third-party Formula artifacts require a future host-controlled
provider and loader contract. Its canonical SHA-256 identifies exact metadata
contents; it is not a publisher signature or proof that Formula code is trusted.

Use a local custom Formula while exploring its mathematics. Register it only
when the application benefits from stable process-local discovery.
