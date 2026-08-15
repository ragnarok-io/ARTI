# Custom Recall Formulas

Recall has one state contract: a Formula receives the current state and its
ordered factors and returns the complete next state.

```text
state   [..., D]
factors [..., F, D]
result  [..., D]  # next_state
```

## Built-ins

Built-in Formula IDs are canonical and versioned:

| ID | Factors | Use |
| --- | ---: | --- |
| `arti/delta@1` | 1 | direct content update |
| `arti/affine@1` | 2 | scale and shift update |
| `arti/state@1` | 17 | structured multi-factor update |

```python
import arti

recall = arti.Recall(768, slots=32, formula="arti/state@1")
next_state = recall(hidden)
```

Old short names such as `delta-v1`, `single`, `product`, and `state` are not
aliases. They are rejected so a checkpoint cannot silently select a different
Formula implementation.

## Custom Formula

A custom Formula is an ordinary `torch.nn.Module` with an explicit contract:

```python
import torch
import arti


class SignedGate(torch.nn.Module):
    output_semantics = "next_state"
    factor_names = ("content", "gate")

    def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        content, gate = factors.unbind(dim=-2)
        return state + torch.tanh(gate) * content


recall = arti.Recall(768, slots=32, formula=SignedGate())
```

Formulas must preserve shape, dtype, and device, avoid mutation and hidden
side effects, and never create an optimizer. `arti.validate_formula` checks
these properties before a long run.

## Contract and Lock

For a reusable Formula, put the declarative metadata on the module instead of
relying on a filename or an implicit factor order:

```python
class SignedGate(torch.nn.Module):
    recall_formula_contract = arti.RecallFormulaContract(
        factors=(
            arti.FactorSpec("content", init="zero"),
            arti.FactorSpec("gate", init="zero"),
        ),
        identity_preserving=True,
        composition="custom",
        capabilities=("torch.compile", "torch.eager"),
        execution=arti.RecallFormulaExecutionSpec(
            vectorization="scalar_vmap",
            supports_compile=True,
        ),
    )

    def forward(self, state, factors):
        content, gate = factors.unbind(dim=-2)
        return state + torch.tanh(gate) * content


recall = arti.Recall(768, slots=32, formula=SignedGate())
lock = recall.formula_lock
payload = lock.to_dict()  # suitable for a JSON artifact or release lock
```

The contract is the Formula's plan: it describes factor names, routes,
identity values, composition, and supported execution capabilities. The lock
is the applied instance: it binds that contract to `hidden_dim`, `slots`, and
the `torch` backend. Loading code can call `RecallFormulaLock.from_dict` and
compare the contract fingerprint before loading weights. This follows the
useful part of Nix/Terraform without adding a package manager or a hidden
apply step. Shape metadata also gives future Triton or other compiled
backends a stable admission check; declaring `torch.compile` does not itself
install Triton or change execution.

The lock contains no module, optimizer, or tensor values. It is therefore
safe to version separately from `state_dict` weights and to reject stale or
mis-sized bank artifacts early.

The Formula contract and lock schema are version 2 in ARTI 3.0. Old 2.x
contracts are intentionally rejected rather than silently reinterpreted.

`execution.vectorization` is conservative by default: `scalar_vmap` preserves
the original Formula ABI, while `batched` opts into the flat hot path
`[M, D] + [M, F, D] -> [M, D]`. A batched Formula must be row-independent:
changing one batch/token row must not change any other row. The validator
probes this invariant before the Formula is admitted. `supported_dtypes` is
also enforced at runtime; `accumulation_dtype`, `deterministic`, and
`supports_autograd` remain explicit execution-contract declarations rather
than hidden conversions.

Process-local factories may be registered with an exact identity:

```python
arti.register_formula("acme/signed-gate@1", factory=SignedGate)
```

Third-party Formula identities are process-local unless the caller supplies a
separately authorized artifact policy.

## RecallRefiner

`RecallRefiner` accepts only a module declaring `output_semantics="next_state"`.
It computes `next_state - state`, optionally applies `Half`, and repeats the
update. Residual or undeclared producers are rejected rather than guessed.
