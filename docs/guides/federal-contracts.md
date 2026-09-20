# Shape-autonomous Program contracts

ARTI Core is tensor-native, but it does not impose one global same-shape TITO
rule. Same-shape remains a convenient `arti.fit()` attachment profile. A
standalone ARTI component owns its input schema, output schema, shape relation,
and gradient contract.

The `FederatedProgram` execution region starts with data contracts rather than a second Retrieve
implementation:

```python
import arti
from arti import mechanisms

local_input = mechanisms.TensorSchema(
    dtype="float32",
    device_class="any",
    dimensions=("B", 17, 64),
    semantic_axes=("batch", "slot", "feature"),
)

terminal = mechanisms.TerminalOutputABI(
    fields=(
        mechanisms.TerminalField(
            "value",
            mechanisms.TensorSchema(
                dtype="float32",
                device_class="any",
                dimensions=("B", "T", 32),
                semantic_axes=("batch", "token", "feature"),
            ),
            "terminal-value",
        ),
        mechanisms.TerminalField(
            "validity",
            mechanisms.TensorSchema(
                dtype="boolean",
                device_class="any",
                dimensions=("B",),
                semantic_axes=("batch",),
            ),
            "terminal-validity",
        ),
        mechanisms.TerminalField(
            "score",
            mechanisms.TensorSchema(
                dtype="float32",
                device_class="any",
                dimensions=("B",),
                semantic_axes=("batch",),
            ),
            "terminal-score",
        ),
    ),
    factor_order=(),
    validity_contract="all positions are valid",
    packing_contract="one named dense tensor",
    score_contract="score supplied by the caller",
    consumer_contract="consumer accepts the named value field",
    gradient_contract=mechanisms.GradientContract.autograd(),
)
```

The generic ABI may describe other structured outputs. `FederatedProgram@1`
specifically requires exactly one `terminal-score` field and one
`terminal-validity` field so hard winner selection has explicit semantics.

`ProgramExecutionSignature` binds an autonomous program to one exact terminal
ABI fingerprint. Two programs may use different ranks, lengths, feature sizes,
Formula programs, Fold/UnFold layouts, or execution policies. They can participate
in one federation only when their explicit terminal adapters produce the same
named ABI.

The signature also requires a fixed, deterministic, non-trainable Query and
program-local normalization. A graph-wide softmax is not part of this
contract. Flat Bank concatenation is only a later execution optimization for
compatible signatures, not the definition of program composition.

`ShapeRelation` expressions are declarative contract text in this stage. They
are fingerprinted and inspected, but are not evaluated as code or used as an
implicit tensor adapter.

`FederatedProgram@1` is the eager reference coordinator for these contracts. It
dispatches each autonomous Program independently, validates every explicit child
tensor against the child input schema, keeps at most one global `K` paths, and
selects one valid terminal record. Runtime paths are returned only through
`ProgramExecutionTrace`; they are not operand state or persisted artifacts.

The first reference batches terminal fields only when their declared tensors
can be explicitly concatenated on the batch axis. Ragged terminal ABIs require
their own offsets/packing fields and a later matching backend; the coordinator
never inserts padding, projection, reshape, broadcast, or device conversion.

The reference coordinator deliberately does not promise heterogeneous kernel
fusion, compiled execution, physical savings, or distributed execution. Those
are backend work after terminal parity, not part of the FederatedProgram definition.

## Program-owned Query assets

`FederatedProgram@1` is the fixed-K reference path for a Program that owns its Query.
A Query may be trained with its Program, selected on validation data, then
sealed and saved as a Program asset. Sealing fixes the registered implementation,
configuration, parameter/buffer roles, tensor values, schemas, retrieval
contract, normalization contract, and gradient contract. Query parameters no
longer receive gradients, while autograd may still cross the Query into its
input.

```python
sealed_query = mechanisms.seal_bank_query(trained_query)
mechanisms.save_bank_query(sealed_query, "vision-query.arti.st")

query = mechanisms.load_bank_query(
    "vision-query.arti.st",
    MyRegisteredQuery(...),
)
```

Custom Query classes must inherit `mechanisms.BankQuery` and have a component
registration before sealing. The registration supplies the canonical artifact
identity; an unregistered Python subclass is intentionally runtime-only.

```python
arti.register_component(
    "example/my-bank-query@1",
    component_type=MyBankQuery,
    lifecycle="stable",
    constructible=False,
    config_builder=lambda query: query.contract_config(),
    dependency_builder=lambda _query: (
        "arti/gradient-contract@1",
        "arti/tensor-schema@1",
    ),
)
```

A `BankOwnedQueryProgram` is constructed in two phases. First create the full
program state. Then derive its signature from the registered program and bind
it. This prevents callers from inventing program configuration or state-schema
fingerprints.

```python
class MyBank(mechanisms.BankOwnedQueryProgram):
    def __init__(self, query, terminal_abi):
        super().__init__(bank_id="vision", query=query)
        self.local_formula = MyFormula(...)
        signature = mechanisms.BankExecutionSignatureV2.from_program(
            self,
            input_schema=query.signature.input_schema,
            output_schema=terminal_value_schema,
            shape_relation=mechanisms.ShapeRelation.arbitrary_to_terminal(
                "the local adapter produces the terminal ABI"
            ),
            query_signature=query.signature,
            local_normalization_contract=query.signature.normalization_contract,
            local_formula_ref="example/my-formula@1",
            local_iteration_ref="arti/local-iteration-policy@1",
            terminal_adapter_ref="example/my-terminal-adapter@1",
            terminal_abi_ref="arti/terminal-output-abi@1",
            terminal_abi_fingerprint=terminal_abi.fingerprint,
            score_contract="one Bank-local terminal score",
            gradient_contract=mechanisms.GradientContract.autograd(),
        )
        self.bind_signature(signature)
```

The Program and terminal adapter references must also be registered for a
portable Federation artifact. `FederatedProgram@1` defaults to a configurable
fixed width of `K=8`. Each retained candidate owns an independent Program-local
iteration trajectory: after a Formula changes its tensor, the next step re-runs
that Program's sealed Query on the candidate's latest tensor. Expansion is pruned
back to at most `K` after every local and Federation step, so width remains `K`
rather than growing as `K ** iteration_steps`. `K=1` is the exact serial degenerate
case.

The coordinator never averages candidate values and never applies a
Federation-wide softmax. Program-local normalization remains owned by each Program;
the terminal ABI supplies the final score and `hard_one_winner` selects exactly
one result. Candidate-weighted merging is a separate, explicitly requested
mode elsewhere in ARTI and is not the Federation default.

The v2 implementation remains an eager semantic reference. K candidates are
logically independent paths; this contract does not claim fused batching,
compiled routing, hot mounting, or a physical throughput improvement.

## Shape-polymorphic Program execution

The current alpha execution region admits a `TensorView`
rather than one fixed-rank tensor schema. A view carries named logical axes,
axis roles, and an optional explicit index map. Its Program-owned Query observes
the current view and may be pretrained with that program, then sealed for runtime
use.

Each program-local execution step follows the same order:

```text
query the current TensorView
-> select one Formula action
-> execute that action
-> obtain a new TensorView
-> query again
```

Formula actions are the only payload transformations. The coordinator does not
insert an MLP, projection, padding, reshape, or terminal cleanup layer. Reshape,
permutation, lookup, Fold/UnFold transport, and other changes must be explicit
Formula programs. A Program exits only when its current view satisfies the declared
terminal ABI and its exit action is selected after the minimum local iteration
depth.

`TensorViewPattern`, `TensorViewBankQuery`, `ProgramExecutionSignatureV3`,
`RoutedProgram`, and `FederatedProgram` currently report lifecycle `alpha`.
Their contracts may evolve independently from the stable `FederatedProgram@1`
component. Historical `TensorViewFormulaProgram` and
`FederalRecallV3` names are legacy inspection interfaces, not new construction
contracts.
