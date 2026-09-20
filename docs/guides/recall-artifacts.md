# Recall Bank Artifacts

Recall Bank artifacts are strict `*.recall.arti.st` packages containing only
the trainable Bank values plus a compatibility contract. The current artifact
schema is v4 and its pure-data provenance schema is v1. The contract binds the
host model, shared reader, Formula, optional Updater, Bank layout, shapes,
dtypes, and SHA-256 fingerprints. Provenance records identities and schemas;
it never stores executable code or imports a class named by an artifact.

An artifact from an older schema is rejected and must be exported again with
the current package. ARTI does not silently migrate version, Formula, Updater,
shape, or layout changes.

The `formula="arti/delta@1"` values below are source declarations at a Python
constructor boundary. Construction resolves them to full
`arti/delta@sha256:<contract-digest>` Formula references. Saved Formula
manifests and Recall Bank provenance persist only those resolved references.

## Save And Load

```python
import torch
import arti

host = torch.nn.Linear(64, 64, bias=False)
recall = arti.Retrieve(64, slots=16, formula="arti/delta@1")

contract = arti.create_recall_bank_contract(host, recall, bank_id="portrait")
arti.freeze_for_recall_bank(host, recall)
arti.save_recall_bank(
    recall,
    "portrait.recall.arti.st",
    host=host,
    bank_id="portrait",
    contract=contract,
)

fresh = arti.Retrieve(64, slots=16, formula="arti/delta@1")
asset = arti.load_recall_bank("portrait.recall.arti.st", fresh, contract=contract)
print(asset.bank_id, asset.artifact_version)
```

When a bank belongs to an explicit Formula/Updater pair, record both roles and
pass the same objects or references when saving and loading:

```python
updater = arti.mechanisms.RecallValueUpdater(
    hidden_dim=64,
    slots=16,
    workspace_dim=128,
    depth=1,
)
contract = arti.create_recall_bank_contract(
    host,
    recall,
    bank_id="portrait",
    updater=updater,
)
arti.freeze_for_recall_bank(host, recall)
arti.save_recall_bank(
    recall,
    "portrait.recall.arti.st",
    host=host,
    bank_id="portrait",
    contract=contract,
    updater=updater,
)
arti.load_recall_bank(
    "portrait.recall.arti.st",
    arti.Retrieve(64, slots=16, formula="arti/delta@1"),
    contract=contract,
    updater=updater,
)
```

`contract.provenance` exposes the recorded `reader`, `formula`, `updater`, and
`bank_layout` descriptors. The descriptors are fingerprints and declarative
metadata, not a second execution or serialization mechanism.

Only the Bank parameters may be trainable at export time. Host and shared
reader parameters are frozen so the artifact remains independently composable.
The `RecallBankError` exception exposes a stable `code`, `path`, `field`, and
`action` for invalid kind or version failures.

## Composition

Different named Banks can share one host/reader contract. Their asset IDs are
not part of the shared compatibility fingerprint, so they can be concatenated
without mixing their values through a second routing or output layer:

```python
template = arti.Retrieve(64, slots=16, formula="arti/delta@1")
assembly = arti.RecallBankAssembly(template, contract, updater=updater)
assembly.add("portrait.recall.arti.st")
assembly.add("lighting.recall.arti.st")
merged, layout = assembly.materialize()
print(layout.bank_ids)
```

Composition requires identical host, shared-reader, Formula, shape, and dtype
contracts. It fails before mutating the template when a bank is incompatible.

## Forward Updater

The forward state-update mechanism is a separate stable feature, not an old TTT
session:

```python
updater = arti.mechanisms.RecallValueUpdater(hidden_dim=64, slots=16)
next_values = updater(trace, previous_values, mask=mask)
```

The caller owns detaching, persistence, validation, and any training schedule.
The updater itself is tensor-in/tensor-out and does not create an optimizer or
write an artifact implicitly.

## Explicit Migration

Migration is an opt-in operation with a target reader and target contract. It
can apply a caller-provided tensor transform, but it never guesses how an old
Formula, Updater, shape, or layout maps to a new one:

```python
arti.migrate_recall_bank(
    "old.recall.arti.st",
    "new.recall.arti.st",
    target_expert=new_recall,
    target_host=host,
    target_contract=new_contract,
    state_transform=lambda state: {
        name: value for name, value in state.items()
    },
)
```

The output training metadata records the source package and both contract
fingerprints. Private tensor extensions are not copied implicitly.
