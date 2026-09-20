# Volatile Tensor Runtime

`arti.mechanisms.VolatileTensorRuntime` is the storage-only S1 reference runtime for
volatile tensors. It provides immutable snapshots and private copy-on-write transactions
over runtime-owned CPU tensors.

```python
runtime = arti.mechanisms.VolatileTensorRuntime(
    {"bank": torch.zeros(4, 8)},
    world_id="session",
)
snapshot = runtime.snapshot()
transaction = runtime.begin(snapshot, transaction_id="step-1", branch_id="main")
observed = transaction.read("bank")
transaction.stage(
    "bank",
    observed.value + 1,
    expected_version=observed.ref.version,
    provenance_fingerprint="0" * 64,
)
receipt = transaction.commit(idempotency_key="step-1")
```

The same immutable proposal may be reconstructed from the same snapshot and
passed to `runtime.replay(..., idempotency_key="step-1")`. An identical request
returns the original receipt without advancing the epoch; a different request
under the same key fails closed.

The v1 contract is single-process, single-writer, in-memory, CPU-only, and
volatile. Ingress and reads use owned clones; snapshots do not expose mutable
runtime storage. Snapshot reads go through `runtime.read(snapshot, key)`, which
validates runtime ownership and returns a clone. Commit uses strict whole-root
compare-and-swap and publishes
the new root, provenance head, receipt, and idempotency index together.

Existing logical pages keep a stable dtype and shape. A stale-root conflict
receipt includes the page references that changed since the base snapshot for
diagnostics, but v1 still rejects the complete transaction rather than merging
unrelated pages.

The runtime's private Python attributes are encapsulation, not a security
sandbox. Untrusted Python code requires process isolation.

This runtime does not execute Formula, Recall, Fold, Updater, routing, or an
operation log. It has no WAL, checkpoint, restart recovery, GPU residency,
cross-process coordination, merge, or distributed semantics. Those capabilities
must use later, separately versioned contracts.
