# Runtime Checkpoint Alpha

`save_runtime_checkpoint` writes one complete committed
`VolatileTensorRuntime@1` root to a `*.runtime.arti.st` SafeTensors artifact.
It may also include one fixed `BoundHotPagePool@1` and exact
`ExternalTensorBinding@1` provenance.

```python
receipt = arti.mechanisms.save_runtime_checkpoint(
    runtime,
    runtime.snapshot(),
    "session.runtime.arti.st",
    resident=bound_pool,
    bindings=[target_binding],
)

restored = arti.mechanisms.load_runtime_checkpoint(
    "session.runtime.arti.st",
    expected_abi_fingerprint=abi,
    expected_component_refs=["arti/target-bank-updater@2"],
    resident_device="cuda:0",
)
```

The artifact preserves the committed root, page versions and hashes, page
provenance, idempotency receipts, external bindings, and optional resident
payload/validity/generation/version metadata. Open transaction overlays are not
part of the runtime root and cannot enter the artifact. CUDA pointers are never
serialized; restore allocates fresh storage and produces a new pointer receipt.

Saving holds the runtime commit lock from current-root validation through the
atomic file replacement, so the persisted CPU root has one explicit
linearization point. Save and load also share one process-local lock per
resolved artifact path, preventing a same-process replacement from splitting
manifest, tensor, and file-hash reads across different artifact generations.
This is not cross-process fencing. An optional resident pool is a separately validated
deployment snapshot in the same artifact; it is not claimed to be an atomic
cross-device transaction with that CPU root.

Repeated saves of the same immutable root preserve the same canonical manifest,
root fingerprint, and tensor values. `artifact_sha256` authenticates the exact
physical file produced by one save; it is not a cross-save content address,
because SafeTensors does not make ARTI's canonical-manifest guarantee for its
container byte layout. A failed atomic replacement leaves an existing target
untouched and removes the temporary artifact. Repeated loads create independent
runtime objects: committing into one restored runtime cannot mutate another.

This is a same-ABI, clean-restart, single-process alpha checkpoint. It is not a
WAL and does not claim crash-consistent incremental persistence, cold paging,
cross-process fencing, cross-model ABI translation, or distributed agency.
