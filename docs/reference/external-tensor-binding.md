# External Tensor Binding

`arti.mechanisms.ExternalTensorBinding` connects an exact runtime page and snapshot
to the component contract that produced or consumes it. It is a host-side
adapter, not a neural executor and not a persistent address space.

```python
bound = arti.mechanisms.bind_external_tensor(
    runtime,
    snapshot,
    "target-bank",
    address_namespace="session",
    partition_id="target",
    logical_id="target-bank",
    role="target-bank",
    authority=arti.mechanisms.TensorAuthority.READ_WRITE,
    component_ref="arti/target-bank-updater@2",
    component_config_fingerprint=config_fingerprint,
    state_schema_ref="arti/target-bank-state@1",
    producer_state_fingerprint=state_fingerprint,
    provenance_fingerprint=provenance_fingerprint,
)

candidate = updater(trace, bound.read.value)
proposal = arti.mechanisms.ExternalTensorProposal(
    bound.binding,
    candidate.detach(),
    producer_ref="arti/target-bank-updater@2",
    producer_config_fingerprint=config_fingerprint,
    producer_state_fingerprint=state_fingerprint,
)
arti.mechanisms.stage_external_proposal(transaction, proposal)
receipt = transaction.commit(idempotency_key="step-1")
```

Only the host transaction can publish a new root. Formula, Fold, and Updater
remain ordinary tensor computations that return candidate values.
Crossing the volatile storage boundary is explicit and non-differentiable;
training code must finish its bounded gradient window before detaching a
candidate for persistence.

The following versions are deliberately independent:

- runtime root epoch and `TensorRef.version` describe published storage;
- `FormulaArenaState.version` describes local SSA slot lineage;
- `FoldRecord.permutation` describes one reversible transport;
- Refine step counts describe computation, not page versions.

`ExternalTensorProposal@1` stages only `complete_next_state`. A delta must be
materialized into a complete candidate before crossing this boundary, which
prevents accidental double application. `FoldAddressBinding@1` binds stable
logical IDs and an explicit write set to a `FoldRecord@1` fingerprint. It
records the logical-ID order actually derived from each permutation row; the
permutation itself is never treated as a persistent address.

This alpha contract has no paging, WAL, restart recovery, branch merge,
cross-process ownership, GPU residency, or automatic persistence.
