# GPU Resident Hot-Only Runtime

`arti.mechanisms.HotPagePool` and `BoundHotPagePool` form a single-GPU,
single-process, fixed-bucket alpha runtime. They do not replace the CPU
transaction runtime and they do not implement paging. Pool payload,
generation/version metadata, page references, worksets, route tensors, and
scratch remain on one CUDA device after binding.

The hot-only runtime can also receive an authorized K-wide branch search result via
`arti.mechanisms.bind_resident_branch_run`. That bridge scores the K Recall
trajectories on CUDA, exposes only bounded scalar scores to host authority, and
commits a selected winner or convex mixture through the same fixed page pool.
It does not download K full tensor proposals and does not make runtime branch
receipts portable model artifacts.

The hot path is deliberately narrow:

```text
fixed PageRefs
-> preallocated gather workset
-> caller-owned tensor operation
-> preallocated output/staging
-> generation-bound scatter and version increment
```

`FormulaResidentOperation` is only an adapter to the existing
`FormulaFabricCompute`; it contains no Formula implementation. Route planning,
manifest validation, page allocation, migration, adaptive stopping, logging,
and error handling remain outside the hot path.

`TopologyFormulaResidentOperation` is the corresponding composed adapter for
the existing `Fold@2 -> FormulaFabricCompute -> UnFold@2` path. Fold and UnFold
retain their canonical component identities and exact recorded transport
semantics; the resident adapter does not implement a second topology engine.
The bucket and operation must declare the same fixed refine depth.

## Lifecycle And Ownership

`HotPagePool` owns CUDA page storage. `BoundHotPagePool` borrows that pool and
owns only its fixed references and preallocated work buffers.
`CapturedHotStep` borrows the bound workset and owns its CUDA Graph handle.
All three expose `lifecycle_state`, `closed`, `assert_open()`, idempotent
`close()`, and context-manager cleanup.

Closing a bound workset does not close a potentially shared pool unless the
caller explicitly selects `close_pool=True`. Closing a pool invalidates and
releases every live binding and captured step derived from it. Release performs
a device synchronization while holding the same authority lock used by eager
execution, graph replay, binding, and checkpoint snapshots. Calls after close
fail closed; a closed resident pool cannot be checkpointed. These are
single-process ownership guarantees, not multi-process lease semantics.

Every binding owns a deep CPU snapshot of its `FixedPageRefs`; mutating the
caller's tensors after binding cannot change execution or checkpoint metadata.
All bindings of one pool also share a CUDA event fence. The host lock orders
submission, while the event fence orders physical work when callers use
different CUDA streams. Checkpoint and close first quiesce that fence. A failed
close remains in `closing`, rejects new work, and may be retried; it never
reopens a partially closed object.

`eager_step()` and `CapturedHotStep.replay()` normally return the borrowed,
preallocated `workset_output`. It remains valid only until the next operation
using that binding. Callers that retain or pass the result across threads or
streams must select `copy_output=True`, which enqueues an owned clone before the
authority fence is published. `synchronize()` waits for all pool work submitted
through the authority and surfaces asynchronous CUDA failures. Raw tensor
aliases retained by callers remain ordinary PyTorch references and can keep
storage alive after wrapper close; wrapper lifecycle cannot revoke external
Python references.

`torch.compile(fullgraph=True, dynamic=False)` applies to the caller-owned
tensor operation. CUDA Graph capture is attempted only after warmup with fixed
shape, dtype, route, and addresses, and is restricted to `commit=False`.
Authority-bearing eager commits synchronize their CUDA stream before releasing
the shared host lock. Commit-capable graph capture fails closed because an
asynchronous replay cannot establish that host authority remains held through
GPU completion. Graph failure or fallback is an unsupported result, not
permission to report graph performance.

`ResidentLatencyReceipt@1` reports CUDA-event latency, allocator observations,
pointer stability, and an explicit unknown allocation-call count. Page miss,
eviction, and prefetch are zero by the hot-only runtime contract, not by a
hardware counter. `CUDAActivityReceipt@1` parses a scoped PyTorch CUPTI
activity trace and reports actual kernel count and H2D/D2H/D2D bytes for that
window. Trace export success does not imply that CUPTI reported a dropped-record
count; that counter remains explicitly unavailable. HBM/L2/occupancy metrics
still require Nsight Compute. Any counter not collected by an authoritative
source remains unavailable; unknown values are never encoded as zero.

This stable surface does not claim cold paging, durable persistence,
multi-process consistency, distributed agency, zero-copy transfer, zero
allocation, or speedup. Those claims require separate physical receipts and
their own fail-closed gates.
