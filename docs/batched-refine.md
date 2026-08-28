# Batched Refine Alpha

`arti.nn.Recall` now exposes this breadth execution through `Recall@4` by
default. The ordinary call still returns one tensor: K trajectories execute,
then the highest-scoring route wins. Use `return_branches=True` to inspect the
underlying `BatchedRefineResult`, or call the alpha functions below when a host
needs explicit scoring, publication authority, Formula/Topology composition,
or branch persistence. Route-weighted merging remains an explicit ablation and
is not the default Recall meaning.

ARTI Batched Refine begins with one grouped Recall query. Its Top-K candidates
are preserved as K addressable branch seeds rather than immediately reduced to
one recalled context:

```text
Recall Query
  -> candidate group/value identity [B,N,K]
  -> absolute route mass + selected-mixture weight [B,N,K]
  -> K independent branch states [B,K,N,D]
```

`RecallBranchBatch@3` is the versioned bridge for a single-value Recall.
`RecallFormulaBranchBatch@3` is the factor-aware bridge for product, state, and
versioned custom Recall Formula candidates. Both preserve candidate group and
slot indices, normalized within-group slot weights, raw candidate Bank context,
weighted route mass from the full route axis, selected-mixture weight
from the existing Recall Top-K read, token/branch masks, Recall layout
fingerprint, Formula identity, and source component identity.
These measures are separate because truncating K must not silently turn a
relative mixture weight into candidate route strength. With per-Bank
normalization, route mass is a weighted finite measure whose total is the sum
of enabled expert weights; it is deliberately not renormalized into one global
probability, because that would reintroduce concat dilution. `expand_state()`
allocates independent branch storage; it does not expose one writable expanded
view.

The bridge is runtime-only. It binds the originating Recall instance and the
PyTorch versions of its query, key, group, and value tensors; execution rejects
foreign or stale candidates instead of silently replaying identities against a
changed Bank.

The runtime class/reference/schema triples are strict:

| Runtime class | Canonical reference | Schema |
| --- | --- | --- |
| `RecallBranchBatch` | `arti/recall-branch-batch@3` | 3 |
| `RecallFormulaBranchBatch` | `arti/recall-formula-branch-batch@3` | 6 |

The two classes are not interchangeable even when their tensors happen to have
compatible shapes. Candidate and authority manifests use their own `@2`
internal identities; older runtime manifests are not migrated or persisted.

Recall parameters must be constructed or loaded outside
`torch.inference_mode()`, so PyTorch supplies mutation version counters. Query
and execution may still run inside `torch.inference_mode()`. If the hidden input
is itself an inference tensor, the bridge captures a versioned immutable
snapshot and verifies the live value against it before expanding branches.
Candidate, result, and diagnostic tensors are likewise normalized into
versioned runtime storage. Pointer identity alone is never accepted as mutation
provenance.

The returned candidate group seed has width one and can be fed back into the
same grouped Recall kernel. Explicit routing accepts any fixed width from one
through the component's configured `group_topk`; automatic routing still uses
the configured width.

Grouped single-value and composed Recall Formula candidates are supported.
For a composed Formula, each branch is one complete factor tuple. Two explicit
coherence contracts are available:

- `cross-bank-joint-factor-beam@1` forms joint tuples across the global route
  axis. It requires global Recall normalization and does not accept Bank
  quotas.
- `same-bank-joint-factor-beam@1` constructs complete tuples independently
  inside each Bank, then interleaves those tuples according to a reserved or
  explicit per-Bank quota. It never creates a cross-Bank tuple and filters it
  afterward. This is the default for Recall@3 per-Bank normalization.

Source width C, Formula beam J, and static branch capacity K are separate
limits: `K <= J <= C ** F`. For cross-Bank coherence, J is the bounded joint
search width and does not materialize the full Cartesian product. For same-Bank
coherence, each Bank's explicit quota is its joint search width; J remains a
declared upper-bound contract and does not create another hidden global
frontier. The candidate record preserves every factor's selected route, the
complete joint score, allocation policy, partition coherence, and static Bank
quota. Dense Recall has no bounded Top-K identity in this contract; unsupported
combinations fail closed.

`max_k` and `partition_quota` describe static candidate capacity. A per-sample
`active_k` request may use only a prefix of that interleaved capacity, so a
small ragged request can legitimately activate fewer Bank partitions than the
static quota names. The bridge records `requested_active_k`, effective
`active_k`, `active_partition_mask`, `active_partition_count`, and
`active_branch_count_by_partition` separately. This makes capacity clamping,
the exact per-sample Bank participation, and each Bank's active candidate
contribution observable without forbidding ragged breadth. For cross-Bank
Formula tuples, one branch may contribute to more than one partition count.

`run_batched_refine()` consumes this bridge and runs all K candidate trajectories
through the existing Recall state executor in one tensor call. The K axis is
flattened only at the kernel boundary and restored on values, deltas, and
branch diagnostics. Each branch uses its candidate group at the first step;
later steps query the Bank again from that branch's changed hidden state.

`BatchedRefinePlan@1` may additionally bind an existing
`FormulaResidentOperation@1` or `TopologyFormulaResidentOperation@1`. Every
Recall refine step then follows the same executor path:

```text
candidate-seeded Recall
  -> optional Fold@2
  -> resident Formula execution
  -> optional UnFold@2
  -> branch state update
  -> Recall re-query from the changed branch state
```

The plan does not implement a second Formula, Topology, or Recall engine. Its
Formula route, optional factors, and topology identity are resident operation
state and participate in normal `arti.st` module save/load.

### Execution layout

`BatchedRefinePlan@1` defaults to `execution_layout="static_capacity"` and
also supports `execution_layout="packed_active"`. The static layout executes
the fixed `B * max_k` row capacity. The packed layout computes one canonical
`active_flat_index`, gathers initially eligible rows into `[P,N,D]`, runs the
same Recall and optional Formula/Topology operation, and scatters back to the
fixed `[B,K,...]` result ABI. Inactive branches preserve the original input
and have zero delta.

The same index selects the token mask, branch schedule, selected Recall groups,
resident Formula route and factors, Topology inputs, and branch-origin keyed
RNG. A zero-row batch bypasses Recall, Formula, Topology, Half, dropout, and
refine. Results expose `batched_static_capacity_rows`,
`batched_eligible_branch_rows`, `batched_physical_branch_rows`,
`batched_initially_inactive_rows`, `batched_packing_mode`,
`batched_packed_active_flat_index`, and `batched_executed_branch_mask`.

Only rows selected by `batched_executed_branch_mask` have execution-diagnostic
semantics; inactive rows are fixed-ABI padding. Packed execution currently
compacts only initially eligible K. A branch that stops during refine remains
inside the P-row kernel under the existing device-side logical mask. Dynamic
repacking of stopped branches is not implemented.

Eager execution accepts arbitrary P. `packed_active` is currently eager-only
and fails closed under direct `torch.compile` or CUDA Graph capture. Compiled
and CUDA Graph deployments require a future fixed-P bucket executor with fixed
buffer addresses; a single graph cannot accept arbitrary dynamic P. A
physical-row profile proves the core input layout, not a universal FLOP, HBM,
or latency improvement.

When the existing Recall path uses `Half`, survival is evaluated independently
for every trajectory after the K axis is expanded. A weak correction in one
candidate therefore cannot thin or preserve another candidate's correction.
Stochastic Half and training dropout require an explicit
`ExecutionRNGPlan@2`. Streams are keyed by a stable call-site `stream_key`,
stable sample identity, canonical
branch origin, stochastic phase, and absolute refine step. Replaying one plan
does not consume the global Torch RNG and candidate-axis permutation only
permutes the corresponding trajectories. The alpha promises replay only on
the same backend/device family and algorithm version; it does not claim
CPU/CUDA bit identity. Candidate route exploration and refine re-query use
separate keyed domains, so adding Half or dropout does not consume their random
streams. `ExecutionContextReceipt@3` records the stable stream key and only the
domains actually consumed; passing an unused plan leaves execution
deterministic. An exploratory candidate batch records the query-plan
fingerprint and rejects execution under a different plan.

This is the numerical execution stage, not persistent branch authorization. A
complete runtime path must still attach branch-local COW deltas, scoring,
receipts, and explicit host commit. Ordinary sample batches, score-only Top-K,
and latent K^n credit spaces are not evidence that K branches were executed.

Recall-only plans report zero Formula work. Composed plans derive Formula cell,
fire, commit, route-application, topology permutation, and successful UnFold
receipts from the resident operation that actually ran. Formula work is
reported separately as declared, invoked, and effective work; logical masking
must not be presented as physical GPU work elimination.
## Meaning of batch

`Batched Refine` is breadth search over Recall candidates. One Bank query
returns `K` candidate routes, and ARTI executes `K` independent refine
trajectories. It is not the ordinary model batch axis and it is not the legacy
two-transaction harness.

The numerical path keeps an explicit `[B, K, N, D]` state and flattens only at
the existing Recall kernel boundary. Each trajectory owns its hidden state,
route history, stopping state, and diagnostics. The first iteration is seeded
by its candidate route; later iterations query the Bank again from the changed
hidden state.

The current alpha aligns configured Formula routes and factors with the
flattened `[B*K, ...]` branch order. Global and per-sample candidate-axis
permutation, ragged per-sample requested/effective `active_k`, independent per-branch refine
budgets, and operation-emitted Formula/Topology traces are supported by the
numerical executor. The optional `packed_active` layout physically compacts
rows that are eligible at dispatch time. It does not dynamically remove
branches that stop inside the refine loop.

## Optional publication authority

`BranchBatchHarness` is the alpha host authority for publishing one completed
trajectory into a volatile tensor page. It accepts only factory-owned
`BatchedRefineOverlayProposal` values rooted at one immutable snapshot. A
frozen score receipt binds all `K` candidates to the same external target, and
the harness applies the declared tie policy before committing at most one
winner. Callers cannot pass a winner directly. Alternatively, the host may
publish one finite convex mixture of the K private deltas, or discard every
candidate. These are explicit authority decisions rooted at the same snapshot;
explored branches never update the shared Target page on their own.

The volatile authority runtime remains the portable CPU COW path. BR6.5 also
provides an explicitly separate GPU-resident authority bridge:

```python
executor = arti.alpha.BatchedRefineExecutor.from_resident_result(result)
run = arti.alpha.bind_resident_branch_run(result, executor, spec, future, pool)
score = run.score()  # only K scalar scores cross to the host
decision = run.decide(score, idempotency_key="request-42")
receipt = run.commit(decision)
```

`ResidentBranchRun@1` keeps the complete `[B,K,N,D]` branch values and deltas
on CUDA. Version 1 accepts a detached, contiguous, host-owned future target,
verifies its authority fingerprint on CPU, and uploads it once for scoring.
The frozen MSE scorer then runs on CUDA; only the K scores and bounded
authority metadata return to the host. A GPU-produced target without a trusted
host fingerprint binding is rejected rather than downloaded implicitly. The
host may authorize one canonical winner, one finite convex mixture, or discard.
Winner gather and FP32
fixed-order mixture reduction run on CUDA, and publication goes through the
existing `BoundHotPagePool` generation/version checks. A changed result,
future target, pool pointer layout, generation, or version fails closed.
All runs bound to one page pool share its host authority lock, so the
generation/version check and commit are serialized inside one process. This is
not advertised as a distributed GPU compare-and-swap primitive.

The resident executor binds live tensor identity, pointer, PyTorch version,
shape, dtype, device, and source lineage. It does not compute content hashes by
copying result, delta, diagnostics, or candidate context to CPU. The ordinary
`from_result()` executor retains its content-fingerprinted CPU authority
semantics; resident publication requires `from_resident_result()` explicitly.

The resident runtime contracts are deliberately non-portable:

| Runtime contract | Canonical reference | Artifact policy |
| --- | --- | --- |
| bound run | `arti/resident-branch-run@1` | `host_bound` |
| score receipt | `arti/resident-branch-score@1` | `runtime_only` |
| host decision | `arti/resident-branch-decision@1` | `runtime_only` |
| commit receipt | `arti/resident-branch-commit@1` | `runtime_only` |

They are not serialized as `arti.st` model components and do not impersonate
the CPU `VolatileTensorRuntime` commit receipt. Version 1 requires the same
active branch set for every sample in the ordinary batch; ragged per-sample
authority is deferred rather than silently changing score semantics.

This resident path removes the full-result device-to-host staging step. It can
be paired with either the static-capacity executor or the optional
`packed_active` numerical layout. The latter proves that initially ineligible
rows can be removed from the core execution input; it does not prove that
branches stopped later are removed from an in-flight kernel.

The reproducible authority-window profiler is:

```text
uv run --extra dev python scripts/profile_batched_refine_resident_authority.py
```

The checked RTX 5070 Ti provenance-bound physical receipt at
`docs/reference/batched-refine-resident-authority-profile.json` uses
`B=4,N=64,D=128,K=4`. Full K value and delta payloads are 524,288 bytes each;
the complete profiled authority window observed 10,274 D2H bytes, below one
131,072-byte branch. This supports the narrow claim that no complete branch
payload crossed to the host. It also observed 63 kernels and 33 CUDA runtime
synchronizations, so operation fusion and synchronization remain open work.
HBM read/write counters are explicitly unavailable. The receipt binds the
resident result manifest, executor config/state, execution context, authority
spec, score, decision, commit, pool layout, selected source digests, runtime
environment, and exact profiler trace fingerprint. It remains a bounded
single-run physical receipt, not a downstream efficacy result or a distributed
transaction proof.

The separate packed-layout profile at
`docs/reference/batched-refine-packed-active-profile.json` uses
`B=8,N=64,D=128,K=8`, with two initially eligible branches per sample and four
refine steps. Static and packed results agree within
`2.384185791015625e-7`. Packed execution reduces physical core rows from 64 to
16 and the measured peak allocation increment from 121,022,976 bytes to
30,325,248 bytes. It is not faster in the current eager implementation: median
latency rises from 10.028 ms to 12.211 ms and observed kernels rise from 488 to
545. The supported conclusion is therefore narrower than a general speedup:
initial active-K packing removes physical rows and lowers peak allocation, but
its gather/scatter and launch overhead still need optimization. HBM counters
remain unavailable.

For single-value Recall, `K` is bounded by the configured Recall routing width
(`group_topk`). For composed Recall Formula candidates, source width C,
Formula beam J, and branch capacity K are declared independently, with
`K <= J <= C ** F`. The branch coordinator supports any `K >= 1` supplied by a
valid candidate result.

## Recovery boundary

Candidate queries, live `[B,K,N,D]` trajectories, score workspaces, and
uncommitted COW overlays are runtime-only. They are deliberately excluded from
checkpoint artifacts: restoring them would blur the authority boundary between
an explored candidate and published state.

After the host commits one winner, the resulting Target page, world root, and
idempotent commit receipt use the ordinary `VolatileTensorRuntime` checkpoint:

```python
receipt = harness.decide(score, idempotency_key="request-42")
snapshot = runtime.snapshot()
arti.alpha.save_runtime_checkpoint(runtime, snapshot, "state.runtime.arti.st")
```

A fresh same-ABI load restores only committed state. Losers, discarded runs,
and partially staged branches do not reappear. To explore again after restart,
issue a fresh Recall query and create a new branch spec from the restored root.

## Evidence boundary

The CUDA mechanism receipt compares score-only Top-K, true K-way refine, and a
single trajectory with matched logical token-refine work. It reports CUDA-event
latency, profiler CUDA events, and allocator peaks. HBM byte counters remain
explicitly unavailable unless an external Nsight/CUPTI hardware-counter run is
performed. Its random frozen target tests execution and selection mechanics;
it is not evidence that an untrained Recall Bank improves a downstream task.

The checked receipt at
`docs/reference/batched-refine-cuda-profile.json` also contains a separate
Formula/Topology control. The same frozen candidates run through Recall-only,
Formula-only, and `Fold@2 -> ADD Formula -> UnFold@2` plans. On the checked RTX
5070 Ti run, the topology path was deterministic, verified every UnFold, and
differed from Formula-only by an MSE of `2.1025805473327637`. Its median latency
was `16.452688217163086 ms`, versus `11.308800220489502 ms` for Formula-only.
This proves that the existing composed executor applies a topology-dependent
transition and records its physical cost. It does not prove that the chosen
topology improves a downstream task or accelerates execution.

The analytic breadth task supplies a narrower positive result. One query
returns two mutually exclusive latent hypotheses. A calibration prefix chooses
one branch and a held-out suffix scores it. With equal logical token-refine
work, K=2 breadth plus four refine steps reaches lower held-out error than K=1
plus eight refine steps. It also outperforms score-only Top-K and a reset Bank.
This demonstrates that preserving and evolving multiple query candidates can
cover alternatives that deeper execution of one candidate cannot. It remains
a controlled mechanism task, not a trained-model or downstream-quality claim.

The trained-asset replay uses a frozen Z-Image Recall Bank and real saved hidden
traces. It verifies that the branch executor accepts an existing trained Reader,
preserves K candidate identities, and executes the K trajectories efficiently.
Its full-Top-2 reference is not equal to selecting or mixing branch endpoints,
and the observed absolute Reader perturbations are small. That receipt must not
be cited as evidence of image-quality improvement or trained breadth-search
superiority.
