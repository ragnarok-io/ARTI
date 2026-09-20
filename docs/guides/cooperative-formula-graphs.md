# Cooperative Formula graphs

`arti/formula-program-query@7` is an alpha, native reference composition of
ProgramQuery@6 and the existing named SSA arena. It executes multiple ready
ordinary operations against one snapshot, publishes their completed products,
and lets later operations consume any declared combination of these products.
It does not introduce a separate routing network.

## Binding alternatives

A candidate represents a Formula occurrence and its input-source tuple. Offer
different tuples as different candidates. Use `with_bindings` on a
`FormulaProgramTensorCandidateV4` to share the actual Fabric, trainable operands,
and plastic Bank owner while changing its SSA wiring:

```python
from arti import mechanisms as m

# existing_candidate is a named-output Tensor candidate. These are its actual
# Formula input/output names, not inferred positional connections.
alternative = existing_candidate.with_bindings(
    "use_second_view",
    input_slots={"x": "second_view"},
    output_slots=existing_candidate.candidate.output_slots,
)
```

Different alternatives may compete for the same output slots. Once an
alternative writes them, competing writers become unavailable. Continuations
still name scalar outputs of real executed federation members; those responses
choose both the operation and its binding alternative. The source space is
finite and declared, not an unrestricted enumeration of all tensor pairs.

```python
graph = m.FormulaProgramQueryV7(
    slot_ids=declared_slots,
    candidates=candidates,
    terminal_slots={"answer": "answer"},
    entry_candidates=entry_candidates,
    continuations=executed_response_edges,
    cooperation_width=4,
    max_steps=32,
)
initial = graph.initial_bank_state()
result = graph(inputs, bank_state=initial)
answer = result.outputs["answer"]
replayed = graph.replay(inputs, result.frontiers, bank_state=initial)
```

`cooperation_width` is the maximum number of actual ordinary operations in a
frontier. It is **not** alternative-answer beam width or the number of views
inside an Observation. `max_steps` counts all non-STOP dispatches, including
every ordinary sibling. Child work is additionally visible in
`result.trace.total_dispatches`.

## Original-input Observation

Every Observation occurrence must bind its substrate to the same original
event input `x0`. Earlier products can generate a new displacement and can remain
inputs to other operations:

```text
x0 -> Observe(d1) -> view1 -> ordinary Formula -> d2
x0 ------------------------------------------> Observe(d2) -> view2
view1 -----------------------------------------------------+--> relation
view2 -----------------------------------------------------+
```

Binding the second substrate to `view1` instead of `x0` performs cascaded
resampling, a different computation. `observe_affine` is feature scale/shift;
`observe_fourier` is circular spatial displacement. Neither distinction is
silently rewritten by the graph executor.

## Products, state and gradients

`result.products` contains all completed immutable SSA values, including
nonterminal products. Capacity is bounded by `slot_ids`; tensor sizes follow
their existing Formula types. This is not a byte-budgeted paging system.
Products do not disappear because they were not the final answer. Multiple
consumers share the original tensor and their cotangents sum through the same
producer. Replay executes each recorded occurrence once, in its frontier, with
all recorded input dependencies.

CALL records include recursive child decisions. Serial ProgramQuery@5/@6
children keep their selected action sequence; cooperative @7 children keep
their actual frontiers. Replay follows these decisions even when new inputs or
changed parameters would lead a fresh search elsewhere. Values, operands and
gradients are recomputed, not loaded from the old tensors. Supply the same entry
inputs, parameters and Bank snapshot when numerical reproduction is required.
Child invocations still use normal module calls and hooks. Shared child instances
retain their shared Bank owners, with the parent overlay passed in call order.

Same-frontier siblings cannot consume each other's unfinished outputs. Their
output and empty-slot requirements must be jointly compatible. Effects and
child calls currently run in singleton frontiers. Effects retain the existing
real immediate-predecessor Bank semantics, not a write to every data ancestor.
All selected effects belong to this one cooperative graph; no donor-beam Bank
states are implicitly averaged or adopted. `commit_` remains explicit.

`selection_log_score` records local sequential masked scores for surrogate
training. It is not a probability of the deterministically selected graph and
does not add ancestor route scores again for each consumer. Ordinary value
autograd is exact for the executed DAG; gradients through discrete choices
require a separately chosen estimator and endpoint objective.
`decision_log_score` includes each nested cooperative decision once, including
cooperative grandchildren inside serial calls. Serial @5/@6 children do not
publish their local selection scores through this property; it is not a complete
graph-risk estimator for a mixed-version hierarchy.

## Current boundaries

- Native hard execution handles one episode at a time. Compatible ordinary
  numerical work can use the existing `execute_many` grouping. This is not a
  GPU-captured, device-only scheduler or a throughput claim.
- Native execution retains its own completed products. The repository's
  `search_cooperative_graphs` experiment entry additionally supports live
  cross-path products as described below. Per-input device handles remain work.
- CALL replay requires the recursive decision trace returned by the original
  execution, including invocation identities and named input bindings. A bare
  parent CALL name is not enough to replay the child graph.
- The wiring tests are not evidence that the model has learned where to observe
  or has completed persistent learning. No intermediate view, displacement, or
  route teacher is prescribed by this API.
- Existing ProgramQuery@6 semantics and component identities remain unchanged.

## Cross-path completed products (reference search)

The repository experiment entry in `benchmarks/_federated_recursive_search.py`
uses the same `advance_frontier` implementation as native V7 execution:

```python
from benchmarks._federated_recursive_search import (
    start_recursive_search, search_cooperative_graphs,
)

search = search_cooperative_graphs(
    (start_recursive_search(graph, inputs, bank_state=initial),),
    product_slots=("source0", "source1"),
    publish_slots=("view1", "view2"),
    width=16,
    beam_width=16,
)
result = search.winner.execution
```

Directory slots must already be declared in the graph and reserved exclusively
for imports. Ordinary candidate binding alternatives can reference them. Source
selection still uses actual executed federation responses, not a host scorer.
`width` limits alternative frontier heads; `cooperation_width` limits the actual
ordinary siblings within each frontier. The head fixes one legal action and the
same native scheduler fills its compatible siblings. This is a greedy beam
heuristic, not an exact distribution over graphs.

Each round reads the previous directory snapshot. Completed outputs publish at
the end of that round, before answer pruning, whether or not the donor reached
STOP. The bounded directory retains the first completed products in execution
order; once full it does not admit later products. This is an append-only native
reference, not dynamic paging or a learned retention policy. Product count is
bounded; total bytes still depend on declared tensor shapes.

An imported product is the same Tensor object with its original Bank lineage.
It adds no dispatch, does not install the donor's Bank overlay, and remains
differentiable after the donor answer branch is pruned. A product computed after
a private donor effect can carry gradients to that effect without committing it
in the consumer. Distinct occurrences retain separate identities even when
their Bank owner and integer revision match. Existing effect admission still
checks the actual current Bank value and revision; adoption does not refresh an
old product into a current write target.

The result contains the retained products and named external source references.
Frontiers with external inputs reject ordinary replay; use the dependency
reconstruction entry below instead. Reusing cached no-grad values is not a
substitute. A search graph with only
local inputs can use the existing V7 frontier replay, including saved STOP
identity. CALL executes a complete native child; unfinished child-internal
products are not published across the call boundary.

This entry is an experiment path, not a device scheduler or the
endpoint learning estimator. Its local selection scores omit unadopted donor
decisions; correct value gradients alone do not establish discrete graph credit.

## Dependency replay after no-grad search

The search records actual input-product and Bank-version dependencies in a
runtime-only tape. It does not use cached products as constants during training:

```python
from benchmarks._federated_product_replay import replay_cooperative_dependencies

rebuilt = replay_cooperative_dependencies(
    graph, inputs,
    tape=search.dependency_tape,
    endpoint=search.winner.dependency_endpoint,
)
answer = rebuilt.outputs["answer"]
next_bank = rebuilt.bank_state
```

Each adopted occurrence is recomputed through the original Formula candidate.
Input references resolve to current caller values or a previously reconstructed
product. Bank references identify an initial root snapshot or a particular
effect occurrence, not just an integer revision. Two branches can have the same
owner and revision but different values. Only required data/state ancestors are
recomputed; an unrelated donor prefix or later same-owner write is not included.

The endpoint's write participation is separate. Reconstructing a private donor
effect to obtain a product does not commit that effect in the consumer's state.
Returned proposals follow the endpoint's actual write order, while shared value
gradients still reach private donor effects. Initial fast states are excluded
from an optimizer by the training harness, not detached by this replay code.

`replay_cooperative_dependencies_many` shares one reconstruction cache across
several retained endpoints of the same event. Combine their losses before
backward. Each result's `executed_occurrences` reports only additional work done
for that result; a shared producer or effect is not counted/executed twice.

Supply `initial_states` to use fresh compatible fast-state tensors, one per
search root. With unchanged inputs, parameters and states, replay reproduces
the saved numerical choices. After changing parameters, it evaluates the same
choices with new values; it does not reproduce the old numerical output or
perform a new search. Bank-only candidates still receive current root arena
batch/device context, without inventing a data dependency on an unused input.

This is a value/state replay, not replay of a graph-policy probability. It
returns no synthetic full execution trace or stale decision score. A CALL is
explicitly an indivisible complete child invocation with its recorded decisions
and normal hooks. Child-internal per-output slicing, dynamic directory paging,
device scheduling and endpoint discrete-choice credit remain separate work.

## Prepared device input sources

The internal device execution path accepts a port-level `FormulaDeviceSources`
view. Before selection its shape is `[row, action, input_port]`; numerical
dispatch receives `[packed_action, input_port]`. Port order follows each
candidate's named input mapping, not its unique SSA slots. Two named inputs
bound to the same local slot can therefore consume different completed values.

`FormulaDeviceSourceBindings` resolves local inputs or completed directory
references into pool handles, availability, numerical validity and original
producer/Bank lineage. A directory reference also identifies its actual
occurrence and output port. Unavailable references do not fall back to local
inputs. Only successful, allocated numeric outputs are published through
`completed_sources`; their payloads stay in the existing pools.

Admission, joint typed-shape matching, grouped numerical execution and frame
publication use the same resolved sources. CALL entry imports the selected
values and lineage without installing donor Bank state. A self-effect on an old
product still requires the receiver's current Bank handle and revision to
match. Ordinary consumers can continue using the old numerical product.

The existing `FormulaDeviceSearchWave` accepts views covering all `2K` frame
rows. It admits sources before ranking and maps selected rows back through
pruning/forking before group packing. `FormulaDeviceCapturedExecutionWave`
supports fixed-address source buffers; replay can change their contents without
replacing the buffers. The standalone captured numerical wrapper does not yet
accept this view.

Each prepared row/action currently has one source tuple. The root cooperative
search below connects these views to publication and pruning. Complete CALL
publication and root numerical dependency replay are described below.
Local source occurrence fields alone are not a dependency tape. No throughput
or learning improvement is implied by these execution checks.

## Prepared cooperative device rounds

`FormulaDeviceCooperativeWave` reuses the existing decision, grouped execution,
typed pools and frames for one V7 round. It separates parent rows, expansion
heads and cooperating ordinary nodes as `[R, W, C]`. It scores current producer
responses once, constructs each frontier against the same entry snapshot, and
deduplicates equivalent sets before numerical dispatch. Encountering a higher
ranked control operation ends an ordinary frontier; it does not skip that
operation to select a lower ranked ordinary candidate.

Successful siblings merge their declared outputs and original lineage into one
frame. A failed sibling prevents publication of the entire frontier, while
already attempted pool allocation remains accounted for. Effect and control
operations occupy singleton frontiers. Numeric products receive occurrence IDs
in parent/head/sibling order, independent of execution grouping. CALL entry only
enters the child frame; complete CALL-return publication is not supplied here.

Selection consumes raw response logits, not already normalized scores that can
lose smaller differences after a dominant head is removed. FP16/BF16 responses
accumulate in FP32. Each action promotes to FP64 only at the point required by
its actual reference/source values; an unused FP64 bucket cannot change it.
Mixed-pool storage preserves these per-row and per-action arithmetic semantics.
Sequential masked scores
match the reference selector but are not a joint graph-policy probability.
One prepared round can be captured as a CUDA Graph and replayed with changed
inputs at stable buffer addresses. The older single-action `SearchWave` rejects
V7 instead of silently running it with serial semantics.

The round returns all surviving expansion heads and pending numeric products.
The search layer, rather than this round, manages directory lifetime and beam
pruning.

## Prepared root cooperative search

`FormulaDeviceCooperativeSearchWave` connects V7 rounds to a fixed K active and
completed beam. Every round resolves named port alternatives from its entry
directory, executes expansion heads, publishes completed products, and only
then prunes answer paths. A donor need not reach STOP or survive pruning for a
later receiver to consume its real product. Receivers retain their own Bank
snapshots; importing a product does not install a donor's Bank or its writes.

The bounded directory stores metadata, not copied payloads. It retains the
oldest publications in occurrence/port order and reports capacity drops. Pool
allocations remain live for the event even when paths are pruned. Completed
answers carry their frames and scores without executing another STOP. Per-path
score precision follows forks and pruning: an unused FP64 pool does not change
FP32 cumulative arithmetic, while a genuinely promoted path stays FP64.

Independent events can run through `forward_batch`, with separate pools,
directories, cursors and frames. A fixed four-round sharing graph is covered by
CUDA Graph replay with changed input values. Its receiver performs four numeric
operations while the search performs five, including the pruned donor. Counters
report selected numeric attempts and successful-frontier numeric operations;
they do not measure padded GPU lanes, FLOPs or latency.

Existing finite binding alternatives remain the source-choice
surface; it does not invent arbitrary source tuples. `product_bindings` can bind
a specific `(candidate_id, input_name)` to a directory slot. Two named ports
sharing a declared local slot can therefore select distinct products.

### Recursive CALL completion

The search completes each selected root CALL through the existing frame stack
before publishing root products or pruning. Nested calls and shared child
modules retain separate invocation frames. Each child uses native local
selection: one hard-selected action for V5/V6 or one cooperative frontier for
V7. Child decisions do not add to the root beam score. V5/V6 compare finite raw
logits even when an unused normalized low score overflows; V7 retains its
conditional score contract. Legacy Query network precision is preserved even
when its payload pools use a lower precision.

Only an actual RETURN publishes the declared CALL outputs. Their external
occurrence and port identify the root CALL, while numerical producer and Bank
version metadata retain the actual child's lineage. Child-private products are
not exposed to the root directory. All root products publish in original
frontier order, not child completion order. A returned row is frozen while
other calls finish; unsuccessful calls publish nothing and do not reclaim
attempted pool allocations.

The prepared child-round bound counts call edges, including separate calls to
the same child module. Pending masks keep semantics correct but do not imply
zero physical work for unused lanes. `call_steps` retains before/after frames
and frontier grouping; separate dispatch/return counters supplement numeric
attempt/completion counts. Tests cover nested calls, real Bank effects, pruned
CALL donors, independent batched events and whole-search CUDA Graph replay.
These records also feed the numerical CALL replay connection below.

## Device records to numerical replay

The private experiment decoder converts the completed root search's actual
accepted frontiers, port references and prune mappings into the existing
`CooperativeDependencyTape`. It transfers metadata after an event, not payloads
or per-node values inside a device wave. Product identities survive donor
pruning. Root identity and writer occurrence distinguish Bank versions even
when their revision numbers match.

Dependency inputs are keyed by Formula input port name. When distinct ports
share a declared local slot but have different references, the original ordinary
candidate receives the reconstructed named inputs directly. Other cases retain
its normal binding path, including existing causal-attention input adapters.
Numerical operations, fast-state ownership and output lineage still belong to
the existing candidates; the decoder does not implement another Fabric.

Replay uses fresh inputs, current slow operands and explicitly supplied initial
Bank snapshots. Adopted donor values and their necessary private effects are
recomputed once per occurrence, and shared consumers accumulate gradients.
Endpoint writes remain separate from this numerical closure. A donor effect can
receive a meta-gradient without being retained in the receiver's Bank. Tests
cover retained writes, mixed second derivatives, distinct same-revision roots,
non-symmetric same-slot input ports, and replay after CUDA-captured search even
after its old numerical pool is cleared.

For CALLs, the decoder rebuilds the existing serial or cooperative child trace
from actual device decisions and invocation frames. It preserves nested calls,
frontier grouping, original execution lineage and named return ports. The native
child module replays this fixed trace using current tensors and Bank versions;
it does not select a replacement route when the input or slow operands change.
A complete CALL is one memoized dependency occurrence. Its internal proposals
remain ordered, and only its explicitly retained root occurrence installs those
writes into the replay endpoint. Multiple consumers of a donor CALL do not
duplicate its execution or adopt its private writes.

CALL input views preserve both the value and original lineage per named port.
When two ports share a declared parent slot but select distinct products, the
native child entry receives both independently. Output lineage can therefore
remain a valid predecessor for a later child's effect, without renaming it to
the CALL wrapper. Tests compare nested traces, changed-input fixed routes,
multihead donor gradients, ordered shared-Bank writes and mixed derivatives.
CUDA-captured search records can be replayed after both old numeric pools are
cleared; no cached payload supplies a training value.

The default decoder produces numerical replay of a selected root graph.
Opting into `include_decisions=True` additionally reconstructs choice credit as
described below; neither mode defines a joint graph probability. The decoder
runs on the host after search. Physical event-plus-backward cost remains to be
measured, and backward is not CUDA-captured by this connection.

## Fixed-panel endpoint training

The private reference search now records decision dependencies separately from
numerical products and retained writes. Each frozen decision includes its
pre-frontier product and Bank-version view, actual sibling order, predecessor
decisions and execution counts. Replay reconstructs the full response snapshot
and calls the native forced frontier selector. Unselected competitors remain in
the normalization denominator. Completed prefix CALL traces are restored only
for calls actually made by that branch; importing a donor output does not
pretend its CALL executed in the receiver.

An endpoint's energy counts its decisions and adopted numerical ancestors'
producing decisions once per occurrence. Computing a competing response may
require further numerical dependencies without adding their decisions to this
energy or their effects to the endpoint Bank. A shared occurrence can contribute
to multiple endpoint losses while being reconstructed once within a group.
CALLs remain opaque complete invocations; their recorded cooperative child
decisions contribute once. Choice credit currently requires V7 child traces.
Serial V5/V6 children remain usable for numerical replay, but are not silently
accepted as having complete choice credit.

For a frozen completed panel, let `s` be these energies and let
`L = L_answer + lambda * L_readonly_reuse(B_return)`. The training adapter uses
`p = softmax(s / T)` and reports `R = sum(p * L)`. At the same parameter point it
freezes `alpha = p` and `beta = p * (L - R) / T`, then accumulates gradients of
`sum(alpha * L_replay + beta * s_replay)` over bounded replay groups. The latter
scalar carries the first-order gradient; it is not the reported risk value.

The same answer and readonly Bank evaluators run in both passes. Their questions,
targets and random samples must be fixed outside the writing search. The adapter
does not install any candidate Bank or perform an optimizer update. The caller
updates unique slow parameters once after all groups, excluding fast Bank
state. Incoming differentiable Bank state from an earlier event remains
connected; neither the returned Bank nor shared response ancestry is detached.

This objective differentiates a fixed finite panel, not greedy pruning or a
joint search probability. It adds no node-count reward or extra legacy routing
loss, and deployment still selects a real graph rather than averaging outputs
or Bank states. Numeric-only device tapes explicitly do not provide the
required decision snapshots. Full event throughput and downstream learning
remain separate work from this training connection.

## Device choice replay

Device search records the pre-frontier frame and each sibling's actual remaining
action mask. The optional decoder preserves response producer identity, the
chosen sequence and per-candidate named-port sources. It does not install donor
products into local SSA to simulate admission: two ports declaring the same
local slot can still refer to distinct completed products.

Differentiable replay rebuilds current response tensors through the existing
occurrence memo. It reuses the device response arithmetic and frozen-sequence
scorer, including global action width and per-row FP32/FP64 accumulation. No
argmax or new admission runs in this scorer. Unselected competitors remain in
their recorded denominators. Unused lower-precision reductions receive finite
placeholder inputs so a valid FP64 response cannot poison backward through an
overflowing FP32 branch.

Choice metadata is owned by the decoded tape; a subsequent captured search can
reuse its buffers without changing the prior tape. Numerical data is never
recovered from cached search payloads. Root device scores use recorded masks;
V7 CALL child energy still comes from native fixed-trace numerical replay and
is counted once per invocation, not once per output port. Serial child choice
credit remains outside the supported scope.

The same fixed-panel answer-plus-readonly-reuse adapter accepts this tape.
Tests compare full slow-parameter gradients with direct finite-panel risk,
including Bank update ancestry, denominator-only responses, private donor
writes and post-capture replay. This establishes training connectivity, not
task-learning success or an entirely device-resident training implementation.
