# Component Provenance

ARTI records the identity of a mechanism separately from the Python package
version and from the `arti.st` file format. This makes a saved component graph
auditable without putting version strings into parameter names or changing the
tensor computation performed by a layer.

## Four Versions

- **Package version**: the installed ARTI distribution version.
- **Artifact format version**: the serialization contract for `arti.st`.
- **Source declaration**: a human-facing registry input, for example
  `arti/fold@1` or `arti/half@1`. It is accepted only at an explicit input
  boundary and is never persisted.
- **Component contract address**: the immutable persisted identity,
  `arti/<name>@sha256:<full-contract-digest>`.
- **Schema fingerprints**: normalized hashes of the concrete configuration and
  parameter/buffer shapes and dtypes.
- **State contract**: a separate hash binding a model graph to the exact
  state-dict names, shapes, dtypes, and declared scope. A raw PyTorch
  `state_dict` remains metadata-free; callers can validate it with
  `arti.component_state_contract()` before loading it.
- **Runtime state identity**: the caller-owned forward `RecallState` is a
  values-only component whose source declaration is `arti/recall-state@1` and
  whose persisted identity is its resolved full contract address.
  Its runtime contract binds the declared Formula and Updater behavior, tensor
  structure, Bank layout, and state schema. It is a structural compatibility
  contract, not a content hash of trained Reader or Updater weights. Persisted
  state that must be tied to one exact trained asset should travel with that
  asset's external lock or artifact identity.

The component reference is not a class suffix. ARTI does not create classes
such as `FoldV1`, and component versions never appear in `state_dict` keys.
The same rule applies to nested dependencies such as the `Half` and `Fold`
parts of the current learned `Pulse`.

## Registered Source Declarations

The following identities are registered by the package.
Their lifecycle is part of the load contract.

| Public surface | Source declaration | Variant | Lifecycle |
| --- | --- | --- | --- |
| `ARTILayer` | `arti/layer@3` | FederatedProgram host | alpha |
| Legacy `ARTILayer` | `arti/layer@1` | monolithic historical layer | legacy |
| Internal classic layer | `arti/classic-layer@1` | composed block implementation | stable |
| `Half` | `arti/half@1` | default | stable |
| `Fold` | `arti/fold@1` | soft workspace compaction | stable |
| `UnFold` | `arti/unfold@1` | trainable layout expansion | stable |
| Reversible `Fold` | `arti/fold@2` | reversible forward transport | stable |
| Reversible `UnFold` | `arti/unfold@2` | recorded exact inverse | stable |
| `ReversibleTopology` | `arti/reversible-topology@1` | permutation partition | stable |
| `FixedTopologyPolicy` | `arti/fixed-topology-policy@1` | fixed index policy | stable |
| `StablePriorityPartition` | `arti/stable-priority-partition@1` | valid-first stable hard operator | stable |
| `SoftTopKTopologySurrogate` | `arti/topology-surrogate@1` | backward-only topology estimator | stable |
| `LearnedTopologyPolicy` | `arti/learned-topology-policy@1` | direct learned priority scorer | stable |
| `BankFormulaTopologyPolicy` | `arti/bank-formula-topology-policy@1` | fixed Query and Bank Formula priority | stable |
| `TopologyOperandBank` | `arti/topology-operand-bank@1` | fixed-address trainable operands | stable |
| `TopologyPriorityFormula` | `arti/topology-priority-formula@1` | versioned operand interpretation | stable |
| `InverseTopologyContract` | `arti/inverse-topology-contract@1` | recorded inverse without learner retention | stable |
| `FoldRecord` / `FoldedTensor` | `arti/fold-record@1` / `arti/fold-state@1` | runtime topology state | stable |
| `Pulse` / `LearnedPulse` | `arti/pulse@1` | learned | stable |
| `AdaptivePulse` | `arti/pulse@2` | composable staged pulse | stable |
| `AdaptiveObservation` | `arti/adaptive-observation@1` | bounded observation trajectory | stable |
| `LearnedObservationPolicy` | `arti/learned-observation-policy@1` | input-conditioned bounded trajectory | stable |
| `StateAffineObservationOperator` | `arti/state-affine-observation-operator@1` | bounded state-conditioned feature frame | stable |
| `FormulaAttention` | `arti/formula-attention@1` | intervention support selection | stable |
| `SelectiveCompute` | `arti/selective-compute@1` | pack, apply, scatter | stable |
| `ReunionAggregate` | `arti/reunion-aggregate@1` | post-reunion aggregation host | stable |
| `PulseCompressor` | `arti/pulse-legacy@1` | explicit | legacy |
| `FusionPulse` | `arti/fusion-pulse@1` | multi-source | stable |
| Global `Recall` | `arti/recall@2` | globally normalized formula-driven routing | stable |
| Partitioned `Recall` | `arti/recall@3` | per-Bank normalization with explicit member asset identity | stable |
| K-wide `Recall` | `arti/recall@4` | independent candidate iteration with hard winner selection | stable |
| `RecallExecutor` | `arti/recall-executor@2` | runtime-policy-adapter | stable |
| `ExecutionPolicy` | `arti/execution-policy@1` | runtime-only | stable |
| `AdaptiveExecutionPolicy` | `arti/execution-policy@2` | adaptive-runtime | stable |
| `ExecutionBudget` | `arti/execution-budget@1` | runtime-only | stable |
| `ExecutionStop` | `arti/execution-stop@1` | runtime-only | stable |
| `RefinePolicy` / `RefineBudget` / `RefineStop` | `arti/refine-*` | historical runtime contracts | legacy |
| `RetrievalRoutePlan` | `arti/recall-route-plan@1` | runtime-only | stable |
| `RetrievalRouteStack` | `arti/recall-route-stack@1` | runtime-only | stable |
| `RecallState` | `arti/recall-state@1` | values-only | stable |
| `RecallValueUpdater` | `arti/updater@1` | value update | stable |
| Affine updater | `arti/affine-updater@1` | affine value update | stable |
| Normalized updater | `arti/normalized-updater@1` | normalized value update | stable |
| Stacked updater | `arti/stacked-updater@1` | multi-site value update | stable |
| `TensorContext` / `FrameContext` | `arti/tensor-context@1` / `arti/frame-context@1` | context | stable |
| `EmissionRouter` | `arti/emission-router@1` | stream routing | stable |
| Built-in Formula `delta` | `arti/delta@1` input declaration | one-factor state transition | stable |
| Built-in Formula `affine` | `arti/affine@1` input declaration | two-factor state transition | stable |
| Built-in Formula `state` | `arti/state@1` input declaration | structured state transition | stable |

Formula source declarations such as `arti/delta@1` are accepted only at an
explicit API or registry input boundary. Resolution produces a full immutable
reference, for example `arti/delta@sha256:<contract-digest>`. Artifact
manifests, locks, dependency closures, and graph fingerprints record only that
resolved reference; the requested declaration is not serialized. A Formula
update therefore produces a new contract digest and a fresh graph fingerprint.

An interactive resolver may accept a 12--63 character SHA-256 prefix when it
selects exactly one registered contract. It returns a resolution receipt that
records the requested input and the resolved full address. Short addresses and
receipts are convenience data only: manifests always contain all 64 digest
characters, and `ComponentRef.parse()` accepts full addresses only.

Recall@2 and Recall@3 are distinct artifact contracts. Recall@3 records Bank
names, route ranges, weights, influences, and member artifact fingerprints;
loading the same-shaped state into a differently assembled Bank fails closed.
Runtime candidate/result batches are non-persistent components and must be
recomputed after a fresh load.

Recall@4 is the default constructor identity. It records `breadth`,
`breadth_mode`, and `breadth_aggregation`; it supports either global or
per-Bank normalization. The default `winner` aggregation forwards one complete
trajectory, while `route_weighted` is an explicit opt-in mode. Resolving
Recall@2 or Recall@3 forces their historical mixed-route semantics, so an old
artifact cannot silently acquire K-wide execution.

## Adaptive Pulse Contract

`arti/pulse@2` owns one versioned stage graph. Every stage is independently
enabled or off, and the manifest is revalidated against the bound module
identity and configuration before execution and provenance generation.

The canonical observation path accepts `[B, N, D]` and produces
`[B, T, N, D]`. Reversible Fold and UnFold operate on `N` independently for
each `(B, T)` pair. Aggregate is the first stage allowed to flatten the middle
instance axes.

FormulaAttention selects intervention support `I` from exposed support `E`; it
does not implement QKV attention or change values. SelectiveCompute changes
only values in `I`. Values outside `I` remain identical, while exposed source
values in `E` may still receive gradients through computations that update
`I`.

`PulseOutput.source_supports` deliberately names the pre-aggregation support
space. It must not be interpreted as masks over an aggregated Pulse envelope.
An optional Bank update returns an explicit next `BankState`; it never mutates
the caller-owned state or hides a state transition in diagnostics.

Pulse@2 uses an internal active-K overlay backend for reversible topology. It
gathers only the K active values, retains the original substrate as the
preserved base, and scatters the processed K values back when reunion is
materialized. This is an execution optimization only: canonical artifacts and
records remain `arti/fold@2`, `arti/unfold@2`, and `arti/fold-record@1`, and the
public FoldedTensor path keeps its complete active and folded payload contract.

Runtime-only Recall policy and route objects are versioned so experiments and
integrations can validate their contracts, but they are not model assets. Route
plans own independent tensor snapshots, route stacks fingerprint their ordered
recursive child structure, and neither appears in a model `state_dict` or saved
`arti.st` component graph.

`Pulse` is the canonical loading alias for the current `LearnedPulse` path.
The old explicit pulse-id implementation remains available only as the
legacy `PulseCompressor` identity. An alias is convenient at construction
time; saved provenance always uses the canonical reference.

The registry exposes `arti.component_catalog()` for release tooling. It lists
resolved full contract addresses, ordinary construction aliases, and explicitly
deprecated aliases. Aliases are never written into saved provenance; loading
always records and checks the canonical reference.

## Inspect and Construct

```python
import arti

fold = arti.resolve_component("arti/fold@1", k=16, dim=64)
print(arti.component_ref(fold))

provenance = arti.component_provenance(fold)
arti.validate_component_provenance(provenance)
```

For a nested model, `component_provenance(model)` returns a deterministic
component graph. Each entry includes its module path, full contract address,
lifecycle, normalized configuration, configuration fingerprint, parameter
schema fingerprint, and direct dependencies. The parameter schema describes
tensor names, shapes, and dtypes; it intentionally does not include the
transient `requires_grad` flag.

## `arti.st` and Web Artifacts

`arti.save()` writes the graph into the architecture manifest and stores the
same graph fingerprint in the SafeTensors header. `arti.load()` verifies both
copies before restoring weights. A model with a different `k`, `dim`, enabled
subcomponent, parameter shape, or registered mechanism is rejected rather
than silently receiving an incompatible state.

The same manifest also carries a state contract. It checks the saved tensor
schema and scope, while direct `state_dict` workflows can use:

```python
contract = arti.component_state_contract(model, model.state_dict(), scope="all")
arti.validate_component_state_contract(
    contract,
    state_dict=model.state_dict(),
    model=model,
)
```

The Python Web exporter copies the same graph into the Web manifest. The
TypeScript runtime treats it as declared metadata and executes the exported
tensor graph; it does not reimplement Half, Fold, Pulse, UnFold, or Recall
rules.

## Compatibility and Migration

| Change | Default result | Required action |
| --- | --- | --- |
| Package patch with identical graph | accepted | none |
| Different component reference/version | rejected | export or migrate explicitly |
| Different config or parameter schema | rejected | reconstruct the exact module or migrate |
| Different `k`, `dim`, slots, or enabled feature | rejected | use a matching model |
| Legacy component in an artifact | rejected | pass `allow_legacy=True` after review |
| Old `PulseCompressor` to current `Pulse` | rejected | write an explicit migration |
| Missing provenance in an old artifact | rejected | re-save it with a current ARTI build |

`allow_legacy=True` is an admission switch, not an automatic conversion. ARTI
does not infer a migration from a class name, tensor shape, or alias. A future
migration must be an explicit function that declares its source and target
references and produces a new artifact with fresh fingerprints.

## Independent Feature Switches

Optional mechanisms remain independently disableable. For example,
`LearnedPulse(use_half=False)` records the learned Pulse without a `Half`
dependency, and `Recall(activation="none")` records no Half dependency. The
component graph therefore describes the actual execution path rather than a
maximum feature set. This preserves tensor-in/tensor-out composition without
forcing coordinates, masks, visibility, Pulse, or Recall onto unrelated data.
