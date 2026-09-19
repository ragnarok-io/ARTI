# Stability Policy

ARTI 3.0.12 remains the stable public mechanism release. The 3.0.13a5 package
is a prerelease that adds bounded Federal graph compilers, an RCC host adapter,
alpha NeuralPlasticity, and self-effect topology-search components without
expanding the stable compatibility surface. Stability covers
documented component identities, tensor contracts, artifact schemas, and
composition semantics. It does not imply a production service SLA or universal
model-quality and performance claims.

## Stable Surface

- `arti.ARTILayer` / `arti.torch.ARTILayer`: `arti/layer@2`, backed by one
  composable `AdaptivePulse` graph.
- `arti.nn`: practical tensor layers including `Half`, `Fold`, `UnFold`,
  `Pulse`, `Recall`, and `RecallRefiner`.
- `arti.mechanisms`: versioned Observation, topology, Formula Fabric,
  Batched Refine, Bank update, Federal Bank, runtime state, and tensor-operation
  contracts promoted from the 3.0.7-3.0.12 development line.
- `ARTI.attach`, attachment configuration, incremental SafeTensors state,
  fresh reload, and reversible detach.
- `arti.st` format version 1 and the documented Recall Bank artifact format.
- component provenance, mask, coordinate, visibility, output, dtype, device,
  and diagnostics contracts documented for each component.

`arti.alpha` remains an identity alias for `arti.mechanisms` so previously
recorded scripts can resolve their imports. New code should use the stable
namespace. Canonical component references do not contain the Python namespace.

## Legacy

The retired monolithic `ARTILayer@1`, `LayerRecall`, and `StatefulRecall` are
available only through `arti.legacy`. They are not used by new attachment or
the default layer. Legacy components may be loaded for historical inspection,
but they do not receive new mechanism features.

## Experimental

The Python-first browser exporter remains under `arti.experimental.web`, and
the TypeScript runtime remains an alpha npm package. Web artifacts are
deployment products, not portable training checkpoints.

`FormulaProgramQuery@1`, `FormulaProgramQuery@3` through `FormulaProgramQuery@7`, `FormulaFabric@3` through
`FormulaFabric@5`, the NeuralPlasticity effect atoms, `FederalRecall@3`, and the
shape-polymorphic TensorView query contracts ship for composition experiments
with lifecycle `alpha`. Their presence in the package does not freeze those
component contracts.

Prepared device dispatch, grouped differentiation and caller-owned CUDA Graph
recipes are experimental execution paths. CUDA with PyTorch 2.11 or newer uses
automatic compilation by default; explicit native execution remains available.
CPU and older PyTorch retain native behavior. Compiled dispatch is no-grad; differentiable execution uses
the original Formula graph or the grouped VJP path. Whole-dispatch fusion excludes
effect instructions. Capture requires fixed storage and gradient participation,
and does not automatically include an optimizer. The acceleration recipes are
validated on PyTorch 2.11; the base package's older-PyTorch support does not imply
support for every compiler or CUDA Graph option. No private benchmark result is
a package-wide throughput guarantee.

The 3.0.13a2 correction removes the alpha `FormulaProgramQuery@2` effect-owned
state arena. Reconstruct those programs with ordinary producer-owned Bank
slots; old alpha state artifacts are not silently migrated. Query@3 has
commit-visible successors, while Query@4 adds branch-local producer
re-execution. Stable component identities are unchanged.

Experimental integrations may change without the stable mechanism deprecation
window. A stable mechanism used by an experimental integration does not make
the entire integration stable.

## Compatibility

- Patch releases fix defects without intentionally changing stable contracts.
- Minor 3.x releases may add optional APIs or component versions while keeping
  existing canonical references resolvable.
- A breaking stable API change requires a new major package version or a new
  canonical component version with an explicit migration path.
- Serialized artifacts never infer behavior from a Python class name alone;
  canonical reference, schema, configuration, and provenance must agree.
- Normal `.arti.st` loading does not depend on Python pickle.

The package version and component version are independent. For example,
`arti/fold@1`, `arti/fold@2`, and `arti/reversible-topology@1` retain different
semantics even when distributed in the same `arti-fit` release.

## Deprecation

A stable API is deprecated before removal. Deprecation includes a runtime
warning, a documented replacement, and release-note coverage. Removal occurs
no earlier than the next major release unless a security or correctness defect
requires a documented exception.

## Support Matrix

| Component | Support |
| --- | --- |
| Python | 3.10, 3.11, and 3.12 in public CI |
| PyTorch | 2.2 or newer |
| CPU | Supported |
| NVIDIA CUDA | Supported through CUDA-enabled PyTorch |
| JAX | Optional functional subset; not full PyTorch parity |
| WebGPU/WASM | Experimental Python-owned artifact runtime |
| Transformers, PEFT, Diffusers | Optional host integrations |
| Artifact format | `arti.st` format version 1 |

## Release Gate

A stable release requires a clean public checkout, the Python CI matrix,
package build and isolated wheel/sdist installation checks, stable API smoke
tests, documentation review, and a privacy scan. CUDA or downstream quality
claims require their own recorded evidence and are not implied by this label.

No ARTI release is LTS unless its release notes publish a maintenance window,
supported Python/PyTorch window, security policy, and artifact read-compatibility
commitment.
