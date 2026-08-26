# Stability Policy

ARTI 3.0.6 remains the Stable Candidate baseline. ARTI 3.0.8a1 is a
prerelease for the versioned vNext composition contracts described below; it
does not promote those alpha components to the supported surface and is not an
LTS promise.

## Supported 3.x Surface

The following contracts are frozen across compatible 3.x releases:

- Core tensor contracts for `[B, D]` and `[B, N, D]` inputs.
- `arti.nn` core layers: `Layer`, `Half`, `Fold`, `UnFold`, `Pulse`, and
  `RecallRefiner`.
- Root core layers: `ARTILayer`, `ARTIResidualBlock`, `ARTISequenceBlock`,
  `ARTIPooledBlock`, and `ARTIClassifier`.
- `ARTI.attach`, attachment configuration, save/load, and reversible removal.
- The `arti.st` format-version 1 reader and writer.
- Mask, coordinate, visibility, output, and diagnostics tensor contracts.

Modules and parameters explicitly documented as alpha, experimental, or
legacy are not frozen at the same level. They will still follow the deprecation
process below.

`FusionPulse` is introduced as an alpha API in 1.6.0. Its tensor-in/tensor-out
contract is documented, but its structural-loss defaults are not part of the
frozen core surface yet.

`arti.nn.Recall` and the Recall Formula contract are introduced as alpha APIs
in 1.8.0. Built-in formula identifiers and tensor shape rules are documented,
but custom-formula serialization and third-party provider portability are not
part of the frozen core surface.

Runtime Recall refinement controls were introduced as alpha APIs in 1.9.0.
Their exact-depth semantics and atomic schedule validation are documented, but
automatic schedule selection is not part of the supported surface.

`arti.alpha.TargetBankUpdater` is introduced in 3.0.5 as an alpha API. It
repeatedly addresses the Bank being updated, supports an optional private Bank
partition, and applies bounded write refinement. Its training objectives and
task-specific state lifecycle remain application concerns.

`arti.alpha.Fold` and `arti.alpha.UnFold` are introduced in 3.0.6 as the
versioned `arti/fold@2` and `arti/unfold@2` reversible-topology operations.
They preserve every original tensor instance through a recorded permutation;
their learned topology policies and Bank Formula policies remain alpha. The
supported `arti.nn.Fold` and `arti.nn.UnFold` names continue to resolve to the
existing `@1` contracts.

`arti.alpha.AdaptivePulse`, `AdaptiveObservation`, FormulaAttention,
SelectiveCompute, typed topology operands, and the Pulse stage graph are
introduced for controlled experimentation in 3.0.7a1. Their canonical
component identities, support-mask contracts, and disabled-stage identity
semantics are versioned, but their stage composition and performance are not
part of the Stable Candidate surface. Fourier observation currently supports
eager training and compiled forward; compiled FFT backward is not promised.

`arti.alpha.FormulaFabric`, bounded Bank-driven Formula routing, iterative
route re-query, Objective-controlled commit strength, pairwise topology
surrogates, and Fold source-contract binding are introduced in 3.0.8a1. Their
component identities and bounded tensor contracts are versioned, but their
training objectives, task quality, runtime performance, and persistence policy
remain application concerns. Objective queries are explicit current/past-only
inputs; future targets and losses remain outside the forward graph.

ARTI 3.0 removes the experimental `RecallTTTSession` API and introduces the
Formula contract/manifest schema v2. Current Recall,
Formula, artifact, policy, and workspace APIs do not own an optimizer or an
implicit online-training session.

## Compatibility

- Patch releases fix defects without intentionally breaking supported APIs.
- Minor 3.x releases may add optional parameters and APIs with compatible
  defaults.
- Breaking supported APIs requires a new major release.
- ARTI 3.x reads valid format-version 1 `arti.st` artifacts produced by the
  pre-public 0.x and public 1.x lines. Recall state migration preserves the
  value Bank and query basis but does not promise identical routing behavior.
  Artifact format compatibility is independent of the package version.
- Serialized artifacts must not depend on Python pickle for normal `.arti.st`
  loading.

## Deprecation

A supported API is deprecated before removal. Deprecation must include a
runtime warning, a documented replacement, and coverage in the release notes.
Removal occurs no earlier than the next major release. Security or correctness
issues may require an exception, which must be documented.

## Support Matrix

| Component | Stable Candidate support |
| --- | --- |
| Python | 3.10 or newer; release CI validates 3.10, 3.11, and 3.12 |
| PyTorch | 2.2 or newer |
| CPU | Supported |
| NVIDIA CUDA | Supported through CUDA-enabled PyTorch |
| JAX | Optional functional backend; smaller surface than PyTorch |
| WebGPU | Alpha Python-first runtime for artifact v2 tensor graphs and explicit-state artifact v3 graphs |
| Transformers, PEFT, Diffusers | Optional integrations |
| Artifact format | `arti.st` format version 1; Formula manifest schema version 2 |

## Promotion To Stable

The Stable label requires the public CI matrix to pass from a clean checkout,
wheel and sdist installation checks to pass, the supported API inventory to be
reviewed, and at least one release-candidate feedback cycle to complete without
an unresolved compatibility defect.

## Future LTS

An LTS release will be declared only with a published maintenance window of at
least 12 months, security and critical-defect support, a documented Python and
PyTorch support window, and continued read compatibility for supported
`arti.st` artifacts. Until that declaration, no ARTI release carries an LTS
commitment.
