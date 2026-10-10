# Release Surface

The public distribution is `arti-fit`; import it as `arti`. Version 3.2.0
connects neural programs to a complete documented local training workflow.

## Runtime And Mechanisms

`arti.ARTILayer` and `arti.torch.ARTILayer` host a configured `FederatedProgram`
or `ProgramGraph`. An unconfigured layer is strict identity. Reusable Formula,
resource, topology, observation and execution operations live in
`arti.mechanisms`; retired monolithic layers live in `arti.legacy`.

Canonical component references are content-addressed. Component lifecycle,
configuration and artifact policy remain explicit and independent of the
package version. Advanced host/graph APIs retain their registry-declared alpha
status; standalone stable mechanisms retain their declared contracts.

## Training And Tasks

Use `arti.trainer(model, ...)`, `model.arti.trainer(...)`, or
`workflow.trainer()` to reach one `ARTITrainingSession`. The local Torch profile
provides periodic assets, continuation and state-preserving evaluation.

- [Training Workflow](training-workflow.md)
- [Classification Recipe](task-recipes.md)
- [Qwen SFT Workflow](qwen-sft-workflow.md)

Both recipes retain real data splits, task objectives, preprocessing and raw-ID
evaluation. Inference assets work without their original training data. Optional
ASGI serving loads once and processes request batches.

## Other Backends

[Native JAX](jax-backend.md) binds the shared Formula and ProgramGraph contracts
with explicit array parameters and state. Custom Torch implementations require
native overrides. Full Torch/JAX parity is not implied.
[Web inference](webgpu-alpha.md) remains experimental.

See [STABILITY.md](https://github.com/ragnarok-io/ARTI/blob/main/STABILITY.md)
and [Release Checklist](release.md) for compatibility and validation boundaries.
