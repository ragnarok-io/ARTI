# Stability Policy

`arti-fit 3.2.0` is a stable package release. Neural component lifecycle is
independent of the package version and remains explicit in the component registry.

## Supported Workflows

- PyTorch native models, attachments and pretrained integrations share one
  training session and task-objective contract.
- Local single-process Torch runs qualify periodic checkpoints, supported
  optimizer/scheduler/RNG continuation and state-preserving evaluation.
- Classification and text-only causal-LM recipes connect data, task loss,
  continuation, evaluation and portable inference. Their exact supported
  preprocessing and deployment contracts are documented in the recipe guides.
- Native JAX bindings execute shared Formula and static resource/program graphs
  with explicit PyTrees; Flax integration is optional.
- SafeTensors, component provenance and supported native graph sidecars provide
  validated assets. Arbitrary custom Python implementations remain caller-owned.

## Component Boundaries

`ARTILayer` is an identity-safe host for a configured `FederatedProgram` or
`ProgramGraph`. Its empty form preserves the input. Advanced graph construction,
dynamic branch search and some Formula mechanisms retain their declared alpha
lifecycle; a stable package version does not silently relabel those contracts.
Standalone mechanisms remain independently versioned.

Canonical saved references are content-addressed. Python names are convenience
entry points, not artifact identities. `arti.legacy` contains retired layered
Recall implementations and does not define the default execution shell.

## Compatibility And Limits

Stable contracts receive explicit deprecation before ordinary removal, except
documented correctness or security repairs. Evolving alpha contracts can change;
assets must retain their exact component and schema provenance.

The local recipes do not qualify arbitrary sharded optimizer resume, hosted
multi-tenant serving, vLLM/SGLang/TensorRT-LLM compatibility, full JAX parity,
unbounded dynamic graphs or broad task-quality superiority. CUDA performance
depends on the chosen program and hardware. Evaluation must use the actual task,
not substitute an internal diagnostic metric for successful answers.

| Area | Status |
| --- | --- |
| Python | 3.10, 3.11, 3.12 release matrix |
| PyTorch | 2.2+; optional integrations can require newer upstream versions |
| CPU | Supported |
| CUDA | Through CUDA-enabled PyTorch; profile-specific performance |
| JAX | Native Formula and bounded/static ProgramGraph bindings |
| Flax | Optional native module/model adaptation |
| Serving | Optional load-once FastAPI/Uvicorn integration |
| Web runtime | Experimental |

Release gates validate the exact code and package candidate. Hardware scaling,
distributed deployment and task effectiveness require their own evidence.
