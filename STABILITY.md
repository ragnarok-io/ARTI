# Stability Policy

`arti-fit 3.1.0a1` is an alpha release of ARTI's current program-runtime
surface. It publishes the current package rather than preserving the retired
layered Recall surface as the default.

## Current Surface

- `arti.ARTILayer` is `arti/layer@3`: an identity-safe host for a federated
  program. Without an installed program it preserves its input.
- `arti.ARTI.attach`, `.arti.st` artifacts, component provenance, attachment,
  and fresh reload remain supported public workflows.
- `arti.mechanisms` contains reusable operations including Formula Fabric,
  topology, observation, pulse, resource, and program-runtime components.
- `arti.legacy` contains historical `LayerRecall` and `StatefulRecall`
  implementations for artifact inspection only. New integrations should not
  depend on them.

Component references and package versions are independent. A component's
reference, schema, configuration, and provenance describe its artifact
contract; Python class names alone do not select artifact behavior.

## Alpha Boundary

`arti/layer@3`, federated-program composition, branch search, resource graphs,
and advanced Formula Fabric programs are alpha contracts. They are published
for developer experimentation, but may evolve before a stable API commitment.
The package does not claim broad downstream superiority, a production service
SLA, universal throughput gains, full JAX parity, or a general replacement for
an arbitrary host model's context state.

## Compatibility

Normal `.arti.st` loading uses explicit component provenance and does not
depend on Python pickle. Stable component contracts are deprecated before
removal except for documented correctness or security fixes. Alpha contracts
may change between alpha releases; artifacts should record their canonical
component reference and schema.

## Support

| Area | Status |
| --- | --- |
| Python | 3.10, 3.11, 3.12 |
| PyTorch | 2.2 or newer |
| CPU | Supported |
| NVIDIA CUDA | Supported through CUDA-enabled PyTorch; workload-specific performance varies |
| JAX | Optional functional subset; not full parity |
| Web runtime | Experimental |

Every release is expected to pass package-build, isolated-install, public API,
documentation, and privacy checks. CUDA or task-quality claims require their
own recorded evidence.
