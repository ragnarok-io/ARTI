# ARTI

ARTI = AI x RT. A programmable neural runtime for PyTorch and JAX, with explicit
tensor resources and a complete local training-to-deployment lifecycle.

The public distribution is `arti-fit`; import it as `arti`.

```bash
pip install "arti-fit==3.2.0"
pip install "arti-fit[qwen,serve]==3.2.0"
```

## Train A Real Task

- [Classification Recipe](guides/task-recipes.md): labeled data, train-only
  preprocessing, loss, continuation, independent evaluation and deployment.
- [Qwen SFT Workflow](guides/qwen-sft-workflow.md): native conversations and task
  objective, tokenizer-bound assets and optional batched ASGI serving.
- [Training Workflow](guides/training-workflow.md): one session for native models,
  attachments and pretrained integrations.

## Build A Neural Program

Formula Fabric composes typed tensor operations. Program graphs add explicit
resources, named ports, connections, parallel stages, joins and loops.
Federated programs provide local iterative execution and routed composition.
The empty `ARTILayer` host is exact identity until a program is configured.

- [New Developer Guide](guides/new-developer.md)
- [Formula Fabric](guides/formula-fabric.md)
- [Program Contracts](guides/federal-contracts.md)
- [Learning Boundaries](guides/learning-boundaries.md)
- [Native JAX](guides/jax-backend.md)

## Contracts And Delivery

Use content-addressed component provenance and supported SafeTensors/graph assets
to preserve the trained result. Package versions and component lifecycle are
independent; optional runtimes have explicit limitations.

- [Release Surface](guides/stable-release-surface.md)
- [API Quick Reference](api/quick-reference.md)
- [Release Checklist](guides/release.md)

ARTI is a reusable library, not a GPU cloud or a task-specific oracle. Correct
software execution, measured learning and successful downstream answers remain
distinct kinds of evidence.
