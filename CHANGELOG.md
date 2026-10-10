# Changelog

## 3.2.0

- Unify native models, model attachments and pretrained integrations around
  `ARTITrainingConfig`, `arti.trainer()` and one `ARTITrainingSession`.
- Add durable local training runs with periodic/first/final checkpoints,
  optimizer/scheduler/RNG continuation, explicit data cursors and state-preserving
  evaluation. The host task objective is the default; tensor alignment is explicit.
- Add executable classification and Qwen causal-LM recipes covering explicit
  splits, preprocessing, real task loss, continuation, held-out evaluation and
  portable prediction assets through the same API and `arti recipe` CLI.
- Add final-assistant/EOS supervision, token-weighted microbatch accumulation,
  native-meta planning and tokenizer-bound Qwen deployment. Custom adapter
  factories are declared and required explicitly on reload.
- Add optional load-once FastAPI/Uvicorn serving with batched prediction and chat.
- Add native JAX Formula/ProgramGraph bindings, explicit parameter/state PyTrees,
  `jit`/`grad`/`vmap`/`scan` execution and optional Flax integration.
- Preserve external Formula operand ownership during shared graph lowering and
  embed supported native graph state in ordinary ARTI assets.
- Refresh public documentation and CI for current workflows. Keep public
  content-addressed checkpoint checks, Python 3.10 support and release privacy
  audits; exclude tests requiring private research benchmarks.

The package version is stable; advanced component lifecycle is still declared
independently. This release qualifies the documented local workflows, not every
distributed checkpoint, serving engine or task-quality claim.

## 3.1.0a3

- Add explicit credit boundaries and graph-derived learning regions, including
  independent optimizer domains and region-specific objectives in one update.
- Add typed Fabric module registration and differentiable node candidates,
  plus position encoding as a Formula operation.
- Extend resource programs with multi-port nodes, dynamic routes, joins,
  bounded loops, nested execution and compiled graph credit paths.
- Move retired layered Recall topology screening into `arti.legacy`.

These are alpha APIs. Compiled graph support is bounded by the declared
program and tensor contracts; this release makes no general speed claim.

## 3.1.0a2

- Make tagged releases wait for the same package and documentation CI used by
  branch builds before package publication.
- Align release readiness checks and the release guide with the public package
  layout, removing requirements for private benchmark reports.

This release hardens the release pipeline; the public API remains alpha. See
`STABILITY.md` for compatibility and claim boundaries.

## 3.1.0a1

- Rebased the public package on the current program-runtime API.
- Promoted the current `ARTILayer` host, Formula Fabric, federated programs,
  resource graph, and execution interfaces into the published alpha surface.
- Moved retired layered Recall implementations under `arti.legacy` and
  refreshed public documentation, examples, schemas, and tests.

This release is an alpha API transition. See `STABILITY.md` for compatibility
and claim boundaries.
