# Changelog

## 3.3.0

- Add model-independent `TaskRecipe`/`TrainingBinding` workflows with explicit
  application factories, native data/loss semantics, checkpoint selection and
  qualified local recovery through the existing training session.
- Add immutable record/binary-asset snapshots, logical roles/splits/groups,
  reference availability, relocatable bindings and versioned preparation reuse.
- Extend prediction journals with staged named-file outputs, atomic committed
  closures, format validators and remainder-only recovery; retain JSON-only use.
- Add independently declared policy/environment episodes with state lifetimes,
  outcome traces, local recovery and refusal of ambiguous external-action replay.
- Add reference/output/external evaluation suites preserving original coverage,
  paired trial identities, explicit criteria and distinct evidence origins.
- Add trusted runtime/handoff profiles and portable result archives with exact
  ID/file closure; formatting export reuses results and performs no submission.
- Add atomic task scaffolding, unified project checks and packaged JSON Schemas,
  read-only task status/explain and a complete installed-package tutorial.
- Add validation-selected checkpoints, prepared supervision, explicit local
  SFT imports, token budgets, cross-request batching and optional native
  Transformers/TRL, Ray Train and offline reporting integration paths.
- Preserve the selected Windows CPython virtual environment while managing the
  actual direct worker, avoiding redirector-only termination on a local deadline.

Support is bounded by documented local CPU and engine contracts. This release
does not qualify arbitrary native episode-state codecs, distributed/sharded
recovery, process-tree/network/memory isolation, providers, hardware, throughput
or task quality. Declared, observed and enforced limits remain separate.

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
