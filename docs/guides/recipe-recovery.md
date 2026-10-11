# Recover a Task Run

Classification and Qwen SFT recipes can recover an existing run without manually
finding its latest checkpoint or calculating the remaining optimizer steps.
Recovery uses the same session, task loss, data loader, evaluation and export
path as ordinary training. It does not introduce another trainer.

Local Torch and single-process Accelerate recipe checkpoints are supported.
Recovery keeps the recorded engine; it does not convert engines or qualify
distributed/sharded state. See [Train With Local Accelerate](accelerate-training.md).

[Custom Task Workflows](custom-task-workflows.md) also use this same recovery
lifecycle. Their named runtime resources can be relocated with
`--resources-json` / `resume_recipe_run(resources={...})`, independent of any
model family.

## Inspect Before Loading a Model

```bash
arti recipe resume-run outputs/first --plan-only --output recovery-plan.json
```

```python
import arti

plan = arti.plan_recipe_resume("outputs/first")
print(plan["action"], plan["checkpoint_step"], plan["remaining_optimizer_steps"])
```

Planning reads `resolved_recipe.json`, `run.json`, `task_result.json` when present,
and the latest completed training checkpoint's validated metadata. Package
integrity checks may read/hash checkpoint files; they do not decode tensor
payloads, instantiate the model, load host weights or allocate CUDA. A configured
CUDA recipe can therefore be inspected on a CPU-only machine.

| Action | Behavior |
| --- | --- |
| `resume_training` | Restore the full checkpoint and perform the remaining updates toward the original target |
| `finalize` | Restore the completed training state, then evaluate and export with zero new optimizer updates |
| `already_completed` | Return the existing successful task result without constructing a model |

Progress comes from the completed checkpoint, not a potentially unsaved tail in
`run.json`. If no completed checkpoint exists, recovery cannot recreate missing
weights/moments; start a new run from the resolved recipe instead.

## Recover the Original Task

```bash
arti recipe resume-run outputs/first --run-dir outputs/recovered --max-seconds 300
```

Stop the old process before starting its recovery. A fresh attempt directory
preserves the source run and stores its parent lineage, recovery plan, target and
new artifacts. Omit `--run-dir` to use the normal generated run directory.

For example, a task targeting step 200 with a checkpoint at step 80 resumes 120
updates, not 200 additional updates. Recovering that attempt again keeps the same
absolute target. A budget/requested stop can export a usable intermediate model
without completing the original horizon; its next recovery still trains the
remaining updates.

Each manually started attempt has its own declared training-time allowance.
`--max-seconds` overrides the source run's allowance; otherwise it is inherited.
This is not an automatic retry service or a cumulative compute allowance. Time
is checked at optimizer-update boundaries by the existing run lifecycle, not as
a deadline for model loading, evaluation or export. Inspect the new `run.json`
for the effective settings.

Python uses the same entry point for either recipe:

```python
import arti

plan = arti.plan_recipe_resume("outputs/first")
result = arti.resume_recipe_run(
    "outputs/first",
    output_dir="outputs/recovered",
    max_seconds=300,
)
print(result["optimizer_updates_this_attempt"], result["model"])
```

Task/data/label/tokenizer/optimizer compatibility is checked by the existing
recipe checkpoint loader. Keep the original data snapshot and pinned host model
available, or provide their current local locations below. Stateful data
position, optimizer, scheduler, RNG and validation
selection state are restored through that loader. Deployment assets are not
training checkpoints.

Constructing the recipe directly and calling its `resume_run` with an explicit
`run=ARTITrainingRunConfig(...)` allows reporter/cadence/run
configuration in Python. Validation policy and cadence still have to agree with
the saved training state. `reporter`, `stop_requested`, `validation_fn` and the
critical `on_checkpoint` handoff hook remain available.

## Recover After Moving Local Resources

Data, cache and native base paths are resource locations, not a new task. Use
the same source run and identical data/model content at their new locations:

```bash
arti recipe resume-run moved/run --plan-only --data snapshots/chat.jsonl --base-model models/qwen-native --cache-dir caches/qwen
arti recipe resume-run moved/run --data snapshots/chat.jsonl --base-model models/qwen-native --cache-dir caches/qwen --run-dir outputs/recovered
```

```python
result = arti.resume_recipe_run(
    "moved/run",
    data_path="snapshots/chat.jsonl",
    base_model="models/qwen-native",
    cache_dir="caches/qwen",
    output_dir="outputs/recovered",
)
print(result["recovery"]["resources"])
```

Recovery prefers the tokenizer saved in `run/tokenizer`, without a Hub lookup.
If absent, it retains the original pinned-model tokenizer loading policy. Paths
to a native base supplied in a previous recovery attempt are inherited from its
run metadata unless `--base-model` overrides them. Paths
inside the old run directory are remapped to the moved run. External data paths
require `--data` when no longer available. A missing imported source manifest
uses the run's `data_source_manifest.json` copy; `--source-manifest` can provide
its current location explicitly. A missing external prepared cache after a run
move defaults to `moved/run/prepared-chat`; `--cache-dir` selects another cache.
An unchanged moved cache reuses its token artifact without re-encoding.

The plan reports effective paths, their origins and `missing_local_paths`.
Planning does not establish data equivalence: actual recovery prepares/checks
the tokenizer and encoded dataset against the checkpoint **before host model or
optimizer allocation**. Changed data, tokenization or source provenance are
not accepted as recovery. The full loader still checks the original optimizer,
engine, evaluation profile and task/architecture contract. Device, precision,
token budget and engine are not silently changed.

For an installed custom program, the original recipe descriptor remains in
force. An explicitly supplied factory must match it:

```bash
arti recipe resume-run moved/run --data snapshots/chat.jsonl --base-model models/qwen-native --layer-factory my_package.layers:build_layer --factory-kwargs-json factory.json
```

Python also accepts `layer_factory=LayerFactoryConfig(...)` or the original
callable when the run used a caller-supplied factory. Keep its installed code
and behavior identical. Local native base weights are caller-provided: the
pinned logical model/task and architecture contract are preserved, but this is
not a byte-identity audit of arbitrary replacement base weights.

The new attempt records `recovery_resources` in its run metadata and result.
It does not edit the source checkpoint or resolved recipe. A completed no-op
does not need the old data, tokenizer or base to remain available; an explicit
refresh/finalize does. These resource options apply to causal-LM recipes only;
classification recovery keeps its existing data configuration.

## Finish Evaluation or Export Without Retraining

If training completed but evaluation/export failed, use the same command. It
performs zero optimizer updates and retries the remaining workflow stages.
An earlier validation-selected best model remains distinct from the coherent
last-state training checkpoint; final evaluation/export uses that selected model
without rewinding live weights or optimizer moments.

Learned early stopping is also a terminal training outcome. Recovery does not
ignore patience and continue to the original hard maximum. Its original target
remains recorded, but the remaining updates are zero. A subsequent explicit
`recipe.fit(steps=N)` is a new continuation, not recovery of that stopped task.

A completed task is idempotent. To deliberately reevaluate/reexport it, use:

```bash
arti recipe resume-run outputs/first --refresh-completed --run-dir outputs/refreshed
```

When a complete run directory is copied to another location, its no-op result
uses the artifact paths under that actual source directory rather than stale
absolute output paths from the earlier task report.

Missing or invalid derived inference artifacts are rebuilt from the valid full
training checkpoint. That does not authorize replacing a missing or invalid
training checkpoint. Newly exported assets work with the ordinary
`recipe evaluate`, `recipe predict` and `recipe serve` commands.

## Recovery Versus Additional Training

`recipe.fit(steps=N, resume_from_checkpoint=...)` and `recipe train --steps N`
request **N additional updates**. `recipe.resume_run(...)` and
`recipe resume-run` instead finish the **original absolute target**. Choose the
former to extend training intentionally, and the latter to recover a failed or
partially completed task.

The executable wrapper accepts an existing run without preparing new data:

```bash
python examples/recipe_run_recovery.py outputs/first --plan-only
python examples/recipe_run_recovery.py outputs/first --refresh-completed --run-dir outputs/refreshed
```

The qualified profile is local Torch and single-process Accelerate with
classification and Qwen SFT. It does not claim cross-device bitwise parity,
automatic remote recovery, streaming-loader exactness,
sharded checkpoints, distributed retries or native Trainer checkpoint recovery.
See [Ray Train Checkpoint Handoff](ray-training.md) and
[Transformers and TRL Training](transformers-training.md) for those separate
engine entrances and their current boundaries.
