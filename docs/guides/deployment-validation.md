# Deployment Validation

For multiple independent reference/output/external views, pass
`EvaluationSuiteConfig` or use `--evaluation-suite` instead of a single scorer.
See [Task Evaluation Suites](evaluation-suites.md). The same prediction/recovery
and report pipeline remains authoritative; unavailable references, view failure,
model reload evidence and caller policy remain separate.

Validate a native classification, custom task or language inference asset using
the existing [delivery](task-deployment.md), [prediction](prediction-runs.md)
and [offline scoring](prediction-scoring.md) APIs. No trainer, second journal
or automatic promotion service is introduced.

## One Entrance

```python
import arti

report = arti.validate_recipe_deployment(
    "outputs/delivery", "requests.jsonl",
    arti.PredictionRunConfig("outputs/check", batch_size=32),
    arti.PredictionScoringConfig("my_tasks.metrics:score", "my-task-rubric-v1"),
    targets="original_labels.jsonl",
    task_factory="my_tasks.models:build_task",
    criteria=criteria,
)
```

Omit `task_factory` for the built-in classifier. Custom tasks retain their own
preprocessing, output schema and metrics. Installed factory/scorer code is an
explicit caller dependency; validation does not download a model or install code.
`predict_kwargs`, ID fields, batch size and post-load time allowance remain the
existing `PredictionRunConfig` settings, not a new execution contract.

```bash
arti recipe validate-deployment outputs/delivery --data requests.jsonl --run-dir outputs/check --task-factory my_tasks.models:build_task --scorer my_tasks.metrics:score --scorer-id my-task-rubric-v1 --targets original_labels.jsonl --criteria-json criteria.json
```

## Explicit Criteria

Criteria map task metric names to `minimum` and/or `maximum`. For example, the
following is an illustrative policy, not an ARTI default or recommended target:

```json
{"accuracy": {"minimum": 0.9}, "error": {"maximum": 0.1}}
```

Bounds are inclusive, finite numbers; booleans, unknown bound names and a
minimum greater than maximum are invalid before inference starts. Both bounds
can define an interval. ARTI does not infer metric direction or label a training
loss as a deployment quality metric. The scorer defines units and aggregation.

No criteria, or `{}`, means evaluation only: `status="evaluated"`,
`criteria_passed=null`. It is not an acceptance decision. With criteria,
`status="passed"` or `"rejected"` records their actual result. A missing or
non-numeric metric is `not_evaluable` in its check and does not pass.
CLI returns 1 for a rejected policy, 0 for successful evaluation/passed criteria
or a cleanly stopped job; always examine `status` and completion fields.

Criteria passing is specific to this dataset, scorer and policy. It does not
authorize production traffic, establish hosted availability, prove ground
truth quality or mark a model globally eligible for deployment.

## Stop, Resume And Rescore

The underlying inference commits durable partitions. `max_seconds` and
`stop_requested` stop new batches, not an active callback or scoring stage.
Stopped jobs report `inference_completed=false`, `scoring_completed=false` and
no criteria result. Partial data is not silently graded as a full population.

```python
report = arti.validate_recipe_deployment(
    "outputs/delivery", None, run, scoring, resume=True,
    task_factory="my_tasks.models:build_task",
)
```

Only unfinished predictions execute. Once every partition is committed,
completion/merge repair, scoring and a changed policy need no model, task
factory or original request file:

```python
report = arti.validate_recipe_deployment(
    None, None, run, scoring, resume=True, criteria=new_criteria,
)
```

```bash
arti recipe validate-deployment --run-dir outputs/check --resume --scorer my_tasks.metrics:score --scorer-id my-task-rubric-v1 --criteria-json revised-criteria.json
```

Scorer code and explicit external labels remain dependencies. Without `targets`,
the pinned request records are used as labels and must actually contain the
task's original targets. Changing a scorer, target set or criterion creates
new derived evidence, not new inference. Changed prediction inputs, model or
package follow the existing pinned-run mismatch checks.

## Runtime Evidence

`runtime_reload_verified` is established only after at least one successful
committed batch from this validation invocation. `runtime_evidence.origin`
distinguishes `current_attempt`, a content-checked `prior_validation` receipt
for the same prediction plan, and `not_run`. Reusing an unrelated completed
prediction run does not claim a fresh model reload. A previous failed criterion
can still contain valid runtime evidence; runtime and task quality are separate.

This is local runtime evidence from the explicitly loaded predictor, not a
proof of third-party code bytes, arbitrary external services or a production
worker. A cold consumer process remains a separate qualification check.

## Durable Reports

The default derived report is a sibling of the inference directory, for
example `outputs/check.validation.json`. Use `output` / `--output` to choose
another path outside the prediction journal and packaged model directory.
Standalone raw `.st` files allow sibling reports, but their weight/manifest/
lock/graph dependencies cannot be overwritten. Reports remain separate so
inspection failure cannot turn a fresh prediction directory into an invalid job.

Reports pin model, task, delivery, prediction plan, input, original-label and
scorer identities, full sample coverage, raw metrics, checks and runtime origin.
Each score artifact is content-addressed inside a sibling report score folder;
old score files remain available when criteria or scorer changes. The summary
has its own content fingerprint, is retained in a sibling `.reports` folder
under that fingerprint, then is atomically published as the latest report.
Previous runtime evidence remains retrievable even when the latest pointer changes.

Runtime/scoring/publication exceptions propagate and write a separate
`.failure.json` with the failing stage and exception type. Existing accepted
reports, score artifacts and committed predictions remain intact. There is no
automatic retry or background deployment: rerun/resume explicitly. Invalid
argument declarations fail before workload preparation.

One local writer and immutable source resources are the qualified scope.
Scoring uses the existing O(N) host-memory population; validation introduces
no streaming/distributed aggregation or GPU throughput claim. A caller's
evaluation rubric, labels, code and acceptance policy remain their responsibility.
