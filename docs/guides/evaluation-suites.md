# Task Evaluation Suites

Evaluate committed structured/file results through independent application-owned
views. Reuse the prediction journal, per-ID scorer validation, explicit metric
criteria and content-addressed report history. No model/task factory, tensor
decoding, regeneration, built-in benchmark protocol or automatic promotion is
needed. Existing `score_prediction_run` remains the single-scorer API.

## Define Independent Views

```python
suite = arti.EvaluationSuiteConfig((
    arti.EvaluationViewConfig(
        "quality",
        arti.PredictionScoringConfig("my_package.metrics:quality", "my-rule-revision"),
        requires="reference", reference_field="answer", availability_field="available",
    ),
    arti.EvaluationViewConfig(
        "format",
        arti.PredictionScoringConfig("my_package.metrics:file_format", "my-format-revision"),
        requires="output", files=True,
    ),
    arti.EvaluationViewConfig(
        "environment", requires="external", external_protocol="my-protocol-revision",
    ),
))
report = arti.evaluate_prediction_suite(
    "outputs/prediction", suite, references="references.jsonl",
    output="outputs/evaluation.json",
)
```

Each local view uses the existing installed `PredictionScoringConfig`. The
scorer receives a batch of original pairs with `id`, `input`, `prediction`,
`target`, `execution` and declared `trial` context. It returns finite JSON
`metrics` and exactly one `scores` record per eligible ID, as before. Score
records may retain task-specific units, successes, failures and diagnostics;
`status=scored` means measurement occurred, not that the model answered correctly.
Scorers receive copies, so argument preprocessing cannot rewrite saved evidence.

`requires="reference"` means that explicit reference rows must be supplied.
They are never guessed from the model's request or generated output. Absent
rows, a false/missing declared availability flag, a missing/null declared
reference field or unavailable required reference-role assets produce
`status=unavailable`, a reason and `score=None`. They do not produce fake labels,
zero scores or a scorer invocation on an empty population. Non-boolean declared
availability is a data-contract failure, not an implicit truthy value.

`requires="output"` passes `target=None` and evaluates the committed output only.
This is useful for format/alignment/validity even when task quality cannot be
measured. `requires="external"` imports separately produced evidence instead of
running a local scorer. It is not a model's self-reported assertion of success.

Each view preserves scored, unavailable and failed IDs; selected/committed/
original total/uncommitted counts; selected and total fractions; original
individual scores and the reference source identity. Extra reference IDs are
listed as unused. An unfinished journal needs explicit `allow_partial=True`;
its original denominator never shrinks to the completed prefix.

## File And Binary Reference Context

With `files=True`, local scorers also receive `directory`,
`reference_directory` and `files_field`. The first is the current prediction
root; committed paths remain relative to it. It is runtime context, not an
absolute path added to the scorer's semantic kwargs. JSON-only scorers need not
accept these keywords. Files are verified by the existing journal before scoring.

For binary references, pass a content-verified `DatasetSnapshot` directory or
object and declare `reference_role` on the view. The pair includes named
`reference_assets` declarations, with relative paths, digests, byte counts and
metadata. The scorer opens them under `reference_directory`. Missing role assets
remain unavailable. Snapshot content is verified at evaluation admission, and
moving the snapshot does not alter its content identity. Plain JSONL references
bind their record content; they do not magically pin arbitrary files read by
custom scorer code. Use the snapshot contract for that claim.

## Aggregate And Apply Policy

There is no default weighted score or metric direction. An explicit installed
aggregator receives copies of all view reports and returns a finite
`{"metrics": {...}}` object, with optional JSON metadata:

```python
suite = arti.EvaluationSuiteConfig(
    views,
    aggregator=arti.PythonFactoryConfig(
        "my_package.metrics:combine", {"weights": my_weights},
    ),
    aggregation_id="my-aggregation-revision",
    criteria={"my_composite": {"minimum": my_minimum}},
)
```

Per-view `criteria` and composite criteria use the shared minimum/maximum
checks. Missing metrics are `not_evaluable`, not passed. No criteria means
`criteria_passed=None`. Coverage is available to an aggregator when the caller
wants an explicit completeness criterion; there is no invented tolerance or
cohort reweighting. `complete_evidence` remains separate from criteria status.
Partial evidence uses `evaluated_with_gaps` or `passed_with_gaps`, as applicable.

A scorer failure is recorded per view and per eligible ID while other views
can complete. Aggregator/scorer failures cannot be concealed as policy success.
Reports retain all outcomes; the latest atomic report has a content-addressed
`.reports/<sha>.json` history, preserving earlier reports. Configuration/source
errors fail before publishing; a deliberately completed evaluation with a
failed view is a truthful failed report, not a newly accepted model.

## Trials, Recovery And Comparison

Declare original input fields when an application has repeated trials:

```python
trials = arti.EvaluationTrialFields(
    initial_state="state_revision", seed="seed", trial="trial_number",
    phase="application_phase",  # optional application context
)
comparison = arti.compare_prediction_suites(
    "outputs/candidate", "outputs/baseline", suite,
    references="references.jsonl", output="outputs/comparison.json",
)
```

Trial fields are optional; declared fields must exist and a declared seed must
be a nonnegative integer. Default comparison pairs the same committed IDs and
exact original input content, including declared initial state and seed. IDs
should identify the independent sample/trial, not just a task category. Explicit
`pairing="intersection"` permits partial overlap, records excluded IDs and keeps
both original populations. No automatic matching of unrelated trials occurs.

New prediction part receipts record actual execution `attempt` and
`phase=initial|recovered`. Recovery does not rewrite earlier parts. Old receipts
without this evidence report `unrecorded`, not an inferred initial attempt.
This execution phase is separate from a declared application's phase and from
externally measured outcomes. The phase map has its own content identity.

Each view's delta is computed only when both arms have measured aggregates on
exactly the same scored ID set. Deltas are candidate-minus-baseline; their
direction and units remain task-owned. An external aggregate for a larger
population is not reused on a selected intersection: individual outcomes remain
available, but the aggregate is unavailable until that producer supplies one for
the actual cohort. Composite comparison also requires comparable view evidence.

## Import External Evidence

Give an external evaluator `arti.evaluation_source_identity(prediction_dir)`.
It must return that exact source identity, the declared protocol/revision,
its producer identity, finite metrics and per-ID scores:

```python
evidence = {
    "source": arti.evaluation_source_identity("outputs/prediction"),
    "protocol": "my-protocol-revision",
    "producer": {"name": "my-evaluator", "revision": "my-evaluator-revision"},
    "metrics": external_metrics,
    "scores": external_per_id_outcomes,
}
arti.import_evaluation_evidence(
    "outputs/prediction", suite, "environment", evidence,
    output="outputs/external-evidence.json",
)
report = arti.evaluate_prediction_suite(
    "outputs/prediction", suite, references="references.jsonl",
    external={"environment": "outputs/external-evidence.json"},
)
```

The importer checks source model/task/input/output/execution identities,
protocol and known committed IDs. It seals the raw declaration and outcomes;
it does not perform the external evaluation or prove that an untrusted producer
really executed its claimed protocol. Origin always remains external,
evaluator-declared and not locally reproduced. Unknown/changed source, view or
protocol is not silently rebound. A comparison supplies separate evidence maps
under `candidate` and `baseline`.

## Connect Deployment Validation

Pass `EvaluationSuiteConfig` instead of `PredictionScoringConfig` to the existing
`validate_recipe_deployment`:

```python
checked = arti.validate_recipe_deployment(
    package, requests, prediction_run, suite,
    targets=references, external=external_evidence,
    task_factory="my_package.tasks:build_task",
)
```

The same runner performs remaining prediction, suite evaluation, derived score
publication and explicit deployment criteria. Complete retries rescore without
model replay. Current/prior/not-run model reload evidence remains separate from
view availability, failures and quality policy. Full per-ID evidence is saved in
the existing content-addressed score archive; the validation summary retains
view coverage/status/checks. Reports cannot overwrite original references or
result files. Inference completion is not full evaluation completeness.

## CLI

```bash
arti recipe score-suite outputs/prediction --suite suite.json --references references.jsonl --output evaluation.json
arti recipe compare-suites outputs/candidate --baseline outputs/baseline --suite suite.json --references references.jsonl --output comparison.json
arti recipe import-evaluation outputs/prediction --suite suite.json --view environment --evidence raw-evidence.json --output external-evidence.json
arti recipe validate-deployment outputs/package --data requests.jsonl --run-dir outputs/prediction --evaluation-suite suite.json --references references.jsonl --task-factory my_package.tasks:build_task --output validation.json
```

JSON/TOML suite files mirror `to_dict()` with
`format="arti.evaluation-suite-config"`. `--external-json` maps view names to
evidence paths (or candidate/baseline mappings for comparison), resolved relative
to that manifest. CLI summaries omit raw per-ID answers; the saved report retains
them. No dependencies, external tools, evaluation providers or hardware are
automatically allocated.

Structured/file prediction and the shared interactive episode runner use these
same per-ID scorer/report contracts. Episode rows preserve their actual trial
execution, initial/recovered outcome availability and trace phases. An interrupted
initial attempt has no fabricated terminal outcome; the later observed outcome
is attributed to the recovered attempt. Configure `EvaluationTrialFields` for
the original input fields; cross-arm comparison additionally verifies actual
committed episode state/seed/trial identities even when each run selected
different aliases. Identical raw input JSON alone is insufficient for pairing.
Reference gaps, local environment outcomes and imported external producer evidence
remain distinct. See [Independent Task Episodes](task-episodes.md) for explicit
state lifetimes, batch independence and recovery boundaries. These workflow
contracts do not add a second evaluation authority or qualify physical effects,
native mutable-model state or task quality.

Reference checked 2026-10-11:
[MLflow precomputed evaluation and separate validation](https://mlflow.org/docs/latest/ml/evaluation/)
separates stored prediction assessment, application metrics and threshold policy.
Transfer those mechanisms; do not add its metric engine, GenAI judges, registry,
automatic defaults or interoperability claims as dependencies.
