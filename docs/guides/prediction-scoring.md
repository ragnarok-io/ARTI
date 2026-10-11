# Offline Prediction Scoring

Score a [prediction run](prediction-runs.md) without loading its model, task
factory, training data or optimizer. The scorer consumes committed results and
original labeled records. Regression, classification, language and developer
tasks share this workflow; ARTI does not invent a task metric or alter outputs.

## Scorer Contract

An explicitly installed callable receives a list of paired records:

```python
{
    "id": "sample-1",
    "input": {...},       # original pinned inference record
    "target": {...},      # original labeled record
    "prediction": {...},  # unchanged task output, including its ID
}
```

It returns a finite JSON-compatible metric object and one score record per ID:

```python
def score(pairs, *, prediction_field="prediction", target_field="target"):
    scores = [
        {
            "id": pair["id"],
            "squared_error": (
                pair["prediction"][prediction_field]
                - pair["target"][target_field]
            ) ** 2,
        }
        for pair in pairs
    ]
    return {
        "metrics": {"mse": sum(row["squared_error"] for row in scores) / len(scores)},
        "scores": scores,
    }
```

Outputs may be returned in any ID order, but IDs must cover exactly the input
pairs, once each. Scores can contain task-owned booleans, numbers or structured
values. Metric names and directions belong to the task, not ARTI. The examples
in `examples/prediction_scorers.py` include original-unit regression and
classification; arbitrary loss objectives are not assumed to be deployment metrics.

## Score Saved Results

```python
import arti

scoring = arti.PredictionScoringConfig(
    scorer=arti.PythonFactoryConfig(
        "my_project.metrics:score", {"target_field": "measured_value"}
    ),
    scorer_id="original-unit-mse-v1",
    target_id_field="sample_id",
)
report = arti.score_prediction_run(
    "outputs/prediction", scoring,
    targets="held_out_labels.jsonl", output="outputs/scores.json",
)
```

```bash
arti recipe score-prediction outputs/prediction --targets held_out_labels.jsonl --scorer my_project.metrics:score --scorer-id original-unit-mse-v1 --scorer-kwargs-json scoring-options.json --target-id-field sample_id --output outputs/scores.json
```

Without `targets`, the run's pinned inputs are the target records. This is useful
when requests already contain original labels; it does not infer labels from
predictions. Target IDs default to the run's input ID field. Label files may be
reordered. Missing/duplicate IDs are errors; extra target IDs are explicitly
listed as unused. A changed label set produces a different target/report identity.
The scorer owns the meaning and validity of labels; file identity is not proof
of ground truth quality.

Only committed parts are authoritative. A missing final merged JSONL does not
require inference or a repair before scoring. Source metadata/part fingerprints
are checked through the existing prediction inspector; source files are never
modified. A moved run needs no model asset. Unfinished runs require explicit
`allow_partial=True` / `--allow-partial`, and report committed coverage,
uncommitted count and unused targets. Zero committed records cannot be scored.

## Compare Models

```python
comparison = arti.compare_prediction_runs(
    "outputs/candidate", "outputs/baseline", scoring,
    targets="held_out_labels.jsonl", output="outputs/comparison.json",
)
```

```bash
arti recipe compare-predictions outputs/candidate --baseline outputs/baseline --targets held_out_labels.jsonl --scorer my_project.metrics:score --scorer-id original-unit-mse-v1 --output outputs/comparison.json
```

Both arms use the same original target set and scoring declaration. Samples are
joined by ID, not row/partition order. Their full original inference records
must match on paired IDs. Batch partition sizes and model/task contracts may
differ; both plans, model manifests and task identities are recorded, not
silently treated as equivalent conditions.

Default `pairing="exact"` requires the same committed sample IDs. Use explicit
`pairing="intersection"` / `--pairing intersection` to compare partial coverage.
Each arm reports excluded IDs, paired/committed and paired/total fractions, and
uncommitted counts. A comparison cannot silently score a smaller denominator.

`delta` includes only shared finite numeric metrics, candidate minus baseline.
It does not label the difference as an improvement: lower MSE and higher
accuracy have different directions. Scores are recomputed on the paired subset,
not subtracted from reports with different sample populations. Raw sample
inputs, labels, predictions and task score records remain inspectable.

## Evidence And Scope

The self-contained report records scorer reference, declared rule identity and
kwargs, prediction/source/plan/task hashes, full and used target hashes, coverage,
metrics and per-sample records. It has a content fingerprint and is published by
one atomic file replacement. Scorer or publication failures leave an existing
report unchanged. Reports can live in a separate evaluation subdirectory of
the run, but cannot overwrite its input, plan, progress, merged predictions,
committed parts or the label file.

The installed scorer is trusted application code. Its `scorer_id` is a declared
version/rubric, not an automatic audit of installed package bytes or a guarantee
of reproducible external state. Package dependencies should be pinned by the
consumer. It receives private copies for preprocessing; saved source values
remain unchanged. No remote judge or metric package is downloaded automatically.

This offline API materializes records/pairs/scores in O(N) host memory. It is not
distributed, streaming aggregation or a GPU-throughput feature. Nonadditive
metrics are computed by the task over the actual complete selected population.
One local writer per report, and no concurrent mutation of a source run during
scoring, are the qualified scope. Model-dependent metrics require the predictor
when their necessary observations were not saved; offline scoring cannot invent
logits, unseen targets or unsaved intermediate tensors.
