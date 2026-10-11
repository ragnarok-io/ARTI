# Compare Qwen Evaluation Results

An aggregate loss can improve while particular replies get worse. Compare the
candidate with a baseline on the same examples before deciding which asset to
deploy. ARTI can reuse existing evaluation receipts; comparison does not load a
model, read its weights, run generation, allocate CUDA or create an optimizer.

## Record Both Evaluations

Use the same labeled records and an explicit generated-answer sample count:

```bash
arti recipe evaluate deployments/baseline --data evaluation.jsonl --generate-samples 32 --output evaluations/baseline.json
arti recipe evaluate deployments/candidate --data evaluation.jsonl --generate-samples 32 --output evaluations/candidate.json
```

Each evaluation saves a JSON report and a sibling `.predictions.jsonl`. Reports
record native NLL, its token denominator, task/data identity, the evaluation
profile and generation configuration. New prediction rows record an input
fingerprint, the original target and generated text/tokens/finish reason. Gold
answers are not included in generation prompts.

Match the saved generation budget and input/template settings for a controlled
comparison. Evaluation itself runs the model; only the subsequent comparison is
offline. Default `generate_samples=0` still means no generated answers. Choose a
positive value when creating receipts to compare replies.

## Compare Without Running Models Again

```bash
arti recipe compare-evaluations evaluations/candidate.json --baseline evaluations/baseline.json --output evaluations/comparison.json
```

Python can compare receipts or already loaded raw generation rows:

```python
import arti

result = arti.compare_causal_lm_evaluations(
    "evaluations/candidate.json",
    "evaluations/baseline.json",
    output="evaluations/comparison.json",
)
print(result["outcomes"])

paired = arti.compare_causal_lm_predictions(
    candidate_rows, baseline_rows, comparison="exact_text",
)
```

Rows are paired by stable ID, not their file order. Known input fingerprints and
original targets must agree for each pair. The output contains:

- `paired_samples` and per-arm coverage/excluded IDs.
- `improved`, `regressed`, `both_correct`, `both_incorrect` outcome counts.
- Each arm's score, literal exact match, truncations and recorded generated tokens.
- Candidate-minus-baseline deltas, which may be positive, negative or zero.
- Conditions marked `matched`, `different` or `unrecorded`.
- Original source reports and fingerprints, without replacing their metrics.

`comparison.json` stores the summary; `comparison.pairs.jsonl` stores the aligned
raw replies and per-case outcomes. The Python result also returns these paired
rows under `predictions`. Inspect regressions and truncations as well as the
headline score. Generated token counts describe reply length, not measured
latency, FLOPs or GPU throughput.

## Keep The Scoring Rule Explicit

Without an override, report comparison reuses a common declared scoring profile,
or literal `exact_text` for older receipts. Different declared profiles require
an explicit common rule. Both arms are rescored from their raw replies; stored
match flags do not determine the result.

```bash
arti recipe compare-evaluations evaluations/candidate.json --baseline evaluations/baseline.json --comparison numeric --output evaluations/numeric-comparison.json
```

`numeric` compares the entire finite numeric response, not an extracted final
number. `42.0` can match target `42`; `The answer is 42` does not. Literal exact
match remains available separately. Neither scoring rule judges arbitrary
semantic equivalence or factual quality.

## Partial And Historical Receipts

Default `pairing=exact` requires identical generated ID sets. Use intersection
only intentionally:

```bash
arti recipe compare-evaluations evaluations/candidate.json --baseline evaluations/baseline.json --pairing intersection --output evaluations/partial-comparison.json
```

Intersection reports each arm's full generated count, matched fraction and
excluded IDs. Its score denominator is the intersection, not the full dataset.
It never silently treats absent replies as wins or drops known regressions.

Older receipts can still be compared. Missing input fingerprints or generation
budgets are reported as unverified/unrecorded rather than invented or recovered
by another model forward. Recorded differences in input settings or generation
budgets are disclosed, not converted into an automatic pass/fail policy.

Teacher-forced loss/perplexity/token-accuracy deltas refer to the full evaluation
dataset, not the generated intersection. They are emitted only when recorded
encoded-data identity, sample/token counts, native objective and weighting agree.
For different tokenizations or denominators these deltas are unavailable, while
paired reply scores can still be useful.

Comparison does not choose a checkpoint, promote a model, rewrite an export or
enforce an application quality threshold. Keep training selection on validation
data; use held-out test results for final reporting. A short integration run does
not qualify an application or establish model superiority.

`examples/qwen_evaluation_comparison.py` is the offline executable wrapper. The
design follows the same-data/same-scoring, per-example inspection pattern in
[MLflow's evaluation-driven development workflow](https://www.mlflow.org/cookbook/eval-driven-development/),
without requiring its server, account or model calls.
