# Evaluate Generated Answers

Teacher-forced token loss and generated-answer correctness are different metrics.
Configure generation checks in the same Qwen recipe used for training, recovery
and deployment. No new trainer or training loss is introduced.

[Evaluate A Qwen Asset Independently](qwen-independent-evaluation.md) explains
new held-out data, selected-split preparation and evaluation without constructing
optimizer or training state.

## Declarative Validation

```toml
[evaluation]
generate_samples = 32
sampling = "seeded"
sample_seed = 17
comparison = "numeric"

[generation]
max_new_tokens = 64
batch_size = 8

[run]
evaluation_steps = 50

[run.validation]
monitor = "generation_score"
mode = "max"
selection = "best"
patience = 6
```

The profile governs scheduled validation and the selected model's final
evaluation. Defaults remain `generate_samples=0`, `sampling="first"`,
`comparison="exact_text"`; automatic selection/early stopping is still opt-in.
Only the validation split selects a checkpoint. Evaluate test separately.

`seeded` selects by a deterministic hash of seed and stable record ID. It does
not consume training RNG, and selecting from the same ID set is independent of
source ordering. The report records actual generated IDs and sample count. A
request larger than the split uses all available records, not duplicated samples.

The plan reports `evaluation_plan.max_generated_tokens_per_check` as a static
upper bound, not latency, FLOPs or observed output length. Generation checks
add real work; schedule their cadence and count deliberately. Custom callbacks
can add work outside that builtin profile estimate. The existing
training-time allowance is checked at optimizer boundaries and is not a deadline
for loading, evaluation or export.

## Comparing Replies

`exact_text` compares the complete reply and target after outer whitespace is
stripped. It does not ignore punctuation/case or extract an answer from prose.

`numeric` is explicitly for numeric-only targets. The entire stripped reply must
be a finite decimal number, optionally signed, with correctly grouped thousands,
a decimal point or an exponent. Decimal equality avoids floating-point rounding:
`42.0` can match `42`, and `1,000` can match `1e3`. `The answer is 42`, fenced
code, units, malformed commas and nonfinite values do not match. Targets that
are not numeric are a configuration error, checked before a training update.

Reports always retain `generation_exact_match`. The configured rule adds
`generation_score`, and numeric mode adds `generation_numeric_exact_match`.
`generation_truncated_samples` separately counts replies stopped by the generation
length budget. Original targets/replies/token IDs/counts/finish reasons remain in
the prediction receipt; a scorer does not rewrite or invent them.

## One Policy Across Entrances

```bash
arti recipe train recipe.toml --run-dir outputs/first
arti recipe resume-run outputs/first --run-dir outputs/recovered
arti recipe evaluate recipe.toml --model outputs/recovered/model --split test
arti recipe evaluate outputs/recovered/model --data new-labeled.jsonl
```

Omitting `--generate-samples` uses the configured profile. An explicit
`--generate-samples 0` disables generation for that evaluation call. Python
`recipe.evaluate(..., generate_samples=None)` behaves the same way.

The builtin evaluator's sample/comparison profile is checkpoint workflow
provenance, not the neural/task fingerprint. Resuming validation history requires
the same builtin profile. A new post-training evaluation can intentionally use
a different profile without pretending that it is a continuation of that history.
The exported inference asset retains the default evaluation profile; asset-only
evaluation therefore does not need the old training recipe.

An explicit Python `validation_fn` still overrides scheduled validation. Its
implementation is caller-supplied, not serialized. Recipe continuation of a
checkpoint marked caller-supplied requires that callback again. The declared
profile still controls the recipe's final evaluation/export stage.

Teacher-forcing and generation run in separate snapshots of owned ARTI state,
runtime resources, host buffers, RNG and modes. The final assistant target is
never passed to the generation branch. RNG preservation follows the shared
session's supported Torch/Python/CUDA state, not arbitrary external generators.
Evaluation restores state without copying
all frozen host parameters. Unregistered Python state and arbitrary mutation of
frozen host parameters remain outside this qualified state contract.

## Rescore Existing Receipts

Raw generation receipts can be rescored without loading a model or generating
new answers:

```bash
python examples/qwen_score_predictions.py outputs/test.predictions.jsonl --comparison numeric --output outputs/numeric-scores.json
```

The example reports how many source records were actually generated and how many
had only teacher-forced metrics. Only generated records form the scoring
denominator. Python `arti.score_causal_lm_predictions(generated_records, ...)`
returns separate metrics and scored copies of the original records.

This profile currently qualifies local single-process Torch text-only Qwen/SFT.
It does not claim new task accuracy, semantic judges, code execution, arbitrary
callback reconstruction or sharded/distributed validation.
