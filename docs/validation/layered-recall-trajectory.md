# Layered Recall Trajectory Validation

This controlled CUDA experiment asks whether three small, depth-local Recall
branches compose processing traces better than one larger Recall branch with
the same total parameter budget.

## Locked comparison

The backbone is a frozen three-stage Transformer-like nonlinear residual
pipeline. Every condition receives identical train/evaluation batches,
optimizer steps, examples, corruption schedule, and CUDA device:

- three Recall branches, rank 4 each: 726 parameters;
- one Recall branch at early, middle, or late depth, rank 12: 722 parameters;
- three ordinary low-rank adapters, rank 5 each: 756 parameters;
- three Recall branches without Half: 726 parameters;
- no Recall and depth/order/artifact controls.

Training alternates single and combined corruptions. Its only targets are
detached clean hidden traces from the same sample at the same named layer.
Labels are generated only for the evaluation accuracy calculation; answers,
labels, and future tokens are forbidden Recall targets.

## Results

Across seeds 17, 31, and 53, layered Recall beats the best of all three
single-layer locations on both corruption families. Mean results are:

| Condition | Single MSE | Combined MSE | Combined accuracy | Unseen influence |
| --- | ---: | ---: | ---: | ---: |
| Layered Recall | 0.00727 | 0.02761 | 0.995 | 0.0324 |
| Best single-layer mean | 0.01578 | 0.04269 | 0.978 | 0.0062 |
| No Recall | 0.04294 | 0.16041 | 0.551 | 0.0000 |
| Matched adapter | 0.00397 | 0.01580 | 0.995 | 0.1844 |

The Layered Recall unseen influence is only 0.2%-1.6% of its in-distribution
combined-trace influence per seed. Absolute cross-layer correction similarity
stays below 0.09, so the three branches are not merely emitting one duplicated
delta.

Every seed's combined MSE improves monotonically as early, middle, and late
branches are enabled. Removing any one branch worsens combined MSE. Rotating
artifacts across layer paths or repeating the early artifact at every depth
more than doubles error, demonstrating that layer identity matters.

## Boundary

The matched ordinary adapter achieves lower direct reconstruction MSE than
Recall on this task. It also produces roughly 5.7 times more influence on
unseen inputs. Therefore the result supports a narrower claim: depth-local
Recall provides compositional repair and stronger abstention than a
single-layer Recall of equal size. It does not establish broad superiority over
ordinary adapters or LoRA.

Half improves single-corruption MSE and lowers unseen influence, but the no-Half
variant is slightly better on combined MSE. This is recorded as a selectivity
tradeoff, not hidden behind a scalar winner claim.

## Reproduce

```bash
uv run --extra dev python benchmarks/run_layered_recall_trajectory.py --device cuda --steps 360
uv run --extra dev python benchmarks/verify_layered_recall_trajectory.py
uv run --extra dev python -m pytest tests/test_layered_recall.py tests/test_layered_recall_benchmark.py
```

The verifier checks every seed separately, all single-layer locations,
parameter and training budgets, local-loss improvement, monotonic depth curves,
layer-order controls, unseen influence, CUDA resource fields, and the
label/future-token prohibition.
