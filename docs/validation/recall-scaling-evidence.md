# Recall Scaling And Line Capacity

This validation freezes feature development and studies Recall capacity under
controlled parameter and runtime budgets. It does not use short-run task
accuracy as a verdict on sufficient-training capability.

## Cached Screen

The frozen Qwen hidden-trace cache supports a model-free CPU screen over:

- insertion depth: 1 through 4;
- complete lines per depth: 1 through 4;
- rank: 2, 3, 4, 6, 8, 12, and 16;
- slots: 8, 24, and 64;
- Half enabled and disabled;
- total parameter budgets: 25k, 50k, and 100k.

The committed screen completed 48 balanced candidates in 77.96 seconds. Pareto
frontiers minimize normalized repair MSE, unseen delta MSE, and parameter cost.
This screen is a local proxy and does not model cross-layer propagation.

```bash
uv run --extra qwen python benchmarks/screen_recall_scaling_matrix.py \
  --device cpu --steps 30 --max-candidates 48 --max-runtime-seconds 120
```

## Matched Qwen Training

The real-Qwen comparison locks every trainable condition to 25k parameters
within 5%:

| Condition | Parameters |
| --- | ---: |
| Ordinary residual adapter | 24,576 |
| LoRA | 24,576 |
| Single-line Recall | 24,914 |
| Same-depth two-line Recall | 24,916 |
| Multi-depth Recall | 24,724 |

All methods optimize the same frozen-hidden reconstruction objective from
corrupted inputs, with an unseen-input preservation term. Three seeds run for
80 steps each. Every batch is resumable at the condition level and completed
in less than two minutes, below the nine-minute hard limit.

The aggregate result is:

| Condition | Convergence ratio | Validation alignment | Hidden cosine | NLL | Unseen hidden drift | Unseen KL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen | n/a | 3.7188 | 0.8651 | 1.8435 | 0.0000 | 0.000000 |
| Adapter | 0.4794 | 1.7969 | 0.9307 | 1.1862 | 24.2821 | 0.319359 |
| LoRA | 0.5789 | 2.1562 | 0.9178 | 1.2521 | 17.1225 | 0.506582 |
| Single-line Recall | 0.9354 | 3.4635 | 0.8727 | 1.8115 | 5.6430 | 0.010399 |
| Same-depth two-line Recall | 0.9122 | 3.3698 | 0.8754 | 1.7922 | 7.2434 | 0.017825 |
| Multi-depth Recall | 0.9186 | 3.4167 | 0.8729 | 1.8041 | 8.5742 | 0.089570 |

Same-depth two-line Recall improves validation alignment over single-line Recall
in every seed at the same parameter budget. Recall also produces substantially
less unseen drift than the faster-fitting Adapter and LoRA baselines.

The Recall convergence ratios remain above 0.91. Adapter and LoRA optimize this
short common objective faster, but the protocol does not establish a Recall
capacity ranking because Recall is not near convergence. No task-accuracy
superiority claim is made.

## Complete-Line Concat

Two independently trained single-line artifacts are appended as complete
lines. Query, emit, gate, recognition, bank, and Half state remain independent;
the first line is not remixed or overwritten.

- old-line SHA-256 before and after append is identical;
- line count grows from 1 to 2;
- parameters grow from 24,914 to 49,828;
- the better individual alignment loss is 3.4062;
- direct append improves alignment to 3.3125;
- hidden semantic cosine rises to 0.8770.

This supports the narrow claim that adding an independent line expands the
observed recoverable representation boundary. It is not yet a capacity scaling
law over converged training.

## Reproduce And Verify

Each GPU invocation must be reviewed and remain below 540 seconds:

```bash
uv run --extra qwen python benchmarks/run_qwen_recall_scaling.py --seeds 503 --resume
uv run --extra qwen python benchmarks/run_qwen_recall_scaling.py --seeds 509 --resume
uv run --extra qwen python benchmarks/run_qwen_recall_scaling.py --seeds 521 --resume
uv run --extra qwen python benchmarks/run_qwen_recall_line_concat.py
uv run --extra dev python benchmarks/verify_recall_scaling_evidence.py
```

The committed result retains every response string, train/validation alignment,
NLL, semantic hidden cosine, unseen hidden and logit drift, line utilization,
per-parameter gain, throughput, peak CUDA memory, and independent SafeTensors
artifact.
