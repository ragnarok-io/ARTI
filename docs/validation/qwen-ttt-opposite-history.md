# Qwen TTT Opposite-History Gate (retired diagnostic)

> Status: retired as a mechanism gate. The protocol and evaluator remain in
> the private repository as an auditable record, but must not be used to claim
> that TTT Recall succeeds or fails.

This was an early private diagnostic for a narrow claim. It is retained to
preserve provenance, not as the current validation contract.

The machine-readable record is
`benchmarks/qwen_ttt_opposite_history_protocol.json`. It records the existing
Updater artifact and rollout corpus by SHA-256, together with the held-out
dialogue range, pairing seeds, controls, metrics, and thresholds used by the
retired run.

The retired implementation allowed the student path to include a
teacher-forced response prefix and did not guarantee that the latest-message
position frame was independent of the history length. Those are causal-design
confounds, not evidence about Recall.

Controls are: matching history, paired wrong history, reversed exchange order,
an exact zero Bank, no active runtime, and the full-context teacher. The gate
also compares whole-exchange writing with token-by-token writing.

Run the non-model preflight first:

```bash
uv run --extra qwen python benchmarks/evaluate_qwen_ttt_opposite_history.py --plan-only
uv run --extra dev pytest -q tests/test_qwen_ttt_opposite_history_gate.py
```

The first model-backed run is limited to the locked 24 held-out pairs and does
not train or change any model, Updater, Query, Bank, or threshold. A passing
result supports only the stated causal-history claim. A failure is classified
as zero-state preservation, chunk consistency, write specificity, read
causality, or teacher/reset alignment; it must not be repaired by silently
increasing capacity or changing the primary metric after observing the result.

## First diagnostic result

The first model-backed run completed in 18.27 seconds on a checkpoint trained
for only 416 optimizer steps. That run is not a mechanism test: the checkpoint
was not trained to convergence, the student path contained a response-prefix
confound, and the causal input contract was not exact.

The run did not establish history-specific memory. Its numerical observations
are useful only for debugging that checkpoint and that protocol.

The diagnostic therefore exposed three protocol/checkpoint questions:

- chunk consistency: token and exchange writes do not represent one transition;
- write specificity: the state captures a general record-query task prior but
  not the paired record identity;
- read causality: changing to the paired wrong state does not swap the answer.

No positive or negative mechanism claim should be derived from this result.
The successor protocol must first pass an input-integrity Gate 0, a
Bank-to-logits reachability probe, and a train-to-criterion small-task Gate 1.
Only then may held-out history and free-running generation be evaluated.
