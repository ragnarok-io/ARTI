# Recall Capacity And Depth Scaling

This suite measures Recall as bounded experience replay under a frozen host.
Every cell receives the same optimizer-step budget. Adaptation sees complete
support traces and internally corrupted support views; query targets remain
evaluation-only. The scan varies item count, Recall depth, expert count, and
the per-expert slot budget.

It reports four things together: reconstruction gain while experience fits,
the degradation curve after a field fills, evaluation-only misrecall rate,
false-recall behavior, and the
parameter/time cost of increasing fields. A fixed-size bank is expected to
have a boundary. The relevant engineering question is whether that boundary
is explicit and whether overflow is handled predictably.

`single_allow` is the intentionally unprotected control. `single_abstain`
falls through to the frozen host when an episode exceeds its declared capacity.
`deep_2`, `sharded_2`, and `deep_sharded_4` add independent bounded Recall
fields. This lets the experiment distinguish added capacity from a claim that
one field is infinite.

Run on CUDA and verify the recorded artifact:

```bash
uv run --extra dev python benchmarks/run_recall_scaling_law.py --device cuda
uv run --extra dev python benchmarks/verify_recall_scaling_law.py
```

The deployment rule is simple: scale slots, stages, or `*.recall.arti.st`
experts for retained experience; when no compatible bounded field can accept
the episode, route it elsewhere or fall through without Recall. The benchmark
does not claim a universal scaling exponent or task-level generalization.
