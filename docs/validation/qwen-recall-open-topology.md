# Qwen Recall Open-Topology Confirmation

This experiment replaces the fixed early/middle/late assumption with an open
topology search under matched parameter and runtime budgets.

## Workflow

1. One bounded Qwen pass cached clean, single-corruption,
   combined-corruption, and unseen hidden tensors at seven depths.
2. A model-free CPU screen evaluated 32 balanced candidates spanning single,
   uniform, nonuniform, repeated-key, Half, recognition, and abstention axes.
3. Pareto selection advanced five candidates to full Qwen confirmation on
   untouched prompts and seeds 401, 409, and 419.
4. Each GPU batch completed in about five minutes, below the locked nine-minute
   limit, and all complete response strings were retained.

## Result

Evidence integrity passes; broad task promotion does not. Across every seed:

- every selected topology improves frozen-Qwen conditional NLL;
- every selected topology improves frozen-Qwen hidden semantic cosine;
- the best multi-line candidate beats the matched single-line candidate on NLL;
- removing any path worsens its trained topology;
- removing any one of the repeated early lines worsens NLL;
- every compatible layer-state reorder worsens NLL.

The aggregate frontier is not one optimum:

| Topology | NLL | Semantic cosine | Unseen KL | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Five-line nonuniform | 1.7808 | 0.7690 | 0.03608 | strongest repair, highest drift |
| Repeated early two-depth | 1.8173 | 0.7765 | 0.00418 | strong repair/selectivity compromise |
| Two-line nonuniform | 1.9762 | 0.7740 | 0.00328 | intermediate frontier point |
| Two-line uniform | 2.4211 | 0.7290 | 0.00193 | conservative multi-line option |
| Matched single line | 2.6814 | 0.7292 | 0.00015 | strongest abstention, weakest repair |
| Frozen Qwen | 2.7435 | 0.7133 | 0.00000 | unchanged reference |

No candidate consistently exceeds frozen Qwen's exact marker accuracy on both
corruption families. The supported conclusion is therefore narrower: adding
and redistributing independent Recall lines expands the latent repair and
semantic frontier, with a measurable selectivity cost. It does not yet prove
broad answer-accuracy improvement.

## Reproduce

Trace extraction is a GPU action and must be reviewed before execution:

```bash
uv run --extra qwen python benchmarks/cache_qwen_layered_recall_traces.py --max-runtime-seconds 120
```

Offline screening does not execute Qwen:

```bash
uv run --extra qwen python benchmarks/screen_qwen_recall_topologies.py --device cpu --steps 120 --max-runtime-seconds 120
```

Full confirmation is resumable and hard-limited to nine minutes per batch:

```bash
uv run --extra qwen python benchmarks/run_qwen_recall_topology_confirmation.py --seeds 401 --resume
uv run --extra dev python benchmarks/verify_qwen_recall_topology_confirmation.py
```

The complete table, response strings, resource measurements, and causal
controls are stored in `benchmarks/results/qwen_recall_topology_confirmation.*`.
