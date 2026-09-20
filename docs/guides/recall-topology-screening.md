# Recall Topology Screening

Layered Recall topology is a hyperparameter space, not a fixed three-layer
architecture. ARTI separates cheap layer-local screening from expensive
full-model confirmation so developers can explore that space without repeatedly
executing a pretrained backbone.

## Candidate and budget

```python
import arti
from arti.legacy import LayerRecallSpec, LayeredRecallConfig

candidate = arti.LayeredRecallCandidate(
    "early-heavy-five-layer",
    LayeredRecallConfig(
        layers=(
            LayerRecallSpec("model.layers.2", dim=1024, rank=16, slots=24),
            LayerRecallSpec("model.layers.8", dim=1024, rank=4, slots=24),
            LayerRecallSpec("model.layers.14", dim=1024, rank=4, slots=24),
            LayerRecallSpec("model.layers.20", dim=1024, rank=4, slots=24),
            LayerRecallSpec("model.layers.26", dim=1024, rank=4, slots=24),
        )
    ),
    abstention_weight=1.0,
    tags=("nonuniform", "early-heavy"),
)

budget = arti.LayeredRecallBudget(
    max_parameters=30_000,
    max_steps=40,
    max_runtime_seconds=120,
    max_candidates=32,
)

cost = arti.estimate_layered_recall_cost(candidate, tokens=512)
```

The static estimate uses each branch's actual dimension, rank, slot count, and
recognition operator. Parameter matching is therefore independent of a
particular early/middle/late convention.

## Frozen trace cache

`LayeredRecallTraceCache` stores clean, single-corruption,
combined-corruption, and unseen hidden tensors for each selected path. The
cache uses safetensors plus a JSON manifest and source fingerprint. Screening
loads this cache and never executes Qwen or another source model.

```python
cache = arti.LayeredRecallTraceCache.load("hidden-traces.safetensors")
score = arti.screen_layered_recall_candidate(candidate, cache, budget)
```

The score reports normalized repair MSE, normalized unseen delta MSE, parameter
count, runtime, completed steps, and per-layer metrics. The proxy trains each
layer-local branch independently. It is useful for rejecting weak or expensive
topologies, but it does not model cross-layer propagation and cannot replace
full-model confirmation.

## Pareto selection

```python
frontier = arti.pareto_layered_recall(scores)
```

A score is removed only when another candidate is no worse in repair,
abstention, parameters, and runtime and is strictly better in at least one.
This keeps selective, efficient, and high-repair alternatives instead of
collapsing topology search into one arbitrary scalar optimum.

## Qwen tools

The extraction command performs one bounded Qwen pass per trace view. It must
only be run after the developer has reviewed the layer count, expected time,
and GPU memory:

```bash
uv run --extra qwen python benchmarks/cache_qwen_layered_recall_traces.py --max-runtime-seconds 120
```

After extraction, screening is resumable and defaults to CPU with a hard
two-minute wall-clock limit:

```bash
uv run --extra qwen python benchmarks/screen_qwen_recall_topologies.py \
  --device cpu --max-candidates 32 --max-runtime-seconds 120 --resume
```

The enumerator includes single layers, uniformly distributed depths,
early-heavy, late-heavy, alternating capacity profiles, and repeated-key
topologies with multiple independent lines at one physical layer, with Half,
recognition, and abstention variants under a matched parameter target. Only
the Pareto frontier should advance to complete-string confirmation.

## Evidence boundary

The original v1 result and interrupted fixed-three-layer v2 result remain
diagnostic baselines. The latter is frozen in
`benchmarks/results/qwen_layered_recall_v2_partial_diagnostic.json`; it is not a
completed confirmation result. New confirmation runs must use untouched seeds,
declare their expected time and memory before execution, remain below ten
minutes per batch, and resume without repeating completed conditions.
