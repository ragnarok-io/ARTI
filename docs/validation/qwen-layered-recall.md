# Qwen Layered Recall Validation

This is the real-language-model counterpart to the controlled Layered Recall
trajectory benchmark. It runs `Qwen/Qwen3-0.6B` on an RTX 5070 Ti through its
native tokenizer, chat template, causal LM forward, and greedy string
generation boundary.

## Locked protocol

The protocol fixes layers 6, 13, and 20; seeds 17, 31, and 53; 240 optimizer
steps; FP32 precision; support, query, recognition-negative, and ordinary
dialogue prompts; two corruption families; and 24 generated tokens. Qwen's
embedding, 28 decoder layers, norm, and LM head remain frozen. Only Recall or
the parameter-matched adapter control is trainable.

Recall receives no answer strings, next-token labels, future tokens, query
prompts, or generated responses. Its objective is same-support clean/corrupt
hidden alignment plus a clean unseen-input zero-influence term.

## Result

The preregistered verifier **does not pass**. Layered Recall improves frozen
Qwen's conditional NLL in all three seeds, and shuffled layer artifacts degrade
NLL in all three seeds. Seed 17 improves task accuracy from 0.375 to 0.500.
Seeds 31 and 53 tie the single-layer controls on task accuracy, however, and
the early/middle/late depth curve is not strictly monotonic.

The selectivity boundary is stronger: all Layered Recall runs preserve the
ordinary dialogue guard, keep unseen-query semantic overlap above 0.9, and
remain below the locked unseen first-token KL threshold. The matched ordinary
adapter is competitive on NLL but causes more ordinary-dialogue drift in two
seeds.

A post-run frozen-Qwen final-hidden cosine check provides a model-based semantic
metric without changing the promotion gate. Layered Recall improves corrupted
support cosine over frozen Qwen in every seed: 0.8703 to 0.8998, 0.8726 to
0.8823, and 0.8811 to 0.8890. Its unseen clean cosine is 0.9934 in every seed.

This supports a limited finding: learned layer-addressed traces affect real
Qwen generation, can improve NLL, and are sensitive to layer order. It does
not yet support the stronger claim that three-layer Recall consistently
outperforms every equal-parameter single-layer Recall.

## Artifacts and reproduction

The result JSON contains every complete response string, per-layer repair
metrics, Recall influence, cross-layer trajectory cosine, latency, throughput,
peak CUDA memory, frozen-Qwen hidden-state semantic cosine, and all causal
controls. The semantic cosine is descriptive and was not added to the locked
promotion gate after seeing results. Each seed also exports three
layer-bound `.recall.arti.st` files; wrong-layer loading is rejected and two
compatible artifacts can be concatenated from 8 to 16 slots. The report also
records the result of independently removing layer 6, 13, or 20 from each
trained three-layer model.

```bash
uv run --extra qwen python benchmarks/run_qwen_layered_recall.py
uv run --extra qwen python benchmarks/render_qwen_layered_recall_report.py
uv run --extra dev python benchmarks/verify_qwen_layered_recall.py
```

The final verifier command intentionally exits nonzero for the current
evidence. See `benchmarks/results/qwen_layered_recall.md` for the complete
table and failed gates.
