# Recall Recognition Modes

ARTI supports two implementations of the requirement that unseen signals
should not produce a recalled trace.

`explicit` compares the current latent trace with the retrieved trace and uses
a configured threshold and temperature. `alignment` has no fixed familiarity
criterion. Its small recognizer is trained by complete/corrupted trace
alignment plus a zero-influence target for unrelated unseen signals. `none` is
the ablation in which every retrieved candidate can flow into Half.

The controlled CUDA benchmark uses the same trace bank, optimizer budget, and
three seeds for all modes. Identity labels and query targets are absent.

| Mode | Seen corrupted MSE | Seen influence | Near-unseen influence | Random-unseen influence |
| --- | ---: | ---: | ---: | ---: |
| Explicit | 0.0149 | 3.7809 | 2.1828 | 0.2341 |
| Alignment | 0.0008 | 3.9915 | 0.1094 | 0.0809 |
| None | 0.0000 | 3.9932 | 3.5364 | 3.1051 |

Both recognition paths suppress unrelated unseen Recall while retaining strong
influence for experienced corrupted traces. In this synthetic setting,
alignment also rejects difficult near-neighbor negatives more strongly. This
is mechanism evidence, not yet a universal task-level ranking.

```bash
uv run --extra dev python benchmarks/run_recall_recognition_modes.py --device cuda
uv run --extra dev python benchmarks/verify_recall_recognition_modes.py
```
