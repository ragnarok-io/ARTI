# Pulse Visual Field Concat

This CUDA mechanism test asks whether Pulse can expand its glyph observation
range by concatenating rigid visual fields before Half/Fold compaction.

Each synthetic glyph has two independent identity bits. One is visible only in
the left half and one only in the right half, producing four classes. A single
half-field therefore has a two-class information ceiling. Every condition uses
the same Pulse and classifier parameter count.

| Condition | Fields | Fragments | Parameters | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Left only | 1 | 2 | 1,359 | 49.89% |
| Right only | 1 | 2 | 1,359 | 50.03% |
| Full bitmap field | 1 | 4 | 1,359 | 100% |
| Left + right concat | 2 | 4 | 1,359 | 100% |

Concat conserves the source bitmap pixels up to floating-point reduction error
and reaches the same accuracy as one full field. No resize, interpolation,
Pulse-output mixture, or additional trainable parameter is used. This supports
Visual Field concat as an input-line expansion mechanism; it does not yet prove
superiority on natural text corpora or arbitrary font distributions.

```bash
uv run --extra dev python benchmarks/run_visual_field_concat.py --device cuda
uv run --extra dev python benchmarks/verify_visual_field_concat.py
```
