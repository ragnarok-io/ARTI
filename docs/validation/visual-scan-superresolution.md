# VisualScan Pixel-Shift Super-Resolution

This alpha validation asks a narrow physical question: can `VisualScan` recover
detail present across complementary low-resolution observations but not
identifiable from one observation? It does not test generic image enhancement.

## Locked protocol

High-resolution procedural glyphs contain one of four confusable subpixel marks.
Low-resolution frames are generated only through

\[
y_t = D H T_{\delta_t} x + \epsilon_t.
\]

The result artifact records glyph classes, train and held-out fonts, train and
unseen shifts, blur, Gaussian noise, random-data offsets, and three seeds. At
inference every model receives only low-resolution frames, registered shifts,
and masks. The high-resolution glyph remains a target.

The comparison includes nearest single-frame reconstruction, bicubic
interpolation, multi-frame averaging, classical registered shift-and-add, a
parameter-matched neural baseline, `VisualScan`, and `VisualScan` with finite
persistence. Metrics include MSE, PSNR, SSIM, four-way micro-glyph accuracy,
reconstruction entropy, latency, throughput, parameters, and CUDA peak memory.

## Causal controls

- Classical shift-and-add must recover the micro-position, proving that the
  complementary samples contain usable information independently of ARTI.
- Repeating one phase and masking all but one frame must remove the VisualScan
  gain.
- Supplying wrong shift metadata must damage recovery.
- Reordering paired frames and shifts must preserve the result.
- The learned baseline must remain within five percent of VisualScan's parameter
  count.

These controls reject the interpretation that a learned prior simply creates
plausible high-frequency detail.

## Current evidence

The generated CUDA artifact reports near-perfect VisualScan micro-glyph
recognition and better reconstruction than the parameter-matched neural
baseline. The gain collapses to chance for repeated phases and no-concat input;
wrong registration also degrades recovery. Finite persistence retains the
complementary identity. The JSON result contains per-seed measurements and the
exact locked protocol.

The held-out continuous shifts retain micro-glyph identity and reconstruction
gain because displacement is applied as an inverse registration operator rather
than learned as a categorical feature. Held-out font classification remains
strong, but pixel reconstruction does not preserve the unseen font style. This
supports registered complementary sampling, not arbitrary-motion
super-resolution, generic OCR, or hallucination-free enhancement.

## Reproduce

```bash
uv run --extra dev python benchmarks/run_visual_scan_superresolution.py --device cuda
uv run --extra dev python benchmarks/verify_visual_scan_superresolution.py
uv run --extra dev python -m pytest tests/test_visual_scan.py tests/test_visual_scan_benchmark.py
```

The verifier fails if the physical contract or locked split is absent, if a seed
or metric is missing, if high-resolution targets leak into inference, if the
negative controls retain the claimed gain, or if the baseline is not parameter
matched.
