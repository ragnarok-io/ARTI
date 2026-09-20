# Text Tensor

`arti.text_tensor` is an alpha text-physical tensor interface. It keeps visible
text, invisible controls, and simple page/line/x layout metadata in the tensor
contract so downstream runtime vocab and pulse experiments do not silently lose
text identity.

This module does not replace HarfBuzz, Pango, Skia, DirectWrite, or Core Text.
The current backend is a dependency-free Unicode fallback. Future shaping
backends can fill the same contract with real glyph ids, glyph positions, and
font-specific cluster mapping.

## Basic Use

```python
from arti import TextTensorConfig, render_text_layout

layout = render_text_layout("r\u200br", config=TextTensorConfig(normalization="raw"))

sequence = layout.to_sequence_tensor()
pulse = layout.to_pulse_tensor(pulse_count=4)
```

The layout contains:

```text
codepoints     [L]
grapheme_ids   [L]
cluster_ids    [L]
control        [L, K]
visible_mask   [L]
coord          [L, 5]  # page, line, x, y, advance
sequence       [L, D]
```

`sequence` concatenates fallback glyph appearance features, control channels,
and layout coordinates. It is suitable as an external runtime vocab tensor
source. Codepoints are preserved in `layout.codepoints` for audit and
normalization checks, but they are not part of the default model-facing
sequence. Visible character identity should come from appearance, not a Unicode
codebook.

## Identity Modes

`TextTensorConfig.identity_mode` controls whether codepoint information is added
as an auxiliary signal:

```text
glyph_only
glyph_plus_codepoint_aux
control_codepoint_aux
```

`glyph_only` is the default. It encodes visible natural-language characters by
glyph appearance, not by codepoint. This is the human-facing path.

`glyph_plus_codepoint_aux` appends an explicit auxiliary codepoint channel for
all positions. Use it only when an experiment needs round-trip identity,
confusable/phishing diagnostics, or exact fallback differentiation.

`control_codepoint_aux` appends the auxiliary codepoint channel only where it is
useful as an engineering signal: invisible/control characters, combining marks,
replacement characters, or visible fallback glyphs that would otherwise all
share the same `?` appearance. Known visible glyphs remain glyph-first.

## Control Channels

The control tensor has a stable channel order exposed as
`TEXT_CONTROL_CHANNELS` and `TextControlKind`:

```text
visible
space
tab
newline
page_break
zero_width_space
zero_width_joiner
zero_width_non_joiner
word_joiner
bom
combining_mark
format
control
replacement
```

Invisible controls are not discarded. For example, `a\u200bb` has a hidden
zero-width-space position with `visible_mask=False`, while the control channel
remains active. Newlines update line coordinates; page breaks update page
coordinates.

## Normalization

`TextTensorConfig.normalization` supports:

```text
raw
NFC
NFD
```

Use `raw` when byte/string identity matters, such as adversarial text or exact
runtime vocabulary experiments. Use `NFC` or `NFD` when canonical Unicode
equivalence should be folded intentionally.

## Runtime Vocab Pairing

```python
from arti import RuntimeVocabPulseAdapter, TextTensorConfig, render_text_layout

config = TextTensorConfig(glyph_height=7, glyph_width=5, identity_mode="glyph_only")
vocab = torch.stack([
    render_text_layout(text, config=config).to_pulse_tensor(pulse_count=4).pulse.squeeze(0).flatten()
    for text in ["rr", "r\u200br", "r\nr"]
])

adapter = RuntimeVocabPulseAdapter(
    context_dim=vocab.shape[-1],
    vocab_tensor_dim=vocab.shape[-1],
    hidden_dim=64,
)
```

This keeps the output softmax bound to the current runtime vocab candidate
tensors while preserving zero-width, newline, and page-break distinctions.

## Boundaries

The fallback implementation approximates grapheme ids by attaching combining
marks to the previous base codepoint. It does not perform full UAX #29
segmentation, Unicode line breaking, bidi resolution, font shaping, kerning, or
pagination. Industrial integrations should use those systems upstream and map
their output into this tensor contract.

The text principle is explicit: appearance is the primary input for visible
natural-language characters. Codepoints are metadata by default and can be
enabled only as an auxiliary engineering channel; they should not replace glyph
appearance as the model-facing identity of ordinary visible text.
