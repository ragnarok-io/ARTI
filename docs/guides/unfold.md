# UnFold

`UnFold` is a value-preserving, layout-changing tensor expansion. It queries
new values from the input, combines them with every original element, and
learns a sample-conditioned output layout.

```python
import torch
from arti.nn import UnFold

x = torch.randn(4, 16, 64)
layer = UnFold(dim=64, exposed=8)
y = layer(x)

assert y.shape == (4, 24, 64)
```

`exposed` is the maximum trainable expansion capacity, not a required output
increment for every call. Use `target_length` when one shared UnFold must serve
workspaces of different sizes:

```python
layer = UnFold(dim=64, exposed=32)

single = layer(torch.randn(4, 8, 64), target_length=12)
combined = layer(torch.randn(4, 32, 64), target_length=48)

assert single.shape == (4, 12, 64)
assert combined.shape == (4, 48, 64)
```

The active exposed count is `target_length - N` and must not exceed the
configured capacity. Only that prefix of independent query and modulation
parameters participates in the call. Omitting `target_length` preserves the
default `[B, N + exposed, D]` contract.

The original values are transported with a hard gather. Every input instance
therefore occurs exactly once in the output, without averaging, interpolation,
or projection. Their positions and adjacency are not preserved.

```python
y, exposed_mask, source_index = layer(
    x,
    return_exposed_mask=True,
    return_source_index=True,
)
```

`source_index < N` identifies an original input instance. Values at
`source_index >= N` are queried exposed values. The mapping is returned per
sample because different inputs may produce different layouts.

Exposed values use a fixed-size bank of shared feature operators. Each exposed
query learns independent operator mixing coefficients and independent
per-feature scale/bias values. `value_operators` controls the bank size and
defaults to 8. Value-generation parameters therefore grow approximately as
`value_operators * D^2 + exposed * (value_operators + 2D)`, rather than
allocating a separate `D x D` matrix for every exposed slot.

`value_rank` optionally factorizes each shared operator through a narrower
feature rank. It is an explicit quality/performance trade-off and is disabled
by default:

```python
layer = UnFold(dim=256, exposed=512, value_rank=64)
```

Aggressive ranks can reduce semantic expansion quality. Choose them through a
task-level validation rather than deriving them from tensor shape alone.

For memory-constrained execution, query and operator work can be chunked
without changing the output contract:

```python
layer = UnFold(
    dim=256,
    exposed=512,
    query_chunk_size=128,
    operator_chunk_size=4,
)
```

Chunking trades additional kernel launches for smaller temporary tensors. It
is disabled by default for eager latency. `torch.compile(mode="reduce-overhead")`
can substantially reduce the launch overhead for fixed deployment shapes.

For CUDA inference, ARTI can automatically use an internal fused Triton path
for the measured compact-workspace region. It computes query attention without
materializing `[B, exposed, N]` weights and applies the complete operator bank
without materializing `[B, exposed, operators, D]`. The optimization does not
change parameters or the state dict. Training, gradients, `torch.compile`,
factorized operators, unsupported dtypes and unmeasured shapes automatically
fall back to the ordinary PyTorch path.

The current automatic gate requires CUDA compute capability 8.0 or newer,
FP32 or BF16, `128 <= D <= 256`, at least 128 exposed slots, no query chunking,
and at most 128 compact input slots. FP32 is additionally limited to 64 input
slots because the larger tile exceeds the measured shared-memory budget. These
are implementation boundaries, not tensor-contract restrictions.

During training, the forward result remains the exact hard layout. A soft
layout supplies an approximate gradient to the layout scorer. The default
`hard_backend="sort"` learns one sample-conditioned rank per candidate and
uses CUDA `argsort` plus `gather` at inference, avoiding a dense layout matrix.
It can be
inspected with `return_soft_layout=True`; it is diagnostic and is not the
returned tensor value. `max_length=128` guards the dense `greedy` and
`auction` research backends; the default sort backend is not bound by that
limit. `hard_backend="auction"` is available as
a small-layout correctness experiment, but the pure PyTorch implementation is
not intended for throughput-sensitive inference.

A boolean input mask is transported by the same hard layout. Exposed values
are valid when the sample contains at least one valid input, and invalid
positions are sorted to the end.

## Optional layout inputs

`guide` separates the values being transported from the tensor used to choose
their layout. It is optional, anonymous, and aligned with the input candidates:

```python
layer = UnFold(dim=64, exposed=8, guide_dim=6, condition_dim=4)

x = torch.randn(4, 16, 64)
guide = torch.randn(4, 16, 6)
condition = torch.randn(4, 4)
y = layer(x, guide=guide, condition=condition)
```

Candidate-varying guide channels are normalized over valid candidates using
FP32 statistics. Padding does not participate. `condition` is sample-level and
is not normalized over the candidate axis. Either input may be omitted even
when its dimension is configured; omitting it supplies a deterministic zero
feature. A module constructed without `guide_dim` or `condition_dim` rejects
the corresponding input and creates no parameters for it.

The guide affects layout scores only. The final output is still produced by a
hard gather from the original candidate values and queried exposed values.
Finite, non-degenerate positive affine changes to candidate guide channels are
therefore expected to preserve their normalized representation within numeric
tolerance. `UnFold` does not promise invariance to arbitrary monotonic
transforms or exact ordering of nearly tied guide values.

The guide must be available at inference time and must not contain a target
permutation, future label, or other information unavailable to the deployed
model. ARTI treats it as an anonymous tensor and cannot determine its external
provenance.

Like standard PyTorch layers, `UnFold` always validates tensor rank, shape,
dtype, and device, but does not scan accelerator values for NaN or infinity on
every forward. Such a scan synchronizes CUDA and is unsuitable for a hot path.
Use `validate_values=True` while debugging untrusted tensor pipelines:

```python
layer = UnFold(dim=64, exposed=8, guide_dim=1, validate_values=True)
```

### Canonical guide layout

Use `layout_mode="canonical"` when a one-dimensional guide is the required
ordering coordinate:

```python
layer = UnFold(
    dim=64,
    exposed=8,
    guide_dim=1,
    layout_mode="canonical",
)
y = layer(x, guide=guide)
```

Canonical layout sorts original candidates directly by their normalized guide.
The learned layout scorer cannot reverse their relative order. Queried exposed
candidates generate their own coordinates from the attended input, query
identity, and optional condition.

If the exposed coordinates are known, provide them explicitly for a completely
specified layout:

```python
exposed_guide = torch.randn(4, 8, 1)
y = layer(x, guide=guide, exposed_guide=exposed_guide)
```

`exposed_guide` overrides internally queried coordinates. It is accepted only
by canonical layout. Canonical layout intentionally requires `guide_dim=1`:
a general multidimensional tensor has no intrinsic total order.

### Dynamic valid expansion

`exposed_mask` controls which active exposed candidates are valid for each
sample. Its final dimension is `target_length - N`, or `exposed` when no target
length is supplied:

```python
requested = torch.rand(4, 8) > 0.5
y, output_mask = layer(
    x,
    mask,
    guide=guide,
    exposed_mask=requested,
)
```

The output shape is fixed for a particular call and can therefore remain
batch-friendly while varying between calls. Disabled exposed candidates are
marked invalid and sorted after valid candidates. They are not given separate
parameters per batch sample.

`UnFold` is unrelated to `torch.nn.Unfold`, which extracts image patches.
