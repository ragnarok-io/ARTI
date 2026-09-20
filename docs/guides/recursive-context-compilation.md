# Recursive Context Compilation

RCC recursively reuses an internal pseudo-token sequence `C` without exposing it as a
normal downstream activation or as a control condition.

For each recursive node, already compiled child contexts and local lexical embeddings
enter the same model input head. Placement is bound by the consuming node, so a saved
`C` has no intrinsic absolute position. The network then produces two logically distinct
results:

- `Y`: the ordinary downstream hidden state used by the language head.
- `C`: a bounded, variable-width sequence of native pseudo-token states updated inside
  the shared backbone. Later RCC nodes consume it only at their input head.

```python
from arti.alpha import RecursiveContextCompiler

compiler = RecursiveContextCompiler(
    dim=256,
    context_width=32,
    heads=8,
    layers=6,
    readout_layers=(2, 4, 6),
)
```

For an existing `nn.Module`, `RCCHostAdapter` can compile C from any named host
layers and consume a **previously compiled** C at an independent layer. The
callback is the host integration point for its native position encoding and
attention masks:

```python
from arti.alpha import RCCHostAdapter

def bind_context(context, args, kwargs):
    # Host-specific: position C with the model's own position mechanism,
    # update masks/position_ids as needed, and return the consumer's inputs.
    return host.bind_rcc_context(context, args, kwargs)

adapter = RCCHostAdapter(
    host,
    compiler,
    capture_paths=("blocks.8", "blocks.16", "blocks.23"),
    consume_path="input_head",  # or a named intermediate module
    bind_context=bind_context,
)
first = adapter(input_tensor)  # first.output is the ordinary host result
next_run = adapter(next_input_tensor, context=first.context)
adapter.close()  # remove hooks when this integration is no longer used
```

The adapter captures outputs during one host forward, then compiles C while
preserving autograd. A C passed to `context=` is consumed during that forward;
the newly compiled C is returned for a later call. Capture and consumption
paths have no ordering constraint. Captured values must be `[B,L,dim]` for the
compiler. `capture_inputs` can select a tensor from a structured layer output,
or return `ContextInput(value, valid, placement)` to preserve the host's mask.
The compiler is any `nn.Module` whose `forward(captures)` returns C as `[B,M,D]`;
the captures arrive in `capture_paths` order. It owns projections, masks,
compilation depth, and output width. The built-in `RecursiveContextCompiler`
accepts plain `[B,L,D]` tensors as well as `ContextInput` values. A custom
compiler can consume different feature dimensions from different host layers.
The callback can prepend pseudo-tokens at the input head or place them at an
intermediate layer. It must use the adapted model's own position and mask rules;
the adapter does not assign positions to C. If the host computes masks or
positions before the intermediate consumer, `prepare_host(context, args, kwargs)`
can adjust root-call metadata before the host forward. This only works when the
host exposes a way to pass all affected metadata. A generic module hook cannot
repair position or cache state held in private host-forward locals; such a host
needs an explicit integration seam. Hooks are installed for the adapter lifetime,
with capture state isolated per invocation, and removed by `close()`.

`adapter.save(directory, integration_ref="my-model/rcc@1")` saves weights and the
capture/consumer declaration in SafeTensors plus JSON. `RCCHostAdapter.load(...)`
requires caller-constructed host and compiler modules, the same `integration_ref`,
and the corresponding binding/selection callbacks. Python callback code is not
serialized. Trainable callbacks or capture selectors must be `nn.Module` instances
to be included in the adapter state dict. The host and compiler architecture must
match the saved weights.
The complete runnable example is [RCC host adapter](https://github.com/Thiocy/ARTI/blob/main/examples/rcc_host_adapter.py).

Placement is a consumer-bound relation coordinate, not an absolute position stored
inside `C`. The default encoder is an explicit learned linear map with no hidden
frequency scale. Replace it when the host has a different relation geometry:

```python
from arti.alpha import FourierContextPlacementEncoder

compiler = RecursiveContextCompiler(
    dim=256,
    context_width=32,
    heads=8,
    layers=6,
    relation_dim=2,
    placement_encoder=FourierContextPlacementEncoder(
        relation_dim=2,
        dim=256,
        bands=8,
        base=10000.0,
    ),
)
```

Custom encoders subclass `ContextPlacementEncoder` and are registered as normal
ARTI components. Their reference and configuration are saved with the compiler;
there is no implicit positional encoding choice hidden in the serialized model.

Lexical token positions are a separate input-head contract. `LexicalEmbedding`
does not choose a trunk position convention implicitly; the host must provide a
registered `TokenPositionEncoder`. The standard sinusoidal reference can be
selected explicitly, or the host can provide `LearnedTokenPositionEncoder` or
another registered encoder without changing the late-bound placement of `C`:

```python
from arti.alpha import LearnedTokenPositionEncoder, LexicalEmbedding

lexical = LexicalEmbedding(
    vocab_size=32000,
    dim=256,
    position_encoder=LearnedTokenPositionEncoder(dim=256, max_length=4096),
)
```

The lexical encoder and the `C` placement encoder are serialized independently.
This keeps the model's main token-position convention explicit instead of hiding
it in the compiler implementation.

`arti/recursive-context-compiler@1` is the compiler source declaration and
`arti/recursive-language@1` is the language-model artifact schema label. A
component graph records the compiler as its resolved full contract address;
the schema label remains separate from component identity.

`context_width` is the maximum C capacity. A call can choose `context_width=m`
with `1 <= m <= max`. The C positions participate in each backbone layer; selected
internal layers contribute to their final readout. This is not a learned-query
pooling pass over a completed token sequence. C is placed as a prefix of the shared
trunk, before the lexical source positions, so a causal host can expose it to every
later lexical position. The downstream Y readout still selects only the lexical
source side; C is not emitted as an ordinary language-head state.

For one source chunk, `compile(channels, depth=k, context_width=m)` computes
`x -> C1 -> ... -> Ck`. Rounds after the first receive only the latest C at the
input head; they do not reread the source or retain earlier hidden states. This
same-level compilation depth `k` is separate from `prefix_plan(depth=...)`, which
controls the number of parent/child abstraction levels. A compiled C has local
sequence order but no intrinsic parent position; each consumer binds placement.

The reference evaluator builds a validated prefix DAG. Internal nodes request `C`; roots
request `Y`. Training uses ordinary next-token cross entropy through the language head,
and gradients flow through recursive C consumption and the native C positions.
The implementation establishes the structural ABI, not semantic equivalence to a
full raw prefix. That requires trained source-replacement comparisons and held-out
language evaluation; no speedup is claimed from this reference evaluator.
