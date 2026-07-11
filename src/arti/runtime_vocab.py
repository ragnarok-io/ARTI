"""Runtime vocabulary binding alpha API.

The runtime vocabulary tensor is treated as rigid external input. Learning
happens in the neural readers and scorers after that tensor, not in a fixed
vocabulary-index embedding row.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class LiteralVocabCache:
    """Encoded literal vocabulary view reusable across decoding steps.

    ``keys`` is ``[K, D]`` for a shared vocabulary or ``[B, K, D]`` for
    per-sample vocabulary views. ``mask`` marks valid vocabulary rows when a
    padded batch is used. The cache contains no learned identity table: it is
    derived from the literal tensors supplied for the current call. Reuse a
    detached cache for inference; rebuild it after each optimizer update while
    training the literal encoder.
    """

    keys: Tensor
    mask: Tensor | None = None


class RuntimeVocabEncoder(nn.Module):
    """Encode rigid runtime vocabulary tensors into latent keys.

    ``vocab_tensor`` has shape ``[K, ...]``. The trailing dimensions are
    flattened and processed by a small MLP. This alpha API intentionally keeps a
    single runtime vocabulary view per forward call.
    """

    def __init__(self, vocab_tensor_dim: int, hidden_dim: int, key_dim: int | None = None) -> None:
        super().__init__()
        resolved_key_dim = hidden_dim if key_dim is None else key_dim
        self.vocab_tensor_dim = vocab_tensor_dim
        self.hidden_dim = hidden_dim
        self.key_dim = resolved_key_dim
        self.net = nn.Sequential(
            nn.Linear(vocab_tensor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, resolved_key_dim),
        )

    def forward(self, vocab_tensor: Tensor) -> Tensor:
        flat = flatten_vocab_tensor(vocab_tensor)
        if flat.shape[-1] != self.vocab_tensor_dim:
            raise ValueError(f"vocab tensor last flattened dim must be {self.vocab_tensor_dim}, got {flat.shape[-1]}")
        return self.net(flat)


class LiteralInput(nn.Module):
    """Read input ids from an explicitly supplied literal input vocabulary.

    The input vocabulary is independent of the vocabulary later supplied to
    :class:`LiteralOutputHead`.
    """

    def __init__(self, vocab_tensor_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = RuntimeVocabEncoder(vocab_tensor_dim, hidden_dim, hidden_dim)

    def forward(self, token_ids: Tensor, input_vocab: Tensor) -> Tensor:
        return gather_runtime_vocab(self.encoder(input_vocab), token_ids)


class OutputLexiconContext(nn.Module):
    """Inject a supplied output vocabulary into hidden states before the body.

    This is a lightweight cross-attention residual. It lets the model body
    observe the available output literals once, while keeping final local-slot
    scoring in a thin :class:`LiteralOutputHead`. Use :meth:`prepare` to encode
    a vocabulary once and reuse it during autoregressive decoding.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_tensor_dim: int,
        context_dim: int | None = None,
        *,
        encoder: RuntimeVocabEncoder | None = None,
    ) -> None:
        super().__init__()
        resolved_dim = hidden_dim if context_dim is None else context_dim
        self.hidden_dim = hidden_dim
        self.context_dim = resolved_dim
        self.encoder = encoder or RuntimeVocabEncoder(vocab_tensor_dim, hidden_dim, resolved_dim)
        if self.encoder.key_dim != resolved_dim:
            raise ValueError("encoder key_dim must match context_dim")
        self.norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, resolved_dim, bias=False)
        self.value = nn.Linear(resolved_dim, resolved_dim, bias=False)
        self.output = nn.Linear(resolved_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim + resolved_dim, hidden_dim)
        self.scale = resolved_dim**-0.5

    def prepare(
        self,
        output_vocab: Tensor,
        *,
        mask: Tensor | None = None,
        batched: bool = False,
        detach: bool = False,
    ) -> LiteralVocabCache:
        """Encode a shared ``[K, ...]`` or batched ``[B, K, ...]`` vocabulary."""

        keys = _encode_literal_vocab(self.encoder, output_vocab, batched=batched)
        checked_mask = _validate_literal_mask(mask, keys)
        if detach:
            keys = keys.detach()
            checked_mask = None if checked_mask is None else checked_mask.detach()
        return LiteralVocabCache(keys=keys, mask=checked_mask)

    def forward(
        self,
        hidden: Tensor,
        output_vocab: Tensor | LiteralVocabCache,
        *,
        output_mask: Tensor | None = None,
        batched_vocab: bool = False,
    ) -> Tensor:
        if hidden.ndim not in {2, 3}:
            raise ValueError("hidden must have shape [B, D] or [B, N, D]")
        cache = (
            output_vocab
            if isinstance(output_vocab, LiteralVocabCache)
            else self.prepare(output_vocab, mask=output_mask, batched=batched_vocab)
        )
        if isinstance(output_vocab, LiteralVocabCache) and output_mask is not None:
            raise ValueError("output_mask is already carried by LiteralVocabCache")
        sequence = hidden.unsqueeze(1) if hidden.ndim == 2 else hidden
        normalized = self.norm(sequence)
        keys, mask = _expand_literal_cache(cache, sequence.shape[0])
        query = self.query(normalized)
        scores = torch.einsum("bnd,bkd->bnk", query, keys) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, :], torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        if mask is not None:
            weights = weights * mask[:, None, :].to(weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
        context = torch.einsum("bnk,bkd->bnd", weights, self.value(keys))
        update = self.output(context)
        gate = torch.sigmoid(self.gate(torch.cat([normalized, context], dim=-1)))
        conditioned = sequence + gate * update
        return conditioned[:, 0] if hidden.ndim == 2 else conditioned


class LiteralOutputHead(nn.Module):
    """Thin dynamic softmax head that scores the supplied output literals."""

    def __init__(
        self,
        hidden_dim: int,
        vocab_tensor_dim: int,
        key_dim: int | None = None,
        *,
        encoder: RuntimeVocabEncoder | None = None,
    ) -> None:
        super().__init__()
        resolved_dim = hidden_dim if key_dim is None else key_dim
        self.encoder = encoder or RuntimeVocabEncoder(vocab_tensor_dim, hidden_dim, resolved_dim)
        if self.encoder.key_dim != resolved_dim:
            raise ValueError("encoder key_dim must match key_dim")
        self.hidden_proj = nn.Linear(hidden_dim, resolved_dim, bias=False)
        self.scale = resolved_dim**-0.5

    def prepare(
        self,
        output_vocab: Tensor,
        *,
        mask: Tensor | None = None,
        batched: bool = False,
        detach: bool = False,
    ) -> LiteralVocabCache:
        keys = _encode_literal_vocab(self.encoder, output_vocab, batched=batched)
        checked_mask = _validate_literal_mask(mask, keys)
        if detach:
            keys = keys.detach()
            checked_mask = None if checked_mask is None else checked_mask.detach()
        return LiteralVocabCache(keys=keys, mask=checked_mask)

    def forward(
        self,
        hidden: Tensor,
        output_vocab: Tensor | LiteralVocabCache,
        *,
        output_mask: Tensor | None = None,
        batched_vocab: bool = False,
    ) -> Tensor:
        if hidden.ndim not in {2, 3}:
            raise ValueError("hidden must have shape [B, D] or [B, N, D]")
        cache = (
            output_vocab
            if isinstance(output_vocab, LiteralVocabCache)
            else self.prepare(output_vocab, mask=output_mask, batched=batched_vocab)
        )
        if isinstance(output_vocab, LiteralVocabCache) and output_mask is not None:
            raise ValueError("output_mask is already carried by LiteralVocabCache")
        keys, mask = _expand_literal_cache(cache, hidden.shape[0])
        projected = self.hidden_proj(hidden)
        if hidden.ndim == 2:
            logits = torch.einsum("bd,bkd->bk", projected, keys) * self.scale
        else:
            logits = torch.einsum("bnd,bkd->bnk", projected, keys) * self.scale
        if mask is not None:
            mask_view = mask if logits.ndim == 2 else mask[:, None, :]
            logits = logits.masked_fill(~mask_view, torch.finfo(logits.dtype).min)
        return logits


class LiteralVocabModel(nn.Module):
    """Reference model with independent input and output literal vocabularies."""

    def __init__(self, input_vocab_tensor_dim: int, output_vocab_tensor_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input = LiteralInput(input_vocab_tensor_dim, hidden_dim)
        output_encoder = RuntimeVocabEncoder(output_vocab_tensor_dim, hidden_dim, hidden_dim)
        self.output_context = OutputLexiconContext(
            hidden_dim,
            output_vocab_tensor_dim,
            encoder=output_encoder,
        )
        self.body = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.head = LiteralOutputHead(
            hidden_dim,
            output_vocab_tensor_dim,
            encoder=output_encoder,
        )

    def prepare_output_vocab(
        self,
        output_vocab: Tensor,
        *,
        mask: Tensor | None = None,
        batched: bool = False,
        detach: bool = False,
    ) -> LiteralVocabCache:
        return self.output_context.prepare(output_vocab, mask=mask, batched=batched, detach=detach)

    def forward(
        self,
        token_ids: Tensor,
        input_vocab: Tensor,
        output_vocab: Tensor | LiteralVocabCache,
        *,
        output_mask: Tensor | None = None,
        batched_output_vocab: bool = False,
    ) -> Tensor:
        cache = (
            output_vocab
            if isinstance(output_vocab, LiteralVocabCache)
            else self.prepare_output_vocab(
                output_vocab,
                mask=output_mask,
                batched=batched_output_vocab,
            )
        )
        if isinstance(output_vocab, LiteralVocabCache) and output_mask is not None:
            raise ValueError("output_mask is already carried by LiteralVocabCache")
        hidden = self.input(token_ids, input_vocab)
        hidden = self.output_context(
            hidden,
            cache,
        )
        hidden = self.body(hidden)
        return self.head(
            hidden,
            cache,
        )


class RuntimeVocabInput(nn.Module):
    """Read token representations from a runtime vocabulary view."""

    def __init__(self, vocab_tensor_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = RuntimeVocabEncoder(vocab_tensor_dim, hidden_dim, hidden_dim)

    def forward(self, token_ids: Tensor, vocab_tensor: Tensor) -> Tensor:
        keys = self.encoder(vocab_tensor)
        return gather_runtime_vocab(keys, token_ids)


class RuntimeVocabHead(nn.Module):
    """Dynamic output head that directly reads the current runtime vocabulary."""

    def __init__(self, hidden_dim: int, vocab_tensor_dim: int, key_dim: int | None = None) -> None:
        super().__init__()
        resolved_key_dim = hidden_dim if key_dim is None else key_dim
        self.encoder = RuntimeVocabEncoder(vocab_tensor_dim, hidden_dim, resolved_key_dim)
        self.hidden_proj = nn.Linear(hidden_dim, resolved_key_dim, bias=False)
        self.scale = resolved_key_dim**-0.5

    def forward(self, hidden: Tensor, vocab_tensor: Tensor) -> Tensor:
        if hidden.ndim not in {2, 3}:
            raise ValueError("hidden must have shape [B, D] or [B, N, D]")
        keys = self.encoder(vocab_tensor)
        projected = self.hidden_proj(hidden)
        return torch.einsum("...d,kd->...k", projected, keys) * self.scale


class RuntimeVocabPulseAdapter(nn.Module):
    """Adapter head for pulse-compressed runtime vocabulary candidates.

    The adapter consumes an upstream context tensor, such as a frozen LLM hidden
    state, and a per-sample runtime vocabulary view. The runtime vocabulary is
    already represented as rigid external tensors; when those tensors come from
    token streams, callers should pulse-compress them before this layer.

    Shapes:

    ```text
    context            [B, C]
    runtime_vocab      [B, K, ...] or [K, ...]
    logits             [B, K]
    ```

    The output index is a local slot in the supplied runtime vocabulary view,
    not a permanent model vocabulary row.
    """

    def __init__(self, context_dim: int, vocab_tensor_dim: int, hidden_dim: int, key_dim: int | None = None) -> None:
        super().__init__()
        resolved_key_dim = hidden_dim if key_dim is None else key_dim
        self.context_dim = context_dim
        self.vocab_tensor_dim = vocab_tensor_dim
        self.hidden_dim = hidden_dim
        self.key_dim = resolved_key_dim
        self.query = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, resolved_key_dim))
        self.key = RuntimeVocabEncoder(vocab_tensor_dim, hidden_dim, resolved_key_dim)
        self.scale = resolved_key_dim**-0.5

    def forward(self, context: Tensor, runtime_vocab: Tensor) -> Tensor:
        if context.ndim != 2:
            raise ValueError("context must have shape [B, C]")
        if runtime_vocab.ndim < 2:
            raise ValueError("runtime_vocab must have shape [B, K, ...] or [K, ...]")
        if runtime_vocab.ndim == 2 or runtime_vocab.shape[0] != context.shape[0]:
            keys = self.key(runtime_vocab).unsqueeze(0).expand(context.shape[0], -1, -1)
        else:
            batch, vocab_size = runtime_vocab.shape[:2]
            flat = runtime_vocab.reshape(batch * vocab_size, *runtime_vocab.shape[2:])
            keys = self.key(flat).reshape(batch, vocab_size, self.key_dim)
        query = self.query(context)
        return torch.einsum("bd,bkd->bk", query, keys) * self.scale


class RuntimeVocabModel(nn.Module):
    """Minimal reference model with runtime vocab input and output binding."""

    def __init__(self, vocab_tensor_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input = RuntimeVocabInput(vocab_tensor_dim, hidden_dim)
        self.body = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.head = RuntimeVocabHead(hidden_dim, vocab_tensor_dim)

    def forward(self, token_ids: Tensor, vocab_tensor: Tensor) -> Tensor:
        x = self.input(token_ids, vocab_tensor)
        hidden = self.body(x)
        return self.head(hidden, vocab_tensor)


def flatten_vocab_tensor(vocab_tensor: Tensor) -> Tensor:
    """Flatten runtime vocabulary item tensors while preserving the vocab axis."""

    if vocab_tensor.ndim < 2:
        raise ValueError("vocab_tensor must have shape [K, ...]")
    return vocab_tensor.flatten(start_dim=1)


def gather_runtime_vocab(vocab_keys: Tensor, token_ids: Tensor) -> Tensor:
    """Gather runtime vocab keys for token ids interpreted in the current view."""

    if token_ids.ndim not in {1, 2}:
        raise ValueError("token_ids must have shape [N] or [B, N]")
    if vocab_keys.ndim == 2:
        return vocab_keys.index_select(0, token_ids.reshape(-1).to(torch.long)).reshape(*token_ids.shape, vocab_keys.shape[-1])
    raise ValueError("vocab_keys must have shape [K, D]")


def _encode_literal_vocab(encoder: RuntimeVocabEncoder, vocab: Tensor, *, batched: bool) -> Tensor:
    if not batched:
        return encoder(vocab)
    if vocab.ndim < 3:
        raise ValueError("batched output vocab must have shape [B, K, ...]")
    batch, size = vocab.shape[:2]
    flat_items = vocab.reshape(batch * size, *vocab.shape[2:])
    return encoder(flat_items).reshape(batch, size, encoder.key_dim)


def _validate_literal_mask(mask: Tensor | None, keys: Tensor) -> Tensor | None:
    if mask is None:
        return None
    checked = mask.to(device=keys.device, dtype=torch.bool)
    expected = keys.shape[:-1]
    if tuple(checked.shape) != tuple(expected):
        raise ValueError(f"output mask must have shape {tuple(expected)}, got {tuple(checked.shape)}")
    if not bool(checked.any(dim=-1).all()):
        raise ValueError("every output vocabulary view must contain at least one valid item")
    return checked


def _expand_literal_cache(cache: LiteralVocabCache, batch: int) -> tuple[Tensor, Tensor | None]:
    keys = cache.keys
    mask = cache.mask
    if keys.ndim == 2:
        keys = keys.unsqueeze(0).expand(batch, -1, -1)
        if mask is not None:
            mask = mask.unsqueeze(0).expand(batch, -1)
    elif keys.ndim == 3:
        if keys.shape[0] != batch:
            raise ValueError(f"batched output vocabulary has batch {keys.shape[0]}, expected {batch}")
    else:
        raise ValueError("cached literal keys must have shape [K, D] or [B, K, D]")
    return keys, mask


def permute_runtime_vocab(vocab_tensor: Tensor, permutation: Tensor) -> Tensor:
    """Return a runtime vocab view permuted along the vocab dimension."""

    if vocab_tensor.ndim < 2:
        raise ValueError("vocab_tensor must have shape [K, ...]")
    return vocab_tensor.index_select(0, permutation.to(device=vocab_tensor.device, dtype=torch.long))


def remap_token_ids(token_ids: Tensor, permutation: Tensor) -> Tensor:
    """Map old runtime ids into ids under ``permute_runtime_vocab``."""

    inverse = torch.empty_like(permutation)
    inverse[permutation.to(torch.long)] = torch.arange(permutation.numel(), device=permutation.device)
    return inverse.to(device=token_ids.device).index_select(0, token_ids.reshape(-1).to(torch.long)).reshape_as(token_ids)


def attach_runtime_vocab_semantics(
    vocab_tensor: Tensor,
    semantic_tensor: Tensor,
    *,
    vocab_scale: float = 1.0,
    semantic_scale: float = 1.0,
) -> Tensor:
    """Append per-item semantic anchors to rigid runtime vocabulary tensors.

    ``vocab_tensor`` is flattened per vocab item and remains the identity
    channel. ``semantic_tensor`` must share the same leading vocab axes and
    supplies an optional learned or external semantic anchor. This helper keeps
    the contract tensor-native: it does not know what the semantics mean, only
    how to combine aligned runtime-vocab fields.

    Examples:

    ```text
    vocab_tensor      [K, ...]
    semantic_tensor   [K, S]
    output            [K, flat(vocab_tensor item) + S]

    vocab_tensor      [B, K, ...]
    semantic_tensor   [B, K, S]
    output            [B, K, flat(vocab_tensor item) + S]
    ```
    """

    if vocab_tensor.ndim < 2:
        raise ValueError("vocab_tensor must have shape [K, ...] or [B, K, ...]")
    if semantic_tensor.ndim < 2:
        raise ValueError("semantic_tensor must have shape [K, S] or [B, K, S]")
    if vocab_tensor.shape[: semantic_tensor.ndim - 1] != semantic_tensor.shape[:-1]:
        raise ValueError(
            "semantic_tensor leading axes must match vocab_tensor item axes: "
            f"got vocab {tuple(vocab_tensor.shape)} and semantic {tuple(semantic_tensor.shape)}"
        )
    flat_vocab = vocab_tensor.flatten(start_dim=semantic_tensor.ndim - 1)
    flat_semantic = semantic_tensor.flatten(start_dim=semantic_tensor.ndim - 1)
    return torch.cat([flat_vocab * vocab_scale, flat_semantic * semantic_scale], dim=-1)
