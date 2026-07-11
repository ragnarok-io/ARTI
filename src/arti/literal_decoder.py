"""Character- or fragment-level decoding over an external literal vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Mapping

import torch
import torch.nn as nn
from torch import Tensor

from .runtime_vocab import LiteralOutputHead, LiteralVocabCache, OutputLexiconContext, RuntimeVocabEncoder

if TYPE_CHECKING:
    from .literal_fit import LiteralFitResult


@dataclass(frozen=True)
class LiteralSequenceOutput:
    """Dynamic-vocabulary sequence decoder output.

    ``logits`` is ``[B, T, K]`` and ``local_ids`` is ``[B, T]``. Local ids are
    meaningful only under the output vocabulary supplied to the decoder.
    ``lengths`` is populated by :meth:`LiteralSequenceDecoder.generate`.
    """

    logits: Tensor
    local_ids: Tensor
    lengths: Tensor | None = None


class _ResidualContextAdapter(nn.Module):
    def __init__(self, dim: int, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, width), nn.GELU(), nn.Linear(width, dim))
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_norm(x + self.net(self.norm(x)))


class LiteralSequenceDecoder(nn.Module):
    """Decode a sequence of local ids from a supplied literal output vocab.

    The decoder is useful when the upstream model and output vocabulary use
    different segmentation. For example, a frozen language model may provide a
    BPE-level context state while this module emits character glyph literals.

    ``LiteralOutputHead`` remains the final scorer. By default,
    ``OutputLexiconContext`` exposes the current output range before the
    recurrent decoder body. Set ``condition_on_vocab=False`` for a terminal-head
    ablation that still reads every literal in the final head.
    """

    def __init__(
        self,
        context_dim: int,
        vocab_tensor_dim: int,
        *,
        hidden_dim: int = 256,
        key_dim: int = 128,
        encoder_hidden_dim: int | None = None,
        adapter_width: int = 128,
        condition_on_vocab: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or key_dim <= 0 or adapter_width <= 0:
            raise ValueError("hidden_dim, key_dim, and adapter_width must be positive")
        encoder_width = hidden_dim if encoder_hidden_dim is None else encoder_hidden_dim
        self.context_dim = context_dim
        self.vocab_tensor_dim = vocab_tensor_dim
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.encoder_hidden_dim = encoder_width
        self.adapter_width = adapter_width
        self.condition_on_vocab = condition_on_vocab
        self.encoder = RuntimeVocabEncoder(vocab_tensor_dim, encoder_width, key_dim)
        self.lexicon_context = (
            OutputLexiconContext(context_dim, vocab_tensor_dim, key_dim, encoder=self.encoder)
            if condition_on_vocab
            else None
        )
        self.context_adapter = _ResidualContextAdapter(context_dim, adapter_width)
        self.initial_state = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.Tanh())
        self.step_context = nn.Sequential(nn.Linear(context_dim, key_dim), nn.Tanh())
        self.recurrent = nn.GRUCell(key_dim * 2, hidden_dim)
        self.head = LiteralOutputHead(hidden_dim, vocab_tensor_dim, key_dim, encoder=self.encoder)
        self.bos = nn.Parameter(torch.zeros(key_dim))

    def serialization_config(self) -> dict[str, int | bool]:
        """Return constructor fields stored in an ``arti.st`` manifest."""

        return {
            "context_dim": self.context_dim,
            "vocab_tensor_dim": self.vocab_tensor_dim,
            "hidden_dim": self.hidden_dim,
            "key_dim": self.key_dim,
            "encoder_hidden_dim": self.encoder_hidden_dim,
            "adapter_width": self.adapter_width,
            "condition_on_vocab": self.condition_on_vocab,
        }

    def prepare_output_vocab(
        self,
        output_vocab: Tensor,
        *,
        mask: Tensor | None = None,
        batched: bool = False,
        detach: bool = False,
    ) -> LiteralVocabCache:
        """Encode an output vocabulary once for the context, loop, and head."""

        return self.head.prepare(output_vocab, mask=mask, batched=batched, detach=detach)

    def fit(
        self,
        batches: Iterable[Mapping[str, object]],
        *,
        steps: int,
        lr: float = 1e-3,
        optimizer: torch.optim.Optimizer | None = None,
        grad_clip_norm: float | None = 1.0,
    ) -> LiteralFitResult:
        """Run ARTI's small tensor-native literal decoder fit recipe."""

        from .literal_fit import fit_literal_sequence

        return fit_literal_sequence(
            self,
            batches,
            steps=steps,
            lr=lr,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
        )

    def forward(
        self,
        context: Tensor,
        output_vocab: Tensor | LiteralVocabCache,
        *,
        teacher_ids: Tensor | None = None,
        steps: int | None = None,
        output_mask: Tensor | None = None,
        batched_vocab: bool = False,
    ) -> LiteralSequenceOutput:
        """Run teacher-forced training or fixed-step greedy decoding.

        Exactly one sequence length source is required: ``teacher_ids`` with
        shape ``[B, T]`` or a positive ``steps`` value. Teacher ids are shifted
        internally; the first prediction always consumes a learned BOS vector.
        """

        if context.ndim != 2:
            raise ValueError("context must have shape [B, C]")
        if teacher_ids is None and (steps is None or steps <= 0):
            raise ValueError("provide teacher_ids or a positive steps value")
        if teacher_ids is not None:
            if teacher_ids.ndim != 2 or teacher_ids.shape[0] != context.shape[0]:
                raise ValueError("teacher_ids must have shape [B, T]")
            if steps is not None and steps != teacher_ids.shape[1]:
                raise ValueError("steps must match teacher_ids sequence length")
            resolved_steps = teacher_ids.shape[1]
        else:
            resolved_steps = int(steps or 0)
        cache = self._resolve_cache(output_vocab, output_mask=output_mask, batched_vocab=batched_vocab)
        conditioned, state = self._condition(context, cache)
        step_context = self.step_context(conditioned)
        keys = _batched_keys(cache, context.shape[0])
        decoder_input = self.bos.unsqueeze(0).expand(context.shape[0], -1)
        logits_rows = []
        id_rows = []
        for step in range(resolved_steps):
            state = self.recurrent(torch.cat([decoder_input, step_context], dim=-1), state)
            logits = self.head(state, cache)
            predicted = logits.argmax(dim=-1)
            logits_rows.append(logits)
            id_rows.append(predicted)
            next_ids = teacher_ids[:, step] if teacher_ids is not None else predicted
            decoder_input = _gather_keys(keys, next_ids)
        return LiteralSequenceOutput(logits=torch.stack(logits_rows, dim=1), local_ids=torch.stack(id_rows, dim=1))

    def generate(
        self,
        context: Tensor,
        output_vocab: Tensor | LiteralVocabCache,
        *,
        eos_local_ids: Tensor,
        max_steps: int,
        output_mask: Tensor | None = None,
        batched_vocab: bool = False,
    ) -> LiteralSequenceOutput:
        """Greedily decode until each item emits its current local EOS id."""

        if context.ndim != 2:
            raise ValueError("context must have shape [B, C]")
        if eos_local_ids.shape != (context.shape[0],):
            raise ValueError("eos_local_ids must have shape [B]")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        cache = self._resolve_cache(output_vocab, output_mask=output_mask, batched_vocab=batched_vocab)
        conditioned, state = self._condition(context, cache)
        step_context = self.step_context(conditioned)
        keys = _batched_keys(cache, context.shape[0])
        decoder_input = self.bos.unsqueeze(0).expand(context.shape[0], -1)
        finished = torch.zeros(context.shape[0], device=context.device, dtype=torch.bool)
        lengths = torch.full((context.shape[0],), max_steps, device=context.device, dtype=torch.long)
        logits_rows = []
        id_rows = []
        eos = eos_local_ids.to(device=context.device, dtype=torch.long)
        for step in range(max_steps):
            state = self.recurrent(torch.cat([decoder_input, step_context], dim=-1), state)
            logits = self.head(state, cache)
            predicted = logits.argmax(dim=-1)
            predicted = torch.where(finished, eos, predicted)
            logits_rows.append(logits)
            id_rows.append(predicted)
            just_finished = ~finished & (predicted == eos)
            lengths = torch.where(just_finished, torch.full_like(lengths, step + 1), lengths)
            finished = finished | just_finished
            decoder_input = _gather_keys(keys, predicted)
            if bool(finished.all()):
                break
        return LiteralSequenceOutput(
            logits=torch.stack(logits_rows, dim=1),
            local_ids=torch.stack(id_rows, dim=1),
            lengths=lengths,
        )

    def _resolve_cache(
        self,
        output_vocab: Tensor | LiteralVocabCache,
        *,
        output_mask: Tensor | None,
        batched_vocab: bool,
    ) -> LiteralVocabCache:
        if isinstance(output_vocab, LiteralVocabCache):
            if output_mask is not None:
                raise ValueError("output_mask is already carried by LiteralVocabCache")
            return output_vocab
        return self.prepare_output_vocab(output_vocab, mask=output_mask, batched=batched_vocab)

    def _condition(self, context: Tensor, cache: LiteralVocabCache) -> tuple[Tensor, Tensor]:
        conditioned = context
        if self.lexicon_context is not None:
            conditioned = self.lexicon_context(conditioned, cache)
        conditioned = self.context_adapter(conditioned)
        return conditioned, self.initial_state(conditioned)


def _batched_keys(cache: LiteralVocabCache, batch: int) -> Tensor:
    if cache.keys.ndim == 2:
        return cache.keys.unsqueeze(0).expand(batch, -1, -1)
    if cache.keys.ndim == 3 and cache.keys.shape[0] == batch:
        return cache.keys
    raise ValueError("literal cache keys must have shape [K, D] or matching [B, K, D]")


def _gather_keys(keys: Tensor, local_ids: Tensor) -> Tensor:
    if local_ids.ndim != 1 or local_ids.shape[0] != keys.shape[0]:
        raise ValueError("local ids must have shape [B]")
    if bool(((local_ids < 0) | (local_ids >= keys.shape[1])).any()):
        raise ValueError("local ids must index the current output vocabulary")
    return keys.gather(1, local_ids[:, None, None].expand(-1, 1, keys.shape[-1])).squeeze(1)


__all__ = ["LiteralSequenceDecoder", "LiteralSequenceOutput"]
