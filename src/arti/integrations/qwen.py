"""Qwen integration helpers for ARTI runtime glyph vocab experiments.

The adapter deliberately keeps two paths separate:

* normal dialogue uses the original frozen Qwen ``generate`` and logits path;
* glyph runtime vocab reads use ARTI bitmap tensors and a separate readout head.

This is an alpha developer interface for controlled adaptation. It is not a full
tokenizer replacement and it does not mutate Qwen base weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..runtime_vocab import RuntimeVocabPulseAdapter
from ..text_bitmap import assert_bitmap_vocab_distinct, render_text_bitmap, render_text_vocab


@dataclass(frozen=True)
class QwenGlyphRuntimeConfig:
    """Configuration for ``QwenGlyphRuntimeAdapter``."""

    model_id: str = "Qwen/Qwen3-0.6B"
    height: int = 14
    width: int = 96
    hidden_dim: int = 192
    device: str | None = None
    font_path: str | Path | None = None
    raw_logit_scale: float = 10.0
    check_vocab_distinct: bool = True


@dataclass(frozen=True)
class GlyphRuntimeReadout:
    """Local-slot logits for an external glyph runtime vocabulary."""

    logits: Tensor
    probabilities: Tensor
    local_index: int
    text: str | None
    diagnostics: dict[str, Tensor | float | int | str]


@dataclass(frozen=True)
class QwenDialogueDrift:
    """Difference between two ordinary Qwen dialogue-logit snapshots."""

    max_abs_logit_delta: float
    mean_abs_logit_delta: float
    mean_kl_divergence: float
    top1_preserved: bool
    prompt_count: int


class QwenGlyphRuntimeAdapter(nn.Module):
    """Frozen Qwen dialogue path plus an independent ARTI glyph vocab readout.

    Parameters can be constructed with ``from_pretrained`` for a real Qwen model
    or with injected ``model``/``tokenizer`` objects for tests. The Qwen model is
    placed in eval mode and all base parameters are frozen.
    """

    def __init__(
        self,
        model: object,
        tokenizer: object,
        *,
        config: QwenGlyphRuntimeConfig | None = None,
        qwen_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.config = QwenGlyphRuntimeConfig() if config is None else config
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(self.config.device or _infer_model_device(model))
        if hasattr(self.model, "eval"):
            self.model.eval()
        if hasattr(self.model, "requires_grad_"):
            self.model.requires_grad_(False)
        self.qwen_hidden_dim = int(qwen_hidden_dim or _infer_hidden_dim(model))
        glyph_dim = self.config.height * self.config.width
        self.query_context = nn.Sequential(
            nn.Linear(self.qwen_hidden_dim + glyph_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.runtime_head = RuntimeVocabPulseAdapter(
            context_dim=self.config.hidden_dim,
            vocab_tensor_dim=glyph_dim,
            hidden_dim=self.config.hidden_dim,
        )
        self.raw_logit_scale = nn.Parameter(torch.tensor(float(self.config.raw_logit_scale)))
        self.to(self.device)

    @classmethod
    def from_pretrained(cls, model_id: str = "Qwen/Qwen3-0.6B", **kwargs: object) -> "QwenGlyphRuntimeAdapter":
        """Load a frozen Qwen model through Transformers.

        Requires the optional ``qwen`` extra. Keyword arguments are split between
        adapter config fields and ``AutoModelForCausalLM.from_pretrained``.
        """

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError("QwenGlyphRuntimeAdapter requires Transformers. Install with `uv sync --extra qwen`.") from exc

        config_fields = set(QwenGlyphRuntimeConfig.__dataclass_fields__)
        config_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in config_fields}
        config = QwenGlyphRuntimeConfig(model_id=model_id, **config_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_kwargs = {"torch_dtype": "auto", "device_map": "auto", **kwargs}
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        return cls(model, tokenizer, config=config)

    def generate(self, prompt: str, **kwargs: object) -> str:
        """Generate text with the untouched Qwen dialogue path."""

        encoded = self._encode([prompt])
        with torch.no_grad():
            output_ids = self.model.generate(**encoded, **kwargs)
        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return str(decoded[0])

    def dialogue_logits(self, prompts: str | Sequence[str]) -> Tensor:
        """Return original Qwen next-token logits for ordinary dialogue prompts."""

        prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
        encoded = self._encode(prompt_list)
        with torch.no_grad():
            out = self.model(**encoded, use_cache=False)
        lengths = encoded["attention_mask"].sum(dim=1) - 1
        logits = out.logits[torch.arange(len(prompt_list), device=lengths.device), lengths]
        return logits.detach().to(torch.float32)

    def dialogue_drift(self, prompts: Sequence[str], operation: Callable[[], object] | None = None) -> QwenDialogueDrift:
        """Measure whether an operation changes the original dialogue logits."""

        before = self.dialogue_logits(prompts).detach().cpu()
        if operation is not None:
            operation()
        after = self.dialogue_logits(prompts).detach().cpu()
        diff = (after - before).abs()
        before_logp = F.log_softmax(before, dim=-1)
        after_logp = F.log_softmax(after, dim=-1)
        kl = (before_logp.exp() * (before_logp - after_logp)).sum(dim=-1)
        return QwenDialogueDrift(
            max_abs_logit_delta=float(diff.max().item()),
            mean_abs_logit_delta=float(diff.mean().item()),
            mean_kl_divergence=float(kl.mean().item()),
            top1_preserved=bool(torch.equal(before.argmax(dim=-1), after.argmax(dim=-1))),
            prompt_count=len(prompts),
        )

    def render_glyph_vocab(self, texts: Sequence[str]) -> Tensor:
        """Render external visible words into rigid bitmap vocab tensors."""

        vocab = render_text_vocab(texts, height=self.config.height, width=self.config.width, font_path=self.config.font_path)
        if self.config.check_vocab_distinct:
            assert_bitmap_vocab_distinct(vocab)
        return vocab.to(self.device)

    def read_glyph_vocab(
        self,
        prompt: str,
        vocab: Sequence[str] | Tensor,
        *,
        query_text: str | None = None,
        query_tensor: Tensor | None = None,
    ) -> GlyphRuntimeReadout:
        """Score the local slots of an external glyph runtime vocabulary.

        ``query_text`` or ``query_tensor`` supplies the currently observed glyph.
        When no query is supplied, a zero query is used; this is useful for API
        smoke tests but trained adapters should provide an observed glyph tensor.
        """

        texts = None if isinstance(vocab, Tensor) else list(vocab)
        vocab_tensor = vocab.to(self.device) if isinstance(vocab, Tensor) else self.render_glyph_vocab(texts)
        flat_vocab = vocab_tensor.flatten(start_dim=1)
        query = self._query_tensor(query_text=query_text, query_tensor=query_tensor, like=flat_vocab)
        context = self.context_hidden([prompt])
        hidden = self.query_context(torch.cat([context, query], dim=-1))
        learned_logits = self.runtime_head(hidden, flat_vocab.unsqueeze(0)).squeeze(0)
        raw_logits = F.cosine_similarity(query, flat_vocab, dim=-1) * self.raw_logit_scale.clamp(min=0.0, max=50.0)
        logits = learned_logits + raw_logits
        probabilities = F.softmax(logits, dim=-1)
        local_index = int(logits.argmax(dim=-1).item())
        return GlyphRuntimeReadout(
            logits=logits,
            probabilities=probabilities,
            local_index=local_index,
            text=None if texts is None else texts[local_index],
            diagnostics={
                "vocab_size": int(flat_vocab.shape[0]),
                "glyph_dim": int(flat_vocab.shape[-1]),
                "raw_logit_scale": float(self.raw_logit_scale.detach().cpu().item()),
                "qwen_hidden_dim": self.qwen_hidden_dim,
            },
        )

    def context_hidden(self, prompts: Sequence[str]) -> Tensor:
        """Extract frozen Qwen final-token hidden states for adapter readouts."""

        encoded = self._encode(list(prompts))
        with torch.no_grad():
            body = getattr(self.model, "model", self.model)
            out = body(**encoded, output_hidden_states=True, use_cache=False)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden_states = getattr(out, "hidden_states", None)
            if hidden_states is None:
                raise RuntimeError("Qwen body output must expose last_hidden_state or hidden_states")
            hidden = hidden_states[-1]
        lengths = encoded["attention_mask"].sum(dim=1) - 1
        return hidden[torch.arange(len(prompts), device=lengths.device), lengths].to(torch.float32)

    def _encode(self, prompts: Sequence[str]) -> dict[str, Tensor]:
        encoded = self.tokenizer(list(prompts), padding=True, return_tensors="pt")
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _query_tensor(self, *, query_text: str | None, query_tensor: Tensor | None, like: Tensor) -> Tensor:
        if query_tensor is not None and query_text is not None:
            raise ValueError("pass only one of query_text or query_tensor")
        if query_tensor is None and query_text is None:
            return torch.zeros(1, like.shape[-1], device=self.device, dtype=like.dtype)
        if query_tensor is None:
            query_tensor = render_text_bitmap(query_text or "", height=self.config.height, width=self.config.width, font_path=self.config.font_path)
        flat = query_tensor.to(self.device, dtype=like.dtype).flatten().unsqueeze(0)
        if flat.shape[-1] != like.shape[-1]:
            raise ValueError(f"query tensor flattened dim must be {like.shape[-1]}, got {flat.shape[-1]}")
        return flat


def _infer_model_device(model: object) -> str:
    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(parameter.device)


def _infer_hidden_dim(model: object) -> int:
    config = getattr(model, "config", None)
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("qwen_hidden_dim is required when the model config does not expose hidden_size")
