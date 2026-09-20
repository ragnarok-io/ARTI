from types import SimpleNamespace

import torch
import torch.nn as nn
from pathlib import Path

import arti

from arti.integrations.qwen import QwenGlyphRuntimeAdapter, QwenGlyphRuntimeConfig


class FakeTokenizer:
    def __call__(self, prompts, padding=True, return_tensors="pt"):
        width = max(len(prompt.split()) for prompt in prompts)
        input_ids = torch.zeros(len(prompts), width, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, prompt in enumerate(prompts):
            tokens = prompt.split()
            attention_mask[row, : len(tokens)] = 1
            input_ids[row, : len(tokens)] = torch.arange(1, len(tokens) + 1)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_decode(self, output_ids, skip_special_tokens=True):
        return [f"decoded:{row.tolist()}" for row in output_ids]


class FakeBody(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask, output_hidden_states=False, use_cache=False):
        base = input_ids.to(torch.float32).unsqueeze(-1)
        offsets = torch.arange(self.hidden_size, device=input_ids.device, dtype=torch.float32)
        hidden = base + offsets
        return SimpleNamespace(last_hidden_state=hidden, hidden_states=(hidden,))


class FakeQwen(nn.Module):
    def __init__(self, hidden_size=16, vocab_size=32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.anchor = nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size
        self.model = FakeBody(hidden_size)

    def forward(self, input_ids, attention_mask, output_hidden_states=False, use_cache=False):
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, self.vocab_size, device=input_ids.device)
        logits[..., 3] = 1.0
        logits[..., 7] = input_ids.to(torch.float32)
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, attention_mask, **kwargs):
        next_token = torch.full((input_ids.shape[0], 1), 9, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, next_token], dim=1)


def make_adapter() -> QwenGlyphRuntimeAdapter:
    return QwenGlyphRuntimeAdapter(
        FakeQwen(),
        FakeTokenizer(),
        config=QwenGlyphRuntimeConfig(height=14, width=96, hidden_dim=24, device="cpu", raw_logit_scale=25.0),
    )


def test_qwen_glyph_adapter_preserves_dialogue_logits_during_readout():
    adapter = make_adapter()
    prompts = ["User: hello\nAssistant:", "User: 2 plus 3\nAssistant:"]
    drift = adapter.dialogue_drift(
        prompts,
        operation=lambda: adapter.read_glyph_vocab("read the glyph", ["apple", "banana", "phase"], query_text="phase"),
    )
    assert drift.prompt_count == 2
    assert drift.max_abs_logit_delta == 0.0
    assert drift.mean_kl_divergence == 0.0
    assert drift.top1_preserved is True


def test_qwen_glyph_adapter_reads_external_bitmap_vocab_slot():
    adapter = make_adapter()
    readout = adapter.read_glyph_vocab("read the external vocab", ["apple", "banana", "phase"], query_text="banana")
    assert readout.local_index == 1
    assert readout.text == "banana"
    assert readout.logits.shape == (3,)
    assert readout.probabilities.argmax().item() == 1


def test_qwen_glyph_adapter_generate_uses_base_model_path():
    adapter = make_adapter()
    assert adapter.generate("User: hi\nAssistant:").startswith("decoded:")


def test_qwen_glyph_adapter_trainable_arti_st_round_trip(tmp_path: Path):
    torch.manual_seed(23)
    adapter = make_adapter().eval()
    expected = adapter.read_glyph_vocab("read", ["apple", "banana", "phase"], query_text="banana").logits.detach()

    saved = arti.save(adapter, tmp_path / "qwen-adapter.st", scope="trainable")
    restored = make_adapter().eval()
    loaded = arti.load(saved.weights_path, model=restored)
    actual = restored.read_glyph_vocab("read", ["apple", "banana", "phase"], query_text="banana").logits.detach()

    assert torch.allclose(actual, expected)
    assert loaded.manifest["weight_scope"] == "trainable"
    assert loaded.missing_keys
    assert all(not key.startswith("model.") for key in loaded.state_dict)
