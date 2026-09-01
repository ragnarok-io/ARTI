from __future__ import annotations

import copy

import pytest
import torch

import arti


transformers = pytest.importorskip("transformers")


def qwen_config():
    return transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


def trainable_layer() -> arti.ARTILayer:
    return arti.ARTILayer(
        arti.mechanisms.AdaptivePulse(
            half=arti.Half(stochastic=False, learnable=True),
        )
    )


def test_qwen_train_save_reload_and_generate_consistency(tmp_path, paired_rng) -> None:
    torch.manual_seed(19)
    base = transformers.Qwen3ForCausalLM(qwen_config())
    initial = copy.deepcopy(base.state_dict())
    preview = arti.ARTI.preview(base, trainable_layer())
    model = arti.ARTI.attach(base, trainable_layer())
    input_ids = torch.tensor([[1, 7, 11, 13], [1, 5, 17, 19]])
    optimizer = torch.optim.AdamW(model.arti.parameters(), lr=1e-3)

    model(input_ids=input_ids, labels=input_ids).loss.backward()
    optimizer.step()
    model.eval()
    artifact = tmp_path / "tiny-qwen.arti.st"
    model.arti.save(artifact)

    restored = transformers.Qwen3ForCausalLM(qwen_config())
    restored.load_state_dict(initial)
    arti.ARTI.load(restored, artifact, layer=trainable_layer())
    restored.eval()
    with torch.no_grad():
        expected, actual = paired_rng(
            lambda: (
                model(input_ids=input_ids).logits,
                model.generate(input_ids[:1], max_new_tokens=2, do_sample=False),
            ),
            lambda: (
                restored(input_ids=input_ids).logits,
                restored.generate(input_ids[:1], max_new_tokens=2, do_sample=False),
            ),
        )
    expected_logits, expected_tokens = expected
    actual_logits, actual_tokens = actual

    assert type(model) is transformers.Qwen3ForCausalLM
    assert model.arti.paths == ("model.layers.0", "model.layers.1")
    assert preview.trainable_parameters == model.arti.summary().trainable_parameters
    assert torch.equal(expected_logits, actual_logits)
    assert torch.equal(expected_tokens, actual_tokens)
