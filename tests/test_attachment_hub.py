from __future__ import annotations

import json

import pytest
import torch

import arti


transformers = pytest.importorskip("transformers")


def tiny_qwen():
    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return transformers.Qwen3ForCausalLM(config)


def trainable_layer() -> arti.ARTILayer:
    return arti.ARTILayer(
        arti.mechanisms.AdaptivePulse(
            half=arti.Half(stochastic=False, learnable=True),
        )
    )


def test_hub_bundle_references_base_without_copying_weights_and_loads_in_one_call(
    tmp_path,
    paired_rng,
) -> None:
    base_dir = tmp_path / "base"
    bundle_dir = tmp_path / "arti-bundle"
    base = tiny_qwen()
    base.save_pretrained(base_dir)
    loaded_base = transformers.AutoModelForCausalLM.from_pretrained(
        base_dir,
        local_files_only=True,
    )
    attached = arti.ARTI.attach(loaded_base, trainable_layer())
    input_ids = torch.tensor([[1, 4, 7]])
    attached.eval()
    saved = attached.arti.save_pretrained(bundle_dir, base_model="../base")

    restored = arti.ARTI.from_pretrained(
        bundle_dir,
        layer=trainable_layer(),
        model_kwargs={"local_files_only": True},
    )
    restored.eval()
    with torch.no_grad():
        expected, actual = paired_rng(
            lambda: attached(input_ids=input_ids).logits,
            lambda: restored(input_ids=input_ids).logits,
        )

    assert saved.manifest_path.exists()
    assert arti.load_attach_config(saved.config_path).layer["layers"] == (
        "model.layers.0",
        "model.layers.1",
    )
    assert torch.equal(expected, actual)
    assert not any(
        path.name in arti.attachment_hub.BASE_WEIGHT_NAMES
        for path in bundle_dir.iterdir()
    )
    assert not any(path.name.startswith("model-") for path in bundle_dir.iterdir())
    manifest = json.loads((bundle_dir / "arti-hub.json").read_text(encoding="utf-8"))
    assert manifest["base_model"]["source"] == "../base"
    assert arti.torch.ARTIDoctorReport is arti.ARTIDoctorReport


def test_doctor_reports_device_dtype_freezing_and_gradient_checkpointing() -> None:
    model = tiny_qwen()
    model.gradient_checkpointing_enable()
    model = arti.ARTI.attach(model, trainable_layer())
    model.to(dtype=torch.bfloat16)
    report = model.arti.doctor()

    assert report.ok
    assert report.devices == ("cpu",)
    assert report.dtypes == ("torch.bfloat16",)
    assert report.backbone_frozen
    assert report.gradient_checkpointing
    assert report.layers == ("model.layers.0", "model.layers.1")


def test_hub_manifest_and_host_integrity_are_enforced(tmp_path) -> None:
    model = arti.ARTI.attach(tiny_qwen(), trainable_layer())
    bundle = tmp_path / "bundle"
    model.arti.save_pretrained(bundle, base_model="Qwen/test")
    manifest = bundle / "arti-hub.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["base_model"]["source"] = "changed"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        arti.ARTI.from_pretrained(
            bundle,
            model=tiny_qwen(),
            layer=trainable_layer(),
        )


def test_existing_base_weight_in_bundle_is_rejected(tmp_path) -> None:
    model = arti.ARTI.attach(tiny_qwen(), trainable_layer())
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.safetensors").write_bytes(b"not-a-base-model")
    with pytest.raises(ValueError, match="must not contain base model weights"):
        model.arti.save_pretrained(bundle, base_model="Qwen/test")
