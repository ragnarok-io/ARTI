from __future__ import annotations

from pathlib import Path
import json
import importlib

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import arti


class TinyAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.out_proj(x)


class TinyMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.down_proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.down_proj(torch.relu(x))


class TinyDecoderBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_attn = TinyAttention(dim)
        self.mlp = TinyMLP(dim)

    def forward(self, x):
        return self.mlp(self.self_attn(x))


class TinyCausalLM(nn.Module):
    def __init__(self, vocab: int = 17, dim: int = 8) -> None:
        super().__init__()
        self.config = type("Config", (), {"use_cache": True, "_commit_hash": "fixed-qwen-revision"})()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([TinyDecoderBlock(dim)])
        self.lm_head = nn.Linear(dim, vocab)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return {
            "logits": self.lm_head(hidden),
            "past_key_values": ((hidden[:, -1:].detach(),),) if use_cache else None,
        }

    def generate(self, input_ids, attention_mask=None, **kwargs):
        token = self.forward(input_ids, attention_mask, use_cache=True)["logits"][:, -1:].argmax(dim=-1)
        return torch.cat([input_ids, token], dim=1)

    def save_pretrained(self, path):
        return path


class TinyViTBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.proj = nn.Linear(dim, dim)
        self.mlp = nn.Module()
        self.mlp.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.mlp.fc2(torch.relu(self.attn.proj(x)))


class TinyViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("ViTConfig", (), {"_commit_hash": "fixed-vit-revision"})()
        self.blocks = nn.ModuleList([TinyViTBlock(6)])
        self.head = nn.Linear(6, 3)

    def forward(self, pixel_values):
        return {"logits": self.head(self.blocks[0](pixel_values))}


class TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1)

    def forward(self, sample):
        return self.conv(sample)


class TinyPipeline:
    def __init__(self) -> None:
        self.unet = TinyUNet()
        self.components = {"unet": self.unet}

    def __call__(self, sample):
        return self.unet(sample)

    def save_pretrained(self, path):
        return path


class TinyPeftModel(TinyCausalLM):
    def __init__(self) -> None:
        super().__init__()
        self.peft_config = {"default": {"type": "LORA"}}

    def enable_adapters(self):
        return None


def qwen_workflow(seed: int = 4) -> arti.ARTIPretrained:
    torch.manual_seed(seed)
    model = TinyCausalLM()
    sample = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    workflow = arti.pretrained(model, provider="transformers", task="causal-lm", revision="main")
    workflow.scan(sample).plan(
        features={"recall": {"enabled": True, "steps": 1, "slots": 2}},
        where="mlp",
        max_adapters=1,
        training={"engine": "torch", "steps": 1},
    )
    return workflow


def test_qwen_style_plan_apply_preserves_generate_and_kv_cache() -> None:
    workflow = qwen_workflow()
    input_ids = torch.tensor([[1, 2, 3]])
    original_logits = workflow.model(input_ids)["logits"].detach()
    assert workflow.plan_value is not None
    assert workflow.plan_value.source["resolved_revision"] == "fixed-qwen-revision"
    assert workflow.plan_value.components[0].selected[0]["name"] == "layers.0.mlp.down_proj"
    capabilities = workflow.plan_value.native_capabilities
    assert "generate" in capabilities
    assert "kv-cache" in capabilities
    assert workflow.plan_value.insertion["identity_gate"] is False
    assert workflow.plan_value.insertion["zero_init_output"] is True

    workflow.apply()
    generated = workflow.generate(input_ids)
    cached = workflow.model(input_ids, use_cache=True)

    assert torch.equal(cached["logits"], original_logits)
    assert generated.shape == (1, 4)
    assert cached["past_key_values"] is not None
    assert workflow.doctor()["applied"] is True
    assert workflow.projects["model"].report().adapter_parameters == workflow.plan_value.components[0].adapter_parameters


def test_plan_round_trip_and_structure_drift_rejection(tmp_path: Path) -> None:
    workflow = qwen_workflow()
    assert workflow.plan_value is not None
    path = workflow.plan_value.write(tmp_path / "qwen.plan.json")
    restored = arti.ARTIPlan.read(path)
    assert restored.fingerprint == workflow.plan_value.fingerprint

    workflow.model.layers[0].mlp.down_proj = nn.Linear(8, 9)
    with pytest.raises(ValueError, match="structure changed"):
        workflow.apply(restored)


def test_pretrained_plan_rejects_unmatched_selector_early() -> None:
    model = nn.Sequential(nn.Linear(4, 4))
    workflow = arti.pretrained(model, provider="torch").scan(torch.zeros(1, 4))

    with pytest.raises(ValueError, match="selected no insertion points"):
        workflow.plan(where="missing.*")


def test_external_phase_requires_and_accepts_runtime_context() -> None:
    model = TinyCausalLM()
    input_ids = torch.tensor([[1, 2, 3]])
    workflow = arti.from_pretrained(
        model,
        provider="transformers",
        sample_batch={"input_ids": input_ids},
        features={
            "phase": {
                "enabled": True,
                "mode": "external",
                "coord_dim": 2,
                "frame_mode": "paired_rotation",
            }
        },
        where="mlp",
        max_adapters=1,
    )

    with pytest.raises(ValueError, match="requires workflow.context"):
        workflow.generate(input_ids)
    generated = workflow.generate(
        input_ids,
        arti_context={"coord": torch.zeros(1, 3, 2)},
    )

    assert generated.shape == (1, 4)


def test_qwen_style_arti_st_and_pretrained_lock_round_trip(tmp_path: Path) -> None:
    workflow = qwen_workflow(seed=11).apply()
    input_ids = torch.tensor([[1, 2, 3]])
    expected = workflow.model(input_ids)["logits"].detach()
    exported = workflow.export(tmp_path / "arti.st")

    restored = qwen_workflow(seed=11).apply()
    restored.load_weights(exported.saved.weights_path)
    actual = restored.model(input_ids)["logits"].detach()
    lock = arti.validate_pretrained_lock(exported.lock_path)

    assert torch.allclose(actual, expected)
    assert lock["source"]["resolved_revision"] == "fixed-qwen-revision"
    assert lock["weights_sha256"] == exported.saved.weights_sha256


def test_pretrained_lock_rejects_environment_drift(tmp_path: Path, monkeypatch) -> None:
    workflow = qwen_workflow(seed=13).apply()
    exported = workflow.export(tmp_path / "arti.st")
    module = importlib.import_module("arti.pretrained")
    current = module._environment_versions()
    monkeypatch.setattr(module, "_environment_versions", lambda: {**current, "torch": "different"})

    with pytest.raises(ValueError, match="environment mismatch"):
        arti.validate_pretrained_lock(exported.lock_path)


def test_vit_transformers_provider_adapts_native_image_model() -> None:
    pixels = torch.randn(2, 4, 6)
    workflow = arti.from_pretrained(
        TinyViT(),
        provider="transformers",
        task="image-classification",
        sample_batch={"pixel_values": pixels},
        where="vision-transformer",
        max_adapters=1,
    )

    assert workflow.model(pixel_values=pixels)["logits"].shape == (2, 4, 3)
    assert workflow.plan_value.components[0].selected[0]["name"] == "blocks.0.attn.proj"


def test_diffusers_provider_mutates_component_without_replacing_pipeline_api(tmp_path: Path) -> None:
    pipeline = TinyPipeline()
    sample = torch.randn(2, 4, 8, 8)
    workflow = arti.pretrained(pipeline, provider="diffusers", components=["unet"])
    workflow.scan(component_samples={"unet": sample}).plan(where={"unet": "conv"}, max_adapters=1)
    workflow.apply()

    assert workflow.model is pipeline
    assert pipeline(sample).shape == sample.shape
    exported = workflow.export(tmp_path / "diffusion.st")
    assert exported.saved.weights_path.exists()
    assert arti.validate_pretrained_lock(exported.lock_path)["component_structures"]["unet"]


def test_peft_provider_preserves_adapter_api_and_reports_adapter_names() -> None:
    model = TinyPeftModel()
    workflow = arti.from_pretrained(
        model,
        provider="peft",
        sample_batch={"input_ids": torch.tensor([[1, 2]])},
        where="mlp",
        max_adapters=1,
    )

    assert callable(workflow.enable_adapters)
    assert workflow.plan_value.provider_metadata["peft_adapters"] == ["default"]


def test_accelerate_engine_trains_and_keeps_model_identity() -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    inputs = torch.randn(8, 4)
    targets = torch.randn(8, 2)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)
    workflow = arti.from_pretrained(
        model,
        provider="torch",
        sample_batch=inputs[:2],
        where="0",
        max_adapters=1,
        training={"engine": "accelerate", "steps": 2, "mixed_precision": "no"},
    )

    result = workflow.fit(loader)

    assert result.model is model
    assert result.engine == "accelerate"
    assert result.steps == 2
    assert len(result.loss_history) == 2


def test_pretrained_checkpoint_resumes_optimizer_and_scheduler(tmp_path: Path) -> None:
    inputs = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16
    targets = torch.zeros(4, 2)
    data = [(inputs, targets)]

    def make_workflow():
        torch.manual_seed(41)
        model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))
        return arti.from_pretrained(
            model,
            provider="torch",
            sample_batch=inputs[:2],
            where="0",
            max_adapters=1,
            training={"engine": "torch", "steps": 1, "learning_rate": 1e-3},
        )

    continuous = make_workflow()
    continuous_optimizer = torch.optim.AdamW(
        [parameter for parameter in continuous.model.parameters() if parameter.requires_grad], lr=1e-3
    )
    continuous_scheduler = torch.optim.lr_scheduler.StepLR(continuous_optimizer, step_size=1, gamma=0.8)
    first = continuous.fit(data, optimizer=continuous_optimizer, scheduler=continuous_scheduler)
    exported = first.export(tmp_path / "resume.st")
    continuous.fit(data, optimizer=continuous_optimizer, scheduler=continuous_scheduler)

    restored = make_workflow()
    restored_optimizer = torch.optim.AdamW(
        [parameter for parameter in restored.model.parameters() if parameter.requires_grad], lr=1e-3
    )
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1, gamma=0.8)
    loaded = restored.load_weights(
        exported.saved.weights_path,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )
    restored.fit(data, optimizer=restored_optimizer, scheduler=restored_scheduler)

    assert loaded.training_state["engine"] == "torch"
    assert loaded.training_state["plan_fingerprint"] == restored.plan_value.fingerprint
    for expected, actual in zip(continuous.model.parameters(), restored.model.parameters(), strict=True):
        assert torch.allclose(actual, expected)
    assert restored_scheduler.state_dict() == continuous_scheduler.state_dict()


def test_distributed_training_requires_accelerate_launch() -> None:
    model = nn.Sequential(nn.Linear(4, 4))
    data = DataLoader(TensorDataset(torch.randn(4, 4), torch.randn(4, 4)), batch_size=2)
    workflow = arti.from_pretrained(
        model,
        provider="torch",
        sample_batch=torch.zeros(1, 4),
        where="0",
        max_adapters=1,
        training={"engine": "accelerate", "steps": 1, "distributed": True},
    )

    with pytest.raises(RuntimeError, match="torchrun"):
        workflow.fit(data)


def test_provider_report_and_actionable_missing_dependency(monkeypatch) -> None:
    report = {row["name"]: row for row in arti.provider_report()}
    assert {"torch", "transformers", "peft", "diffusers"}.issubset(report)

    provider = arti.get_pretrained_provider("diffusers")
    monkeypatch.setattr(type(provider), "available", property(lambda self: False))
    with pytest.raises(arti.ARTIProviderError, match="uv sync --extra sd"):
        provider.require()


def test_pretrained_adapter_inherits_base_dtype() -> None:
    model = nn.Sequential(nn.Linear(4, 4, dtype=torch.float16))
    sample = torch.randn(2, 4, dtype=torch.float16)
    workflow = arti.from_pretrained(model, provider="torch", sample_batch=sample, where="0", max_adapters=1)

    output = workflow.model(sample)
    wrapper = workflow.model[0]

    assert output.dtype == torch.float16
    assert next(wrapper.adapter.parameters()).dtype == torch.float16
    assert wrapper.output_gate is None
    assert wrapper.adapter.zero_init_output is True
    assert wrapper.adapter.out.linear.weight.dtype == torch.float16
    assert wrapper.adapter.out.radius.dtype == torch.float32


def test_pretrained_zero_output_is_exact_identity_without_scalar_throttle() -> None:
    model = nn.Sequential(nn.Linear(4, 4))
    sample = torch.randn(2, 4)
    workflow = arti.from_pretrained(model, provider="torch", sample_batch=sample, where="0", max_adapters=1)
    wrapper = workflow.model[0]
    calls = []
    handle = wrapper.adapter.register_forward_hook(lambda *args: calls.append(True))
    try:
        workflow.model.eval()
        with torch.no_grad():
            assert torch.equal(workflow.model(sample), wrapper.base(sample))
        assert calls == [True]
        workflow.model(sample).sum().backward()
        assert calls == [True, True]
        assert wrapper.output_gate is None
        assert wrapper.adapter.out.radius.grad is not None
        assert torch.count_nonzero(wrapper.adapter.out.radius.grad) > 0
        assert wrapper.adapter.out.linear.weight.grad is not None
        assert torch.count_nonzero(wrapper.adapter.out.linear.weight.grad) == 0
        workflow.model.train()
        workflow.model(sample)
        assert calls == [True, True, True]
    finally:
        handle.remove()


def test_pretrained_identity_gate_remains_an_explicit_legacy_option() -> None:
    model = nn.Sequential(nn.Linear(4, 4))
    sample = torch.randn(2, 4)
    workflow = arti.from_pretrained(
        model,
        provider="torch",
        sample_batch=sample,
        where="0",
        max_adapters=1,
        identity_gate=True,
        zero_init_output=False,
    )

    wrapper = workflow.model[0]
    assert wrapper.output_gate is not None
    assert wrapper.adapter.zero_init_output is False


def test_pretrained_cli_plan_fit_export_and_validate_lock(tmp_path: Path, monkeypatch, capsys) -> None:
    module_path = tmp_path / "pretrained_fixture.py"
    module_path.write_text(
        "import torch\n"
        "import torch.nn as nn\n"
        "def make_model():\n"
        "    torch.manual_seed(7)\n"
        "    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))\n"
        "def make_sample():\n"
        "    return torch.zeros(2, 4)\n"
        "def make_data():\n"
        "    x = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16\n"
        "    y = torch.zeros(4, 2)\n"
        "    return [(x[:2], y[:2]), (x[2:], y[2:])]\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config = {
        "model": {"factory": "pretrained_fixture:make_model", "provider": "torch"},
        "sample": {"factory": "pretrained_fixture:make_sample"},
        "features": {"recall": {"enabled": True, "steps": 1, "slots": 2}},
        "insertion": {"where": "0", "max_adapters": 1, "freeze_base": True},
        "training": {"engine": "torch", "steps": 2, "learning_rate": 0.001},
        "data": {"factory": "pretrained_fixture:make_data"},
    }
    config_path = tmp_path / "arti-pretrained.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    plan_path = tmp_path / "reviewed.plan.json"
    weights_path = tmp_path / "arti.st"

    from arti.cli import main

    assert main(["pretrained", "plan", str(config_path), "--output", str(plan_path)]) == 0
    plan_output = json.loads(capsys.readouterr().out)
    assert plan_output["kind"] == "pretrained-plan"
    assert main(
        [
            "pretrained",
            "fit",
            str(config_path),
            "--plan",
            str(plan_path),
            "--weights",
            str(weights_path),
        ]
    ) == 0
    fit_output = json.loads(capsys.readouterr().out)
    assert fit_output["kind"] == "pretrained-fit"
    assert fit_output["steps"] == 2
    assert main(["pretrained", "validate-lock", "--lock", fit_output["lock"]]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "pretrained-lock"
