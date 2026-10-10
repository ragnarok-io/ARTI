from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import arti
from test_training_run import TaskModel, data, graph_layer


def run_config(path):
    return arti.ARTITrainingRunConfig(output_dir=path, checkpoint_steps=2)


class NativeProgram(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = graph_layer()

    def forward(self, inputs, labels):
        value = self.layer(inputs)
        return {"loss": (value - labels).square().mean(), "prediction": value}


def workflow(*, accumulation=1, precision="no", steps=2, engine="torch"):
    torch.manual_seed(73)
    model = nn.Sequential(nn.Linear(4, 4))
    return arti.from_pretrained(model, provider="torch", sample_batch=torch.ones(2, 4), where="0", max_adapters=1, training={"engine": engine, "steps": steps, "gradient_accumulation_steps": accumulation, "mixed_precision": precision, "device": "cpu"})


def paired_data():
    return [(torch.full((2, 4), x), torch.full((2, 4), x / 2)) for x in (1.0, 2.0, 3.0)]


def test_config_and_trainer_are_shared_across_namespaces():
    assert arti.ARTIAttachTrainingConfig is arti.ARTITrainingConfig
    assert arti.torch.ARTITrainingConfig is arti.ARTITrainingConfig
    assert arti.torch.trainer is arti.trainer
    session = arti.trainer(NativeProgram())
    assert type(session) is arti.ARTITrainingSession
    assert session.plan()["binding_kind"] == "model"
    assert "attachment" not in session.plan()


@pytest.mark.parametrize(
    "settings",
    [
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"learning_rate": True},
        {"steps": 1.5},
        {"steps": True},
        {"gradient_accumulation_steps": 1.5},
        {"gradient_accumulation_steps": True},
    ],
)
def test_training_settings_do_not_silently_accept_invalid_update_counts(settings):
    with pytest.raises(ValueError):
        arti.ARTITrainingConfig(**settings)


def test_native_program_periodic_save_resume_and_evaluation(tmp_path):
    model = NativeProgram()
    session = arti.trainer(model, learning_rate=0.02)
    before = session.evaluate(data()).loss
    first = session.fit(data(), steps=3, run=run_config(tmp_path / "first"))
    restored = NativeProgram()
    resumed = arti.trainer(restored, learning_rate=0.02, resume_from_checkpoint=first.checkpoint_path)
    second = resumed.fit(data(), steps=2, run=run_config(tmp_path / "second"))
    assert second.steps == 5
    assert resumed.micro_batches_seen == 5
    assert resumed.evaluate(data()).loss < before
    assert second.checkpoint_path.exists()
    assert resumed.check(data()[0])["ok"]


def test_native_and_attachment_gradients_use_the_same_torch_loop(tmp_path):
    native = NativeProgram()
    attached = arti.ARTI.attach(TaskModel(), graph_layer(), layers="features")
    native_session = arti.trainer(native, gradient_accumulation_steps=2)
    attached_session = attached.arti.trainer(gradient_accumulation_steps=2)
    native_session.fit(data(), steps=3, run=run_config(tmp_path / "native"))
    attached_session.fit(data(), steps=3, run=run_config(tmp_path / "attached"))
    assert native_session.loss_history == attached_session.loss_history
    for a, b in zip(native.parameters(), attached.arti.parameters(), strict=True):
        assert torch.equal(a, b)


def test_pretrained_session_accumulates_all_microbatches(tmp_path):
    current = workflow(accumulation=3, steps=2)
    calls = []

    def loss_fn(output, target):
        calls.append(1)
        return (output - target).square().mean()

    result = current.fit(paired_data(), loss_fn=loss_fn, run=run_config(tmp_path))
    assert type(result.session) is arti.ARTITrainingSession
    assert len(calls) == 6 and result.session.micro_batches_seen == 6
    assert result.steps == 2 and len(result.loss_history) == 2
    assert result.checkpoint_path.exists()


def test_pretrained_accumulation_matches_explicit_large_batch(tmp_path):
    accumulated = workflow(accumulation=3, steps=2)
    combined = workflow(accumulation=1, steps=2)
    pairs = paired_data()
    merged = [(torch.cat([x for x, _ in pairs]), torch.cat([y for _, y in pairs]))]
    accumulated.fit(pairs, run=run_config(tmp_path / "accumulated"))
    combined.fit(merged, run=run_config(tmp_path / "combined"))
    for a, b in zip(accumulated.model.parameters(), combined.model.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)


def test_pretrained_checkpoint_preserves_adapter_buffers_and_excludes_host(tmp_path):
    current = workflow()
    wrapper = current.model[0]
    before = {name: value.clone() for name, value in wrapper.adapter.named_buffers()}
    assert before
    result = current.fit(paired_data(), run=run_config(tmp_path / "run"))
    exported = result.export(tmp_path / "model.st")
    inspected = arti.load(exported.saved.weights_path, load_resources=False)
    assert any("recall" in name for name in inspected.state_dict)
    assert any(name.startswith("adapters.") for name in inspected.state_dict)
    assert not any("base." in name or name.endswith("base.weight") for name in inspected.state_dict)
    restored = workflow()
    restored.load_weights(exported.saved.weights_path)
    for name, value in restored.model[0].adapter.named_buffers():
        assert torch.equal(value, dict(wrapper.adapter.named_buffers())[name])
    assert set(before) == set(dict(restored.model[0].adapter.named_buffers()))


def test_pretrained_resume_restores_data_position_and_optimizer(tmp_path):
    continuous = workflow(steps=2)
    continuous.fit(paired_data(), steps=5, run=run_config(tmp_path / "continuous"))
    split = workflow(steps=2)
    first = split.fit(paired_data(), run=run_config(tmp_path / "split"))
    restored = workflow(steps=2)
    final = restored.fit(paired_data(), steps=3, resume_from_checkpoint=first.checkpoint_path, run=run_config(tmp_path / "restored"))
    assert final.steps == 5 and final.session.micro_batches_seen == 5
    assert final.loss_history == continuous.fit_result.loss_history
    for a, b in zip(continuous.model.parameters(), restored.model.parameters(), strict=True):
        assert torch.equal(a, b)


def test_reopening_pretrained_trainer_does_not_reset_progress(tmp_path):
    current = workflow(steps=2)
    first = current.fit(paired_data(), run=run_config(tmp_path / "first"))
    reopened = current.trainer()
    assert reopened is first.session
    assert reopened.global_step == 2
    second = current.fit(paired_data(), steps=1, run=run_config(tmp_path / "second"))
    assert second.session is reopened and second.steps == 3
    assert current.trainer(reset=True).global_step == 0


def test_pretrained_bf16_is_actually_used(tmp_path):
    current = workflow(precision="bf16", steps=1)
    dtypes = []

    def loss_fn(output, target):
        dtypes.append(output.dtype)
        return (output - target).square().mean()

    current.fit(paired_data(), loss_fn=loss_fn, run=run_config(tmp_path))
    assert dtypes == [torch.bfloat16]


def test_pretrained_preserves_the_models_native_labeled_loss(tmp_path):
    class NativeLossModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.loss_calls = 0

        def forward(self, inputs, labels=None):
            output = self.linear(inputs)
            if labels is None:
                return {"logits": output}
            self.loss_calls += 1
            return {"logits": output, "loss": (output - labels).abs().mean() + 7.0}

    model = NativeLossModel()
    current = arti.from_pretrained(model, provider="torch", sample_batch={"inputs": torch.zeros(2, 4)}, where="linear", max_adapters=1, training={"steps": 2})
    result = current.fit(data(), run=run_config(tmp_path))
    assert model.loss_calls == 2
    assert all(value >= 7 for value in result.loss_history)


def test_native_zero_steps_does_not_train_and_still_produces_a_checkpoint(tmp_path):
    model = NativeProgram()
    before = copy.deepcopy(model.state_dict())
    result = arti.trainer(model, steps=0).fit(data(), run=run_config(tmp_path))
    assert result.steps == 0 and not result.loss_history
    assert result.checkpoint_path.exists()
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_model_parameter_subset_is_explicit_and_owned(tmp_path):
    model = nn.Linear(4, 4)
    before = model.weight.detach().clone()
    def objective(host, batch):
        return (host(batch[0]) - batch[1]).square().mean()
    session = arti.trainer(model, parameters=[model.bias], objective=objective)
    session.fit(paired_data(), steps=2, run=run_config(tmp_path))
    assert torch.equal(before, model.weight)
    with pytest.raises(ValueError, match="owned"):
        arti.trainer(model, parameters=[nn.Parameter(torch.ones(()))], objective=objective)
    with pytest.raises(ValueError, match="unique"):
        arti.trainer(model, parameters=[model.bias, model.bias], objective=objective)


def test_pretrained_accelerate_honors_accumulation(tmp_path):
    pytest.importorskip("accelerate")
    current = workflow(accumulation=3, steps=2, engine="accelerate")
    x = torch.randn(6, 4)
    loader = DataLoader(TensorDataset(x, x / 2), batch_size=2)
    result = current.fit(loader)
    assert result.steps == 2
    assert result.session.micro_batches_seen == 6
    assert len(result.loss_history) == 2
    exported = result.export(tmp_path / "accelerate.st")
    assert exported.saved.weights_path.exists()


def test_pretrained_transformers_uses_the_shared_session_and_native_loss(tmp_path):
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=32, pad_token_id=0)
    model = transformers.Qwen3ForCausalLM(config)
    inputs = torch.tensor([[1, 2, 3, 4]])
    current = arti.from_pretrained(model, provider="transformers", sample_batch={"input_ids": inputs}, where="mlp", max_adapters=1, training={"engine": "transformers", "steps": 2, "device": "cpu"})
    dataset = [{"input_ids": inputs[0], "labels": inputs[0]}]
    result = current.fit(dataset, trainer_kwargs={"output_dir": str(tmp_path / "trainer"), "per_device_train_batch_size": 1, "logging_steps": 1, "disable_tqdm": True})
    assert type(result.session) is arti.ARTITrainingSession
    assert result.session.objective._arti_native_model_loss
    assert result.steps == 2 and result.model is model
    assert result.export(tmp_path / "qwen.st").saved.weights_path.exists()


def test_pretrained_pipeline_custom_step_keeps_its_component_binding(tmp_path):
    from test_pretrained_workflow import TinyPipeline

    pipeline = TinyPipeline()
    sample = torch.ones(2, 4, 4, 4)
    current = arti.pretrained(pipeline, provider="diffusers", components=["unet"])
    current.scan(component_samples={"unet": sample}).plan(where={"unet": "conv"}, max_adapters=1, training={"steps": 2})

    def step(bundle, batch):
        return (bundle.components["unet"](batch) - batch).square().mean()

    result = current.fit([sample], step_fn=step, run=run_config(tmp_path))
    assert result.steps == 2
    assert current.model is pipeline
    assert result.checkpoint_path.exists()


def test_pretrained_cli_shares_run_config_and_explicit_resume(tmp_path, monkeypatch):
    from arti.pretrained_cli import pretrained_cli_report
    import importlib
    import json

    source = tmp_path / "shared_cli_factory.py"
    source.write_text("import torch\nfrom torch import nn\ndef model():\n    torch.manual_seed(44)\n    return nn.Sequential(nn.Linear(4, 4))\ndef sample():\n    return torch.ones(2, 4)\ndef data():\n    return [(torch.ones(2, 4), torch.zeros(2, 4))]\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    config = {"model": {"factory": "shared_cli_factory:model", "provider": "torch"}, "sample": {"factory": "shared_cli_factory:sample"}, "insertion": {"where": "0", "max_adapters": 1}, "training": {"steps": 2}, "data": {"factory": "shared_cli_factory:data", "validation_factory": "shared_cli_factory:data"}, "run": {"output_dir": str(tmp_path / "first"), "evaluation_steps": 1}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    first = pretrained_cli_report("fit", path, weights=tmp_path / "first.st")
    config["run"]["output_dir"] = str(tmp_path / "continued")
    path.write_text(json.dumps(config), encoding="utf-8")
    resumed = pretrained_cli_report("fit", path, weights=tmp_path / "continued.st", resume_from_checkpoint=first["checkpoint"], steps=1)
    assert first["steps"] == 2 and resumed["steps"] == 3
    assert resumed["run_directory"] == str(tmp_path / "continued")
    assert resumed["stop_reason"] == "steps_completed"


def test_token_weighted_accumulation_matches_one_native_task_batch(tmp_path):
    class TokenModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(3, 4)

        def forward(self, inputs, labels):
            logits = self.projection(inputs)
            return {"loss": torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)}

    torch.manual_seed(762)
    first = TokenModel()
    second = TokenModel()
    second.load_state_dict(first.state_dict())
    inputs = torch.randn(2, 3, 3)
    labels = torch.tensor([[1, 2, 3], [0, -100, -100]])
    micros = [{"inputs": inputs[index:index + 1], "labels": labels[index:index + 1]} for index in range(2)]
    accumulated = arti.trainer(first, learning_rate=0.1, gradient_accumulation_steps=2,
                               optimizer=torch.optim.SGD(first.parameters(), lr=0.1))
    whole = arti.trainer(second, learning_rate=0.1, optimizer=torch.optim.SGD(second.parameters(), lr=0.1))
    accumulated.fit(micros, steps=1, objective_weight=lambda batch: batch["labels"].ne(-100).sum(),
                    run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "micros"))
    whole.fit([{"inputs": inputs, "labels": labels}], steps=1,
              run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "whole"))
    for actual, expected in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert accumulated.loss_history[0] == pytest.approx(whole.loss_history[0], rel=1e-6)


def test_clipping_is_applied_after_accumulation(tmp_path):
    class ScalarTask(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Parameter(torch.tensor(1.0))

        def forward(self, inputs):
            return {"loss": self.value * inputs}

    model = ScalarTask()
    session = arti.trainer(model, gradient_accumulation_steps=2,
                           optimizer=torch.optim.SGD(model.parameters(), lr=0.1))
    session.fit([{"inputs": torch.tensor(1000.0)}], steps=1, max_grad_norm=1.0,
                run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "clipped"))
    assert float(model.value.detach()) == pytest.approx(0.9)
