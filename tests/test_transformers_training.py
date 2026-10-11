from __future__ import annotations

import copy

import pytest
import torch

import arti

transformers = pytest.importorskip("transformers")


def attached_session(*, resume=None, stochastic=True):
    torch.manual_seed(91)
    config = transformers.Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        pad_token_id=0,
        attention_dropout=0.2 if stochastic else 0.0,
    )
    model = transformers.Qwen3ForCausalLM(config)
    layer = arti.ARTILayer(
        arti.mechanisms.AdaptivePulse(half=arti.Half(stochastic=False, learnable=True))
    )
    arti.ARTI.attach(model, layer, layers=("model.layers.0.mlp",))
    return model.arti.trainer(
        engine="transformers", device="cpu", learning_rate=0.02, resume_from_checkpoint=resume
    )


def dataset():
    return [
        {
            "input_ids": torch.tensor([1, i + 2, i + 3, 2]),
            "labels": torch.tensor([1, i + 2, i + 3, 2]),
        }
        for i in range(12)
    ]


def options(path, **extra):
    return {
        "training_args": {
            "output_dir": str(path),
            "use_cpu": True,
            "disable_tqdm": True,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "logging_steps": 1,
            "save_steps": 1,
            "dataloader_pin_memory": False,
            "seed": 37,
            "data_seed": 53,
            "report_to": [],
            **extra,
        }
    }


class StopAtTwo(transformers.TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 2:
            control.should_save = True
            control.should_training_stop = True
        return control


def track_inputs(session):
    seen = []
    hook = session.model.get_input_embeddings().register_forward_pre_hook(
        lambda _module, args: seen.extend(args[0].detach().tolist())
    )
    return seen, hook


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("row_count", [12, 5])
def test_native_checkpoint_resume_matches_continuous_rng_data_optimizer(tmp_path, row_count):
    rows = dataset()[:row_count]
    full = attached_session()
    frozen = [(p, p.detach().clone()) for p in full.model.parameters() if not p.requires_grad]
    initial = copy.deepcopy(full.binding.state_module().state_dict())
    full_seen, full_hook = track_inputs(full)
    result = full.fit(rows, steps=6, trainer_kwargs=options(tmp_path / "full"))
    full_hook.remove()
    first = attached_session()
    first_seen, hook = track_inputs(first)
    kwargs = {**options(tmp_path / "first"), "callbacks": [StopAtTwo()]}
    interrupted = first.fit(rows, steps=6, trainer_kwargs=kwargs)
    hook.remove()
    checkpoint = tmp_path / "first" / "checkpoint-2"
    assert interrupted.steps == 2 and interrupted.stop_reason == "trainer_stopped"
    assert {
        "model.arti.st",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    }.issubset(p.name for p in checkpoint.iterdir())
    assert not (checkpoint / "model.safetensors").exists()
    continued = attached_session(resume=checkpoint)
    continued_seen, hook = track_inputs(continued)
    resumed = continued.fit(rows, steps=4, trainer_kwargs=options(tmp_path / "resumed"))
    hook.remove()
    assert result.steps == resumed.steps == 6
    assert continued.micro_batches_seen == full.micro_batches_seen == len(full_seen)
    assert full_seen == first_seen + continued_seen
    assert_tree_equal(
        full.binding.state_module().state_dict(), continued.binding.state_module().state_dict()
    )
    assert_tree_equal(full.optimizer.state_dict(), continued.optimizer.state_dict())
    assert_tree_equal(full.scheduler.state_dict(), continued.scheduler.state_dict())
    assert full.loss_history == continued.loss_history
    assert any(
        not torch.equal(initial[k], v) for k, v in full.binding.state_module().state_dict().items()
    )
    for parameter, before in frozen:
        assert torch.equal(parameter, before)


def test_direct_native_trainer_and_auto_resume_preserve_class_and_callbacks(tmp_path):
    class CustomTrainer(transformers.Trainer):
        def compute_loss(self, *args, **kwargs):
            self.native_loss_calls = getattr(self, "native_loss_calls", 0) + 1
            return super().compute_loss(*args, **kwargs)

    session = attached_session()
    kwargs = options(tmp_path / "trainer")
    kwargs["training_args"]["max_steps"] = 6
    kwargs["callbacks"] = [StopAtTwo()]
    trainer = session.transformers_trainer(
        dataset(), trainer_class=CustomTrainer, trainer_kwargs=kwargs
    )
    assert isinstance(trainer, CustomTrainer)
    trainer.train()
    assert trainer.native_loss_calls == session.micro_batches_seen == 4
    assert session.global_step == 2
    resumed = attached_session(resume=True)
    outcome = resumed.fit(dataset(), steps=1, trainer_kwargs=options(tmp_path / "trainer"))
    assert outcome.steps == 3
    assert resumed.engine_object.state.global_step == 3
    assert len(resumed.loss_history) == 3
    exported = tmp_path / "export"
    resumed.engine_object.save_model(str(exported))
    assert (exported / "model.arti.st").exists()
    assert not (exported / "optimizer.pt").exists()


def test_portable_file_is_explicit_warm_resume_and_step_offset_is_not_doubled(tmp_path):
    first = attached_session()
    checkpoint = tmp_path / "portable.arti.st"
    first.fit(
        dataset(), steps=2, checkpoint_path=checkpoint, trainer_kwargs=options(tmp_path / "first")
    )
    resumed = attached_session(resume=checkpoint)
    with pytest.warns(UserWarning, match="portable ARTI file"):
        resumed.fit(dataset(), steps=2, trainer_kwargs=options(tmp_path / "warm"))
    assert resumed.global_step == 4
    assert len(resumed.loss_history) == 4
    assert all(int(value["step"]) == 4 for value in resumed.optimizer.state.values())
    again = attached_session(resume=tmp_path / "warm" / "checkpoint-2" / "model.arti.st")
    again.fit(dataset(), steps=1, trainer_kwargs=options(tmp_path / "again"))
    assert again.global_step == 5
    assert len(again.loss_history) == 5
    assert again.micro_batches_seen == 10


def test_best_asset_and_last_optimizer_remain_separate(tmp_path):
    evaluations = iter([1.0, 2.0, 3.0])

    def metric(_prediction):
        return {"score": next(evaluations)}

    session = attached_session(stochastic=False)
    kwargs = options(
        tmp_path / "run",
        eval_strategy="steps",
        eval_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="score",
        greater_is_better=False,
    )
    kwargs.update(eval_dataset=dataset()[:2], compute_metrics=metric)
    result = session.fit(
        dataset(), steps=3, checkpoint_path=tmp_path / "last.arti.st", trainer_kwargs=kwargs
    )
    best = tmp_path / "run" / "checkpoint-1" / "model.arti.st"
    assert result.best_checkpoint_path == result.selected_checkpoint_path == best
    assert result.checkpoint_path == tmp_path / "last.arti.st"
    assert result.selection["policy"] == "best"
    last = attached_session()
    last.load_checkpoint(result.checkpoint_path)
    assert_tree_equal(
        last.binding.state_module().state_dict(), session.binding.state_module().state_dict()
    )
    assert_tree_equal(last.optimizer.state_dict(), session.optimizer.state_dict())
    chosen = attached_session()
    chosen.binding.load(best, load_checkpoint=False)
    assert any(
        not torch.equal(a, b)
        for a, b in zip(chosen.binding.parameters(), last.binding.parameters())
    )


def test_standalone_native_model_uses_the_same_bridge(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)

        def forward(self, features, labels=None):
            logits = self.linear(features)
            return {"logits": logits, "loss": torch.nn.functional.cross_entropy(logits, labels)}

    rows = [{"features": torch.ones(3), "labels": torch.tensor(i % 2)} for i in range(6)]
    model = Model()
    session = arti.trainer(model, engine="transformers", device="cpu")
    session.fit(rows, steps=2, trainer_kwargs=options(tmp_path / "native"))
    resumed = arti.trainer(
        Model(),
        engine="transformers",
        device="cpu",
        resume_from_checkpoint=tmp_path / "native" / "checkpoint-2",
    )
    resumed.fit(rows, steps=1, trainer_kwargs=options(tmp_path / "continued"))
    assert resumed.global_step == 3
    assert session.binding.kind == "model"


def test_bridge_rejects_competing_parameter_ownership():
    session = attached_session()
    with pytest.raises(TypeError, match="Trainer subclass"):
        session.transformers_trainer(dataset(), trainer_class=object)
    with pytest.raises(ValueError, match="ownership"):
        session.transformers_trainer(
            dataset(), trainer_kwargs={"model_init": lambda: session.model}
        )


def test_native_optimizer_settings_and_explicit_subset_are_respected(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 2))
    selected = tuple(model[1].parameters())
    session = arti.trainer(model, parameters=selected, engine="transformers", device="cpu")
    kwargs = options(tmp_path, learning_rate=0.004, weight_decay=0.02, adam_beta1=0.8)
    trainer = session.transformers_trainer([], trainer_kwargs=kwargs)
    trainer.create_optimizer()
    assert {id(p) for group in trainer.optimizer.param_groups for p in group["params"]} == {
        id(p) for p in selected
    }
    assert all(p.requires_grad for p in model.parameters())
    assert all(
        group["lr"] == 0.004 and group["betas"][0] == 0.8
        for group in trainer.optimizer.param_groups
    )
    assert {group["weight_decay"] for group in trainer.optimizer.param_groups} == {0.0, 0.02}


def test_explicit_optimizer_is_kept_and_gradient_checkpointing_works(tmp_path):
    session = attached_session()
    optimizer = torch.optim.SGD(session.binding.parameters(), lr=0.003, momentum=0.5)
    session = session.model.arti.trainer(engine="transformers", device="cpu", optimizer=optimizer)
    kwargs = options(tmp_path, gradient_checkpointing=True)
    session.fit(dataset(), steps=1, trainer_kwargs=kwargs)
    assert session.optimizer.optimizer is optimizer
    assert session.engine_object.args.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert session.global_step == 1 and optimizer.state
    artifact = session.save_checkpoint(tmp_path / "custom.arti.st")
    restored = attached_session()
    with pytest.raises(ValueError, match="caller-supplied optimizer"):
        restored.load_checkpoint(artifact)
    restored_optimizer = torch.optim.SGD(restored.binding.parameters(), lr=0.003, momentum=0.5)
    restored = restored.model.arti.trainer(
        engine="transformers", device="cpu", optimizer=restored_optimizer
    )
    restored.load_checkpoint(artifact)
    assert_tree_equal(restored.optimizer.state_dict(), session.optimizer.state_dict())


def test_zero_step_resume_restores_native_state_without_consuming_data(tmp_path):
    first = attached_session()
    first.fit(dataset(), steps=2, trainer_kwargs=options(tmp_path / "first"))
    resumed = attached_session(resume=tmp_path / "first" / "checkpoint-2")
    inputs, hook = track_inputs(resumed)
    outcome = resumed.fit(dataset(), steps=0, trainer_kwargs=options(tmp_path / "zero"))
    assert inputs == [] and outcome.steps == 2
    assert_tree_equal(
        first.binding.state_module().state_dict(), resumed.binding.state_module().state_dict()
    )
    assert_tree_equal(first.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_tree_equal(first.scheduler.state_dict(), resumed.scheduler.state_dict())
    resumed.fit(dataset(), steps=1, trainer_kwargs=options(tmp_path / "continued"))
    hook.remove()
    assert resumed.global_step == 3 and len(inputs) == 2


def test_direct_trainer_best_view_saves_coherent_last_state(tmp_path):
    values = iter([1.0, 2.0, 3.0])
    session = attached_session(stochastic=False)
    kwargs = options(
        tmp_path / "direct",
        max_steps=3,
        eval_strategy="steps",
        eval_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="score",
        greater_is_better=False,
    )
    kwargs.update(eval_dataset=dataset()[:2], compute_metrics=lambda _: {"score": next(values)})
    trainer = session.transformers_trainer(dataset(), trainer_kwargs=kwargs)
    trainer.train()
    best_state = copy.deepcopy(session.binding.state_module().state_dict())
    saved = session.save_checkpoint(tmp_path / "resume.arti.st")
    assert_tree_equal(best_state, session.binding.state_module().state_dict())
    reloaded = attached_session()
    reloaded.load_checkpoint(saved)
    last = attached_session()
    last.load_checkpoint(trainer._arti_last_artifact)
    assert_tree_equal(
        reloaded.binding.state_module().state_dict(), last.binding.state_module().state_dict()
    )
    assert_tree_equal(reloaded.optimizer.state_dict(), session.optimizer.state_dict())
    assert session.selection_receipt()["selection"] == "best"


def test_native_trainer_respects_read_only_checkpoint_context(tmp_path):
    session = attached_session()
    artifact = session.save_checkpoint(tmp_path / "view.arti.st")
    trainer = session.transformers_trainer(dataset(), trainer_kwargs=options(tmp_path / "trainer"))
    with (
        session.checkpoint_context(artifact),
        pytest.raises(ValueError, match="model-only checkpoint context"),
    ):
        trainer.train()
    assert session.global_step == 0
