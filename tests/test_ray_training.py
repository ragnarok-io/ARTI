from __future__ import annotations

import copy
import os
import random

import pytest
import torch

import arti
from test_training_run import attached, data, policy

ray = pytest.importorskip("ray")
train = pytest.importorskip("ray.train")
TorchTrainer = pytest.importorskip("ray.train.torch").TorchTrainer
RunConfig, ScalingConfig = train.RunConfig, train.ScalingConfig
pytestmark = pytest.mark.skipif(
    os.environ.get("RAY_TRAIN_V2_ENABLED") != "1",
    reason="Ray Train V2 profile requires RAY_TRAIN_V2_ENABLED=1",
)


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


def local_run(function, config=None, *, name="arti-local"):
    return TorchTrainer(
        function,
        train_loop_config=config,
        scaling_config=ScalingConfig(num_workers=0),
        run_config=RunConfig(name=name),
    ).fit()


def session():
    current = attached().arti.trainer(
        objective=stochastic_loss, gradient_accumulation_steps=2, learning_rate=0.02
    )
    current.scheduler = torch.optim.lr_scheduler.StepLR(current.optimizer, step_size=2, gamma=0.8)
    torch.manual_seed(57)
    random.seed(57)
    return current


def stochastic_loss(model, batch):
    target = batch["labels"] + torch.rand_like(batch["labels"]) * random.random()
    return model(batch["inputs"], target)["loss"]


def test_native_ray_local_checkpoint_handoff_and_fresh_restore(tmp_path):
    full = session()
    full.fit(data(), steps=6, run=policy(tmp_path / "full"))
    partial = session()
    source_state = copy.deepcopy(partial.binding.state_module().state_dict())

    def first(_config):
        assert train.get_context().get_world_size() == 1
        assert train.get_checkpoint() is None
        partial.fit(
            data(),
            steps=2,
            run=policy(tmp_path / "first"),
            on_checkpoint=arti.RayTrainCheckpointHandler(checkpoint_dir=tmp_path / "durable"),
        )
        assert train.get_checkpoint() is not None

    output = local_run(first)
    assert not ray.is_initialized()
    assert output.metrics["optimizer_step"] == 2
    assert output.checkpoint is not None
    assert any(
        not torch.equal(source_state[k], v)
        for k, v in partial.binding.state_module().state_dict().items()
    )
    resumed = session()

    def second(config):
        artifact = arti.restore_ray_checkpoint(
            resumed, directory=tmp_path / "materialized", checkpoint=config["checkpoint"]
        )
        assert artifact.exists() and resumed.global_step == 2
        resumed.fit(
            data(),
            steps=4,
            run=policy(tmp_path / "second"),
            on_checkpoint=arti.RayTrainCheckpointHandler(),
        )

    continued = local_run(second, {"checkpoint": output.checkpoint}, name="arti-continued")
    assert continued.metrics["optimizer_step"] == resumed.global_step == 6
    assert full.micro_batches_seen == resumed.micro_batches_seen == 12
    assert_tree_equal(
        full.binding.state_module().state_dict(), resumed.binding.state_module().state_dict()
    )
    assert_tree_equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert_tree_equal(full.scheduler.state_dict(), resumed.scheduler.state_dict())
    assert full.loss_history == resumed.loss_history
    assert not ray.is_initialized()


def test_reported_validation_best_remains_inside_materialized_checkpoint(tmp_path):
    current = attached().arti.trainer(learning_rate=0.02)
    validation = [
        {"inputs": row["inputs"], "labels": torch.zeros_like(row["labels"])} for row in data()
    ]
    run = policy(tmp_path / "source", evaluation_steps=1, validation=arti.ARTIValidationConfig())

    def train_fn(_config):
        current.fit(
            data(),
            steps=3,
            run=run,
            eval_data=validation,
            on_checkpoint=arti.RayTrainCheckpointHandler(),
        )

    output = local_run(train_fn)
    restored = attached().arti.trainer()
    saved = arti.restore_ray_checkpoint(
        restored, directory=tmp_path / "restored", checkpoint=output.checkpoint
    )
    assert restored.global_step == 3
    assert restored.selection_receipt()["selected_step"] == 1
    assert restored._validation.best_checkpoint.is_relative_to(saved.parent)
    best = restored._validation.best_checkpoint
    with restored.checkpoint_context(best):
        restored_loss = restored.evaluate(validation).loss
    with current.checkpoint_context(current._validation.best_checkpoint):
        original_loss = current.evaluate(validation).loss
    assert restored_loss == original_loss
    restored.fit(
        data(),
        steps=1,
        run=policy(
            tmp_path / "continue", evaluation_steps=1, validation=arti.ARTIValidationConfig()
        ),
        eval_data=validation,
    )
    assert restored.global_step == 4 and restored._validation.best_checkpoint.exists()


def test_default_restore_uses_current_ray_checkpoint(tmp_path):
    original = session()
    restored = session()

    def train_fn(_config):
        assert arti.restore_ray_checkpoint(restored, directory=tmp_path / "empty") is None
        original.fit(
            data(),
            steps=1,
            run=policy(tmp_path / "original"),
            on_checkpoint=arti.RayTrainCheckpointHandler(
                metrics=lambda session: {"custom/step": session.global_step}
            ),
        )
        artifact = arti.restore_ray_checkpoint(restored, directory=tmp_path / "restore")
        assert artifact.exists() and restored.global_step == 1

    result = local_run(train_fn)
    assert result.metrics["custom/step"] == 1
    assert arti.torch.RayTrainCheckpointHandler is arti.RayTrainCheckpointHandler
    assert arti.torch.restore_ray_checkpoint is arti.restore_ray_checkpoint


def test_handoff_does_not_change_next_random_training_trajectory(tmp_path):
    a, b = session(), session()
    torch.manual_seed(57)
    random.seed(57)
    a.fit(data(), steps=3, run=policy(tmp_path / "a"))
    torch.manual_seed(57)
    random.seed(57)

    def train_fn(_config):
        b.fit(
            data(),
            steps=3,
            run=policy(tmp_path / "b"),
            on_checkpoint=arti.RayTrainCheckpointHandler(),
        )

    local_run(train_fn)
    assert a.loss_history == b.loss_history
    assert_tree_equal(a.binding.state_module().state_dict(), b.binding.state_module().state_dict())
