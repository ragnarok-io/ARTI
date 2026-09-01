from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

import arti


def model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))


def trainable_layer() -> arti.ARTILayer:
    return arti.ARTILayer(
        arti.mechanisms.AdaptivePulse(
            half=arti.Half(stochastic=False, learnable=True),
        )
    )


def batches():
    generator = torch.Generator().manual_seed(8)
    x = torch.randn(3, 4, 8, generator=generator)
    target = torch.randn(3, 4, 8, generator=generator)
    return [{"x": x, "target": target}]


def objective(host, batch):
    return torch.nn.functional.mse_loss(host(batch["x"]), batch["target"])


def test_toml_attach_config_roundtrip_and_reproducible_lock(tmp_path) -> None:
    config_path = tmp_path / "arti-attach.toml"
    arti.write_attach_config(
        config_path,
        arti.ARTIAttachConfig(
            layer={"layers": ["0", "2"]},
            training=arti.ARTIAttachTrainingConfig(
                steps=2,
                gradient_accumulation_steps=2,
                mixed_precision="bf16",
            ),
        ),
    )
    loaded = arti.load_attach_config(config_path)
    attached = arti.ARTI.attach(model(), trainable_layer(), config=config_path)
    first = attached.arti.write_lock(tmp_path / "first.lock.json")
    second = attached.arti.write_lock(tmp_path / "second.lock.json")

    assert loaded.training.mixed_precision == "bf16"
    assert attached.arti.paths == ("0", "2")
    assert first.read_bytes() == second.read_bytes()
    assert attached.arti.validate_lock(first)["format"] == "arti.attach.lock"
    artifact = tmp_path / "configured.arti.st"
    attached.arti.save(artifact)
    restored = arti.ARTI.load(model(), artifact, layer=trainable_layer())
    restored_session = restored.arti.trainer()
    assert restored_session.config.mixed_precision == "bf16"
    assert restored_session.config.gradient_accumulation_steps == 2

    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "# changed\n",
        encoding="utf-8",
    )
    changed = arti.load_attach_config(config_path)
    with pytest.raises(ValueError, match="config_sha256"):
        arti.validate_attach_lock(
            first,
            config=changed,
            resolved_layer={"freeze_backbone": True, "layers": []},
            host_structure="wrong",
        )


def test_standard_tensor_alignment_objective_and_bf16_accumulation() -> None:
    attached = arti.ARTI.attach(
        model(),
        trainable_layer(),
        layers=("0", "2"),
    )
    clean = torch.randn(2, 5, 8)
    corrupt = clean.clone()
    corrupt[:, 1:3] = 0
    loss = arti.tensor_alignment_objective(
        attached.arti,
        {"clean_inputs": clean, "corrupt_inputs": corrupt},
    )
    session = attached.arti.trainer(
        objective=objective,
        steps=2,
        gradient_accumulation_steps=2,
        mixed_precision="bf16",
    )
    result = session.fit(batches())

    assert loss.isfinite()
    assert result.steps == 2
    assert len(result.loss_history) == 2
    assert all(torch.isfinite(torch.tensor(result.loss_history)))


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path) -> None:
    torch.manual_seed(33)
    initial = model().state_dict()

    continuous_model = model()
    continuous_model.load_state_dict(copy.deepcopy(initial))
    continuous_model = arti.ARTI.attach(
        continuous_model,
        trainable_layer(),
        layers=("0", "2"),
    )
    continuous = continuous_model.arti.trainer(objective=objective, learning_rate=1e-3)
    continuous.fit(batches(), steps=4)

    split_model = model()
    split_model.load_state_dict(copy.deepcopy(initial))
    split_model = arti.ARTI.attach(
        split_model,
        trainable_layer(),
        layers=("0", "2"),
    )
    split = split_model.arti.trainer(objective=objective, learning_rate=1e-3)
    checkpoint = tmp_path / "resume.arti.st"
    split.fit(batches(), steps=2, checkpoint_path=checkpoint)
    rng_after_split = torch.get_rng_state()

    restored_base = model()
    restored_base.load_state_dict(copy.deepcopy(initial))
    restored_model = arti.ARTI.load(
        restored_base,
        checkpoint,
        layer=trainable_layer(),
    )
    resumed = restored_model.arti.trainer(objective=objective, learning_rate=1e-3)
    resumed.load_checkpoint(checkpoint)
    torch.set_rng_state(rng_after_split)
    resumed.fit(batches(), steps=2)

    assert resumed.global_step == 4
    assert len(resumed.loss_history) == 4
    for expected, actual in zip(
        continuous_model.arti.parameters(),
        restored_model.arti.parameters(),
        strict=True,
    ):
        assert torch.equal(expected, actual)


def test_custom_objective_must_return_scalar() -> None:
    attached = arti.ARTI.attach(model(), trainable_layer(), layers="0")
    session = attached.arti.trainer(objective=lambda host, batch: host(batch["x"]))
    with pytest.raises(ValueError, match="scalar"):
        session.fit(batches(), steps=1)
