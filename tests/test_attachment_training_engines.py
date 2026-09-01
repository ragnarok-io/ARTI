from __future__ import annotations

import copy

import pytest
import torch

import arti


transformers = pytest.importorskip("transformers")


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.rows = [
            {"input_ids": torch.tensor([1, 3, 5, 7]), "labels": torch.tensor([1, 3, 5, 7])},
            {"input_ids": torch.tensor([1, 2, 4, 6]), "labels": torch.tensor([1, 2, 4, 6])},
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def tiny_qwen():
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
    )
    return transformers.Qwen3ForCausalLM(config)


def trainable_layer() -> arti.ARTILayer:
    return arti.ARTILayer(
        arti.mechanisms.AdaptivePulse(
            half=arti.Half(stochastic=False, learnable=True),
        )
    )


def test_transformers_trainer_uses_attachment_optimizer_and_artifact(tmp_path) -> None:
    base = tiny_qwen()
    base_state = copy.deepcopy(base.state_dict())
    model = arti.ARTI.attach(base, trainable_layer())
    session = model.arti.trainer(engine="transformers", objective="model_loss", learning_rate=1e-3)
    result = session.fit(
        TinyDataset(),
        steps=2,
        checkpoint_path=tmp_path / "trainer.arti.st",
        trainer_kwargs={
            "training_args": {
                "per_device_train_batch_size": 1,
                "logging_steps": 1,
                "disable_tqdm": True,
            }
        },
    )

    assert result.engine == "transformers"
    assert result.steps == 2
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert session.engine_object.__class__.__name__ == "AttachmentTrainer"
    restored_base = tiny_qwen()
    restored_base.load_state_dict(base_state)
    restored = arti.ARTI.load(
        restored_base,
        result.checkpoint_path,
        layer=trainable_layer(),
    )
    restored_session = restored.arti.trainer(engine="transformers", objective="model_loss")
    restored_session.load_checkpoint(result.checkpoint_path)
    assert restored_session.global_step == 2


def test_accelerate_engine_honors_gradient_accumulation_and_checkpoint(tmp_path) -> None:
    pytest.importorskip("accelerate")
    host = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.GELU(), torch.nn.Linear(8, 8))
    host = arti.ARTI.attach(host, trainable_layer(), layers=("0", "2"))
    x = torch.randn(2, 4, 8)
    target = torch.randn_like(x)
    loader = torch.utils.data.DataLoader([{"x": x[0], "target": target[0]}, {"x": x[1], "target": target[1]}], batch_size=1)

    def objective(model, batch):
        return torch.nn.functional.mse_loss(model(batch["x"]), batch["target"])

    session = host.arti.trainer(
        engine="accelerate",
        objective=objective,
        steps=2,
        gradient_accumulation_steps=2,
    )
    checkpoint = tmp_path / "accelerate.arti.st"
    result = session.fit(loader, checkpoint_path=checkpoint)

    assert result.steps == 2
    assert len(result.loss_history) == 2
    restored_base = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.GELU(), torch.nn.Linear(8, 8))
    restored = arti.ARTI.load(restored_base, checkpoint, layer=trainable_layer())
    restored_session = restored.arti.trainer(engine="accelerate", objective=objective, gradient_accumulation_steps=2)
    restored_session.load_checkpoint(checkpoint)
    assert restored_session.global_step == 2
    restored_session.fit(loader, steps=1)
    assert restored_session.global_step == 3
