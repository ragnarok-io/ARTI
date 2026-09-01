from __future__ import annotations

import copy

import pytest
import torch

import arti


transformers = pytest.importorskip("transformers")


class Dataset(torch.utils.data.Dataset):
    rows = (
        {"input_ids": torch.tensor([1, 3, 5, 7]), "labels": torch.tensor([1, 3, 5, 7])},
        {"input_ids": torch.tensor([1, 2, 4, 6]), "labels": torch.tensor([1, 2, 4, 6])},
    )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def tiny_qwen():
    return transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
            pad_token_id=0,
            use_cache=False,
        )
    )


def trainable_layer() -> arti.ARTILayer:
    return arti.ARTILayer(
        arti.mechanisms.AdaptivePulse(
            half=arti.Half(stochastic=False, learnable=True),
        )
    )


def test_trainer_rotation_writes_only_arti_and_resumes_exactly(tmp_path, paired_rng) -> None:
    assert issubclass(arti.ARTICheckpointCallback, transformers.TrainerCallback)
    torch.manual_seed(71)
    base = tiny_qwen()
    base_state = copy.deepcopy(base.state_dict())
    base.gradient_checkpointing_enable()
    model = arti.ARTI.attach(base, trainable_layer())
    session = model.arti.trainer(engine="transformers", objective="model_loss", learning_rate=1e-3)
    output_dir = tmp_path / "trainer"
    session.fit(
        Dataset(),
        steps=3,
        trainer_kwargs={
            "training_args": {
                "per_device_train_batch_size": 1,
                "logging_steps": 1,
                "save_strategy": "steps",
                "save_steps": 1,
                "save_total_limit": 2,
                "lr_scheduler_type": "constant",
                "disable_tqdm": True,
                "output_dir": str(output_dir),
            }
        },
    )
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    assert [path.name for path in checkpoints] == ["checkpoint-2", "checkpoint-3"]
    for checkpoint in checkpoints:
        assert (checkpoint / "model.arti.st").exists()
        assert (checkpoint / "arti-checkpoint.json").exists()
        assert not (checkpoint / "model.safetensors").exists()
        assert not (checkpoint / "pytorch_model.bin").exists()
        assert not (checkpoint / "optimizer.pt").exists()

    device = next(model.parameters()).device
    input_ids = torch.tensor([[1, 4, 7]], device=device)
    model.eval()
    restored_base = tiny_qwen()
    restored_base.load_state_dict(base_state)
    restored_base.gradient_checkpointing_enable()
    restored = arti.ARTI.load(
        restored_base,
        checkpoints[-1] / "model.arti.st",
        layer=trainable_layer(),
    )
    restored.to(device)
    restored.eval()
    with torch.no_grad():
        expected, actual = paired_rng(
            lambda: model(input_ids=input_ids).logits,
            lambda: restored(input_ids=input_ids).logits,
        )
    assert torch.equal(expected, actual)

    resumed = restored.arti.trainer(
        engine="transformers",
        objective="model_loss",
        learning_rate=1e-3,
        resume_from_checkpoint=checkpoints[-1],
    )
    resumed.fit(
        Dataset(),
        steps=1,
        trainer_kwargs={
            "training_args": {
                "per_device_train_batch_size": 1,
                "lr_scheduler_type": "constant",
                "disable_tqdm": True,
                "output_dir": str(tmp_path / "resumed"),
            }
        },
    )
    assert resumed.global_step == 4
