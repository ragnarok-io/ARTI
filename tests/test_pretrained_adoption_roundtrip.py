from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import arti


def _torch_workflow(model: nn.Module, sample: torch.Tensor) -> arti.ARTIPretrained:
    workflow = arti.pretrained(model, provider="torch")
    workflow.scan(sample).plan(
        where="0",
        scale="tiny",
        max_adapters=1,
        training={"engine": "torch", "steps": 1},
    )
    return workflow.apply()


def test_torch_fit_export_fresh_process_and_detach_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(71)
    model = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
    initial_state = copy.deepcopy(model.state_dict())
    sample = torch.randn(2, 4)
    original = model(sample).detach()

    workflow = _torch_workflow(model, sample)
    target = original + 0.25
    result = workflow.fit([(sample, target)])
    assert result.steps == 1
    exported = workflow.export(tmp_path / "torch.arti.st")
    adapted = workflow.model(sample).detach()
    assert not torch.equal(adapted, original)

    restored_model = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
    restored_model.load_state_dict(initial_state)
    restored = _torch_workflow(restored_model, sample)
    restored.load_weights(exported.saved.weights_path)
    assert torch.allclose(restored.model(sample), adapted)

    child = r'''
import json
import sys
import torch
import torch.nn as nn
import arti

model = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
model.load_state_dict(torch.load(sys.argv[1], map_location="cpu", weights_only=True))
sample = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
workflow = (
    arti.pretrained(model, provider="torch")
)
workflow.scan(sample).plan(where="0", scale="tiny", max_adapters=1,
                           training={"engine": "torch", "steps": 1})
workflow.apply()
workflow.load_weights(sys.argv[3])
print(json.dumps({"sum": float(workflow.model(sample).sum())}))
'''
    state_path = tmp_path / "base-state.pt"
    sample_path = tmp_path / "sample.pt"
    torch.save(initial_state, state_path)
    torch.save(sample, sample_path)
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            child,
            str(state_path),
            str(sample_path),
            str(exported.saved.weights_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    child_sum = json.loads(completed.stdout)["sum"]
    assert child_sum == pytest.approx(float(adapted.sum()), abs=1e-6)

    detached = workflow.detach()
    assert detached is model
    assert isinstance(model[0], nn.Linear)
    assert torch.equal(model(sample), original)
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert workflow.doctor()["applied"] is False


def _qwen_config(transformers):
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


def _qwen_workflow(model, batch):
    workflow = arti.pretrained(model, provider="transformers", task="causal-lm")
    workflow.scan(batch).plan(
        where="model.layers.0.mlp.down_proj",
        scale="tiny",
        max_adapters=1,
        training={"engine": "torch", "steps": 1},
    )
    return workflow.apply()


def test_transformers_qwen_fit_generate_export_and_detach(tmp_path: Path) -> None:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(73)
    base = transformers.Qwen3ForCausalLM(_qwen_config(transformers))
    initial = copy.deepcopy(base.state_dict())
    batch = {
        "input_ids": torch.tensor([[1, 7, 11, 13], [1, 5, 17, 19]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
    }
    workflow = _qwen_workflow(base, batch)

    def step(model, values):
        output = model(**values)
        return F.cross_entropy(
            output.logits[:, :-1].reshape(-1, output.logits.shape[-1]),
            values["input_ids"][:, 1:].reshape(-1),
        )

    workflow.fit([batch], step_fn=step)
    workflow.model.eval()
    with torch.no_grad():
        expected_logits = workflow.model(**batch).logits
        expected_tokens = workflow.generate(batch["input_ids"][:1], max_new_tokens=1, do_sample=False)
    exported = workflow.export(tmp_path / "qwen.arti.st")

    restored_model = transformers.Qwen3ForCausalLM(_qwen_config(transformers))
    restored_model.load_state_dict(initial)
    restored = _qwen_workflow(restored_model, batch)
    restored.load_weights(exported.saved.weights_path)
    restored.model.eval()
    with torch.no_grad():
        actual_logits = restored.model(**batch).logits
        actual_tokens = restored.generate(batch["input_ids"][:1], max_new_tokens=1, do_sample=False)

    assert torch.equal(expected_logits, actual_logits)
    assert torch.equal(expected_tokens, actual_tokens)
    assert isinstance(workflow.detach(), transformers.Qwen3ForCausalLM)
    assert not any(module.__class__.__name__ == "ARTIAdapterWrapper" for module in base.modules())
    assert callable(base.generate)
