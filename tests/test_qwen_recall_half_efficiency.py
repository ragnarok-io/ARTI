from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

from arti.fit import resolve_fit_config_mechanism


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_qwen_recall_half_efficiency", ROOT / "benchmarks" / "verify_qwen_recall_half_efficiency.py")
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
assert SPEC.loader is not None
SPEC.loader.exec_module(module)

RUNNER_SPEC = importlib.util.spec_from_file_location("run_qwen_recall_half_efficiency", ROOT / "benchmarks" / "run_qwen_recall_half_efficiency.py")
runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = runner
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(runner)


def test_verifier_rejects_missing_required_variant() -> None:
    payload = {"status": "completed", "scope": "Qwen3-0.6B Recall + Half reproducible adaptation benchmark", "config": {"model_id": "Qwen/Qwen3-0.6B", "seeds": [1, 2, 3], "device": "cuda"}, "provenance": {}, "dataset": {}, "fairness": {}, "runs": [], "promotion": {}}
    failures = module.verify(payload)
    assert any("missing variants" in failure for failure in failures)
    assert any("provenance missing" in failure for failure in failures)


def test_committed_benchmark_artifact_passes_without_local_checkpoint() -> None:
    payload = json.loads((ROOT / "benchmarks" / "results" / "qwen_recall_half_efficiency.json").read_text(encoding="utf-8"))
    assert module.verify(payload) == []


def test_dataset_hash_is_sensitive_to_records() -> None:
    payload = {"records": {"train": [{"prompt": "a", "answer": "b"}], "validation": []}}
    original = module._dataset_hash(payload)
    payload["records"]["train"][0]["answer"] = "c"
    assert module._dataset_hash(payload) != original


def test_save_trainable_weights_excludes_frozen_tensors(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    artifact = runner.save_trainable_weights(model, tmp_path / "adapter.safetensors")
    assert Path(artifact["path"]).exists()
    assert artifact["tensor_count"] == 1


def test_recall_activation_can_be_disabled_only_as_an_explicit_config_choice() -> None:
    _, scale = resolve_fit_config_mechanism(
        {"fit": {"scale": "tiny"}, "mechanism": {"recall_steps": 1, "recall_activation": "none"}}
    )
    assert scale.recall_steps == 1
    assert scale.recall_activation == "none"
