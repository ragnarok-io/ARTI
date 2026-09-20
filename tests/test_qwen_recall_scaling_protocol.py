from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_scaling_protocol_is_locked_budget_matched_and_not_accuracy_claim() -> None:
    protocol = json.loads((ROOT / "benchmarks" / "qwen_recall_scaling_protocol.json").read_text(encoding="utf-8"))
    screen = ROOT / "benchmarks" / "results" / "recall_scaling_screen.json"
    assert hashlib.sha256(screen.read_bytes()).hexdigest().upper() == protocol["screen_sha256"]
    assert protocol["parameter_tolerance"] <= 0.05
    assert set(protocol["conditions"]) == {"frozen", "adapter", "lora", "recall-single", "recall-same-depth-2", "recall-multi-depth"}
    assert len(protocol["seeds"]) >= 3
    assert "not sufficient-training accuracy" in protocol["scope"]


def test_scaling_runner_is_resumable_bounded_and_retains_required_metrics() -> None:
    source = (ROOT / "benchmarks" / "run_qwen_recall_scaling.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--resume", action="store_true")' in source
    assert 'parser.add_argument("--refresh-conditions", action="store_true")' in source
    assert "args.max_runtime_seconds <= 540" in source
    for metric in (
        "recall_alignment_loss",
        "semantic_hidden_cosine",
        "unseen_hidden_delta_norm",
        "unseen_logit_kl",
        "tokens_per_second",
        "peak_cuda_bytes",
        "utilization",
        "generations",
    ):
        assert metric in source
    assert ".recall.arti.st" in source
