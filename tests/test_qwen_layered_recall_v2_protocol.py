from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks" / "qwen_layered_recall_v2_protocol.json"
DISCOVERY = ROOT / "benchmarks" / "results" / "qwen_layered_recall_v2_discovery.json"
PARTIAL = ROOT / "benchmarks" / "results" / "qwen_layered_recall_v2_partial_diagnostic.json"
PARTIAL_SHA256 = "4A51C63996883E408E3D84DD5C261339F33F0E40C349B7693B6EBD185199BC64"
SPEC = importlib.util.spec_from_file_location("run_qwen_layered_recall_v2", ROOT / "benchmarks" / "run_qwen_layered_recall_v2.py")
assert SPEC and SPEC.loader
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_v2_discovery_and_confirmation_are_disjoint_and_locked() -> None:
    raw = PROTOCOL.read_bytes()
    protocol = json.loads(raw)
    discovery = json.loads(DISCOVERY.read_text(encoding="utf-8"))
    discovery_prompts = {row["prompt"] for row in protocol["discovery_support"]}
    confirmation_prompts = {row["prompt"] for row in protocol["confirmation_support"]}

    assert set(protocol["discovery_seeds"]).isdisjoint(protocol["confirmation_seeds"])
    assert discovery_prompts.isdisjoint(confirmation_prompts)
    assert set(protocol["discovery_negatives"]).isdisjoint(protocol["confirmation_negatives"])
    assert discovery["protocol_sha256"] == hashlib.sha256(raw).hexdigest()
    assert discovery["selected_abstention_weight"] == 1.0


def test_embedding_erasure_preserves_attention_positions_and_token_count() -> None:
    batch = {
        "input_ids": torch.arange(16).reshape(1, 16),
        "attention_mask": torch.ones(1, 16, dtype=torch.long),
    }
    damaged, erase = RUN.erase_plan(batch, "single", seed=7, neutral_id=0)

    assert damaged["input_ids"].shape == batch["input_ids"].shape
    assert torch.equal(damaged["attention_mask"], batch["attention_mask"])
    assert torch.equal(damaged["input_ids"], batch["input_ids"])
    assert erase.any()
    assert not erase[:, :3].any()
    assert not erase[:, -3:].any()


def test_confirmation_defaults_enforce_short_resumable_batches() -> None:
    source = (ROOT / "benchmarks" / "run_qwen_layered_recall_v2.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--resume", action="store_true")' in source
    assert 'parser.add_argument("--max-conditions", type=int, default=4)' in source
    assert 'parser.add_argument("--max-runtime-seconds", type=float, default=540.0)' in source


def test_interrupted_fixed_topology_result_is_frozen_as_diagnostic_only() -> None:
    raw = PARTIAL.read_bytes()
    payload = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest().upper() == PARTIAL_SHA256
    assert payload["status"] == "running"
    assert len(payload["runs"]) == 11
    assert {run["seed"] for run in payload["runs"]} == {211, 223}
