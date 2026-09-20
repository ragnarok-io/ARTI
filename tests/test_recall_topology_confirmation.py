from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import arti


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks" / "qwen_recall_topology_confirmation_protocol.json"
SCREEN = ROOT / "benchmarks" / "results" / "qwen_recall_topology_screen.json"
CACHE = ROOT / "benchmarks" / "artifacts" / "qwen_layered_recall_hidden_traces.safetensors"
SPEC = importlib.util.spec_from_file_location(
    "run_qwen_recall_topology_confirmation", ROOT / "benchmarks" / "run_qwen_recall_topology_confirmation.py"
)
assert SPEC and SPEC.loader
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_confirmation_inputs_are_hash_locked_and_seeds_are_new() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    previous = {17, 31, 53, 101, 113, 211, 223, 227}

    assert protocol["screen_sha256"] == hashlib.sha256(SCREEN.read_bytes()).hexdigest().upper()
    assert protocol["trace_cache_sha256"] == hashlib.sha256(CACHE.read_bytes()).hexdigest().upper()
    assert previous.isdisjoint(protocol["confirmation_seeds"])
    assert protocol["batch_policy"]["max_runtime_seconds"] == 540
    assert protocol["batch_policy"]["max_conditions"] == 6


def test_selected_candidates_decode_to_distinct_open_topologies() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    candidates = [RUN.decode_candidate(screen["candidates"][name]) for name in protocol["selected_candidates"]]

    assert len(candidates) == 5
    assert len({candidate.name for candidate in candidates}) == 5
    assert any(any(spec.copies > 1 for spec in candidate.config.layers) for candidate in candidates)
    assert {len(candidate.config.layers) for candidate in candidates}.issuperset({1, 2, 5})
    assert all(22000 <= arti.estimate_layered_recall_cost(candidate).parameters <= 26000 for candidate in candidates)


def test_confirmation_runner_has_hard_batch_limits_and_resume() -> None:
    source = (ROOT / "benchmarks" / "run_qwen_recall_topology_confirmation.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--resume", action="store_true")' in source
    assert 'parser.add_argument("--max-runtime-seconds", type=float, default=540.0)' in source
    assert 'parser.add_argument("--max-conditions", type=int, default=6)' in source
