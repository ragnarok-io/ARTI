from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks" / "qwen_layered_recall_protocol.json"
RESULTS = ROOT / "benchmarks" / "results" / "qwen_layered_recall.json"
SPEC = importlib.util.spec_from_file_location("verify_qwen_layered_recall", ROOT / "benchmarks" / "verify_qwen_layered_recall.py")
assert SPEC and SPEC.loader
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def test_protocol_separates_training_support_from_queries_and_answers() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    support = set(protocol["support_prompts"])
    query = set(protocol["query_prompts"])
    negatives = set(protocol["recognition_negative_prompts"])

    assert support.isdisjoint(query)
    assert support.isdisjoint(negatives)
    assert query.isdisjoint(negatives)
    assert protocol["generation"] == {"do_sample": False, "enable_thinking": False}
    assert protocol["layer_indices"] == [6, 13, 20]
    assert protocol["seeds"] == [17, 31, 53]


def test_verifier_rejects_incomplete_results() -> None:
    failures = VERIFY.verify({"runs": [], "training_signal_policy": {}}, PROTOCOL.read_bytes())
    assert any("protocol hash" in failure for failure in failures)
    assert any("missing preregistered runs" in failure for failure in failures)


def test_real_qwen_evidence_is_complete_and_preserves_failed_gates() -> None:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    failures = VERIFY.verify(payload, PROTOCOL.read_bytes())

    assert payload["status"] == "completed"
    assert len(payload["runs"]) == 21
    assert payload["runtime"]["device"] == "NVIDIA GeForce RTX 5070 Ti"
    assert all(len(run["generations"]) == 48 for run in payload["runs"])
    assert all(run["summary"]["coherence_rate"] == 1.0 for run in payload["runs"])
    assert all(run["summary"]["generated_tokens_per_second"] > 0 for run in payload["runs"])
    assert all(-1.0 <= run["summary"]["support_semantic_cosine"] <= 1.0 for run in payload["runs"])
    assert all(all("semantic_cosine" in row for row in run["generations"]) for run in payload["runs"])
    assert any("does not beat every single-layer" in failure for failure in failures)
    assert any("marginal improvement is not strictly positive" in failure for failure in failures)
    assert not any("artifact" in failure for failure in failures)
