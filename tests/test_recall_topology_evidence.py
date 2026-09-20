from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results" / "qwen_recall_topology_confirmation.json"
REPORT = ROOT / "benchmarks" / "results" / "qwen_recall_topology_confirmation.md"
PROTOCOL = ROOT / "benchmarks" / "qwen_recall_topology_confirmation_protocol.json"
SPEC = importlib.util.spec_from_file_location(
    "verify_qwen_recall_topology_confirmation", ROOT / "benchmarks" / "verify_qwen_recall_topology_confirmation.py"
)
assert SPEC and SPEC.loader
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def test_open_topology_confirmation_evidence_is_complete() -> None:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    verified = VERIFY.verify(payload, PROTOCOL.read_bytes())

    assert verified["evidence_passed"] is True
    assert verified["promotion_passed"] is False
    assert len(payload["runs"]) == 18
    assert all(len(run["generations"]) == 44 for run in payload["runs"])
    assert all(run["summary"]["coherence_rate"] == 1.0 for run in payload["runs"])
    assert REPORT.is_file()


def test_open_topology_findings_preserve_repair_and_accuracy_boundary() -> None:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    findings = VERIFY.verify(payload, PROTOCOL.read_bytes())["findings"]

    assert findings["all_candidates_nll_better_than_frozen_every_seed"] is True
    assert findings["all_candidates_semantic_better_than_frozen_every_seed"] is True
    assert findings["multi_line_frontier_better_than_matched_single_every_seed"] is True
    assert findings["path_removals_degrade_every_control"] is True
    assert findings["repeated_line_removals_degrade_every_seed"] is True
    assert findings["compatible_artifact_reorders_degrade"] is True
    assert findings["task_superiority_candidates"] == []
    assert set(findings["selective_candidates"]) == {
        "single-2-half1-alignment-u4",
        "uniform-2-half0-alignment-u1",
    }
