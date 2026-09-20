from __future__ import annotations

import json
from pathlib import Path

from benchmarks.verify_qwen_literal_output_context import verify


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_literal_output_context_evidence_passes() -> None:
    payload = json.loads((ROOT / "benchmarks/results/qwen_literal_output_context_results.json").read_text(encoding="utf-8"))
    assert verify(payload) == []


def test_qwen_literal_output_context_rejects_missing_claim_boundary() -> None:
    payload = json.loads((ROOT / "benchmarks/results/qwen_literal_output_context_results.json").read_text(encoding="utf-8"))
    payload["claim_boundary"] = "too broad"
    assert verify(payload)
