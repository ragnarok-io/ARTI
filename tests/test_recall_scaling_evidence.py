from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_recall_scaling", ROOT / "benchmarks" / "verify_recall_scaling_evidence.py")
assert SPEC and SPEC.loader
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def test_committed_recall_scaling_evidence_passes() -> None:
    result = verifier.verify()
    assert result["passed"], result["failures"]


def test_scaling_report_explicitly_rejects_short_run_accuracy_claim() -> None:
    source = (ROOT / "benchmarks" / "render_recall_scaling_report.py").read_text(encoding="utf-8")
    assert "did not converge" in source
    assert "No task-accuracy superiority claim is made" in source
