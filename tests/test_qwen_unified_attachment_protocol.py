from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_real_qwen_attachment_runner_is_local_bounded_and_roundtrip_complete() -> None:
    source = (ROOT / "benchmarks" / "run_qwen_unified_attachment.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "local_files_only=True" in source
    assert "max-runtime-seconds" in source
    assert "ARTI.preview" in source
    assert "ARTI.attach" in source
    assert ".arti.save" in source
    assert ".arti.detach" in source
    assert "ARTI.load" in source
    assert any(isinstance(node, ast.Raise) for node in ast.walk(tree))
