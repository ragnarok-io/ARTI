from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_hub_smoke_is_local_bounded_independent_and_resumable() -> None:
    source = (ROOT / "benchmarks" / "run_qwen_hub_lifecycle.py").read_text(encoding="utf-8")
    assert "local_files_only=True" in source
    assert "max-runtime-seconds" in source
    assert "snapshot_download" in source
    assert "ARTI.from_pretrained" in source
    assert "save_pretrained" in source
    assert "resume_from_checkpoint=True" in source
    assert "not a task-quality benchmark" in source
