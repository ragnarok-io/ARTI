from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_training_smoke_is_declarative_local_bounded_and_resumable() -> None:
    runner = (ROOT / "benchmarks" / "run_qwen_unified_training.py").read_text(encoding="utf-8")
    protocol = (ROOT / "benchmarks" / "qwen_unified_training.toml").read_text(encoding="utf-8")

    assert "local_files_only=True" in runner
    assert "max-runtime-seconds" in runner
    assert "ARTI.attach(model, layer=layer, config=ATTACH_CONFIG)" in runner
    assert "AdaptivePulse" in runner
    assert ".write_lock(" in runner
    assert ".save_checkpoint" not in runner  # fit(checkpoint_path=...) exercises the public path
    assert ".load_checkpoint(" in runner
    assert "objective = \"model_loss\"" in protocol
    assert "gradient_accumulation_steps = 2" in protocol
    assert "mixed_precision = \"bf16\"" in protocol
