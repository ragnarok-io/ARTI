from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from benchmarks._managed_training import (
    CHECKPOINT_FORMAT,
    ManagedRunConfig,
    ManagedTrainingRun,
    PerformancePolicy,
    acceleration_capabilities,
    managed_training_entrypoint,
)
from benchmarks.run_qwen_vocab_recall_updater_pretraining import (
    VocabularySequenceStream,
)


@pytest.fixture(autouse=True)
def _isolate_accelerate_process_state():
    from accelerate.state import AcceleratorState

    AcceleratorState._reset_state(reset_partial_state=True)
    yield
    AcceleratorState._reset_state(reset_partial_state=True)


def _config(path: Path, *, revision: int = 1) -> ManagedRunConfig:
    return ManagedRunConfig(
        output_dir=path,
        contract={"experiment": "fault-injection", "revision": revision},
        performance=PerformancePolicy(device="cpu", precision="no"),
        checkpoint_interval_seconds=60.0,
        keep_checkpoints=2,
        gradient_clip_norm=1.0,
    )


def test_managed_run_rejects_excessive_finite_gradient_before_step(
    tmp_path: Path,
) -> None:
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    run = ManagedTrainingRun(
        ManagedRunConfig(
            output_dir=tmp_path,
            contract={"experiment": "gradient-limit"},
            performance=PerformancePolicy(device="cpu", precision="no"),
            gradient_clip_norm=1.0,
            max_preclip_gradient_norm=10.0,
        ),
        trainable=model,
        optimizer=optimizer,
    )
    before = model.weight.detach().clone()
    loss = run.prepared_trainable(torch.tensor([[100.0]])).square().sum()

    with pytest.raises(RuntimeError, match="safety limit"):
        run.backward_and_step(loss)

    torch.testing.assert_close(model.weight, before)


def _make_run(path: Path, *, revision: int = 1):
    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    stream = VocabularySequenceStream(17, seed=71)
    run = ManagedTrainingRun(
        _config(path, revision=revision),
        trainable=model,
        optimizer=optimizer,
        scheduler=scheduler,
        stateful={"vocabulary_stream": stream},
    )
    return run, stream


def _step(run: ManagedTrainingRun, x: torch.Tensor) -> float:
    target = torch.tensor([[0.25, -0.5, 0.75]])
    prediction = run.prepared_trainable(x)
    loss = (prediction - target).square().mean()
    norm = run.backward_and_step(loss)
    assert norm is not None
    run.record_step(examples=1, tokens=4)
    return float(loss.detach())


def _assert_state_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_state_equal(left_item, right_item)
    else:
        assert left == right


def test_accumulated_gradient_step_matches_joint_backward(tmp_path: Path) -> None:
    torch.manual_seed(117)
    joint, _ = _make_run(tmp_path / "joint")
    grouped, _ = _make_run(tmp_path / "grouped")
    grouped.trainable.load_state_dict(joint.trainable.state_dict())
    inputs = (torch.randn(1, 4), torch.randn(1, 4))
    target = torch.tensor([[.25, -.5, .75]])
    joint_norm = joint.backward_and_step(sum(
        (joint.prepared_trainable(x) - target).square().mean() / 2 for x in inputs))
    for x in inputs:
        ((grouped.prepared_trainable(x) - target).square().mean() / 2).backward()
    grouped_norm = grouped.step_accumulated_gradients()
    assert grouped_norm == pytest.approx(joint_norm, rel=1e-6)
    torch.testing.assert_close(grouped.trainable.state_dict(), joint.trainable.state_dict())
    torch.testing.assert_close(grouped.optimizer.state_dict(), joint.optimizer.state_dict())
    _assert_state_equal(grouped.scheduler.state_dict(), joint.scheduler.state_dict())
    assert all(parameter.grad is None for parameter in grouped.trainable.parameters())


def test_same_step_checkpoint_saves_new_custom_state(tmp_path: Path) -> None:
    run, stream = _make_run(tmp_path)
    first = run.checkpoint(reason="before-evaluation")
    stream.next(2, 4)
    second = run.checkpoint(reason="after-evaluation")
    assert first != second
    expected = stream.next(2, 4)
    restored, restored_stream = _make_run(tmp_path)
    assert restored.resume() == second.resolve()
    torch.testing.assert_close(restored_stream.next(2, 4), expected, rtol=0, atol=0)
    assert restored.progress.step == 0


def test_managed_checkpoint_restores_exact_training_trajectory(tmp_path: Path) -> None:
    torch.manual_seed(3107)
    first, first_stream = _make_run(tmp_path)
    first_stream.next(2, 3)
    _step(first, torch.randn(1, 4))
    checkpoint = first.checkpoint(reason="fault-injection")

    expected_input = torch.randn(1, 4)
    expected_tokens = first_stream.next(2, 4)
    expected_loss = _step(first, expected_input)
    expected_model = {
        name: tensor.detach().clone()
        for name, tensor in first.trainable.state_dict().items()
    }
    expected_optimizer = first.optimizer.state_dict()
    expected_scheduler = first.scheduler.state_dict()

    torch.manual_seed(9999)
    restored, restored_stream = _make_run(tmp_path)
    assert restored.resume(checkpoint) == checkpoint.resolve()
    actual_input = torch.randn(1, 4)
    actual_tokens = restored_stream.next(2, 4)
    actual_loss = _step(restored, actual_input)

    torch.testing.assert_close(actual_input, expected_input, rtol=0, atol=0)
    torch.testing.assert_close(actual_tokens, expected_tokens, rtol=0, atol=0)
    assert actual_loss == expected_loss
    _assert_state_equal(restored.trainable.state_dict(), expected_model)
    _assert_state_equal(restored.optimizer.state_dict(), expected_optimizer)
    _assert_state_equal(restored.scheduler.state_dict(), expected_scheduler)
    assert restored.progress.step == 2
    assert restored.progress.examples == 2
    assert restored.progress.tokens == 8


def test_checkpoint_manifest_and_latest_pointer_are_explicit(tmp_path: Path) -> None:
    run, _stream = _make_run(tmp_path)
    _step(run, torch.ones(1, 4))
    checkpoint = run.checkpoint(reason="interval")
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))

    assert manifest["format"] == CHECKPOINT_FORMAT
    assert manifest["trusted_internal_state"] is True
    assert manifest["stateful_names"] == ["vocabulary_stream"]
    assert latest["checkpoint"] == checkpoint.relative_to(tmp_path).as_posix()
    assert run.resume() == checkpoint.resolve()


def test_resume_rejects_contract_drift_and_corruption(tmp_path: Path) -> None:
    run, _stream = _make_run(tmp_path)
    _step(run, torch.ones(1, 4))
    checkpoint = run.checkpoint(reason="interval")

    incompatible, _ = _make_run(tmp_path, revision=2)
    with pytest.raises(ValueError, match="contract"):
        incompatible.resume(checkpoint)

    state_file = next(
        path
        for path in checkpoint.iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    state_file.write_bytes(state_file.read_bytes() + b"corrupt")
    restored, _ = _make_run(tmp_path)
    with pytest.raises(ValueError, match="integrity"):
        restored.resume(checkpoint)


def test_exception_checkpoint_preserves_last_finite_state(tmp_path: Path) -> None:
    run, _stream = _make_run(tmp_path)

    @managed_training_entrypoint
    def fail(active: ManagedTrainingRun) -> None:
        _step(active, torch.ones(1, 4))
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        fail(run)
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["format"] == CHECKPOINT_FORMAT


def test_checkpoint_failure_does_not_mask_training_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, _stream = _make_run(tmp_path)

    def fail_checkpoint(*, reason: str) -> Path:
        raise OSError(f"injected checkpoint failure during {reason}")

    monkeypatch.setattr(run, "checkpoint", fail_checkpoint)

    @managed_training_entrypoint
    def fail(active: ManagedTrainingRun) -> None:
        _step(active, torch.ones(1, 4))
        raise RuntimeError("original training failure")

    with pytest.raises(RuntimeError, match="original training failure") as captured:
        fail(run)
    assert any("checkpoint failed" in note for note in captured.value.__notes__)


def test_nonfinite_loss_never_updates_or_checkpoints(tmp_path: Path) -> None:
    run, _stream = _make_run(tmp_path)
    before = {
        name: tensor.detach().clone() for name, tensor in run.trainable.state_dict().items()
    }
    with pytest.raises(RuntimeError, match="non-finite loss"):
        run.backward_and_step(torch.tensor(float("nan"), requires_grad=True))
    _assert_state_equal(run.trainable.state_dict(), before)


def test_performance_policy_is_explicit_and_non_substituting() -> None:
    policy = PerformancePolicy(
        precision="bf16",
        attention_backend="sdpa",
        compile_mode="none",
    )
    assert policy.transformers_load_kwargs() == {
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }
    capabilities = acceleration_capabilities()
    assert set(capabilities) == {
        "accelerate",
        "cuda",
        "bf16",
        "flash_sdpa",
        "liger",
        "torch_compile",
    }
    with pytest.raises(ValueError, match="attention_backend"):
        PerformancePolicy(attention_backend="silent-fallback")


def test_expensive_qwen_loop_uses_managed_lifecycle() -> None:
    source = Path("benchmarks/run_qwen_values_only_affine_ttt.py").read_text(
        encoding="utf-8"
    )
    assert "@managed_training_entrypoint" in source
    assert "ManagedTrainingRun(" in source
    assert "managed.backward_and_step(loss)" in source
    assert "loss.backward()" not in source
    assert "optimizer.step()" not in source
