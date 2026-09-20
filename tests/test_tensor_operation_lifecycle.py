from pathlib import Path

import pytest
import torch

from benchmarks.run_tensor_operation_lifecycle import (
    build_fixture,
    evaluate_backing,
    fork_committed_backing,
    load_committed_backing,
    run_arm,
    run_lifecycle,
    save_committed_backing,
)


def test_persistent_arm_carries_each_committed_root_to_the_next_call() -> None:
    fixture = build_fixture()
    result = run_arm(fixture, tuple(range(4)) * 2, arm="persistent")

    assert result.port.step_index == 8
    assert result.port.resolve().mask.all()
    assert all(record["route"] == record["event"] for record in result.records)
    assert all(
        current["before"] == previous["after"]
        for previous, current in zip(result.records, result.records[1:])
    )
    assert all(record["changed_slots"] == 1 for record in result.records)


def test_frozen_and_reset_controls_execute_but_do_not_accumulate() -> None:
    fixture = build_fixture()
    order = tuple(range(4)) * 2
    persistent = run_arm(fixture, order, arm="persistent")
    frozen = run_arm(fixture, order, arm="frozen")
    reset = run_arm(fixture, order, arm="reset")

    assert len(persistent.records) == len(frozen.records) == len(reset.records) == 8
    assert all(record["proposal"] != record["before"] for record in frozen.records)
    assert not frozen.port.resolve().mask.any()
    assert not reset.port.resolve().mask.any()
    assert not torch.equal(
        evaluate_backing(fixture, persistent.port),
        evaluate_backing(fixture, frozen.port),
    )


def test_checkpoint_reload_fork_and_reset_are_exact(tmp_path: Path) -> None:
    fixture = build_fixture()
    persistent = run_arm(fixture, tuple(range(4)) * 3, arm="persistent")
    checkpoint = tmp_path / "root.arti.st"
    saved = save_committed_backing(persistent.port, checkpoint)
    reloaded = load_committed_backing(fixture.spec, checkpoint)
    fork = fork_committed_backing(reloaded)

    assert saved["digest"]
    assert reloaded.step_index == persistent.port.step_index
    torch.testing.assert_close(reloaded.resolve().value, persistent.port.resolve().value)
    assert torch.equal(reloaded.resolve().mask, persistent.port.resolve().mask)

    event = fixture.worlds[0] * 99
    proposal = fixture.invocation(event, fork.resolve()).operation
    fork.advance(proposal.value, proposal.mask)
    assert not torch.equal(fork.resolve().value, reloaded.resolve().value)
    torch.testing.assert_close(reloaded.resolve().value, persistent.port.resolve().value)

    reset = load_committed_backing(fixture.spec, checkpoint)
    torch.testing.assert_close(reset.resolve().value, persistent.port.resolve().value)
    assert torch.equal(reset.resolve().mask, persistent.port.resolve().mask)


def test_lifecycle_report_requires_longitudinal_causal_controls(tmp_path: Path) -> None:
    report = run_lifecycle(tmp_path, calls=64)

    assert report["passed"] is True
    assert report["calls"] == 64
    assert len(report["records"]) == 64
    assert all(report["checks"].values())
    assert (tmp_path / "committed-final.arti.st").is_file()
    assert (tmp_path / "report.json").is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_lifecycle_smoke_runs_on_cuda(tmp_path: Path) -> None:
    report = run_lifecycle(tmp_path, calls=8, device="cuda")

    assert report["passed"] is True
    assert report["device"] == "cuda"
