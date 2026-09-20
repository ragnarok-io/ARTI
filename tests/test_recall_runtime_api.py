import torch

import arti


def _runtime() -> arti.mechanisms.RecallRuntime:
    updater = arti.mechanisms.NormalizedDeltaRecallValueUpdater(
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        factors=2,
    )
    reader = arti.Recall(4, 3, formula="arti/delta@1")
    return arti.mechanisms.RecallRuntime(updater, reader)


def test_runtime_keeps_writer_and_dynamic_state_separate() -> None:
    runtime = _runtime()
    hidden = torch.randn(2, 5, 4)
    trace = torch.randn(2, 6, 4)
    state = runtime.initial_state(2, reference=hidden)
    next_state = runtime.write(
        trace,
        state,
        mask=torch.ones(2, 6, dtype=torch.bool),
        detach_state=True,
    )

    assert state.step == 0
    assert next_state.step == 1
    assert next_state.value.shape == (2, 3, 4)
    assert next_state.value.device == hidden.device
    assert next_state.value.dtype == hidden.dtype
    assert not next_state.value.requires_grad
    assert not state.value.requires_grad


def test_runtime_reads_external_state_through_recall_formula() -> None:
    runtime = _runtime()
    hidden = torch.randn(2, 5, 4)
    trace = torch.randn(2, 6, 4)
    state = runtime.initial_state(2, reference=hidden)
    output, next_state = runtime(
        hidden,
        trace,
        state,
        trace_mask=torch.ones(2, 6, dtype=torch.bool),
        detach_state=True,
        mask=torch.ones(2, 5, dtype=torch.bool),
    )

    assert output.shape == hidden.shape
    assert next_state.step == 1
    torch.testing.assert_close(
        output,
        runtime.reader(
            hidden,
            memory=next_state.value,
            mask=torch.ones(2, 5, dtype=torch.bool),
        ),
    )


def test_recall_state_safetensors_compatible_payload_round_trip() -> None:
    state = arti.mechanisms.RecallState.zeros(2, 3, 4, dtype=torch.float32).advance(
        torch.randn(2, 3, 4)
    )
    restored = arti.mechanisms.RecallState.from_state_dict(state.state_dict())
    assert restored.step == state.step
    torch.testing.assert_close(restored.value, state.value)


def test_recall_state_lifecycle_stack_fork_reset_and_file_round_trip(tmp_path) -> None:
    runtime = _runtime()
    original = runtime.initial_state(3, dtype=torch.float32).advance(torch.randn(3, 3, 4))
    parts = original.unstack()
    stacked = arti.mechanisms.RecallState.stack(parts)
    forked = stacked.fork()
    reset = forked.reset()
    path = tmp_path / "session.recall.arti.st"
    forked.save(path)
    restored = arti.mechanisms.RecallState.load(path)

    torch.testing.assert_close(stacked.value, original.value)
    torch.testing.assert_close(restored.value, forked.value)
    assert restored.step == forked.step == 1
    assert restored.contract_fingerprint == runtime.contract_fingerprint
    assert reset.step == 0
    assert torch.count_nonzero(reset.value).item() == 0
    assert reset.contract_fingerprint == runtime.contract_fingerprint
    assert forked.value.data_ptr() != stacked.value.data_ptr()


def test_recall_state_stack_rejects_mixed_steps_or_contracts() -> None:
    first = arti.mechanisms.RecallState(torch.zeros(3, 4), step=0)
    later = arti.mechanisms.RecallState(torch.zeros(3, 4), step=1)

    try:
        arti.mechanisms.RecallState.stack((first, later))
    except ValueError as error:
        assert "layout, step, and contract" in str(error)
    else:
        raise AssertionError("mixed-step states should not be stacked")


def test_runtime_scan_preserves_order_and_returns_forks() -> None:
    runtime = _runtime()
    traces = torch.randn(2, 3, 4, 4)
    state = runtime.initial_state(2, dtype=torch.float32)
    final, snapshots = runtime.scan(
        traces,
        state,
        order=(2, 0, 1),
        return_snapshots=True,
        detach_state=True,
    )

    serial = state
    for index in (2, 0, 1):
        serial = runtime.update(traces[:, index], serial, detach_state=True)
    torch.testing.assert_close(final.value, serial.value)
    assert snapshots.shape == (2, 3, 3, 4)
    assert final.step == 3
    assert state.step == 0


def test_recall_rejects_external_memory_with_wrong_shape() -> None:
    reader = arti.Recall(4, 3, formula="arti/delta@1")
    with torch.no_grad():
        try:
            reader(torch.randn(2, 5, 4), memory=torch.randn(2, 2, 4))
        except ValueError as error:
            assert "memory" in str(error)
        else:
            raise AssertionError("invalid external memory should be rejected")


def test_runtime_rejects_writer_reader_dimension_mismatch() -> None:
    updater = arti.mechanisms.NormalizedDeltaRecallValueUpdater(
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
    )
    reader = arti.Recall(4, 2, formula="arti/delta@1")
    try:
        arti.mechanisms.RecallRuntime(updater, reader)
    except ValueError as error:
        assert "slots" in str(error)
    else:
        raise AssertionError("mismatched writer and reader dimensions should be rejected")
