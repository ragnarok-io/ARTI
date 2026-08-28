from __future__ import annotations

import copy

import pytest
import torch

import arti
from arti._recall_state import RECALL_STATE_SCHEMA_VERSION


def _runtime(*, updater_factors: int = 2) -> arti.alpha.RecallRuntime:
    updater = arti.alpha.NormalizedDeltaRecallValueUpdater(
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        factors=updater_factors,
    )
    reader = arti.Recall(4, 3, formula="arti/delta@1")
    return arti.alpha.RecallRuntime(updater, reader)


def test_runtime_contract_binds_reader_updater_and_state_schema() -> None:
    runtime = _runtime()
    contract = runtime.contract
    restored = arti.alpha.RecallRuntimeContract.from_dict(contract.to_dict())

    assert restored == contract
    assert len(contract.fingerprint) == 64
    assert contract.reader["ref"] == "arti/recall@4"
    assert contract.updater["ref"] == "arti/normalized-updater@1"
    assert contract.formula["reference"] == "arti/delta@1"
    assert contract.bank_layout["kind"] == "values-only"
    assert contract.bank_layout["shape"] == ["B", 3, 4]
    assert contract.state["ref"] == "arti/recall-state@1"
    assert contract.state_schema_version == RECALL_STATE_SCHEMA_VERSION


def test_state_round_trip_preserves_contract_fingerprint() -> None:
    runtime = _runtime()
    state = runtime.initial_state(2, dtype=torch.float32)
    restored = arti.alpha.RecallState.from_state_dict(state.state_dict())

    assert restored.contract_fingerprint == runtime.contract_fingerprint
    assert restored.schema_version == runtime.contract.state_schema_version
    runtime.read(torch.randn(2, 5, 4), restored)


def test_runtime_rejects_state_from_different_reader_contract() -> None:
    first = _runtime(updater_factors=2)
    second = _runtime(updater_factors=3)
    state = first.initial_state(1)

    with pytest.raises(ValueError, match="contract fingerprint"):
        second.read(torch.randn(1, 3, 4), state)


def test_runtime_rejects_unbound_tensor_and_legacy_state_payload() -> None:
    runtime = _runtime()
    with pytest.raises(ValueError, match="contract fingerprint"):
        runtime.read(torch.randn(1, 3, 4), torch.zeros(1, 3, 4))

    legacy = arti.alpha.RecallState.from_state_dict(
        {
            "value": torch.zeros(1, 3, 4),
            "step": torch.tensor(0, dtype=torch.int64),
        }
    )
    with pytest.raises(ValueError, match="schema version"):
        runtime.read(torch.randn(1, 3, 4), legacy)


def test_runtime_migration_explicitly_binds_legacy_state() -> None:
    runtime = _runtime()
    legacy = {
        "value": torch.zeros(1, 3, 4),
        "step": torch.tensor(4, dtype=torch.int64),
    }
    migrated = runtime.migrate_state(legacy)
    assert migrated.step == 4
    assert migrated.contract_fingerprint == runtime.contract_fingerprint
    runtime.read(torch.randn(1, 3, 4), migrated)


def test_runtime_contract_rejects_formula_layout_or_state_drift() -> None:
    for field in ("formula", "bank_layout", "state"):
        payload = copy.deepcopy(_runtime().contract.to_dict())
        if field == "formula":
            payload[field]["reference"] = "arti/state@1"
        elif field == "bank_layout":
            payload[field]["slots"] = 99
        else:
            payload[field]["ref"] = "arti/recall-state@2"
        with pytest.raises(ValueError, match="fingerprint"):
            arti.alpha.RecallRuntimeContract.from_dict(payload)


def test_runtime_contract_rejects_tampered_fingerprint() -> None:
    payload = copy.deepcopy(_runtime().contract.to_dict())
    payload["fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint"):
        arti.alpha.RecallRuntimeContract.from_dict(payload)


def test_runtime_scan_and_serial_update_are_equal() -> None:
    runtime = _runtime()
    traces = torch.randn(2, 4, 3, 4)
    initial = runtime.initial_state(2, dtype=torch.float32)
    scanned = runtime.scan(traces, initial, detach_state=True)

    serial = initial
    for index in range(traces.shape[1]):
        serial = runtime.update(traces[:, index], serial, detach_state=True)

    torch.testing.assert_close(scanned.value, serial.value)
    assert scanned.contract_fingerprint == serial.contract_fingerprint
