from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"


def _load_gate():
    sys.path.insert(0, str(BENCHMARKS))
    try:
        spec = importlib.util.spec_from_file_location(
            "objective_bank_engineering_gate_test",
            BENCHMARKS / "run_objective_bank_engineering_gate.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(BENCHMARKS))


def _passing_seed(gate) -> dict[str, object]:
    return {
        "hard_invariants": {key: True for key in gate.HARD_KEYS},
        "cost": {
            "objective_parameters": 16,
            "matched_parameters": 16,
            "p50_symmetric_ratio": 1.0,
            "training_seconds_symmetric_ratio": 1.0,
            "peak_allocated_bytes": 1024,
            "peak_reserved_bytes": 2048,
        },
        "capability": {key: True for key in gate.CAPABILITY_KEYS},
    }


def test_v2_decision_fails_closed_for_every_gate_family() -> None:
    gate = _load_gate()
    passing = [_passing_seed(gate), _passing_seed(gate), _passing_seed(gate)]
    assert gate._decision(
        passing,
        elapsed=1.0,
        deadline_seconds=10.0,
        canonical_contract=True,
    )["classification"] == "ENGINEERING_GO"

    two_of_three = [
        _passing_seed(gate),
        _passing_seed(gate),
        _passing_seed(gate),
    ]
    two_of_three[0]["capability"]["beats_uniform"] = False
    assert gate._decision(
        two_of_three,
        elapsed=1.0,
        deadline_seconds=10.0,
        canonical_contract=True,
    )["classification"] == "ENGINEERING_GO"

    for family, field in (
        ("hard_invariants", "future_taint_ok"),
        ("capability", "shuffle_degrades"),
    ):
        failing = [
            _passing_seed(gate),
            _passing_seed(gate),
            _passing_seed(gate),
        ]
        for seed in failing[:2]:
            seed[family][field] = False
        assert gate._decision(
            failing,
            elapsed=1.0,
            deadline_seconds=10.0,
            canonical_contract=True,
        )["classification"] == "ENGINEERING_OBSERVATION"

    for field, value in (
        ("p50_symmetric_ratio", 2.0),
        ("training_seconds_symmetric_ratio", 2.0),
        ("peak_allocated_bytes", 3 * 1024**3),
        ("peak_reserved_bytes", 5 * 1024**3),
    ):
        failing_cost = [
            _passing_seed(gate),
            _passing_seed(gate),
            _passing_seed(gate),
        ]
        failing_cost[0]["cost"][field] = value
        assert gate._decision(
            failing_cost,
            elapsed=1.0,
            deadline_seconds=10.0,
            canonical_contract=True,
        )["classification"] == "ENGINEERING_OBSERVATION"
    assert gate._decision(
        passing,
        elapsed=11.0,
        deadline_seconds=10.0,
        canonical_contract=True,
    )["classification"] == "ENGINEERING_OBSERVATION"


def test_v2_decision_rejects_incomplete_schema_and_noncanonical_runs() -> None:
    gate = _load_gate()
    passing = [_passing_seed(gate), _passing_seed(gate), _passing_seed(gate)]
    missing = [_passing_seed(gate), _passing_seed(gate), _passing_seed(gate)]
    del missing[0]["hard_invariants"][gate.HARD_KEYS[0]]
    missing_capability = [
        _passing_seed(gate),
        _passing_seed(gate),
        _passing_seed(gate),
    ]
    del missing_capability[0]["capability"][gate.CAPABILITY_KEYS[0]]

    assert gate._decision(
        missing,
        elapsed=1.0,
        deadline_seconds=10.0,
        canonical_contract=True,
    )["classification"] == "ENGINEERING_INVALID"
    assert gate._decision(
        missing_capability,
        elapsed=1.0,
        deadline_seconds=10.0,
        canonical_contract=True,
    )["classification"] == "ENGINEERING_INVALID"
    assert gate._decision(
        passing,
        elapsed=1.0,
        deadline_seconds=10.0,
        canonical_contract=False,
    )["classification"] == "ENGINEERING_OBSERVATION"


def test_v2_canonical_identity_and_exit_codes_are_fixed() -> None:
    gate = _load_gate()
    canonical = SimpleNamespace(
        seeds=list(gate.DEFAULT_SEEDS),
        writer_steps=80,
        controller_steps=160,
        batch_size=256,
        eval_size=512,
        deadline_seconds=240.0,
    )
    assert gate._is_canonical_configuration(canonical)
    for seeds in ([33101], [33101, 33101, 33101], [33113, 33101, 33129]):
        assert not gate._is_canonical_configuration(
            SimpleNamespace(**{**vars(canonical), "seeds": seeds})
        )
    assert gate._classification_exit_code("ENGINEERING_GO") == 0
    assert gate._classification_exit_code("ENGINEERING_OBSERVATION") == 1
    assert gate._classification_exit_code("ENGINEERING_INVALID") == 2


def test_v2_invalid_receipt_is_parseable_and_redacts_error(
    tmp_path: Path, monkeypatch
) -> None:
    gate = _load_gate()
    output = tmp_path / "invalid.json"
    monkeypatch.setattr(sys, "argv", ["gate", "--output", str(output)])

    gate._write_invalid_receipt(RuntimeError(r"C:\private\secret.txt"))

    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["decision"]["classification"] == "ENGINEERING_INVALID"
    assert receipt["decision"]["error_type"] == "RuntimeError"
    assert "secret" not in json.dumps(receipt)


def test_v2_objective_exposure_uses_public_bounded_output() -> None:
    gate = _load_gate()
    bank = gate.ObjectiveExposureBank(
        8,
        2,
        key_layout="circle",
        min_exposure=0.2,
        max_exposure=0.8,
    )
    with torch.no_grad():
        bank.values.copy_(torch.linspace(-3.0, 3.0, 8))
    query = torch.randn(5, 4, 2)
    bounded = bank(query, return_info=True).exposure
    expected = bounded / bounded.sum(dim=-1, keepdim=True)

    actual = gate._controller_exposure(bank, query)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert gate.FORMAT == "arti.objective-bank-engineering-gate.v2"
