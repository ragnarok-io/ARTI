from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"


def _load_gate():
    sys.path.insert(0, str(BENCHMARKS))
    try:
        spec = importlib.util.spec_from_file_location(
            "objective_formula_engineering_gate_test",
            BENCHMARKS / "run_objective_formula_engineering_gate.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(BENCHMARKS))


def _passing(gate) -> dict[str, object]:
    return {
        "hard_invariants": {key: True for key in gate.HARD_KEYS},
        "capability": {key: True for key in gate.CAPABILITY_KEYS},
        "cost": {
            "p50_symmetric_ratio": 1.0,
            "p95_symmetric_ratio": 1.0,
            "peak_allocated_bytes": 1024,
            "peak_reserved_bytes": 2048,
        },
    }


def test_decision_is_fail_closed_and_uses_capability_majority() -> None:
    gate = _load_gate()
    passing = [_passing(gate) for _ in range(3)]
    assert gate._decision(
        passing, elapsed=1.0, deadline_seconds=10.0, canonical_contract=True
    )["classification"] == "ENGINEERING_GO"

    passing[0]["capability"][gate.CAPABILITY_KEYS[0]] = False
    assert gate._decision(
        passing, elapsed=1.0, deadline_seconds=10.0, canonical_contract=True
    )["classification"] == "ENGINEERING_GO"

    passing[1]["capability"][gate.CAPABILITY_KEYS[0]] = False
    assert gate._decision(
        passing, elapsed=1.0, deadline_seconds=10.0, canonical_contract=True
    )["classification"] == "ENGINEERING_OBSERVATION"

    incomplete = [_passing(gate) for _ in range(3)]
    del incomplete[0]["hard_invariants"][gate.HARD_KEYS[0]]
    assert gate._decision(
        incomplete, elapsed=1.0, deadline_seconds=10.0, canonical_contract=True
    )["classification"] == "ENGINEERING_INVALID"


def test_canonical_identity_is_fixed() -> None:
    gate = _load_gate()
    canonical = SimpleNamespace(
        seeds=list(gate.DEFAULT_SEEDS),
        steps=120,
        batch_size=256,
        eval_size=1024,
        deadline_seconds=240.0,
    )
    assert gate._is_canonical(canonical)
    assert not gate._is_canonical(
        SimpleNamespace(**{**vars(canonical), "seeds": [41011, 41011, 41023]})
    )
    assert not gate._is_canonical(
        SimpleNamespace(**{**vars(canonical), "steps": 119})
    )
