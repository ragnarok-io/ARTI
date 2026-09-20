from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


SCRIPT = Path(__file__).parents[1] / "benchmarks" / "unfold_assignment_screen.py"
SPEC = importlib.util.spec_from_file_location("unfold_assignment_screen", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_adversarial_matrix_separates_greedy_from_global_assignment() -> None:
    result = MODULE.adversarial_audit(torch.device("cpu"))
    assert result["greedy_objective"] == 18.0
    assert result["optimal_objective"] == 25.0
    assert result["auction_objective"] == 25.0
    assert result["parallel_objective"] == 25.0
