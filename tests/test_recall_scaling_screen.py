from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

import arti


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("scaling_screen", ROOT / "benchmarks" / "screen_recall_scaling_matrix.py")
assert SPEC and SPEC.loader
screen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(screen)


def cache() -> arti.LayeredRecallTraceCache:
    dims = {f"model.layers.{index}": 16 for index in range(7)}
    tensors = {}
    for path in dims:
        clean = torch.randn(2, 3, 16)
        tensors[arti.recall_topology.trace_key(path, "clean")] = clean
        tensors[arti.recall_topology.trace_key(path, "corrupt_single")] = clean + 0.2
        tensors[arti.recall_topology.trace_key(path, "corrupt_combined")] = clean + 0.4
        tensors[arti.recall_topology.trace_key(path, "unseen")] = torch.randn_like(clean)
    return arti.LayeredRecallTraceCache(tensors, dims, {"model": "synthetic"})


def test_matrix_controls_all_axes_under_matched_budgets() -> None:
    candidates = screen.enumerate_matrix(cache(), budgets=(1000, 2000), tolerance=0.25)
    assert candidates
    names = [candidate.name for candidate in candidates]
    assert {"h0", "h1"}.issubset({name.rsplit("-", 1)[-1] for name in names})
    assert len({tag for candidate in candidates for tag in candidate.tags if tag.startswith("depth-")}) > 1
    assert len({tag for candidate in candidates for tag in candidate.tags if tag.startswith("slots-")}) > 1
    assert len({tag for candidate in candidates for tag in candidate.tags if tag.startswith("rank-")}) > 1
    assert len({tag for candidate in candidates for tag in candidate.tags if tag.startswith("copies-")}) > 1
    for candidate in candidates:
        budget = int(next(tag.split("-", 1)[1] for tag in candidate.tags if tag.startswith("budget-")))
        parameters = arti.estimate_layered_recall_cost(candidate).parameters
        assert abs(parameters - budget) / budget <= 0.25


def test_screen_protocol_is_resumable_and_hard_limited() -> None:
    source = (ROOT / "benchmarks" / "screen_recall_scaling_matrix.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--resume", action="store_true")' in source
    assert "args.max_runtime_seconds <= 120" in source
    assert "repair_gain_per_million_parameters" in source
