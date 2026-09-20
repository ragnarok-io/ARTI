from __future__ import annotations

import torch
import importlib.util
from pathlib import Path

import arti
from arti import legacy
from arti.recall_topology import trace_key


ROOT = Path(__file__).resolve().parents[1]


def candidate(name: str, specs: tuple[legacy.LayerRecallSpec, ...]) -> arti.LayeredRecallCandidate:
    return arti.LayeredRecallCandidate(name, legacy.LayeredRecallConfig(layers=specs), abstention_weight=1.0)


def cache() -> arti.LayeredRecallTraceCache:
    generator = torch.Generator().manual_seed(29)
    tensors = {}
    dims = {"a": 8, "b": 8, "c": 8}
    for path, dim in dims.items():
        clean = torch.randn(4, 5, dim, generator=generator)
        tensors[trace_key(path, "clean")] = clean
        tensors[trace_key(path, "corrupt_single")] = clean + torch.randn(clean.shape, generator=generator) * 0.2
        tensors[trace_key(path, "corrupt_combined")] = clean + torch.randn(clean.shape, generator=generator) * 0.4
        tensors[trace_key(path, "unseen")] = torch.randn(clean.shape, generator=generator) + 4
    return arti.LayeredRecallTraceCache(tensors=tensors, layer_dims=dims, source={"model": "synthetic"})


def test_trace_cache_roundtrip_and_fingerprint(tmp_path) -> None:
    original = cache()
    weights, manifest = original.save(tmp_path / "hidden-traces")
    restored = arti.LayeredRecallTraceCache.load(weights)

    assert weights.suffix == ".safetensors"
    assert manifest.is_file()
    assert restored.fingerprint == original.fingerprint
    assert set(restored.tensors) == set(original.tensors)


def test_static_cost_handles_arbitrary_nonuniform_topology() -> None:
    value = candidate(
        "nonuniform",
        (
            legacy.LayerRecallSpec("a", dim=8, rank=1, slots=2),
            legacy.LayerRecallSpec("b", dim=8, rank=3, slots=7, use_half=False),
            legacy.LayerRecallSpec("c", dim=8, rank=2, slots=4, recognition_mode="none"),
        ),
    )
    cost = arti.estimate_layered_recall_cost(value, tokens=20)

    assert cost.layer_count == 3
    assert cost.parameters > 0
    assert cost.token_multiply_adds > cost.parameters

    repeated = candidate("repeated", (legacy.LayerRecallSpec("a", dim=8, rank=1, slots=2, copies=3),))
    single = candidate("single-copy", (legacy.LayerRecallSpec("a", dim=8, rank=1, slots=2),))
    assert arti.estimate_layered_recall_cost(repeated).parameters == 3 * arti.estimate_layered_recall_cost(single).parameters


def test_budget_filter_and_pareto_frontier() -> None:
    small = candidate("small", (legacy.LayerRecallSpec("a", dim=8, rank=1, slots=2),))
    large = candidate("large", (legacy.LayerRecallSpec("a", dim=8, rank=8, slots=16),))
    budget = arti.LayeredRecallBudget(max_parameters=100, max_steps=5, max_candidates=2)
    accepted = arti.candidates_within_budget((small, large), budget)
    assert accepted == (small,)

    scores = (
        arti.LayeredRecallScore("balanced", 0.8, 0.1, 100, 1.0),
        arti.LayeredRecallScore("dominated", 0.9, 0.2, 120, 1.2),
        arti.LayeredRecallScore("selective", 0.9, 0.01, 110, 1.1),
    )
    assert {score.candidate for score in arti.pareto_layered_recall(scores)} == {"balanced", "selective"}


def test_cached_screening_repairs_without_source_model_execution() -> None:
    value = candidate(
        "two-layer",
        (
            legacy.LayerRecallSpec("a", rank=2, slots=3),
            legacy.LayerRecallSpec("b", rank=2, slots=3),
        ),
    )
    budget = arti.LayeredRecallBudget(max_parameters=1000, max_steps=12, max_runtime_seconds=2)
    score = arti.screen_layered_recall_candidate(value, cache(), budget, learning_rate=1e-2)

    assert score.status == "completed"
    assert score.completed_steps == 12
    assert score.runtime_seconds < 2
    assert set(score.per_layer) == {"a", "b"}
    assert score.normalized_repair_mse < 1.1


def test_cached_screening_promotes_fp16_cache_for_cpu_training() -> None:
    original = cache()
    half_cache = arti.LayeredRecallTraceCache(
        tensors={key: value.half() for key, value in original.tensors.items()},
        layer_dims=original.layer_dims,
        source=original.source,
    )
    value = candidate("half-cache", (legacy.LayerRecallSpec("a", rank=1, slots=2),))
    budget = arti.LayeredRecallBudget(max_parameters=1000, max_steps=2, max_runtime_seconds=2)

    score = arti.screen_layered_recall_candidate(value, half_cache, budget, device="cpu")
    assert score.completed_steps == 2


def test_cached_screening_supports_repeated_independent_lines() -> None:
    value = candidate("repeated-cache", (legacy.LayerRecallSpec("a", rank=1, slots=2, copies=3),))
    budget = arti.LayeredRecallBudget(max_parameters=1000, max_steps=2, max_runtime_seconds=2)
    score = arti.screen_layered_recall_candidate(value, cache(), budget, device="cpu")

    assert score.completed_steps == 2
    assert score.parameters == arti.estimate_layered_recall_cost(value, layer_dims=cache().layer_dims).parameters


def test_public_topology_api_matches_torch_namespace() -> None:
    assert arti.torch.LayeredRecallCandidate is arti.LayeredRecallCandidate
    assert arti.torch.LayeredRecallTraceCache is arti.LayeredRecallTraceCache
    assert arti.torch.pareto_layered_recall is arti.pareto_layered_recall


def test_qwen_candidate_enumeration_retains_open_topology_families() -> None:
    path = ROOT / "benchmarks" / "screen_qwen_recall_topologies.py"
    spec = importlib.util.spec_from_file_location("screen_qwen_recall_topologies", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = module.enumerate_candidates(cache(), target_parameters=200, tolerance=0.5)
    first = values[:32]
    tags = {tag for value in first for tag in value.tags}

    assert {"single", "uniform", "nonuniform", "repeated-key", "early-heavy", "late-heavy"}.issubset(tags)
    assert {value.abstention_weight for value in first} == {0.25, 1.0, 4.0}
    assert {spec.use_half for value in first for spec in value.config.layers} == {True, False}
    assert {spec.recognition_mode for value in first for spec in value.config.layers} == {"alignment", "explicit"}
