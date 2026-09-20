from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

import arti
from arti.component_registry import canonical_contract_reference
from benchmarks.federal_tensor_view_shape_refine import (
    build_runtime,
    input_view,
    run,
)


def test_same_bank_requeries_latest_view_across_rank_and_axis_changes() -> None:
    value = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    output, trace = build_runtime()(input_view(value), return_trace=True)
    expected = value.reshape(1, 2, 2, 2).permute(0, 3, 2, 1)
    torch.testing.assert_close(output["value"], expected)
    local = trace.steps[0].local_refine
    assert [item.input_shape for item in local] == [
        (1, 8),
        (1, 2, 4),
        (1, 2, 2, 2),
        (1, 2, 2, 2),
    ]
    assert [item.input_axes for item in local] == [
        ("batch", "token"),
        ("batch", "row", "column"),
        ("batch", "depth", "row", "lane"),
        ("batch", "lane", "row", "depth"),
    ]
    assert [item.candidate_id for item in local] == [
        "to-plane",
        "to-volume",
        "rotate-axes",
        "exit",
    ]
    assert len({item.query_state_fingerprint for item in local}) == 1
    assert len({item.input_view_fingerprint for item in local}) == 4
    assert [item.output_view_fingerprint for item in local[:-1]] == [
        item.input_view_fingerprint for item in local[1:]
    ]


def test_shape_polymorphic_federation_batches_independent_views() -> None:
    value = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    output, trace = build_runtime()(input_view(value), return_trace=True)
    expected = value.reshape(3, 2, 2, 2).permute(0, 3, 2, 1)
    torch.testing.assert_close(output["value"], expected)
    assert len(trace.steps) == 3
    assert all(len(step.local_refine) == 4 for step in trace.steps)


def test_formula_actions_are_the_only_payload_shape_transforms() -> None:
    runtime = build_runtime()
    atoms = [
        [instruction.atom_ref for instruction in item.action.program.instructions]
        for item in runtime.banks["shape-workshop"].actions
    ]
    assert atoms == [
        ["arti/formula-atom-reshape@1"],
        ["arti/formula-atom-reshape@1"],
        ["arti/formula-atom-permute@1"],
    ]
    assert all(not tuple(action.parameters()) for action in runtime.banks["shape-workshop"].actions)


def test_one_shot_and_wrong_query_cannot_reach_terminal() -> None:
    value = input_view(torch.arange(8, dtype=torch.float32).reshape(1, 8))
    with pytest.raises(ValueError, match="min_steps"):
        build_runtime(max_steps=1)
    with pytest.raises((ValueError, RuntimeError)):
        build_runtime(phase_offset=1)(value)


def test_source_index_map_survives_reshape_and_axis_permutation() -> None:
    value = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    mask = torch.tensor(
        [[True, False, True, False, True, False, True, False]],
    )
    bank = build_runtime().banks["shape-workshop"]
    source = input_view(value)
    current = type(source)(source.value, source.axes, mask=mask)
    for action in bank.actions:
        current = action(current)
    assert current.index_map is not None
    assert current.index_map.source_axes == ("token",)
    assert current.index_map.source_shape == (8,)
    assert current.index_map.target_shape == (2, 2, 2)
    expected = torch.arange(8).reshape(2, 2, 2).permute(2, 1, 0)
    torch.testing.assert_close(current.index_map.coordinates[..., 0], expected)
    assert current.mask is not None
    torch.testing.assert_close(
        current.mask,
        mask.reshape(1, 2, 2, 2).permute(0, 3, 2, 1),
    )


def test_shape_polymorphic_federation_preserves_input_gradients_and_provenance() -> None:
    value = torch.arange(8, dtype=torch.float32).reshape(1, 8).requires_grad_()
    runtime = build_runtime()
    output = runtime(input_view(value))["value"]
    output.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert arti.component_ref(runtime) == canonical_contract_reference("arti/federal-recall@3")
    assert arti.component_ref(runtime.banks["shape-workshop"]) == canonical_contract_reference(
        "arti/tensor-view-formula-program@2"
    )
    assert runtime.banks["shape-workshop"].signature.input_pattern.max_rank == 5


def test_benchmark_emits_replayable_mechanism_receipts() -> None:
    result = run()
    assert result["correct"] is True
    assert result["controls"] == {
        "one_shot": "FederalRecallError",
        "wrong_query": "FederalRecallError",
    }
    assert result["actions"] == ["to-plane", "to-volume", "rotate-axes", "exit"]


def test_shape_polymorphic_federation_safetensors_round_trip(tmp_path: Path) -> None:
    value = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    runtime = build_runtime()
    expected = runtime(input_view(value))["value"]
    provenance = arti.component_provenance(runtime)
    assert arti.validate_component_provenance(provenance) == provenance

    artifact = tmp_path / "federal-tensor-view.arti.st"
    arti.save(runtime, artifact)
    fresh = build_runtime()
    arti.load(artifact, model=fresh)
    actual = fresh(input_view(value))["value"]
    torch.testing.assert_close(actual, expected)

    forged = deepcopy(provenance)
    root = next(item for item in forged["components"] if item["path"] == "$")
    root["dependencies"].remove(
        canonical_contract_reference("arti/tensor-view-pattern@1")
    )
    forged["fingerprint"] = arti.component_graph_fingerprint(forged["components"])
    with pytest.raises(arti.ComponentCompatibilityError, match="dependency closure"):
        arti.validate_component_provenance(forged)


def test_artifact_rejects_a_changed_sealed_tensor_view_query(tmp_path: Path) -> None:
    runtime = build_runtime()
    query = runtime.banks["shape-workshop"].query.query
    with torch.no_grad():
        next(query.parameters()).add_(1.0)
    with pytest.raises(ValueError, match="changed after mounting"):
        arti.save(runtime, tmp_path / "changed-query.arti.st")
