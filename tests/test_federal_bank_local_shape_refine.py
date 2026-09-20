from __future__ import annotations

import pytest
import torch

import arti
from arti import mechanisms
from benchmarks.federal_bank_local_shape_refine import (
    build_runtime,
    expected_output,
    run,
)


def test_bank_local_refine_requires_multiple_shape_forming_formula_steps() -> None:
    value = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2).requires_grad_()
    runtime = build_runtime()

    output, trace = runtime(value, return_trace=True)

    torch.testing.assert_close(output["value"], expected_output(value))
    assert len(trace.steps) == 1
    local = trace.steps[0].local_refine
    assert [item.input_shape for item in local] == [
        (1, 8, 2),
        (1, 4, 2),
        (1, 6, 2),
    ]
    assert [item.output_shape for item in local] == [
        (1, 4, 2),
        (1, 6, 2),
        None,
    ]
    assert [item.action for item in local] == [
        "continue-local",
        "continue-local",
        "terminal",
    ]
    assert trace.winner_paths == ("shape-memory/exit",)

    output["value"].sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_shape_refine_uses_explicit_index_workset_atoms() -> None:
    runtime = build_runtime()
    bank = runtime.banks["shape-memory"]

    assert tuple(item.atom_ref for item in bank.compress.program.instructions) == (
        "arti/formula-atom-gather@1",
    )
    assert tuple(item.atom_ref for item in bank.expand.program.instructions) == (
        "arti/formula-atom-gather@1",
        "arti/formula-atom-scatter@1",
    )
    assert {"arti/formula-atom-gather@1", "arti/formula-atom-scatter@1"}.issubset(
        arti.component_spec(bank.expand).dependencies
    )


def test_one_shot_and_wrong_or_frozen_routes_cannot_form_the_output_abi() -> None:
    value = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)

    with pytest.raises(mechanisms.FederalRecallError, match="max_steps"):
        build_runtime(min_steps=1, max_steps=1)(value)
    with pytest.raises((mechanisms.FormulaBindingError, mechanisms.FederalRecallError)):
        build_runtime(query_mode="frozen")(value)
    with pytest.raises((mechanisms.FormulaBindingError, mechanisms.FederalRecallError)):
        build_runtime(query_mode="wrong")(value)


def test_reset_bank_preserves_route_but_loses_the_bank_owned_payload() -> None:
    value = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    correct, correct_trace = build_runtime()(value, return_trace=True)
    reset, reset_trace = build_runtime(reset_bank=True)(value, return_trace=True)

    assert not torch.equal(reset["value"], correct["value"])
    assert [item.action for item in reset_trace.steps[0].local_refine] == [
        item.action for item in correct_trace.steps[0].local_refine
    ]
    assert reset_trace.winner_paths == correct_trace.winner_paths


def test_shape_refine_state_and_provenance_round_trip(tmp_path) -> None:
    value = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    source = build_runtime()
    restored = build_runtime()
    expected = source(value)["value"]

    provenance = arti.component_provenance(source)
    assert arti.validate_component_provenance(provenance) == provenance
    saved = arti.save(source, tmp_path / "shape-refine.arti.st")
    arti.load(saved.weights_path, model=restored)

    torch.testing.assert_close(restored(value)["value"], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_shape_refine_preserves_cuda_and_gradients() -> None:
    runtime = build_runtime().cuda()
    value = torch.arange(16, dtype=torch.float32, device="cuda").reshape(1, 8, 2)
    value.requires_grad_()

    output = runtime(value)["value"]
    output.square().mean().backward()

    assert output.is_cuda
    assert value.grad is not None and value.grad.is_cuda
    assert torch.isfinite(value.grad).all()


def test_shape_refine_benchmark_reports_replayable_receipts() -> None:
    result = run()

    assert result["correct"] is True
    assert result["reset_bank_matches"] is False
    assert result["controls"] == {
        "one_shot": "FederalRecallError",
        "frozen_route": "FormulaBindingError",
        "wrong_query": "FormulaBindingError",
    }
    assert result["formula_atoms"] == {
        "compress": ["arti/formula-atom-gather@1"],
        "expand": ["arti/formula-atom-gather@1", "arti/formula-atom-scatter@1"],
    }
