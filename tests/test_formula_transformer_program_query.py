from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from arti import mechanisms
from benchmarks.train_formula_transformer_program_query import (
    REQUIRED_STRUCTURE,
    bind_fixed_formula_banks,
    build_fixed_formula_program,
    build_search,
    canonical_path,
    evaluate,
    execute_path,
    matched_wrong_path,
    sample,
    train,
)


def _eligible_ids(
    query: mechanisms.FormulaProgramQuery,
    arena: mechanisms.FormulaProgramArena,
    *,
    steps: int,
) -> set[str]:
    return {
        action_id
        for action_id, enabled in zip(
            query.action_ids,
            query.eligible(arena, steps=steps).tolist(),
            strict=True,
        )
        if enabled
    }


def test_transformer_search_grammar_closes_abandoned_branches() -> None:
    search = build_search(seed=17, device=torch.device("cpu"))
    initial, _target = sample(search, count=3, sequence_length=4, seed=301)
    arena = search.query.arena(initial)
    candidates = {
        candidate.candidate_id: candidate for candidate in search.query.candidates
    }

    assert _eligible_ids(search.query, arena, steps=0) == {
        "skip-attention",
        "query-project",
        "key-project",
        "value-project",
    }
    arena = candidates["query-project"](arena)
    assert "skip-attention" not in _eligible_ids(search.query, arena, steps=1)
    arena = candidates["key-project"](arena)
    arena = candidates["value-project"](arena)
    arena = candidates["attention-scores"](arena)
    assert {
        "score-scaled",
        "score-unit",
        "score-negated",
    }.issubset(_eligible_ids(search.query, arena, steps=4))


def test_fixed_formula_and_canonical_query_path_match_reference() -> None:
    search = build_search(seed=19, device=torch.device("cpu"))
    initial, target = sample(search, count=5, sequence_length=6, seed=302)
    canonical = execute_path(search, initial, canonical_path())
    matched_wrong = execute_path(search, initial, matched_wrong_path())
    program = build_fixed_formula_program(search.config)
    fabric = mechanisms.FormulaFabricV2(program)
    fixed = fabric(
        inputs=initial,
        banks=bind_fixed_formula_banks(program, search.weights),
    ).values[0]

    torch.testing.assert_close(canonical, target, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(fixed, target, atol=1e-6, rtol=1e-5)
    assert F.mse_loss(matched_wrong, target) > 0.1


def test_transformer_program_query_learns_structure_and_reloads() -> None:
    search = build_search(seed=17, device=torch.device("cpu"))
    training = train(search, seed=17, steps=20, batch_size=32)
    report = evaluate(
        search,
        seed=70_018,
        count=8,
        sequence_lengths=(1, 6, 8),
    )

    assert training["max_visited_states"] == 133
    assert report["structure_accuracy"] == 1.0
    assert max(row["mse"] for row in report["lengths"]) < 1e-10
    assert all(
        frozenset(path) == REQUIRED_STRUCTURE for path in report["sample_paths"]
    )

    restored = build_search(seed=999, device=torch.device("cpu"))
    restored.query.load_state_dict(search.query.state_dict(), strict=True)
    initial, _target = sample(search, count=1, sequence_length=7, seed=303)
    expected, expected_trace = search.query(initial, return_trace=True)
    actual, actual_trace = restored.query(initial, return_trace=True)

    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    assert actual_trace == expected_trace


def test_fixed_formula_transformer_gradients_are_finite() -> None:
    search = build_search(seed=23, device=torch.device("cpu"))
    program = build_fixed_formula_program(search.config)
    initial, _target = sample(search, count=4, sequence_length=5, seed=304)
    query_input = initial["query-input"].detach().clone().requires_grad_()
    key_value_input = initial["key-value-input"].detach().clone().requires_grad_()
    bank_names = {
        binding.name
        for binding in program.bindings
        if isinstance(binding, mechanisms.BankBinding)
    }
    weights = {
        name: value.detach().clone().requires_grad_()
        for name, value in search.weights.items()
        if name in bank_names
    }
    output = mechanisms.FormulaFabricV2(program)(
        inputs={
            "query-input": query_input,
            "key-value-input": key_value_input,
            "causal-mask": initial["causal-mask"],
        },
        banks=bind_fixed_formula_banks(program, weights),
    ).values[0]
    output.square().mean().backward()

    assert query_input.grad is not None and torch.isfinite(query_input.grad).all()
    assert key_value_input.grad is not None and torch.isfinite(key_value_input.grad).all()
    assert all(
        value.grad is not None and torch.isfinite(value.grad).all()
        for value in weights.values()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_learned_transformer_structure_has_fullgraph_cuda_formula() -> None:
    search = build_search(seed=31, device=torch.device("cuda"))
    program = build_fixed_formula_program(search.config)
    initial, target = sample(search, count=2, sequence_length=6, seed=305)
    fabric = mechanisms.FormulaFabricV2(program).cuda()
    prepared = fabric.bind_tensors(
        inputs=initial,
        banks=bind_fixed_formula_banks(program, search.weights),
    )
    plan = fabric.execution_plan().cuda()

    expected = plan(prepared)[0]
    actual = torch.compile(plan, fullgraph=True)(prepared)[0]

    torch.testing.assert_close(expected, target, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
