from __future__ import annotations

import runpy
from pathlib import Path

import torch


GATE = runpy.run_path(
    str(Path(__file__).parents[1] / "benchmarks" / "run_formula_fabric_opaque_ssa_v2_gate.py"),
    run_name="formula_fabric_opaque_ssa_gate",
)


def make_episode(batch: int = 8):
    generator = torch.Generator().manual_seed(17)
    return GATE["episode"](batch, device=torch.device("cpu"), generator=generator)


def test_parameter_matched_models_are_exactly_matched() -> None:
    candidate = GATE["FabricCandidate"]()
    baseline = GATE["MatchedNeural"]()
    assert GATE["parameter_count"](candidate) == 161
    assert GATE["parameter_count"](baseline) == 161


def test_oracle_route_matches_independent_target_at_every_depth() -> None:
    item = make_episode()
    model = GATE["FabricCandidate"]()
    for depth in (1, 2, 3):
        actual = model(item, depth, "oracle")
        expected = GATE["oracle_target"](item, depth)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_key_coordinate_permutation_does_not_change_router_logits() -> None:
    item = make_episode()
    router = GATE["OpaqueRouter"](16)
    permutation = torch.tensor([3, 1, 7, 0, 5, 2, 6, 4])
    before = router.logits(item.key, item.query, 3)
    after = router.logits(item.key[..., permutation], item.query[..., permutation], 3)
    torch.testing.assert_close(before, after, rtol=1e-6, atol=1e-7)


def test_candidate_backward_is_finite() -> None:
    item = make_episode()
    model = GATE["FabricCandidate"]()
    loss = GATE["normalized_mse"](
        model(item, 2), GATE["oracle_target"](item, 2)
    )
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
