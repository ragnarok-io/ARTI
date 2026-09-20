from __future__ import annotations

import pytest
import torch

from arti import mechanisms
from benchmarks.federal_bank_owned_query_tree import (
    FEATURE_DIM,
    _orthogonal_basis,
    _transition_program,
    level_layout,
)


@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("branch", [0, 1])
def test_bank_formula_transforms_heterogeneous_layouts_without_external_topology(
    level: int,
    branch: int,
) -> None:
    current_basis = _orthogonal_basis(1200 + level)
    next_basis = _orthogonal_basis(2200 + level * 2 + branch)
    current_layout = level_layout(level)
    next_layout = level_layout(level + 1)
    program, indices, transition = _transition_program(
        current_layout=current_layout,
        next_layout=next_layout,
        current_basis=current_basis,
        next_basis=next_basis,
        branch=branch,
    )
    fabric = mechanisms.FormulaFabricV2(program)
    bank_bindings = {
        binding.name: binding
        for binding in program.bindings
        if isinstance(binding, mechanisms.BankBinding)
    }
    latent = torch.randn(5, FEATURE_DIM)
    value = (latent @ current_basis).reshape(5, *current_layout).requires_grad_()

    result = fabric(
        inputs={"value": value},
        banks={
            "topology.indices": bank_bindings["topology.indices"].bind(indices),
            "transition.matrix": bank_bindings["transition.matrix"].bind(transition),
        },
        return_trace=True,
    )

    expected = (latent @ next_basis).reshape(5, *next_layout)
    torch.testing.assert_close(result.values[0], expected, rtol=1e-5, atol=1e-6)
    assert result.trace is not None
    assert result.trace.atom_refs == (
        "arti/formula-atom-reshape@1",
        "arti/formula-atom-gather@1",
        "arti/formula-atom-reshape@1",
        "arti/formula-atom-permute@1",
        "arti/formula-atom-reshape@1",
        "arti/formula-atom-contract@1",
        "arti/formula-atom-reshape@1",
    )

    result.values[0].square().mean().backward()
    assert value.grad is not None
    assert torch.isfinite(value.grad).all()


def test_federal_formula_architecture_uses_distinct_branch_programs() -> None:
    current_basis = _orthogonal_basis(3100)
    next_basis = _orthogonal_basis(3200)
    programs = [
        _transition_program(
            current_layout=level_layout(0),
            next_layout=level_layout(1),
            current_basis=current_basis,
            next_basis=next_basis,
            branch=branch,
        )[0]
        for branch in (0, 1)
    ]

    assert programs[0].fingerprint != programs[1].fingerprint
    assert programs[0].bank_names == programs[1].bank_names
