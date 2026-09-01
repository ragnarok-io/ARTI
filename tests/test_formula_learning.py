from __future__ import annotations

import copy
import hashlib
import json

import pytest
import torch

import arti
from arti import mechanisms


def _make_bank(
    *,
    members: int = 4,
    key_dim: int = 6,
    input_dim: int = 5,
    output_dim: int = 3,
    rank: int = 2,
    member_ids: tuple[str, ...] | None = None,
) -> mechanisms.FormulaOperandBank:
    return mechanisms.FormulaOperandBank(
        keys=torch.randn(members, key_dim, dtype=torch.float64),
        operands={
            "A": torch.randn(members, rank, input_dim, dtype=torch.float64),
            "B": torch.randn(members, output_dim, rank, dtype=torch.float64),
            "gain": torch.randn(members, dtype=torch.float64),
        },
        source_ref="arti/formula-operand-bank@1",
        bundle_id="lora",
        member_ids=member_ids,
    )


def test_hard_formula_route_is_one_hot_forward_with_explicit_surrogate() -> None:
    logits = torch.tensor(
        [[0.1, 1.2, -0.5], [2.0, -1.0, 0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    selection = mechanisms.hard_formula_route(logits, estimator="straight-through")

    assert selection.estimator == "straight-through"
    assert selection.hard_indices.tolist() == [1, 0]
    assert torch.equal(selection.route.detach(), torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64))
    selection.route.mul(torch.tensor([[1.0, 3.0, 7.0], [2.0, 5.0, 11.0]])).sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0

    hard = mechanisms.hard_formula_route(logits.detach(), estimator="hard")
    assert not hard.route.requires_grad
    assert torch.equal(hard.route, selection.route.detach())


def test_formula_operand_bank_ties_follow_stable_member_identity() -> None:
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    keys = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
    )
    member_ids = ("zeta", "alpha", "other")
    bank = mechanisms.FormulaOperandBank(
        keys=keys,
        operands={"value": torch.randn(3, 2, dtype=torch.float64)},
        member_ids=member_ids,
    )
    original = bank.route(query, estimator="hard")
    assert bank.member_ids[int(original.hard_indices.item())] == "alpha"

    permutation = torch.tensor([2, 0, 1])
    permuted = mechanisms.FormulaOperandBank(
        keys=keys[permutation],
        operands={"value": bank.operands["value"].detach()[permutation]},
        member_ids=tuple(member_ids[index] for index in permutation.tolist()),
    )
    selected = permuted.route(query, estimator="hard")
    assert permuted.member_ids[int(selected.hard_indices.item())] == "alpha"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_formula_operand_bank_constructed_on_cuda_routes_without_device_drift() -> None:
    keys = torch.tensor([[1.0, 0.0], [1.0, 0.0]], device="cuda")
    bank = mechanisms.FormulaOperandBank(
        keys=keys,
        operands={"value": torch.randn(2, 3, device="cuda")},
        member_ids=("zeta", "alpha"),
    )

    selected = bank.route(torch.tensor([[1.0, 0.0]], device="cuda"), estimator="hard")

    assert selected.hard_indices.is_cuda
    assert bank.member_ids[int(selected.hard_indices.item())] == "alpha"


def test_routed_lora_program_matches_selected_bundle_and_gradients() -> None:
    torch.manual_seed(9201)
    members, input_dim, output_dim, rank = 4, 5, 3, 2
    bank = _make_bank(
        members=members,
        input_dim=input_dim,
        output_dim=output_dim,
        rank=rank,
    )
    program = mechanisms.build_routed_lora_program(
        input_dim=input_dim,
        output_dim=output_dim,
        rank=rank,
        candidate_count=members,
        source_ref=bank.source_ref,
        bundle_id=bank.bundle_id,
        member_ids=bank.member_ids,
        dtype="float64",
    )
    fabric = mechanisms.FormulaFabricV2(program)
    query = torch.randn(6, bank.key_dim, dtype=torch.float64, requires_grad=True)
    x = torch.randn(6, 3, input_dim, dtype=torch.float64, requires_grad=True)
    base = torch.randn(6, 3, output_dim, dtype=torch.float64, requires_grad=True)
    selection = bank.route(query, estimator="straight-through", temperature=0.7)
    actual = fabric(
        inputs={"x": x, "base": base, "formula.route": selection.route},
        banks=bank.bind(program),
        return_trace=True,
    )

    selected_a = torch.einsum("bk,kri->bri", selection.route, bank.operands["A"])
    selected_b = torch.einsum("bk,kor->bor", selection.route, bank.operands["B"])
    selected_gain = torch.einsum("bk,k->b", selection.route, bank.operands["gain"])
    hidden = torch.einsum("bsi,bri->bsr", x, selected_a)
    expected = base + selected_gain[:, None, None] * torch.einsum(
        "bsr,bor->bso", hidden, selected_b
    )
    torch.testing.assert_close(actual.values[0], expected, rtol=1e-12, atol=1e-12)
    assert actual.trace is not None
    assert actual.trace.atom_refs == (
        "arti/formula-atom-contract@1",
        "arti/formula-atom-contract@1",
        "arti/formula-atom-contract@1",
        "arti/formula-atom-contract@1",
        "arti/formula-atom-contract@1",
        "arti/formula-atom-scale@1",
        "arti/formula-atom-add@1",
    )
    assert sum(parameter.numel() for parameter in fabric.parameters()) == 0

    loss = actual.values[0].square().mean()
    loss.backward()
    assert query.grad is not None and torch.count_nonzero(query.grad) > 0
    assert bank.keys.grad is not None and torch.count_nonzero(bank.keys.grad) > 0
    assert all(
        value.grad is not None and torch.count_nonzero(value.grad) > 0
        for value in bank.operands.values()
    )


def test_routed_lora_recipe_records_explicit_accumulation_policy() -> None:
    program = mechanisms.build_routed_lora_program(
        input_dim=5,
        output_dim=3,
        rank=2,
        candidate_count=4,
        contract_accumulation_dtype="activation",
        pointwise_accumulation_dtype="float32",
    )
    attributes = [dict(item.attributes) for item in program.instructions]
    assert [item["accumulation_dtype"] for item in attributes] == [
        "activation",
        "activation",
        "activation",
        "activation",
        "activation",
        "float32",
        "float32",
    ]


def test_formula_operand_bank_permutation_preserves_route_and_output() -> None:
    torch.manual_seed(9202)
    bank = _make_bank()
    program = mechanisms.build_routed_lora_program(
        input_dim=5,
        output_dim=3,
        rank=2,
        candidate_count=4,
        source_ref=bank.source_ref,
        bundle_id=bank.bundle_id,
        member_ids=bank.member_ids,
        dtype="float64",
    )
    query = torch.randn(5, bank.key_dim, dtype=torch.float64)
    x = torch.randn(5, 2, 5, dtype=torch.float64)
    base = torch.randn(5, 2, 3, dtype=torch.float64)

    def run(
        operand_bank: mechanisms.FormulaOperandBank,
        formula_program: mechanisms.FormulaProgram,
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        selection = operand_bank.route(query, estimator="hard")
        value = mechanisms.FormulaFabricV2(formula_program)(
            inputs={"x": x, "base": base, "formula.route": selection.route},
            banks=operand_bank.bind(formula_program),
        ).values[0]
        identities = tuple(operand_bank.member_ids[index] for index in selection.hard_indices)
        return value, identities

    original_value, original_identity = run(bank, program)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted_ids = tuple(bank.member_ids[index] for index in permutation.tolist())
    permuted_bank = mechanisms.FormulaOperandBank(
        keys=bank.keys.detach()[permutation],
        operands={
            name: value.detach()[permutation] for name, value in bank.operands.items()
        },
        source_ref=bank.source_ref,
        bundle_id=bank.bundle_id,
        member_ids=permuted_ids,
    )
    permuted_program = mechanisms.build_routed_lora_program(
        input_dim=5,
        output_dim=3,
        rank=2,
        candidate_count=4,
        source_ref=permuted_bank.source_ref,
        bundle_id=permuted_bank.bundle_id,
        member_ids=permuted_bank.member_ids,
        dtype="float64",
    )
    permuted_value, permuted_identity = run(permuted_bank, permuted_program)

    torch.testing.assert_close(permuted_value, original_value, rtol=1e-12, atol=1e-12)
    assert permuted_identity == original_identity
    assert permuted_program.fingerprint != program.fingerprint


def test_formula_operand_bank_binding_is_fail_closed_and_versioned() -> None:
    bank = _make_bank()
    program = mechanisms.build_routed_lora_program(
        input_dim=5,
        output_dim=3,
        rank=2,
        candidate_count=4,
        source_ref=bank.source_ref,
        bundle_id=bank.bundle_id,
        member_ids=bank.member_ids,
        dtype="float64",
    )
    assert arti.component_ref(bank) == "arti/formula-operand-bank@1"
    assert set(bank.bind(program)) == {"lora.A", "lora.B", "lora.gain"}
    provenance = arti.component_provenance(bank)
    assert arti.validate_component_provenance(provenance) == provenance

    forged = copy.deepcopy(provenance)
    root = forged["components"][0]
    root["config"]["member_ids"][1] = root["config"]["member_ids"][0]
    root["config_fingerprint"] = hashlib.sha256(
        json.dumps(root["config"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    forged["fingerprint"] = arti.component_graph_fingerprint(forged["components"])
    with pytest.raises(arti.ComponentCompatibilityError, match="config is invalid"):
        arti.validate_component_provenance(forged)

    wrong_program = mechanisms.build_routed_lora_program(
        input_dim=5,
        output_dim=3,
        rank=2,
        candidate_count=4,
        source_ref=bank.source_ref,
        bundle_id="other",
        dtype="float64",
    )
    with pytest.raises(mechanisms.FormulaBindingError, match="does not match"):
        bank.bind(wrong_program)

    fingerprinted_program = mechanisms.build_routed_lora_program(
        input_dim=5,
        output_dim=3,
        rank=2,
        candidate_count=4,
        source_ref=bank.source_ref,
        bundle_id=bank.bundle_id,
        member_ids=bank.member_ids,
        asset_fingerprint="a" * 64,
        dtype="float64",
    )
    with pytest.raises(mechanisms.FormulaBindingError, match="does not match"):
        bank.bind(fingerprinted_program)

    with pytest.raises(ValueError, match="canonical component reference"):
        mechanisms.FormulaOperandBank(
            keys=torch.randn(2, 3),
            operands={"A": torch.randn(2, 4)},
            source_ref="not-canonical",
        )

    with pytest.raises(ValueError, match="candidate axis"):
        mechanisms.FormulaOperandBank(
            keys=torch.randn(3, 4),
            operands={"A": torch.randn(2, 5)},
        )


def test_formula_operand_bank_state_roundtrip_does_not_own_query() -> None:
    torch.manual_seed(9204)
    bank = _make_bank()
    query = torch.randn(3, bank.key_dim, dtype=torch.float64, requires_grad=True)
    expected_parameter_names = {
        "keys",
        "operands.A",
        "operands.B",
        "operands.gain",
    }

    assert set(dict(bank.named_parameters())) == expected_parameter_names
    assert set(bank.state_dict()) == expected_parameter_names
    query_before = query.detach().clone()
    expected = bank.route(query, estimator="hard")

    restored = _make_bank()
    restored.load_state_dict(copy.deepcopy(bank.state_dict()))
    actual = restored.route(query, estimator="hard")
    state_contract = arti.component_state_contract(
        bank,
        bank.state_dict(),
        scope="trainable",
    )

    torch.testing.assert_close(query, query_before)
    torch.testing.assert_close(actual.route, expected.route)
    torch.testing.assert_close(actual.logits, expected.logits)
    assert actual.hard_indices.tolist() == expected.hard_indices.tolist()
    assert arti.component_spec(restored).parameter_schema_fingerprint == (
        arti.component_spec(bank).parameter_schema_fingerprint
    )
    assert arti.validate_component_state_contract(
        state_contract,
        state_dict=restored.state_dict(),
        model=restored,
    ) == state_contract


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_formula_execution_plan_cuda_inductor_matches_eager_and_gradients() -> None:
    torch.manual_seed(9203)
    bank = _make_bank().cuda()
    program = mechanisms.build_routed_lora_program(
        input_dim=5,
        output_dim=3,
        rank=2,
        candidate_count=4,
        source_ref=bank.source_ref,
        bundle_id=bank.bundle_id,
        member_ids=bank.member_ids,
        dtype="float64",
    )
    fabric = mechanisms.FormulaFabricV2(program).cuda()
    query = torch.randn(6, bank.key_dim, device="cuda", dtype=torch.float64)
    x = torch.randn(6, 3, 5, device="cuda", dtype=torch.float64, requires_grad=True)
    base = torch.randn(6, 3, 3, device="cuda", dtype=torch.float64, requires_grad=True)
    route = bank.route(query, estimator="straight-through").route
    prepared = fabric.bind_tensors(
        inputs={"x": x, "base": base, "formula.route": route},
        banks=bank.bind(program),
    )
    plan = fabric.execution_plan().cuda()
    eager = plan(prepared)[0]
    compiled = torch.compile(plan, fullgraph=True)
    actual = compiled(prepared)[0]

    torch.testing.assert_close(actual, eager, rtol=1e-12, atol=1e-12)
    gradients = torch.autograd.grad(actual.square().mean(), prepared.values)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
