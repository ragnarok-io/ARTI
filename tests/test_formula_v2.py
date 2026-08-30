from __future__ import annotations

import copy
import hashlib
import json

import pytest
import torch

import arti
import arti.formula_v2 as formula_v2
from arti import alpha


def _bind_banks(
    program: alpha.FormulaProgram, values: dict[str, torch.Tensor]
) -> dict[str, alpha.FormulaBankOperand]:
    return {
        binding.name: binding.bind(values[binding.name])
        for binding in program.bindings
        if isinstance(binding, alpha.BankBinding)
    }


def _run_lora(
    *,
    batch: int,
    sequence: int,
    input_dim: int,
    output_dim: int,
    rank: int,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, alpha.FormulaFabricV2, dict[str, torch.Tensor]]:
    program = alpha.build_lora_program(
        input_dim=input_dim,
        output_dim=output_dim,
        rank=rank,
        source_ref="arti/test-lora-bank@1",
        dtype=str(dtype).removeprefix("torch."),
    )
    fabric = alpha.FormulaFabricV2(program)
    values = {
        "x": torch.randn(batch, sequence, input_dim, dtype=dtype, requires_grad=True),
        "base": torch.randn(batch, sequence, output_dim, dtype=dtype, requires_grad=True),
        "lora.gain": torch.randn((), dtype=dtype, requires_grad=True),
        "lora.A": torch.randn(rank, input_dim, dtype=dtype, requires_grad=True),
        "lora.B": torch.randn(output_dim, rank, dtype=dtype, requires_grad=True),
    }
    actual = fabric(
        inputs={key: values[key] for key in ("x", "base", "lora.gain")},
        banks=_bind_banks(program, values),
        return_trace=True,
    ).values[0]
    hidden = torch.matmul(values["x"], values["lora.A"].transpose(-1, -2))
    expected = values["base"] + values["lora.gain"] * torch.matmul(
        hidden, values["lora.B"].transpose(-1, -2)
    )
    return actual, expected, fabric, values


@pytest.mark.parametrize(
    ("batch", "sequence", "input_dim", "output_dim", "rank"),
    ((1, 1, 3, 2, 1), (2, 4, 7, 5, 2), (3, 2, 5, 9, 4)),
)
def test_formula_v2_exact_rank_r_lora(
    batch: int, sequence: int, input_dim: int, output_dim: int, rank: int
) -> None:
    torch.manual_seed(9100 + rank)
    actual, expected, fabric, _values = _run_lora(
        batch=batch,
        sequence=sequence,
        input_dim=input_dim,
        output_dim=output_dim,
        rank=rank,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert tuple(item.atom_ref for item in fabric.program.instructions) == (
        "arti/formula-atom-contract@1",
        "arti/formula-atom-contract@1",
        "arti/formula-atom-scale@1",
        "arti/formula-atom-add@1",
    )
    assert sum(parameter.numel() for parameter in fabric.parameters()) == 0


def test_formula_v2_matches_complete_lora_gradients() -> None:
    torch.manual_seed(9111)
    actual, expected, _fabric, values = _run_lora(
        batch=2, sequence=3, input_dim=6, output_dim=4, rank=3
    )
    ordered = tuple(values[key] for key in ("x", "base", "lora.gain", "lora.A", "lora.B"))

    actual_gradients = torch.autograd.grad(actual.square().mean(), ordered, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().mean(), ordered)

    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)
        assert torch.isfinite(actual_gradient).all()


def test_formula_v2_is_input_and_rank_permutation_equivariant() -> None:
    torch.manual_seed(9121)
    batch, sequence, input_dim, output_dim, rank = 2, 3, 7, 5, 4
    program = alpha.build_lora_program(
        input_dim=input_dim,
        output_dim=output_dim,
        rank=rank,
        source_ref="arti/test-lora-bank@1",
        dtype="float64",
    )
    fabric = alpha.FormulaFabricV2(program)
    x = torch.randn(batch, sequence, input_dim, dtype=torch.float64)
    base = torch.randn(batch, sequence, output_dim, dtype=torch.float64)
    a = torch.randn(rank, input_dim, dtype=torch.float64)
    b = torch.randn(output_dim, rank, dtype=torch.float64)
    gain = torch.tensor(0.7, dtype=torch.float64)

    def run(x_value: torch.Tensor, a_value: torch.Tensor, b_value: torch.Tensor) -> torch.Tensor:
        bank_values = {"lora.A": a_value, "lora.B": b_value}
        return fabric(
            inputs={"x": x_value, "base": base, "lora.gain": gain},
            banks=_bind_banks(program, bank_values),
        ).values[0]

    original = run(x, a, b)
    input_permutation = torch.randperm(input_dim)
    rank_permutation = torch.randperm(rank)
    torch.testing.assert_close(
        run(x[..., input_permutation], a[:, input_permutation], b), original
    )
    torch.testing.assert_close(
        run(x, a[rank_permutation], b[:, rank_permutation]), original
    )


def test_formula_v2_explicit_k_bank_reduce_matches_reference() -> None:
    torch.manual_seed(9131)
    batch, sequence, members, input_dim, output_dim, rank = 2, 3, 4, 6, 5, 3
    program = alpha.build_lora_program(
        input_dim=input_dim,
        output_dim=output_dim,
        rank=rank,
        source_ref="arti/test-lora-bank@1",
        member_count=members,
        dtype="float64",
    )
    fabric = alpha.FormulaFabricV2(program)
    x = torch.randn(batch, sequence, input_dim, dtype=torch.float64, requires_grad=True)
    base = torch.randn(batch, sequence, output_dim, dtype=torch.float64, requires_grad=True)
    a = torch.randn(members, rank, input_dim, dtype=torch.float64, requires_grad=True)
    b = torch.randn(members, output_dim, rank, dtype=torch.float64, requires_grad=True)
    gain = torch.randn(members, dtype=torch.float64, requires_grad=True)

    result = fabric(
        inputs={"x": x, "base": base, "lora.gain": gain},
        banks=_bind_banks(program, {"lora.A": a, "lora.B": b}),
        return_trace=True,
    )
    hidden = torch.einsum("bsi,kri->bskr", x, a)
    deltas = torch.einsum("bskr,kor->bsko", hidden, b)
    expected = base + (deltas * gain.reshape(1, 1, members, 1)).sum(dim=2)

    torch.testing.assert_close(result.values[0], expected, rtol=1e-12, atol=1e-12)
    assert result.trace is not None
    assert result.trace.atom_refs[-2:] == (
        "arti/formula-atom-reduce@1",
        "arti/formula-atom-add@1",
    )


def test_formula_v2_k_bank_matches_complete_gradients_and_member_permutation() -> None:
    torch.manual_seed(9132)
    members, rank, input_dim, output_dim = 5, 3, 7, 4
    member_ids = tuple(f"expert-{index}" for index in range(members))
    program = alpha.build_lora_program(
        input_dim=input_dim,
        output_dim=output_dim,
        rank=rank,
        source_ref="arti/test-lora-bank@1",
        member_count=members,
        member_ids=member_ids,
        dtype="float64",
    )
    bindings = {
        binding.name: binding
        for binding in program.bindings
        if isinstance(binding, alpha.BankBinding)
    }
    assert bindings["lora.A"].bundle_id == bindings["lora.B"].bundle_id == "lora"
    assert bindings["lora.A"].member_ids == bindings["lora.B"].member_ids == member_ids

    x = torch.randn(2, 3, input_dim, dtype=torch.float64, requires_grad=True)
    base = torch.randn(2, 3, output_dim, dtype=torch.float64, requires_grad=True)
    a = torch.randn(members, rank, input_dim, dtype=torch.float64, requires_grad=True)
    b = torch.randn(members, output_dim, rank, dtype=torch.float64, requires_grad=True)
    gain = torch.randn(members, dtype=torch.float64, requires_grad=True)
    fabric = alpha.FormulaFabricV2(program)
    actual = fabric(
        inputs={"x": x, "base": base, "lora.gain": gain},
        banks=_bind_banks(program, {"lora.A": a, "lora.B": b}),
    ).values[0]
    hidden = torch.einsum("bsi,kri->bskr", x, a)
    deltas = torch.einsum("bskr,kor->bsko", hidden, b)
    expected = base + (deltas * gain.reshape(1, 1, members, 1)).sum(dim=2)

    ordered = (x, base, gain, a, b)
    actual_gradients = torch.autograd.grad(actual.square().mean(), ordered, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.square().mean(), ordered)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)

    permutation = torch.randperm(members)
    permuted_ids = tuple(member_ids[index] for index in permutation.tolist())
    permuted_program = alpha.build_lora_program(
        input_dim=input_dim,
        output_dim=output_dim,
        rank=rank,
        source_ref="arti/test-lora-bank@1",
        member_count=members,
        member_ids=permuted_ids,
        dtype="float64",
    )
    permuted = alpha.FormulaFabricV2(permuted_program)(
        inputs={"x": x, "base": base, "lora.gain": gain[permutation]},
        banks=_bind_banks(
            permuted_program,
            {"lora.A": a[permutation], "lora.B": b[permutation]},
        ),
    ).values[0]
    torch.testing.assert_close(permuted, actual, rtol=1e-12, atol=1e-12)


def test_formula_v2_rejects_mismatched_bank_bundle_member_order() -> None:
    payload = alpha.build_lora_program(
        input_dim=4,
        output_dim=3,
        rank=2,
        source_ref="arti/test-lora-bank@1",
        member_count=3,
    ).to_dict()
    bank_b = next(binding for binding in payload["bindings"] if binding["name"] == "lora.B")
    bank_b["member_ids"] = list(reversed(bank_b["member_ids"]))

    with pytest.raises(alpha.FormulaProgramError, match="inconsistent source or member order"):
        alpha.FormulaProgram.from_dict(payload)


def test_dot_is_a_canonical_contract_alias() -> None:
    vector = alpha.TensorType.axes(("B", "D"), sizes=(None, 5), dtype="float32")
    weight = alpha.TensorType.axes(("R", "D"), sizes=(3, 5), dtype="float32")
    left = alpha.InputBinding("x", vector)
    right = alpha.BankBinding("bank.A", "arti/test-bank@1", "A", weight)
    dot_program = alpha.FormulaProgram.build(
        outputs=(alpha.dot(left, right, left_axis="D", right_axis="D"),)
    )
    contract_program = alpha.FormulaProgram.build(
        outputs=(alpha.contract(left, right, reduce_axes=(("D", "D"),)),)
    )

    assert dot_program.to_dict() == contract_program.to_dict()
    assert dot_program.fingerprint == contract_program.fingerprint
    assert dot_program.instructions[0].atom_ref == "arti/formula-atom-contract@1"


def test_formula_v2_program_json_and_state_reload_are_exact() -> None:
    torch.manual_seed(9141)
    program = alpha.build_lora_program(
        input_dim=5,
        output_dim=4,
        rank=2,
        source_ref="arti/test-lora-bank@1",
        member_count=3,
        dtype="float32",
    )
    payload = json.loads(json.dumps(program.to_dict()))
    assert payload["schema_ref"] == "arti/formula-program@2"
    assert payload["limits"]["schema_ref"] == "arti/formula-limits@1"
    assert payload["slots"][0]["value_type"]["schema_ref"] == "arti/formula-tensor-type@1"
    assert payload["instructions"][0]["attributes"]["reduce_axes"] == [["Din", "Din"]]
    restored_program = alpha.FormulaProgram.from_dict(payload)
    assert restored_program.fingerprint == program.fingerprint
    assert restored_program.to_dict() == payload

    first = alpha.FormulaFabricV2(program)
    second = alpha.FormulaFabricV2(restored_program)
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    inputs = {
        "x": torch.randn(2, 3, 5),
        "base": torch.randn(2, 3, 4),
        "lora.gain": torch.randn(3),
    }
    bank_values = {"lora.A": torch.randn(3, 2, 5), "lora.B": torch.randn(3, 4, 2)}
    banks = _bind_banks(program, bank_values)
    restored_banks = _bind_banks(restored_program, bank_values)
    first_value = first(inputs=inputs, banks=banks).values[0]
    second_value = second(inputs=inputs, banks=restored_banks).values[0]
    torch.testing.assert_close(first_value, second_value)

    traced = first(inputs=inputs, banks=banks, return_trace=True).trace
    assert traced is not None
    restored_trace = alpha.FormulaTraceV2.from_dict(
        json.loads(json.dumps(traced.to_dict()))
    )
    assert alpha.FORMULA_TRACE_V1_SCHEMA_VERSION == 1
    assert restored_trace == traced
    assert restored_trace.to_dict()["schema_ref"] == "arti/formula-trace@1"
    assert restored_trace.fingerprint == traced.fingerprint
    restored_trace.verify(restored_program)
    forged_trace = restored_trace.to_dict()
    forged_trace["atom_refs"][0] = "arti/formula-atom-add@1"
    with pytest.raises(alpha.FormulaSchemaError, match="does not match"):
        alpha.FormulaTraceV2.from_dict(forged_trace).verify(restored_program)


def test_formula_v2_components_have_canonical_references_and_dependencies() -> None:
    vector = alpha.TensorType.axes(("B", "D"), sizes=(None, 4))
    matrix = alpha.TensorType.axes(("R", "D"), sizes=(2, 4))
    contract_atom = alpha.ContractAtom(
        vector, matrix, reduce_axes=(("D", "D"),), output_axes=("B", "R")
    )
    scale_atom = alpha.ScaleAtom(vector, alpha.TensorType.scalar())
    add_atom = alpha.AddAtom(vector)
    reduce_atom = alpha.ReduceAtom(vector, axis="D")
    assert arti.component_ref(contract_atom) == "arti/formula-atom-contract@1"
    assert arti.component_ref(scale_atom) == "arti/formula-atom-scale@1"
    assert arti.component_ref(add_atom) == "arti/formula-atom-add@1"
    assert arti.component_ref(reduce_atom) == "arti/formula-atom-reduce@1"

    fabric = alpha.FormulaFabricV2(
        alpha.build_lora_program(
            input_dim=4,
            output_dim=3,
            rank=2,
            source_ref="arti/test-lora-bank@1",
            member_count=2,
        )
    )
    spec = arti.component_spec(fabric)
    assert arti.component_ref(fabric) == "arti/formula-fabric@2"
    assert set(spec.dependencies) == {
        "arti/formula-atom-contract@1",
        "arti/formula-atom-scale@1",
        "arti/formula-atom-add@1",
        "arti/formula-atom-reduce@1",
    }

    provenance = arti.component_provenance(fabric)
    assert arti.validate_component_provenance(provenance) == provenance
    forged = copy.deepcopy(provenance)
    root = forged["components"][0]
    root["config"]["program_fingerprint"] = "0" * 64
    root["config_fingerprint"] = hashlib.sha256(
        json.dumps(root["config"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    forged["fingerprint"] = arti.component_graph_fingerprint(forged["components"])
    with pytest.raises(arti.ComponentCompatibilityError, match="program fingerprint"):
        arti.validate_component_provenance(forged)


def test_formula_fabric_v1_identity_remains_frozen() -> None:
    legacy_program = alpha.FormulaFabricProgram(
        arena_capacity=4,
        feature_dim=2,
        steps=(
            (alpha.FormulaInvocation(alpha.FormulaPrimitive.ADD, 2),),
            (alpha.FormulaInvocation(alpha.FormulaPrimitive.MULTIPLY, 3),),
        ),
    )
    legacy = alpha.FormulaFabric(legacy_program)

    assert legacy_program.fingerprint == (
        "ae99758c53615b27095201ed0c4fece21df276d79e14c6fc9ac5f5cea1770d71"
    )
    assert arti.component_ref(legacy) == "arti/formula-fabric@1"
    spec = arti.component_spec(legacy)
    assert spec.config_schema_version == 2
    assert spec.config_fingerprint == (
        "01b23825275a0d6733195d49ca295d82b897814ac95afecde0a6d620e29dec63"
    )


def test_formula_v2_rejects_missing_unknown_and_wrong_shape_bindings() -> None:
    program = alpha.build_lora_program(
        input_dim=5,
        output_dim=4,
        rank=2,
        source_ref="arti/test-lora-bank@1",
    )
    fabric = alpha.FormulaFabricV2(program)
    inputs = {
        "x": torch.randn(2, 3, 5),
        "base": torch.randn(2, 3, 4),
        "lora.gain": torch.tensor(1.0),
    }
    bank_values = {"lora.A": torch.randn(2, 5), "lora.B": torch.randn(4, 2)}
    banks = _bind_banks(program, bank_values)

    with pytest.raises(alpha.FormulaBindingError, match="keys must be"):
        fabric(inputs={"x": inputs["x"], "base": inputs["base"]}, banks=banks)
    with pytest.raises(alpha.FormulaBindingError, match="keys must be"):
        fabric(inputs={**inputs, "unknown": torch.tensor(0.0)}, banks=banks)
    with pytest.raises(alpha.FormulaBindingError, match="expected 5"):
        fabric(inputs={**inputs, "x": torch.randn(2, 3, 6)}, banks=banks)
    with pytest.raises(alpha.FormulaBindingError, match="conflicting extents"):
        fabric(inputs={**inputs, "base": torch.randn(4, 3, 4)}, banks=banks)
    with pytest.raises(alpha.FormulaBindingError, match="requires FormulaBankOperand"):
        fabric(inputs=inputs, banks=bank_values)

    mixed_dtype_inputs = {**inputs, "base": inputs["base"].to(torch.float64)}
    with pytest.raises(alpha.FormulaBindingError, match="different dtypes"):
        fabric(inputs=mixed_dtype_inputs, banks=banks)
    with pytest.raises(alpha.FormulaBindingError, match="different dtypes"):
        fabric.bind_tensors(inputs=mixed_dtype_inputs, banks=banks)

    wrong_identity = dict(banks)
    wrong_identity["lora.A"] = alpha.FormulaBankOperand(
        bank_values["lora.A"],
        source_ref="arti/wrong-bank@1",
        partition_id="A",
    )
    with pytest.raises(alpha.FormulaBindingError, match="does not match"):
        fabric(inputs=inputs, banks=wrong_identity)


def test_formula_v2_rejects_hidden_or_ill_typed_program_data() -> None:
    with pytest.raises(alpha.FormulaTypeError, match="different domains"):
        alpha.add(
            alpha.InputBinding("a", alpha.TensorType.scalar(domain="a")),
            alpha.InputBinding("b", alpha.TensorType.scalar(domain="b")),
        )
    with pytest.raises(alpha.FormulaTypeError, match="must be a permutation"):
        alpha.contract(
            alpha.InputBinding("x", alpha.TensorType.axes(("B", "D"))),
            alpha.InputBinding("a", alpha.TensorType.axes(("R", "D"))),
            reduce_axes=(("D", "D"),),
            output_axes=("B",),
        )
    with pytest.raises(alpha.FormulaTypeError, match="exactly one reduction"):
        alpha.contract(
            alpha.InputBinding("x", alpha.TensorType.axes(("B", "I", "J"))),
            alpha.InputBinding("a", alpha.TensorType.axes(("R", "I", "J"))),
            reduce_axes=(("I", "I"), ("J", "J")),
        )
    with pytest.raises(alpha.FormulaTypeError, match="alias preserved axes"):
        alpha.contract(
            alpha.InputBinding("x", alpha.TensorType.axes(("B", "D"))),
            alpha.InputBinding("a", alpha.TensorType.axes(("D", "R"))),
            reduce_axes=(("D", "R"),),
        )

    payload = alpha.build_lora_program(
        input_dim=3, output_dim=2, rank=1, source_ref="arti/test-bank@1"
    ).to_dict()
    payload["hidden_parameter"] = 1
    with pytest.raises(alpha.FormulaSchemaError, match="missing or unknown"):
        alpha.FormulaProgram.from_dict(payload)

    payload = alpha.build_lora_program(
        input_dim=3, output_dim=2, rank=1, source_ref="arti/test-bank@1"
    ).to_dict()
    payload["instructions"][0]["attributes"]["unknown"] = True
    with pytest.raises(alpha.FormulaProgramError, match="requires arity"):
        alpha.FormulaProgram.from_dict(payload)

    payload = alpha.build_lora_program(
        input_dim=3,
        output_dim=2,
        rank=1,
        member_count=2,
        source_ref="arti/test-bank@1",
    ).to_dict()
    reduce_instruction = next(
        item
        for item in payload["instructions"]
        if item["atom_ref"] == "arti/formula-atom-reduce@1"
    )
    reduce_instruction["attributes"]["mode"] = "max"
    with pytest.raises(alpha.FormulaProgramError, match="only supports mode='sum'"):
        alpha.FormulaProgram.from_dict(payload)

    payload = alpha.build_lora_program(
        input_dim=3, output_dim=2, rank=1, source_ref="arti/test-bank@1"
    ).to_dict()
    scale_instruction = next(
        item
        for item in payload["instructions"]
        if item["atom_ref"] == "arti/formula-atom-scale@1"
    )
    scale_instruction["attributes"]["factor_axes"] = ["B"]
    with pytest.raises(alpha.FormulaProgramError, match="factor_axes must exactly match"):
        alpha.FormulaProgram.from_dict(payload)


def test_formula_v2_standalone_atoms_share_runtime_validation() -> None:
    vector = alpha.TensorType.axes(("B", "D"), sizes=(None, 4))
    matrix = alpha.TensorType.axes(("R", "D"), sizes=(2, 4))
    contract_atom = alpha.ContractAtom(
        vector, matrix, reduce_axes=(("D", "D"),), output_axes=("B", "R")
    )

    with pytest.raises(alpha.FormulaBindingError, match="same dtype"):
        contract_atom(torch.randn(3, 4, dtype=torch.float32), torch.randn(2, 4, dtype=torch.float64))

    scale_atom = alpha.ScaleAtom(
        alpha.TensorType.axes(("B", "K", "D"), sizes=(None, None, 4)),
        alpha.TensorType.axes(("K",), sizes=(None,)),
    )
    with pytest.raises(alpha.FormulaBindingError, match="conflicting extents"):
        scale_atom(torch.randn(2, 3, 4), torch.randn(5))


def test_formula_v2_rejects_instruction_writes_to_binding_slots() -> None:
    payload = alpha.build_lora_program(
        input_dim=3, output_dim=2, rank=1, source_ref="arti/test-bank@1"
    ).to_dict()
    payload["instructions"].append(
        {
            "instruction_id": "i-extra",
            "step": 5,
            "atom_ref": "arti/formula-atom-add@1",
            "input_slots": ["base", "%2"],
            "output_slot": "base",
            "attributes": {"accumulation_dtype": "activation"},
        }
    )

    with pytest.raises(alpha.FormulaProgramError, match="own produced slot"):
        alpha.FormulaProgram.from_dict(payload)


def test_formula_v2_rejects_malformed_slot_and_output_sequences() -> None:
    with pytest.raises(alpha.FormulaProgramError, match="instruction-produced"):
        alpha.FormulaProgram.build(
            outputs=(alpha.InputBinding("x", alpha.TensorType.axes(("B", "D"))),)
        )

    payload = alpha.build_lora_program(
        input_dim=3, output_dim=2, rank=1, source_ref="arti/test-bank@1"
    ).to_dict()
    payload["instructions"][0]["input_slots"] = "x"
    with pytest.raises(alpha.FormulaSchemaError, match="input_slots must be a sequence"):
        alpha.FormulaProgram.from_dict(payload)

    payload = alpha.build_lora_program(
        input_dim=3, output_dim=2, rank=1, source_ref="arti/test-bank@1"
    ).to_dict()
    payload["outputs"] = payload["outputs"] * 2
    with pytest.raises(alpha.FormulaProgramError, match="must not contain duplicates"):
        alpha.FormulaProgram.from_dict(payload)


def test_formula_v2_scale_and_add_accumulation_policy_is_explicit() -> None:
    value_type = alpha.TensorType.axes(("B", "D"), sizes=(2, 4), dtype="float16")
    factor_type = alpha.TensorType.scalar(dtype="float16")
    value = torch.randn(2, 4, dtype=torch.float16)
    factor = torch.tensor(0.75, dtype=torch.float16)

    scale_atom = alpha.ScaleAtom(
        value_type, factor_type, accumulation_dtype="float32"
    )
    add_atom = alpha.AddAtom(value_type, accumulation_dtype="float32")
    scaled = scale_atom(value, factor)
    actual = add_atom(value, scaled)
    scaled_expected = value.float().mul(factor.float()).to(dtype=torch.float16)
    expected = (value.float() + scaled_expected.float()).to(dtype=torch.float16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    expression = alpha.add(
        alpha.InputBinding("left", value_type),
        alpha.scale(
            alpha.InputBinding("value", value_type),
            alpha.InputBinding("factor", factor_type),
            accumulation_dtype="float32",
        ),
        accumulation_dtype="float32",
    )
    program = alpha.FormulaProgram.build(outputs=(expression,))
    attributes = [dict(item.attributes) for item in program.instructions]
    assert attributes == [
        {"accumulation_dtype": "float32", "factor_axes": ()},
        {"accumulation_dtype": "float32"},
    ]

def test_formula_v2_limits_are_serialized_and_enforced() -> None:
    with pytest.raises(alpha.FormulaProgramError, match="instruction count"):
        alpha.build_lora_program(
            input_dim=3,
            output_dim=2,
            rank=1,
            source_ref="arti/test-bank@1",
            limits=alpha.FormulaLimits(max_instructions=3),
        )

    limits = alpha.FormulaLimits(max_tensor_elements=10)
    program = alpha.build_lora_program(
        input_dim=3,
        output_dim=2,
        rank=1,
        source_ref="arti/test-bank@1",
        limits=limits,
    )
    default_program = alpha.build_lora_program(
        input_dim=3, output_dim=2, rank=1, source_ref="arti/test-bank@1"
    )
    assert program.fingerprint != default_program.fingerprint
    assert program.to_dict()["limits"] == limits.to_dict()

    fabric = alpha.FormulaFabricV2(program)
    values = {
        "x": torch.randn(2, 2, 3),
        "base": torch.randn(2, 2, 2),
        "lora.gain": torch.tensor(1.0),
        "lora.A": torch.randn(1, 3),
        "lora.B": torch.randn(2, 1),
    }
    with pytest.raises(alpha.FormulaBindingError, match="element limit"):
        fabric(
            inputs={key: values[key] for key in ("x", "base", "lora.gain")},
            banks=_bind_banks(program, values),
        )


def test_lora_recipe_records_explicit_accumulation_policy() -> None:
    program = alpha.build_lora_program(
        input_dim=3,
        output_dim=2,
        rank=1,
        source_ref="arti/test-bank@1",
        contract_accumulation_dtype="activation",
        pointwise_accumulation_dtype="float32",
    )
    attributes = [dict(item.attributes) for item in program.instructions]
    assert [item["accumulation_dtype"] for item in attributes] == [
        "activation",
        "activation",
        "float32",
        "float32",
    ]


def test_formula_preflight_accounts_for_float32_working_tensors() -> None:
    limits = alpha.FormulaLimits(max_tensor_bytes=20)
    safe_program = alpha.build_lora_program(
        input_dim=3,
        output_dim=2,
        rank=1,
        source_ref="arti/test-bank@1",
        dtype="float16",
        contract_accumulation_dtype="activation",
        limits=limits,
    )
    strict_program = alpha.build_lora_program(
        input_dim=3,
        output_dim=2,
        rank=1,
        source_ref="arti/test-bank@1",
        dtype="float16",
        contract_accumulation_dtype="float32",
        limits=limits,
    )
    values = {
        "x": torch.randn(1, 2, 3, dtype=torch.float16),
        "base": torch.randn(1, 2, 2, dtype=torch.float16),
        "lora.gain": torch.tensor(1.0, dtype=torch.float16),
        "lora.A": torch.randn(1, 3, dtype=torch.float16),
        "lora.B": torch.randn(2, 1, dtype=torch.float16),
    }
    inputs = {key: values[key] for key in ("x", "base", "lora.gain")}
    alpha.FormulaFabricV2(safe_program).bind_tensors(
        inputs=inputs,
        banks=_bind_banks(safe_program, values),
    )
    with pytest.raises(alpha.FormulaBindingError, match="byte limit"):
        alpha.FormulaFabricV2(strict_program).bind_tensors(
            inputs=inputs,
            banks=_bind_banks(strict_program, values),
        )


def test_formula_preflight_limits_aggregate_live_working_set() -> None:
    safe_limits = alpha.FormulaLimits(max_working_bytes=84)
    strict_limits = alpha.FormulaLimits(max_working_bytes=83)
    safe_program = alpha.build_lora_program(
        input_dim=3,
        output_dim=2,
        rank=1,
        source_ref="arti/test-bank@1",
        dtype="float16",
        limits=safe_limits,
    )
    strict_program = alpha.build_lora_program(
        input_dim=3,
        output_dim=2,
        rank=1,
        source_ref="arti/test-bank@1",
        dtype="float16",
        limits=strict_limits,
    )
    values = {
        "x": torch.randn(1, 2, 3, dtype=torch.float16),
        "base": torch.randn(1, 2, 2, dtype=torch.float16),
        "lora.gain": torch.tensor(1.0, dtype=torch.float16),
        "lora.A": torch.randn(1, 3, dtype=torch.float16),
        "lora.B": torch.randn(2, 1, dtype=torch.float16),
    }
    inputs = {key: values[key] for key in ("x", "base", "lora.gain")}

    alpha.FormulaFabricV2(safe_program).bind_tensors(
        inputs=inputs,
        banks=_bind_banks(safe_program, values),
    )
    with pytest.raises(alpha.FormulaBindingError, match="aggregate working byte limit"):
        alpha.FormulaFabricV2(strict_program).bind_tensors(
            inputs=inputs,
            banks=_bind_banks(strict_program, values),
        )
    assert safe_program.fingerprint != strict_program.fingerprint
    assert safe_program.to_dict()["limits"]["max_working_bytes"] == 84


def test_formula_v2_rejects_dynamic_output_before_tensor_execution(monkeypatch) -> None:
    value_type = alpha.TensorType.axes(
        ("B", "D"), sizes=(None, 2), dtype="float32"
    )
    bank_type = alpha.TensorType.axes(("R", "D"), sizes=(20, 2), dtype="float32")
    x = alpha.InputBinding("x", value_type)
    bank = alpha.BankBinding("bank", "arti/test-bank@1", "A", bank_type)
    program = alpha.FormulaProgram.build(
        outputs=(alpha.contract(x, bank, reduce_axes=(("D", "D"),)),),
        limits=alpha.FormulaLimits(max_tensor_elements=100),
    )
    fabric = alpha.FormulaFabricV2(program)
    called = False

    def fail_if_executed(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("instruction executed before allocation admission")

    monkeypatch.setattr(formula_v2, "_execute_instruction", fail_if_executed)
    with pytest.raises(alpha.FormulaBindingError, match="output allocation"):
        fabric(
            inputs={"x": torch.randn(20, 2)},
            banks={"bank": bank.bind(torch.randn(20, 2))},
        )
    assert not called


def test_formula_v2_bank_sources_require_canonical_component_references() -> None:
    with pytest.raises(alpha.FormulaSchemaError, match="canonical component reference"):
        alpha.BankBinding(
            "bank",
            "not-a-component",
            "A",
            alpha.TensorType.axes(("R", "D"), sizes=(2, 3)),
        )


def test_formula_v2_direct_constructors_reject_string_sequences() -> None:
    value_type = alpha.TensorType.axes(("D",), sizes=(2,))
    with pytest.raises(alpha.FormulaSchemaError, match="axes and sizes must be sequences"):
        alpha.TensorType("D", (2,))
    with pytest.raises(alpha.FormulaSchemaError, match="axes and sizes must be sequences"):
        alpha.TensorType(("D",), "2")
    with pytest.raises(alpha.FormulaSchemaError, match="member_ids must be a sequence"):
        alpha.BankBinding(
            "bank",
            "arti/test-bank@1",
            "A",
            value_type,
            bundle_id="bundle",
            member_ids="ab",
        )
    with pytest.raises(alpha.FormulaBindingError, match="member_ids must be a sequence"):
        alpha.FormulaBankOperand(
            torch.randn(2),
            "arti/test-bank@1",
            "A",
            bundle_id="bundle",
            member_ids="ab",
        )
    with pytest.raises(alpha.FormulaSchemaError, match="input_slots must be a sequence"):
        alpha.FormulaInstructionV2(
            "i0",
            1,
            "arti/formula-atom-add@1",
            "xy",
            "%0",
            (("accumulation_dtype", "activation"),),
        )
    program = alpha.build_lora_program(
        input_dim=2,
        output_dim=2,
        rank=1,
        source_ref="arti/test-bank@1",
    )
    with pytest.raises(alpha.FormulaSchemaError, match="outputs must be a sequence"):
        alpha.FormulaProgram(
            program.bindings,
            program.slots,
            program.instructions,
            program.outputs[0],
            limits=program.limits,
        )
    with pytest.raises(alpha.FormulaSchemaError, match="trace identifiers must be sequences"):
        alpha.FormulaTraceV2(program.fingerprint, "i0", ("atom",), ("output",))


def test_contract_rejects_unequal_dynamic_reduction_extents() -> None:
    left = alpha.InputBinding(
        "left", alpha.TensorType.axes(("B", "I"), sizes=(None, None))
    )
    right = alpha.InputBinding(
        "right", alpha.TensorType.axes(("R", "J"), sizes=(None, None))
    )
    program = alpha.FormulaProgram.build(
        outputs=(
            alpha.contract(
                left,
                right,
                reduce_axes=(("I", "J"),),
                output_axes=("B", "R"),
            ),
        )
    )
    with pytest.raises(alpha.FormulaBindingError, match="reduction extents differ"):
        alpha.FormulaFabricV2(program)(
            inputs={"left": torch.randn(2, 3), "right": torch.randn(4, 5)},
            banks={},
        )


def test_formula_execution_plan_is_positional_compilable_and_versioned() -> None:
    torch.manual_seed(9161)
    program = alpha.build_lora_program(
        input_dim=5,
        output_dim=4,
        rank=2,
        member_count=3,
        source_ref="arti/test-bank@1",
        dtype="float32",
    )
    fabric = alpha.FormulaFabricV2(program)
    values = {
        "x": torch.randn(2, 3, 5, requires_grad=True),
        "base": torch.randn(2, 3, 4, requires_grad=True),
        "lora.gain": torch.randn(3, requires_grad=True),
        "lora.A": torch.randn(3, 2, 5, requires_grad=True),
        "lora.B": torch.randn(3, 4, 2, requires_grad=True),
    }
    inputs = {key: values[key] for key in ("x", "base", "lora.gain")}
    banks = _bind_banks(program, values)
    expected = fabric(inputs=inputs, banks=banks).values
    prepared = fabric.bind_tensors(inputs=inputs, banks=banks)
    plan = fabric.execution_plan()
    actual = plan(prepared)
    compiled = torch.compile(plan, backend="eager", fullgraph=True)
    compiled_actual = compiled(prepared)

    assert arti.component_ref(plan) == "arti/formula-execution-plan@1"
    assert arti.validate_component_provenance(arti.component_provenance(plan))
    assert plan.binding_names == tuple(binding.name for binding in program.bindings)
    assert prepared.binding_names == plan.binding_names
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(compiled_actual[0], expected[0])
    gradients = torch.autograd.grad(compiled_actual[0].square().mean(), prepared.values)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    with pytest.raises(TypeError, match="PreparedFormulaBindings"):
        plan(prepared.values)
    forged = alpha.PreparedFormulaBindings(
        "0" * 64,
        prepared.binding_names,
        prepared.values,
    )
    with pytest.raises(alpha.FormulaBindingError, match="different program"):
        plan(forged)
