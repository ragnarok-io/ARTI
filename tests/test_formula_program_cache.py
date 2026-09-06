from __future__ import annotations

import hashlib
import json
import gc
import weakref
from dataclasses import asdict, fields, replace

import pytest
import torch

import arti.formula_v2 as formula_v2
from arti import mechanisms


def _program(*, dtype: str = "float64", limits=None) -> mechanisms.FormulaProgram:
    value_type = mechanisms.TensorType(("B", "D"), ("B", 3), dtype=dtype, domain="activation")
    weight_type = mechanisms.TensorType(("D",), (3,), dtype=dtype, domain="activation")
    value = mechanisms.InputBinding("x", value_type)
    weight = mechanisms.BankBinding("weight", "arti/program-cache-test@1", "weight", weight_type)
    output = mechanisms.add(mechanisms.scalar_map(mechanisms.scale(value, weight), mode="silu"), value)
    kwargs = {} if limits is None else {"limits": limits}
    return mechanisms.FormulaProgram.build(outputs=(output,), **kwargs)


def _values(program, *, batch: int = 2, dtype: torch.dtype = torch.float64):
    value = (torch.arange(batch * 3, dtype=dtype).reshape(batch, 3) / 7 - 0.4).requires_grad_()
    weight = torch.tensor([0.5, -0.25, 1.25], dtype=dtype, requires_grad=True)
    binding = next(item for item in program.bindings if isinstance(item, mechanisms.BankBinding))
    return {"x": value}, {"weight": binding.bind(weight)}


def _canonical_fingerprint(program) -> str:
    payload = json.dumps(program.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def test_program_fingerprint_serialized_once_across_admission_trace_and_plan(monkeypatch) -> None:
    program = _program()
    expected = _canonical_fingerprint(program)
    original = mechanisms.FormulaProgram.to_dict
    calls = []

    def record(value):
        calls.append(id(value))
        return original(value)

    monkeypatch.setattr(mechanisms.FormulaProgram, "to_dict", record)
    fabric = mechanisms.FormulaFabricV2(program)
    inputs, banks = _values(program)
    for _ in range(3):
        assert program.fingerprint == expected
        assert fabric.bind_tensors(inputs=inputs, banks=banks).program_fingerprint == expected
        assert fabric(inputs=inputs, banks=banks, return_trace=True).trace.program_fingerprint == expected
        assert fabric.execution_plan().program_fingerprint == expected
    assert calls == [id(program)]


def test_program_cache_is_not_a_dataclass_or_serialized_field() -> None:
    program = _program()
    wire = program.to_dict()
    structural = asdict(program)
    previous_hash, previous_repr = hash(program), repr(program)
    lookup = {program: "original"}
    assert "fingerprint" not in vars(program)
    assert program.fingerprint == _canonical_fingerprint(program)
    assert tuple(field.name for field in fields(program)) == (
        "bindings", "slots", "instructions", "outputs", "limits", "schema_version",
    )
    assert program.to_dict() == wire
    assert asdict(program) == structural
    assert hash(program) == previous_hash and repr(program) == previous_repr
    restored = mechanisms.FormulaProgram.from_dict(wire)
    assert restored == program and hash(restored) == previous_hash
    assert lookup[restored] == "original"
    assert "fingerprint" not in vars(restored)
    assert restored.fingerprint == program.fingerprint


def test_binding_plan_reuses_only_metadata_and_keeps_current_values(monkeypatch) -> None:
    program = _program()
    fabric = mechanisms.FormulaFabricV2(program)
    inputs, banks = _values(program)
    fabric.bind_tensors(inputs=inputs, banks=banks)
    latest_inputs, latest_banks = _values(program)
    with torch.no_grad():
        latest_banks["weight"].value.add_(1.0)

    def redundant(*args, **kwargs):
        raise AssertionError("identical binding metadata was revalidated")

    with monkeypatch.context() as patch:
        patch.setattr(formula_v2, "_validate_tensor_metadata_against_type", redundant)
        patch.setattr(formula_v2, "_preflight_program_shapes", redundant)
        prepared = fabric.bind_tensors(inputs=latest_inputs, banks=latest_banks)
    bound = dict(zip(prepared.binding_names, prepared.values, strict=True))
    assert bound["x"] is latest_inputs["x"]
    assert bound["weight"] is latest_banks["weight"].value
    fabric.execution_plan()(prepared)[0].sum().backward()
    assert latest_banks["weight"].value.grad is not None
    assert banks["weight"].value.grad is None
    with torch.no_grad():
        latest_banks["weight"].value.fill_(torch.nan)
    with pytest.raises(mechanisms.FormulaBindingError, match="finite"):
        fabric.bind_tensors(inputs=latest_inputs, banks=latest_banks)
    checked = fabric._bind_for_checked_execution(inputs=latest_inputs, banks=latest_banks)
    assert not bool(fabric.execution_plan().forward_checked(checked)[1])


@pytest.mark.parametrize("change", ("shape", "dtype", "identity", "keys"))
def test_binding_plan_hit_cannot_hide_changed_contract(change) -> None:
    program = _program()
    fabric = mechanisms.FormulaFabricV2(program)
    inputs, banks = _values(program)
    fabric.bind_tensors(inputs=inputs, banks=banks)
    if change == "shape":
        inputs["x"] = torch.ones(2, 4, dtype=torch.float64)
    elif change == "dtype":
        inputs["x"] = inputs["x"].float()
    elif change == "identity":
        banks["weight"] = replace(banks["weight"], partition_id="different")
    else:
        inputs["extra"] = inputs["x"]
    with pytest.raises(mechanisms.FormulaBindingError):
        fabric.bind_tensors(inputs=inputs, banks=banks)


def test_binding_plan_is_bounded_tensor_free_and_not_serialized() -> None:
    program = _program()
    wire, structural = program.to_dict(), asdict(program)
    fabric = mechanisms.FormulaFabricV2(program)
    inputs, banks = _values(program)
    input_ref, bank_ref = weakref.ref(inputs["x"]), weakref.ref(banks["weight"].value)
    prepared = fabric.bind_tensors(inputs=inputs, banks=banks)
    del inputs, banks, prepared
    gc.collect()
    assert input_ref() is None and bank_ref() is None
    for batch in range(1, 36):
        inputs, banks = _values(program, batch=batch)
        fabric.bind_tensors(inputs=inputs, banks=banks)
    assert len(program._binding_plan.metadata) == 32
    assert program.to_dict() == wire and asdict(program) == structural
    changed = replace(program, limits=replace(program.limits, max_tensor_elements=3))
    assert changed._binding_plan is not program._binding_plan
    with pytest.raises(mechanisms.FormulaBindingError, match="limit"):
        mechanisms.FormulaFabricV2(changed).bind_tensors(inputs=inputs, banks=banks)


def test_binding_plan_respects_stride_and_current_dynamic_extent() -> None:
    program = _program()
    fabric = mechanisms.FormulaFabricV2(program)
    inputs, banks = _values(program)
    fabric.bind_tensors(inputs=inputs, banks=banks)
    transposed = torch.randn(3, 2, dtype=torch.float64).transpose(0, 1)
    prepared = fabric.bind_tensors(inputs={"x": transposed}, banks=banks)
    assert dict(zip(prepared.binding_names, prepared.values, strict=True))["x"] is transposed
    assert len(program._binding_plan.metadata) == 2
    larger_inputs, larger_banks = _values(program, batch=5)
    prepared = fabric.bind_tensors(inputs=larger_inputs, banks=larger_banks)
    assert fabric.execution_plan()(prepared)[0].shape == (5, 3)
    assert len(program._binding_plan.metadata) == 3


@pytest.mark.parametrize("change", ("none", "limits", "instruction", "bank"))
def test_program_replace_uses_a_fresh_cache(change: str) -> None:
    program = _program()
    original = program.fingerprint
    kwargs = {}
    if change == "limits":
        kwargs["limits"] = replace(program.limits, max_working_bytes=program.limits.max_working_bytes + 8)
    elif change == "instruction":
        kwargs["instructions"] = tuple(
            replace(item, attributes=(("mode", "relu"),))
            if item.atom_ref == "arti/formula-atom-scalar-map@1" else item
            for item in program.instructions
        )
    elif change == "bank":
        kwargs["bindings"] = tuple(
            replace(item, source_ref="arti/other-program-cache@1")
            if isinstance(item, mechanisms.BankBinding) else item
            for item in program.bindings
        )
    changed = replace(program, **kwargs)
    assert changed is not program and "fingerprint" not in vars(changed)
    assert changed.fingerprint == _canonical_fingerprint(changed)
    assert (changed.fingerprint == original) == (change == "none")
    assert mechanisms.FormulaProgram.from_dict(changed.to_dict()).fingerprint == changed.fingerprint
    with pytest.raises(mechanisms.FormulaProgramError):
        replace(program, limits=replace(program.limits, max_instructions=1))


def test_cached_program_does_not_share_mutable_exports() -> None:
    program = _program()
    original = program.fingerprint
    expected = program.to_dict()
    exported = program.to_dict()
    exported["bindings"][0]["value_type"]["sizes"][0] = 999
    exported["limits"]["max_working_bytes"] = 1
    exported["instructions"][0]["attributes"]["factor_axes"].append("extra")
    exported["outputs"].clear()
    program.slot_types.clear()
    assert program.to_dict() == expected
    assert program.fingerprint == _canonical_fingerprint(program) == original


@pytest.mark.parametrize(
    "invalid", ("input-shape", "input-dtype", "input-finite", "bank-shape", "bank-dtype",
                "bank-finite", "bank-identity", "missing-input"),
)
def test_cached_fingerprint_never_caches_tensor_or_bank_admission(invalid: str) -> None:
    program = _program()
    fabric = mechanisms.FormulaFabricV2(program)
    inputs, banks = _values(program)
    warmed = fabric.bind_tensors(inputs=inputs, banks=banks)
    if invalid == "input-shape":
        inputs["x"] = torch.zeros(2, 4, dtype=torch.float64)
    elif invalid == "input-dtype":
        inputs["x"] = inputs["x"].float()
    elif invalid == "input-finite":
        with torch.no_grad():
            inputs["x"][0, 0] = torch.nan
    elif invalid == "bank-shape":
        banks["weight"] = replace(banks["weight"], value=torch.zeros(4, dtype=torch.float64))
    elif invalid == "bank-dtype":
        banks["weight"] = replace(banks["weight"], value=banks["weight"].value.float())
    elif invalid == "bank-finite":
        with torch.no_grad():
            banks["weight"].value[0] = torch.inf
    elif invalid == "bank-identity":
        banks["weight"] = replace(banks["weight"], source_ref="arti/wrong-cache-bank@1")
    else:
        inputs.clear()
    assert program.fingerprint == warmed.program_fingerprint
    with pytest.raises(formula_v2.FormulaV2Error):
        fabric.bind_tensors(inputs=inputs, banks=banks)
    if invalid not in ("input-finite", "bank-finite"):
        with pytest.raises(formula_v2.FormulaV2Error):
            fabric._bind_for_checked_execution(inputs=inputs, banks=banks)


def test_cached_fingerprint_preserves_runtime_limits() -> None:
    program = _program(limits=mechanisms.FormulaLimits(max_tensor_elements=12))
    fabric = mechanisms.FormulaFabricV2(program)
    inputs, banks = _values(program, batch=2)
    fabric.bind_tensors(inputs=inputs, banks=banks)
    inputs, banks = _values(program, batch=5)
    with pytest.raises(formula_v2.FormulaV2Error):
        fabric.bind_tensors(inputs=inputs, banks=banks)


def test_floating_declaration_cache_allows_fresh_runtime_dtype_and_values() -> None:
    program = _program(dtype="floating")
    fabric = mechanisms.FormulaFabricV2(program)
    inputs, banks = _values(program, dtype=torch.float32)
    prepared = fabric.bind_tensors(inputs=inputs, banks=banks)
    fingerprint = prepared.program_fingerprint
    inputs, banks = _values(program, dtype=torch.float64)
    with torch.no_grad():
        banks["weight"].value.add_(0.5)
    new_prepared = fabric.bind_tensors(inputs=inputs, banks=banks)
    assert new_prepared.program_fingerprint == fingerprint
    assert all(value.dtype == torch.float64 for value in new_prepared.values)
    actual = fabric.execution_plan()(new_prepared)[0]
    expected = torch.nn.functional.silu(inputs["x"] * banks["weight"].value) + inputs["x"]
    torch.testing.assert_close(actual, expected)


def test_plan_output_type_table_is_static_without_shared_dict(monkeypatch) -> None:
    program = _program()
    fabric = mechanisms.FormulaFabricV2(program)
    inputs, banks = _values(program)
    prepared = fabric.bind_tensors(inputs=inputs, banks=banks)
    original = mechanisms.FormulaProgram.slot_types.fget
    calls = []

    def record(value):
        calls.append(id(value))
        return original(value)

    monkeypatch.setattr(mechanisms.FormulaProgram, "slot_types", property(record))
    plan = fabric.execution_plan()
    assert calls == [id(program)]
    for _ in range(3):
        actual = plan(prepared)[0]
    assert calls == [id(program)]
    exported = program.slot_types
    exported.clear()
    torch.testing.assert_close(plan(prepared)[0], actual)
    forged = mechanisms.PreparedFormulaBindings("0" * 64, prepared.binding_names, prepared.values)
    with pytest.raises(mechanisms.FormulaBindingError, match="different program"):
        plan(forged)


def test_cached_program_execution_plan_compile_outputs_gradients_and_dynamic_batch() -> None:
    program = _program()
    fabric = mechanisms.FormulaFabricV2(program)
    # Constructing a plan materializes a cold fingerprint before numerical tracing.
    assert "fingerprint" not in vars(program)
    plan = fabric.execution_plan()
    compiled = torch.compile(plan, backend="eager", fullgraph=True, dynamic=True)
    for batch in (2, 5):
        inputs, banks = _values(program, batch=batch)
        prepared = fabric.bind_tensors(inputs=inputs, banks=banks)
        actual = compiled(prepared)[0]
        eager = fabric(inputs=inputs, banks=banks, return_trace=True)
        torch.testing.assert_close(actual, eager.values[0])
        actual_gradients = torch.autograd.grad(actual.square().sum(), prepared.values)
        eager_gradients = torch.autograd.grad(eager.values[0].square().sum(), prepared.values)
        for actual_gradient, eager_gradient in zip(actual_gradients, eager_gradients, strict=True):
            assert torch.isfinite(actual_gradient).all()
            torch.testing.assert_close(actual_gradient, eager_gradient)
        assert eager.trace.program_fingerprint == program.fingerprint


@pytest.mark.parametrize("bind_method", ("bind_tensors", "_bind_for_checked_execution"))
def test_checked_plan_compile_preserves_outputs_gradients_and_tensor_flag(bind_method) -> None:
    program = _program()
    fabric = mechanisms.FormulaFabricV2(program)
    plan = fabric.execution_plan()
    compiled = torch.compile(plan.forward_checked, backend="eager", fullgraph=True, dynamic=True)
    for batch in (2, 5):
        inputs, banks = _values(program, batch=batch)
        prepared = getattr(fabric, bind_method)(inputs=inputs, banks=banks)
        outputs, finite = compiled(prepared)
        expected = plan(prepared)
        assert finite.shape == () and finite.dtype == torch.bool
        assert not finite.requires_grad and bool(finite)
        torch.testing.assert_close(outputs[0], expected[0])
        actual_gradients = torch.autograd.grad(outputs[0].square().sum(), prepared.values)
        expected_gradients = torch.autograd.grad(expected[0].square().sum(), prepared.values)
        for actual, reference in zip(actual_gradients, expected_gradients, strict=True):
            torch.testing.assert_close(actual, reference)


def test_checked_plan_vmap_reports_intermediate_overflow_even_with_finite_output() -> None:
    value_type = mechanisms.TensorType(("D",), (1,), dtype="float32")
    value = mechanisms.InputBinding("x", value_type)
    gain = mechanisms.InputBinding("gain", value_type)
    output = mechanisms.scalar_map(mechanisms.scale(value, gain), mode="sigmoid")
    program = mechanisms.FormulaProgram.build(outputs=(output,))
    fabric = mechanisms.FormulaFabricV2(program)
    plan = fabric.execution_plan()
    inputs = (
        {"x": torch.ones(1), "gain": torch.ones(1)},
        {"x": torch.tensor([1e20]), "gain": torch.tensor([1e20])},
    )
    prepared = tuple(fabric.bind_tensors(inputs=row, banks={}) for row in inputs)
    stacked = tuple(torch.stack(tuple(row.values[index] for row in prepared))
                    for index in range(len(plan.binding_names)))

    def run(*values):
        return plan.forward_checked(mechanisms.PreparedFormulaBindings(
            plan.program_fingerprint, plan.binding_names, values,
        ))

    grouped = torch.compile(torch.vmap(run), backend="eager", fullgraph=True)
    outputs, flags = grouped(*stacked)
    assert torch.isfinite(outputs[0]).all()
    assert flags.dtype == torch.bool and flags.shape == (2,)
    assert flags.tolist() == [True, False]
    torch.testing.assert_close(outputs[0][0], fabric(inputs=inputs[0], banks={}).values[0])
    with pytest.raises(mechanisms.FormulaBindingError, match="finite"):
        fabric(inputs=inputs[1], banks={})


def test_checked_plan_tracks_inputs_mutated_after_admission() -> None:
    program = _program()
    fabric = mechanisms.FormulaFabricV2(program)
    inputs, banks = _values(program)
    prepared = fabric.bind_tensors(inputs=inputs, banks=banks)
    with torch.no_grad():
        banks["weight"].value[0] = torch.nan
    _, finite = fabric.execution_plan().forward_checked(prepared)
    assert not bool(finite)
    with pytest.raises(mechanisms.FormulaBindingError, match="finite"):
        fabric.bind_tensors(inputs=inputs, banks=banks)


@pytest.mark.parametrize("invalid", ("dtype", "shape"))
def test_checked_plan_keeps_native_output_metadata_checks(monkeypatch, invalid) -> None:
    program = _program()
    fabric = mechanisms.FormulaFabricV2(program)
    inputs, banks = _values(program)
    prepared = fabric.bind_tensors(inputs=inputs, banks=banks)
    plan = fabric.execution_plan()
    original = formula_v2._execute_instruction

    def invalid_result(*args, **kwargs):
        result = original(*args, **kwargs)
        return result.float() if invalid == "dtype" else result[..., :2]

    monkeypatch.setattr(formula_v2, "_execute_instruction", invalid_result)
    for run in (lambda: fabric(inputs=inputs, banks=banks), lambda: plan.forward_checked(prepared)):
        with pytest.raises(mechanisms.FormulaBindingError):
            run()


def test_checked_plan_keeps_instruction_dtype_checks() -> None:
    program = _program(dtype="floating")
    fabric = mechanisms.FormulaFabricV2(program)
    inputs, banks = _values(program)
    prepared = fabric.bind_tensors(inputs=inputs, banks=banks)
    values = tuple(value.float() if name == "weight" else value
                   for name, value in zip(prepared.binding_names, prepared.values, strict=True))
    inconsistent = mechanisms.PreparedFormulaBindings(
        prepared.program_fingerprint, prepared.binding_names, values,
    )
    with pytest.raises(mechanisms.FormulaBindingError, match="dtype"):
        fabric.execution_plan().forward_checked(inconsistent)


@pytest.mark.parametrize("binding_name", ("x", "weight"))
@pytest.mark.parametrize("nonfinite", (float("nan"), float("inf"), float("-inf")))
def test_internal_checked_binding_defers_only_finiteness(binding_name, nonfinite) -> None:
    program = _program()
    fabric = mechanisms.FormulaFabricV2(program)
    inputs, banks = _values(program)
    value = inputs["x"] if binding_name == "x" else banks["weight"].value
    with torch.no_grad():
        value.flatten()[0] = nonfinite
    prepared = fabric._bind_for_checked_execution(inputs=inputs, banks=banks)
    _, finite = fabric.execution_plan().forward_checked(prepared)
    assert finite.shape == () and finite.dtype == torch.bool and not bool(finite)
    for run in (fabric.bind_tensors, fabric):
        with pytest.raises(mechanisms.FormulaBindingError) as caught:
            run(inputs=inputs, banks=banks)
        assert caught.value.code == "FF2_NONFINITE"


def test_internal_checked_binding_never_evaluates_tensor_finiteness(monkeypatch) -> None:
    program = _program()
    fabric = mechanisms.FormulaFabricV2(program)
    inputs, banks = _values(program)
    admitted = fabric.bind_tensors(inputs=inputs, banks=banks)

    def unexpected_finite(*_args, **_kwargs):
        raise AssertionError("internal binding must leave finite checks to forward_checked")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "isfinite", unexpected_finite)
        prepared = fabric._bind_for_checked_execution(inputs=inputs, banks=banks)
    assert prepared.program_fingerprint == admitted.program_fingerprint
    assert prepared.binding_names == admitted.binding_names
    assert all(left is right for left, right in zip(prepared.values, admitted.values, strict=True))
    _, finite = fabric.execution_plan().forward_checked(prepared)
    assert bool(finite)


@pytest.mark.parametrize("changes", (
    {"partition_id": "other-partition"},
    {"asset_fingerprint": "a" * 64},
    {"route_ref": "arti/other-route@1"},
    {"bundle_id": "other-bundle", "member_ids": ("member",)},
))
def test_internal_checked_binding_preserves_bank_provenance(changes) -> None:
    program = _program()
    fabric = mechanisms.FormulaFabricV2(program)
    inputs, banks = _values(program)
    banks["weight"] = replace(banks["weight"], **changes)
    for bind in (fabric.bind_tensors, fabric._bind_for_checked_execution):
        with pytest.raises(mechanisms.FormulaBindingError) as caught:
            bind(inputs=inputs, banks=banks)
        assert caught.value.code == "FF2_BANK_IDENTITY_MISMATCH"


def test_internal_checked_binding_preserves_device_check_without_cuda() -> None:
    program = _program()
    fabric = mechanisms.FormulaFabricV2(program)
    inputs, banks = _values(program)
    banks["weight"] = replace(banks["weight"], value=torch.empty(3, dtype=torch.float64, device="meta"))
    with pytest.raises(mechanisms.FormulaBindingError) as caught:
        fabric._bind_for_checked_execution(inputs=inputs, banks=banks)
    assert caught.value.code == "FF2_DEVICE_MISMATCH"


@pytest.mark.parametrize("limits", (
    mechanisms.FormulaLimits(max_tensor_elements=5),
    mechanisms.FormulaLimits(max_tensor_bytes=40),
    mechanisms.FormulaLimits(max_working_bytes=80),
    mechanisms.FormulaLimits(max_axis_extent=3),
))
def test_internal_checked_binding_preserves_limits_and_preflight(limits) -> None:
    program = _program(limits=limits)
    fabric = mechanisms.FormulaFabricV2(program)
    inputs, banks = _values(program, batch=4 if limits.max_axis_extent == 3 else 2)
    errors = []
    for bind in (fabric.bind_tensors, fabric._bind_for_checked_execution):
        with pytest.raises(mechanisms.FormulaBindingError) as caught:
            bind(inputs=inputs, banks=banks)
        errors.append((caught.value.code, str(caught.value)))
    assert errors[0] == errors[1]
    assert errors[0][0] == "FF2_LIMIT_EXCEEDED"


def test_internal_checked_binding_preserves_dynamic_output_allocation_preflight() -> None:
    value = mechanisms.InputBinding("x", mechanisms.TensorType(("B", "D"), ("B", 2)))
    bank = mechanisms.BankBinding(
        "weight", "arti/program-cache-test@1", "weight", mechanisms.TensorType(("R", "D"), (20, 2)),
    )
    program = mechanisms.FormulaProgram.build(
        outputs=(mechanisms.contract(value, bank, reduce_axes=(("D", "D"),)),),
        limits=mechanisms.FormulaLimits(max_tensor_elements=100),
    )
    fabric = mechanisms.FormulaFabricV2(program)
    for bind in (fabric.bind_tensors, fabric._bind_for_checked_execution):
        with pytest.raises(mechanisms.FormulaBindingError, match="output allocation"):
            bind(inputs={"x": torch.ones(20, 2)}, banks={"weight": bank.bind(torch.ones(20, 2))})
