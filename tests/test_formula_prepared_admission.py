from dataclasses import replace

import pytest
import torch

from arti import mechanisms as m
from arti._formula_candidate_admission import candidate_mask
from arti.formula_v2 import _FormulaBindingPlan


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _query(device, *, version=5, multi=False):
    kind = m.TensorType(("B", "D"), ("B", 3), dtype="floating")
    x = m.InputBinding("x", kind)
    weight = m.BankBinding("weight", "arti/prepared-admission-test@1", "weight", kind)
    value = m.scale(x, weight)
    program = m.FormulaProgram.build(outputs=(value, m.add(x, x)) if multi else (value,))
    options = dict(input_slots={"x": "x"}, operands={"weight": torch.ones(1, 3, device=device)},
                   batch_broadcast_operands=("weight",))
    if multi:
        implementation = m.FormulaProgramCandidateV3("producer", program,
            output_slots=dict(zip(program.outputs, ("out", "aux"), strict=True)), **options)
        candidate = m.FormulaProgramTensorCandidateV4(implementation, plastic_bank_slot="weight")
    else:
        candidate = m.FormulaProgramTensorCandidateV3(m.FormulaProgramCandidateV2(
            "producer", program, output_slot="out", **options), plastic_bank_slot="weight")
    options = dict(slot_ids=("x", "out", "aux"), candidates=(candidate,), max_steps=3,
                   max_tensor_steps=2, min_tensor_steps=1)
    query = m.FormulaProgramQueryV5(**options, terminal_slots={"answer": "out"}) if version == 5 else (
        m.FormulaProgramQueryV4(**options, terminal_slot="out"))
    return query.to(device)


def _mask(query, arenas, *, steps=0):
    return candidate_mask(query, arenas, query.candidates, steps=steps, include_stop=True)


def _reference(query, arenas, *, steps=0):
    return torch.tensor([[query._candidate_eligible(c, arena, steps=steps) for c in query.candidates]
                         + [query._stop_eligible(arena, steps=steps)] for arena in arenas], device=arenas[0].device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("version,multi", ((4, False), (5, False), (5, True)))
def test_metadata_hits_avoid_binding_but_read_live_broadcast_bank_and_values(monkeypatch, device, version, multi):
    query = _query(device, version=version, multi=multi)
    initial = query._arena({"x": torch.ones(2, 3, device=device)})
    assert _mask(query, (initial,))[0, 0]
    candidate = query.candidates[0]
    ref = candidate.bank_slot_ref
    bad_state = initial.bank_state.replace(ref, torch.full((1, 3), torch.nan, device=device), revision=1)
    bad_bank = replace(initial, bank_state=bad_state)
    bad_input = query._arena({"x": torch.full((2, 3), torch.nan, device=device)})
    exhausted = replace(initial, tensor_steps=2)
    occupied = replace(initial, values=initial.values.write("aux" if multi else "out", initial.values.get("x")))
    arenas = (initial, bad_bank, initial, bad_input, exhausted, occupied)
    expected = _reference(query, arenas)
    calls = []
    original = _FormulaBindingPlan.bind
    def record(self, **kwargs):
        calls.append(self)
        return original(self, **kwargs)
    monkeypatch.setattr(_FormulaBindingPlan, "bind", record)
    torch.testing.assert_close(_mask(query, arenas), expected)
    assert not calls
    # A following call cannot retain finite decisions for an in-place mutation.
    initial.values.get("x").fill_(torch.nan)
    assert not _mask(query, (initial,))[0, 0]
    assert not calls


@pytest.mark.parametrize("device", DEVICES)
def test_metadata_miss_uses_original_binding_validator(monkeypatch, device):
    query = _query(device)
    initial = query._arena({"x": torch.ones(1, 3, device=device)})
    _mask(query, (initial,))
    dynamic = query._arena({"x": torch.ones(3, 3, device=device)})
    wrong = query._arena({"x": torch.ones(3, 4, device=device)})
    calls = []
    original = _FormulaBindingPlan.bind
    def record(self, **kwargs):
        calls.append(self)
        return original(self, **kwargs)
    monkeypatch.setattr(_FormulaBindingPlan, "bind", record)
    assert _mask(query, (dynamic, dynamic, wrong))[:, 0].tolist() == [True, True, False]
    assert len(calls) == 2


@pytest.mark.parametrize("device", DEVICES)
def test_prepared_sources_do_not_bypass_bank_identity_or_operand_replacement(device):
    query = _query(device)
    initial = query._arena({"x": torch.ones(1, 3, device=device)})
    _mask(query, (initial,))
    implementation = query.candidates[0].candidate
    binding = implementation._bank_bindings["weight"]
    implementation._bank_bindings["weight"] = replace(binding, partition_id="foreign")
    assert not _mask(query, (initial,))[0, 0]
    implementation._bank_bindings["weight"] = binding
    assert _mask(query, (initial,))[0, 0]
    # Ordinary nonplastic operands also read their current store, not a prepared Tensor.
    query.candidates[0].plastic_bank_slot = None
    owner = query.candidates[0]._bank_owner
    owner.value = torch.full_like(owner.value, torch.nan)
    assert not _mask(query, (initial,))[0, 0]


def test_custom_binding_and_admission_callbacks_keep_native_execution(monkeypatch):
    query = _query("cpu")
    initial = query._arena({"x": torch.ones(1, 3)})
    _mask(query, (initial,))
    candidate = query.candidates[0]
    calls = []
    def accepts(arena):
        calls.append(arena)
        return False
    monkeypatch.setattr(candidate, "accepts", accepts)
    assert not _mask(query, (initial, initial))[:, 0].any()
    assert len(calls) == 2
    monkeypatch.undo()
    calls.clear()
    original = candidate.candidate.fabric.bind_tensors
    def bind_tensors(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(candidate.candidate.fabric, "bind_tensors", bind_tensors)
    assert _mask(query, (initial, initial))[:, 0].all()
    assert len(calls) == 2


@pytest.mark.parametrize("method", ("accepts", "_bindings", "bind_tensors", "bind", "consume", "budget"))
def test_class_level_extension_keeps_native_admission_on_warm_metadata(monkeypatch, method):
    from arti.formula_v2 import BankBinding, FormulaBankOperand, FormulaBindingError, FormulaFabricV2

    query = _query("cpu")
    initial = query._arena({"x": torch.ones(1, 3)})
    _mask(query, (initial,))
    candidate = query.candidates[0]
    classes = {"accepts": type(candidate), "_bindings": type(candidate), "bind_tensors": FormulaFabricV2,
               "bind": BankBinding, "consume": FormulaBankOperand, "budget": type(query)}
    name = "_candidate_budget_eligible" if method == "budget" else method
    calls = []
    def reject(*args, **kwargs):
        calls.append(1)
        if method in ("accepts", "budget"):
            return False
        raise FormulaBindingError("FF2_BINDING_SHAPE", "extension rejected the binding")
    monkeypatch.setattr(classes[method], name, reject)
    assert not _mask(query, (initial, initial))[:, 0].any()
    assert len(calls) == 2


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("version", [4, 5])
def test_native_batch_prepares_shared_tensor_metadata_and_broadcast_once(monkeypatch, device, version):
    from arti import formula_v2

    query = _query(device, version=version)
    initial = query._arena({"x": torch.ones(2, 3, device=device)})
    _mask(query, (initial,))
    scans = []
    original = formula_v2._binding_tensor_metadata

    def record(value):
        scans.append(value)
        return original(value)

    monkeypatch.setattr(formula_v2, "_binding_tensor_metadata", record)
    assert _mask(query, (initial,) * 20)[:, 0].all()
    assert len(scans) == 2  # Shared input and one expanded current Bank view.
    scans.clear()
    assert _mask(query, (initial,) * 20)[:, 0].all()
    assert len(scans) == 2  # Nothing survives a new invocation.


@pytest.mark.parametrize("version", [4, 5])
def test_custom_stop_discards_metadata_and_broadcast_preparation(monkeypatch, version):
    query = _query("cpu", version=version)
    value = torch.ones(2, 3)
    initial = query._arena({"x": value})
    _mask(query, (initial,))
    calls = []

    def stop(arena, *, steps):
        calls.append(steps)
        if len(calls) == 1:
            value.resize_(2, 4).fill_(1)
        return False

    monkeypatch.setattr(query, "_stop_eligible", stop)
    assert _mask(query, (initial, initial))[:, 0].tolist() == [True, False]
    assert len(calls) == 2


@pytest.mark.parametrize("level", ["instance", "class"])
def test_custom_operand_reader_retains_per_row_calls(monkeypatch, level):
    query = _query("cpu")
    candidate = query.candidates[0]
    candidate.plastic_bank_slot = None
    initial = query._arena({"x": torch.ones(2, 3)})
    _mask(query, (initial,))
    store = candidate.candidate.operand_store
    owner = store if level == "instance" else type(store)
    original = getattr(owner, "tensor")
    calls = []

    def reader(*args, **kwargs):
        calls.append(True)
        result = original(*args, **kwargs)
        return result if len(calls) % 2 else torch.full_like(result, torch.nan)

    monkeypatch.setattr(owner, "tensor", reader)
    assert _mask(query, (initial, initial))[:, 0].tolist() == [True, False]
    assert len(calls) == 2


@pytest.mark.parametrize("device", DEVICES)
def test_callback_cannot_retroactively_change_prior_numerical_admission(monkeypatch, device):
    query = _query(device)
    value = torch.ones(2, 3, device=device)
    initial = query._arena({"x": value})
    _mask(query, (initial,))

    def stop(arena, *, steps):
        value.fill_(torch.nan)
        return False

    monkeypatch.setattr(query, "_stop_eligible", stop)
    assert _mask(query, (initial, initial))[:, 0].tolist() == [True, False]


@pytest.mark.parametrize("source", ["input", "bank"])
def test_tensor_extension_runs_outside_capture_and_after_prior_checks(source):
    from arti import formula_v2

    query = _query("cpu")
    first_value = torch.ones(2, 3)
    initial = query._arena({"x": first_value})
    _mask(query, (initial,))
    enabled = False
    calls = []

    class ExtensionTensor(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if enabled:
                assert formula_v2._FINITE_VALIDATION_CAPTURE.get() is None
                calls.append(func)
                first_value.fill_(torch.nan)
            return super().__torch_function__(func, types, args, kwargs or {})

    value = torch.ones(2 if source == "input" else 1, 3).as_subclass(ExtensionTensor)
    other = query._arena({"x": value if source == "input" else torch.ones(2, 3)})
    if source == "bank":
        other = replace(other, bank_state=other.bank_state.replace(
            query.candidates[0].bank_slot_ref, value, revision=1,
        ))
    enabled = True
    assert _mask(query, (initial, other))[:, 0].tolist() == [True, True]
    assert calls
