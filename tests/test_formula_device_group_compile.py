"""Prepared dispatch tests with a standalone public Formula fixture."""

import pytest
import torch

from arti import mechanisms as m
from arti._formula_device_dispatch import FormulaDeviceDispatchLayout, FormulaDeviceNumericalDispatch, formula_device_dispatch_groups
from arti._formula_device_frames import FormulaDeviceFrameKernel
from arti._formula_device_pools import FormulaDevicePoolLayout
from test_formula_device_dispatch import _producer


def fixture(device):
    with torch.device(device):
        query = m.FormulaProgramQueryV5(
            slot_ids=("x", "owned"), candidates=(_producer("read", "owned", "memory"),),
            terminal_slots={"answer": "owned"}, max_steps=1,
        )
        kernel = FormulaDeviceFrameKernel.from_query(query)
        dispatch = FormulaDeviceNumericalDispatch.from_query(query, frame_kernel=kernel, execution_backend="native").to(device)
        layout = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(query)[0]).to(device)
        dl = FormulaDevicePoolLayout.from_samples((torch.ones(1, 3),), 2)
        bl = FormulaDevicePoolLayout.from_samples((torch.ones(1, 3),), 2)
        dispatch.prepare_typed_pools_(dl, bl)
        state = kernel.initial_state(1, torch.tensor([[0, -1]]), bank_value_handles=torch.tensor([0]))
        data = (torch.ones(1, 1, 3).expand(3, 1, 3),)
        banks = bl.allocate(device)
        banks[0][0].fill_(2)
        banks[0][1].fill_(7)
        packet = layout(torch.tensor([[0]]))
        return dispatch, state, packet, data, banks


@torch.no_grad()
def test_default_cpu_dispatch_does_not_require_new_compiler_api(monkeypatch):
    from test_formula_device_dispatch import _numeric_fixture
    _, layout, dispatch, state, data, bank = _numeric_fixture()
    import arti._formula_device_dispatch as module
    class LegacyTorchView:
        def __getattr__(self, name):
            if name == "compiler":
                raise AssertionError("native dispatch must not access the new compiler API")
            return getattr(torch, name)
    monkeypatch.setattr(module, "torch", LegacyTorchView())
    result = dispatch(state, layout(torch.tensor([[0]])), data, bank)
    assert result.numeric_valid.all()
    assert dispatch._compiled_dispatch is False and not dispatch._compiled_groups


@pytest.mark.parametrize("mode", ("groups", "dispatch"))
@pytest.mark.parametrize("device", ("cpu", "cuda"))
@torch.no_grad()
def test_prepared_dispatch_reads_live_bank_and_preserves_contract(mode, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dispatch, state, packet, data, banks = fixture(device)
    def run():
        return dispatch(tuple(state), tuple(packet), data, banks)
    initial = run()
    backend = "inductor" if device == "cuda" else "eager"
    if mode == "groups":
        dispatch.prepare_compiled_groups_(run, group_ids=(0,), backend=backend)
    else:
        dispatch.prepare_compiled_dispatch_(run, backend=backend)
    groups, whole = dispatch._compiled_groups, dispatch._compiled_dispatch
    torch.testing.assert_close(run(), initial)
    previous = initial.output_values[0].clone()
    for change_handle in (False, True):
        if change_handle:
            state.bank_value_handles.fill_(1)
        else:
            banks[0][0].fill_(4)
        dispatch._compiled_groups, dispatch._compiled_dispatch = {}, False
        expected = run()
        dispatch._compiled_groups, dispatch._compiled_dispatch = groups, whole
        actual = run()
        assert actual.numeric_valid.all() and actual.output_present[0].any()
        torch.testing.assert_close(actual, expected)
        assert not torch.equal(actual.output_values[0], previous)
        previous = actual.output_values[0].clone()
    with torch.enable_grad(), pytest.raises(RuntimeError, match="no-grad"):
        (groups[0] if mode == "groups" else whole)()
    with torch.autocast(device, dtype=torch.bfloat16), pytest.raises(ValueError, match="metadata changed"):
        run()
    dispatch.double().float()
    with pytest.raises(RuntimeError, match="dispatch moved"):
        run()


@torch.no_grad()
def test_whole_compilation_rejects_effect_dispatch():
    from test_formula_device_dispatch import _numeric_fixture
    _, _, dispatch, _, data, bank = _numeric_fixture()
    dispatch.prepare_typed_pools_(FormulaDevicePoolLayout.from_samples((data[0],), 2),
                                 FormulaDevicePoolLayout.from_samples((bank[0],), 2))
    with pytest.raises(ValueError, match="ordinary typed groups"):
        dispatch.prepare_compiled_dispatch_(lambda: None, backend="eager")
