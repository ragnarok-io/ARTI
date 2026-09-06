"""Default CUDA acceleration uses the existing numerical and AD contracts."""

import pytest
import torch

from arti._formula_device_dispatch import FormulaDeviceDispatchLayout, FormulaDeviceNumericalDispatch, formula_device_dispatch_groups
from arti._formula_device_frames import FormulaDeviceFrameKernel
from arti._formula_device_pools import FormulaDevicePoolLayout
from arti._formula_grouped_training import _DEFAULT_GROUPED_CACHE, grouped_formula_training


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


@torch.no_grad()
def test_default_cuda_dispatch_compiles_and_reuses_whole_graph():
    from test_formula_device_dispatch import _query
    with torch.device("cuda"):
        query = _query()
        kernel = FormulaDeviceFrameKernel.from_query(query)
        dispatch = FormulaDeviceNumericalDispatch.from_query(query, frame_kernel=kernel).cuda()
        layout = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(query)[0]).cuda()
        pools = FormulaDevicePoolLayout.from_samples((torch.ones(1, 3),), 3)
        dispatch.prepare_typed_pools_(pools, pools)
        data, bank = pools.allocate("cuda"), pools.allocate("cuda")
        data[0][0].fill_(1)
        bank[0][:2].fill_(2)
        state = kernel.initial_state(1, torch.tensor([[0, -1, -1]]), bank_value_handles=torch.tensor([0, 1]))
        packet = layout(torch.tensor([[0]]))
        def run():
            return dispatch(state, packet, data, bank)
        initial = run()
        compiled = dispatch._compiled_dispatch
        assert compiled is not False and dispatch._automatic_ready
        assert len(compiled.calls) == 1
        bank[0][0].fill_(4)
        actual = run()
        assert dispatch._compiled_dispatch is compiled and len(compiled.calls) == 1
        dispatch.execution_backend = "native"
        dispatch._compiled_dispatch = False
        expected = run()
        torch.testing.assert_close(actual, expected)
        assert not torch.equal(initial.output_values[0], actual.output_values[0])
        dispatch.execution_backend = "auto"
        dispatch.prepare_compiled_groups_(run, group_ids=(0,), backend="eager")
        assert dispatch._compiled_dispatch is False and tuple(dispatch._compiled_groups) == (0,)
        torch.testing.assert_close(run(), expected)
        dispatch.prepare_compiled_dispatch_(run, backend="eager")
        assert not dispatch._compiled_groups and dispatch._compiled_dispatch is not False
        torch.testing.assert_close(run(), expected)


@torch.no_grad()
def test_default_mixed_dispatch_does_not_repeat_effect_for_preparation(monkeypatch):
    from test_formula_device_dispatch import _numeric_fixture
    with torch.device("cuda"):
        kernel, layout, dispatch, state, _, _ = _numeric_fixture()
        pools = FormulaDevicePoolLayout.from_samples((torch.ones(1, 3),), 3)
        dispatch.prepare_typed_pools_(pools, pools)
        data, bank = pools.allocate("cuda"), pools.allocate("cuda")
        data[0][0].fill_(1)
        bank[0][0].fill_(2)
        effect = next(g for g in dispatch.groups if g.is_effect)
        original, calls = effect.forward, []
        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)
        monkeypatch.setattr(effect, "forward", counted)
        produced = dispatch(state, layout(torch.tensor([[0]])), data, bank)
        assert len(calls) == 1
        assert dispatch._compiled_groups and dispatch._compiled_dispatch is False
        data[0][1].copy_(produced.output_values[0][0, 0])
        state, event = kernel(state, torch.tensor([0]), torch.tensor([[1]]), produced.numeric_valid, torch.tensor([-1]))
        assert event.accepted.all()
        calls.clear()
        actual = dispatch(state, layout(torch.tensor([[1]])), data, bank)
        assert len(calls) == 1 and any(value.any() for value in actual.bank_successor_present)
        dispatch.execution_backend, dispatch._compiled_groups = "native", {}
        expected = dispatch(state, layout(torch.tensor([[1]])), data, bank)
        torch.testing.assert_close(actual, expected)
        assert bank[0][0].eq(2).all()


def test_default_cuda_training_compiles_without_context_and_preserves_unused_gradients():
    from test_formula_candidate_batch import _ordinary, _query
    candidate = _ordinary("automatic", saturate=True).cuda()
    query = _query((candidate,))
    _DEFAULT_GROUPED_CACHE.clear()
    def run(native):
        inputs = tuple(torch.full((1, 3), float(i + 1), device="cuda", requires_grad=True) for i in range(3))
        def compute():
            rows = query.execute_many(tuple((candidate, query._arena({"x": x})) for x in inputs))
            outputs = tuple(row.values.get("terminal") for row in rows)
            gradients = torch.autograd.grad(outputs[0].sum() + outputs[2].square().mean(),
                (*inputs, candidate.candidate.operand_store.tensor("weight")), allow_unused=True)
            return outputs, gradients
        if native:
            with grouped_formula_training(backend="native"):
                return compute()
        return compute()
    expected = run(True)
    actual = run(False)
    torch.testing.assert_close(actual, expected)
    assert actual[1][1] is None and _DEFAULT_GROUPED_CACHE
    runtime = next(iter(_DEFAULT_GROUPED_CACHE.values()))
    assert runtime.backend == "inductor" and runtime.cache_builds == 2
    run(False)
    assert runtime.cache_builds == 2 and runtime.cache_hits >= 2
