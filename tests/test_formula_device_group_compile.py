"""Compiled dispatch islands preserve actual multi-round search state."""

import pytest
import torch
from torch.utils._pytree import tree_map

from benchmarks._formula_expression_runtime import expression_runtime
from benchmarks.probe_formula_expression_gpu import _compare
from benchmarks.train_formula_expression_search import build_graph, data


def clone(values):
    return tree_map(lambda v: v.detach().clone(), values)


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


@torch.no_grad()
@pytest.mark.parametrize("mode", ("groups", "dispatch"))
def test_prepared_group_graph_matches_full_records_pools_and_operand_refresh(mode):
    query, (values, _) = build_graph(), data(29, positions=3)
    wave, args = expression_runtime(query, values, width=1, heads=3)
    dispatch = wave.wave.execution.dispatch
    group_id = next(g.group_id for g in dispatch.groups if
                    any(c.candidate_id.startswith("swish") for c in g.candidates))
    expected_args = clone(args)
    expected = wave.forward_steps(*expected_args, steps=6)
    def sample():
        return wave.forward_steps(*clone(args), steps=6)
    if mode == "groups":
        info = dispatch.prepare_compiled_groups_(sample, group_ids=(group_id,), backend="eager")[group_id]
    else:
        info = dispatch.prepare_compiled_dispatch_(sample, backend="eager")
    assert info["variants"] > 0 and info["graph_nodes"] > 0
    actual_args = clone(args)
    actual = wave.forward_steps(*actual_args, steps=6)
    _compare(actual, expected, score_metadata=True)
    _compare(actual_args[4:6], expected_args[4:6])
    compiled = dispatch._compiled_groups
    compiled_dispatch = dispatch._compiled_dispatch
    next(c for c in query.candidates if c.candidate_id == "swish.0").candidate.operand_store.tensor("beta").mul_(2)
    dispatch.refresh_operands_()
    beta_args = clone(args)
    wave.forward_steps(*beta_args, steps=6)
    assert any(not torch.equal(a, b) for a, b in zip(beta_args[4], actual_args[4], strict=True))
    args[4][0][:2].mul_(-0.7)
    fresh_args, changed_args = clone(args), clone(args)
    dispatch._compiled_groups = {}
    dispatch._compiled_dispatch = False
    fresh = wave.forward_steps(*fresh_args, steps=6)
    dispatch._compiled_groups = compiled
    dispatch._compiled_dispatch = compiled_dispatch
    changed = wave.forward_steps(*changed_args, steps=6)
    _compare(changed, fresh, score_metadata=True)
    _compare(changed_args[4:6], fresh_args[4:6])
    assert dispatch._compiled_groups is compiled
    with torch.enable_grad():
        with pytest.raises(RuntimeError, match="no-grad"):
            (compiled[group_id] if mode == "groups" else compiled_dispatch)()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="metadata changed"):
            wave.forward_steps(*clone(args), steps=6)
    dispatch.double().float()
    with pytest.raises(RuntimeError, match="dispatch moved"):
        wave.forward_steps(*clone(args), steps=6)


def test_compilation_requires_typed_layout_and_real_group_ids():
    query, (values, _) = build_graph(), data(29, positions=3)
    wave, args = expression_runtime(query, values, width=1, heads=3)
    with pytest.raises(ValueError, match="ordinary group IDs"):
        wave.wave.execution.dispatch.prepare_compiled_groups_(lambda: None, group_ids=(-1,))


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@torch.no_grad()
def test_compiled_group_reads_current_predecessor_bank_and_handle(device):
    from arti._formula_device_pools import FormulaDevicePoolLayout
    from test_formula_device_dispatch import _numeric_fixture
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with torch.device(device):
        _, layout, dispatch, state, data, bank = _numeric_fixture()
        dl = FormulaDevicePoolLayout.from_samples((data[0],), 2)
        bl = FormulaDevicePoolLayout.from_samples((bank[0],), 2)
        dispatch.prepare_typed_pools_(dl, bl)
        # Expanded storage must not be silently made contiguous for tracing.
        data = (data.expand(3, 1, 3),)
        banks = bl.allocate(device)
        banks[0][0].fill_(2)
        banks[0][1].fill_(7)
        packet = layout(torch.tensor([[0]]))
        def run():
            return dispatch(tuple(state), tuple(packet), data, banks)
        initial = run()
        assert initial.numeric_valid.all() and initial.output_present[0].any()
        selected = tuple(g.group_id for g in dispatch.groups if not g.is_effect)
        dispatch.prepare_compiled_groups_(run, group_ids=selected,
                                          backend="inductor" if device == "cuda" else "eager")
        compiled = dispatch._compiled_groups
        _compare(run(), initial)
        previous = initial.output_values[0].clone()
        for changed_handle in (False, True):
            if changed_handle:
                state.bank_value_handles.fill_(1)
            else:
                banks[0][0].fill_(4)
            dispatch._compiled_groups = {}
            expected = run()
            dispatch._compiled_groups = compiled
            actual = run()
            assert actual.numeric_valid.all() and actual.output_present[0].any()
            _compare(actual, expected)
            assert not torch.equal(actual.output_values[0], previous)
            previous = actual.output_values[0].clone()
