from dataclasses import replace
import gc
import weakref

import pytest
import torch

from arti import mechanisms as m
from arti import formula_v2 as formula


def _fabric():
    value_type = m.TensorType(("B", "D"), ("B", "D"), dtype="floating", domain="activation")
    value = m.InputBinding("value", value_type)
    return m.FormulaFabricV2(m.FormulaProgram.build(outputs=(m.scalar_map(value, mode="tanh"),)))


def test_shape_success_cache_does_not_cache_values_or_hold_tensors():
    fabric = _fabric()
    cache = formula._preflight_program_shape_metadata
    cache.cache_clear()
    value = torch.ones(2, 3)
    reference = weakref.ref(value)
    fabric.bind_tensors(inputs={"value": value}, banks={})
    first = cache.cache_info()
    assert first.misses == 1
    fabric.bind_tensors(inputs={"value": value * 2}, banks={})
    # The binding plan now bypasses repeated preflight preparation entirely.
    assert cache.cache_info() == first
    assert len(fabric.program._binding_plan.metadata) == 1
    value.fill_(float("nan"))
    with pytest.raises(m.FormulaBindingError) as error:
        fabric.bind_tensors(inputs={"value": value}, banks={})
    assert error.value.code == "FF2_NONFINITE"
    del error, value
    gc.collect()
    assert reference() is None
    assert cache.cache_info().maxsize == 2048


def test_shape_cache_distinguishes_shapes_dtypes_strides_and_limits():
    fabric = _fabric()
    cache = formula._preflight_program_shape_metadata
    cache.cache_clear()
    for value in (
        torch.ones(2, 3), torch.ones(2, 3, dtype=torch.float64),
        torch.ones(3, 2).t(), torch.ones(4, 3),
    ):
        fabric.bind_tensors(inputs={"value": value}, banks={})
    assert cache.cache_info().misses == 4
    strict = m.FormulaFabricV2(replace(
        fabric.program, limits=m.FormulaLimits(max_working_bytes=25),
    ))
    for _ in range(2):
        with pytest.raises(m.FormulaBindingError) as error:
            strict.bind_tensors(inputs={"value": torch.ones(2, 3)}, banks={})
        assert error.value.code == "FF2_LIMIT_EXCEEDED"
    assert cache.cache_info().misses == 6
    assert cache.cache_info().currsize == 4


def test_shape_cache_is_equivalent_to_uncached_forward_and_gradient():
    fabric = _fabric()
    value = torch.randn(2, 3, requires_grad=True)
    formula._preflight_program_shape_metadata.cache_clear()
    first = fabric(inputs={"value": value}, banks={}).values[0]
    first_grad = torch.autograd.grad(first.sum(), value)[0]
    second = fabric(inputs={"value": value}, banks={}).values[0]
    second_grad = torch.autograd.grad(second.sum(), value)[0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first_grad, second_grad, rtol=0, atol=0)
