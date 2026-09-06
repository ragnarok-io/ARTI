import math

import pytest
import torch
from torch import nn

from arti import mechanisms as m
from arti.formula_program_query_v4 import _scaled_population_std, _scaled_root_mean_square


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _query(device, *, encoder=False, dtype=torch.float32):
    kind = m.TensorType(("B", "D"), ("B", 3), dtype="float32")
    value = m.InputBinding("value", kind)
    candidate = m.FormulaProgramTensorCandidateV3(m.FormulaProgramCandidate(
        "double", m.FormulaProgram.build(outputs=(m.add(value, value),)),
        input_slots={"value": "x"}, output_slot="terminal", operands={},
    ))
    return m.FormulaProgramQueryV4(
        slot_ids=("x", "empty", "second", "third", "terminal"),
        candidates=(candidate,), terminal_slot="terminal", max_steps=1, hidden_dim=4,
        tensor_encoder=m.FormulaProgramQueryTensorEncoderV1(3, 4) if encoder else None,
    ).to(device=device, dtype=dtype)


def _reference(query, arena):
    parameter = next(query.network.parameters())
    rows = []
    for value in arena.values:
        if value is None:
            width = 8 if query.tensor_encoder is None else query.tensor_encoder.output_width
            rows.append(parameter.new_zeros((arena.batch_size, width)))
        elif query.tensor_encoder is not None:
            rows.append(query.tensor_encoder(value))
        else:
            numeric = value.to(dtype=parameter.dtype).reshape(arena.batch_size, -1)
            if not bool(torch.isfinite(numeric).all()):
                raise ValueError("ProgramQuery arena values must be finite")
            rows.append(torch.cat((
                numeric.new_ones((arena.batch_size, 1)),
                (numeric / numeric.shape[-1]).sum(-1, keepdim=True),
                _scaled_population_std(numeric, dim=-1).unsqueeze(-1),
                (numeric.abs() / numeric.shape[-1]).sum(-1, keepdim=True),
                numeric.amax(-1, keepdim=True), numeric.amin(-1, keepdim=True),
                _scaled_root_mean_square(numeric, dim=-1).unsqueeze(-1),
                numeric.new_full((arena.batch_size, 1), math.log1p(numeric.shape[-1])),
            ), -1))
    return torch.cat(rows, -1)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("encoder", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_native_summary_matches_original_values_and_gradients(device, encoder, dtype):
    torch.manual_seed(71)
    query = _query(device, encoder=encoder, dtype=dtype)
    first = torch.randn(2, 4, 3, device=device, dtype=dtype, requires_grad=True)
    second = torch.randn(2, 3, 4, device=device, dtype=dtype, requires_grad=True)
    third = torch.randn(2, 1, 3, device=device, dtype=dtype, requires_grad=True)
    arena = m.FormulaProgramArena(query.slot_ids, (first, None, second.transpose(1, 2), third, first))
    expected = _reference(query, arena)
    actual = query._summarize_values(arena)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.linspace(0.1, 1, actual.numel(), device=device, dtype=dtype).reshape_as(actual)
    parameters = (first, second, third, *query.parameters())
    expected_grad = torch.autograd.grad((weights * expected).sum(), parameters, allow_unused=True)
    actual_grad = torch.autograd.grad((weights * actual).sum(), parameters, allow_unused=True)
    for left, right in zip(actual_grad, expected_grad, strict=True):
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("encoder", [False, True])
def test_one_host_predicate_per_native_summary(device, encoder, monkeypatch):
    query = _query(device, encoder=encoder)
    values = tuple(torch.randn(2, 3, device=device) for _ in query.slot_ids)
    arena = m.FormulaProgramArena(query.slot_ids, values)
    original = torch.Tensor.__bool__
    calls = []

    def checked(value):
        calls.append(value.device)
        return original(value)

    monkeypatch.setattr(torch.Tensor, "__bool__", checked)
    query._summarize_values(arena)
    assert calls == [values[0].device]
    calls.clear()
    _reference(query, arena)
    assert calls == [values[0].device] * len(values)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("encoder", [False, True])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_summary_rejects_nonfinite_after_dtype_conversion(device, encoder, bad):
    query = _query(device, encoder=encoder)
    first = torch.ones(2, 3, device=device)
    invalid = first.clone()
    invalid[1, 2] = bad
    arena = m.FormulaProgramArena(query.slot_ids, (first, None, invalid, None, first))
    with pytest.raises(ValueError, match="finite"):
        query._summarize_values(arena)
    cast_overflow = torch.full((2, 3), 1e100, dtype=torch.float64, device=device)
    arena = m.FormulaProgramArena(query.slot_ids, (first, None, cast_overflow, None, first))
    with pytest.raises(ValueError, match="finite"):
        query._summarize_values(arena)


@pytest.mark.parametrize("encoder", [False, True])
def test_summary_has_no_stale_input_cache(encoder):
    query = _query("cpu", encoder=encoder)
    value = torch.ones(2, 3)
    arena = m.FormulaProgramArena(query.slot_ids, (value, None, value, None, None))
    before = query._summarize_values(arena)
    value.mul_(3)
    after = query._summarize_values(arena)
    torch.testing.assert_close(after, _reference(query, arena), rtol=0, atol=0)
    assert not torch.equal(before, after)


def test_encoder_hooks_and_custom_modules_keep_original_call_boundary(monkeypatch):
    query = _query("cpu", encoder=True)
    value = torch.ones(2, 3)
    arena = m.FormulaProgramArena(query.slot_ids, (value, None, value * 2, None, None))
    seen = []
    hook = query.tensor_encoder.register_forward_hook(lambda *args: seen.append("encoder"))
    query._summarize_values(arena)
    hook.remove()
    assert seen == ["encoder", "encoder"]
    seen.clear()

    class CustomActivation(nn.SiLU):
        def forward(self, value):
            seen.append("custom")
            return super().forward(value)

    query.tensor_encoder.network[1] = CustomActivation()
    original = query.tensor_encoder.forward
    monkeypatch.setattr(query.tensor_encoder, "forward", lambda value: (seen.append("forward"), original(value))[1])
    query._summarize_values(arena)
    assert seen == ["forward", "custom", "forward", "custom"]


@pytest.mark.parametrize("cls", [nn.Sequential, nn.Linear, nn.SiLU])
def test_class_level_encoder_callback_keeps_incremental_validation(cls, monkeypatch):
    query = _query("cpu", encoder=True)
    first, second = torch.ones(2, 3), torch.ones(2, 3)
    arena = m.FormulaProgramArena(query.slot_ids, (first, None, second, None, None))
    target = next(module for module in query.tensor_encoder.modules() if type(module) is cls)
    original = cls.forward
    seen = []

    def changed(module, value):
        if module is target:
            seen.append("changed")
            second.fill_(float("inf"))
        return original(module, value)

    monkeypatch.setattr(cls, "forward", changed)
    with pytest.raises(ValueError, match="finite"):
        query._summarize_values(arena)
    assert seen == ["changed"]


def test_tensor_subclass_keeps_serial_summary(monkeypatch):
    query = _query("cpu")
    value = torch.ones(2, 3).as_subclass(type("CustomTensor", (torch.Tensor,), {}))
    arena = m.FormulaProgramArena(query.slot_ids, (value, None, value, None, None))
    monkeypatch.setattr(query, "_summarize_native_values", lambda *args: pytest.fail("custom Tensor was batched"))
    torch.testing.assert_close(query._summarize_values(arena), _reference(query, arena))


def test_encoder_parameter_subclass_keeps_incremental_validation():
    query = _query("cpu", encoder=True)
    first, second = torch.ones(2, 3), torch.ones(2, 3)
    arena = m.FormulaProgramArena(query.slot_ids, (first, None, second, None, None))
    seen = []

    class WeightTensor(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if func is torch.nn.functional.linear:
                seen.append("linear")
                second.fill_(float("inf"))
            return super().__torch_function__(func, types, args, kwargs or {})

    linear = query.tensor_encoder.network[0]
    linear.weight = nn.Parameter(linear.weight.detach().as_subclass(WeightTensor))
    with pytest.raises(ValueError, match="finite"):
        query._summarize_values(arena)
    assert seen


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("magnitude", [0., 1., 1e21, 1e38])
def test_batched_statistics_preserve_scaled_extremes_and_zero_gradients(device, magnitude):
    query = _query(device)
    first = (torch.tensor([[0., 0., 0.], [0.5, 0.75, 1.]], device=device) * magnitude).requires_grad_()
    second = torch.full((2, 3), magnitude, device=device, requires_grad=True)
    arena = m.FormulaProgramArena(query.slot_ids, (first, None, second, None, first))
    actual, expected = query._summarize_values(arena), _reference(query, arena)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    left = torch.autograd.grad(actual, (first, second), grad_outputs=torch.full_like(actual, 1e10))
    right = torch.autograd.grad(expected, (first, second), grad_outputs=torch.full_like(expected, 1e10))
    for a, b in zip(left, right, strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_grouped_statistics_respect_chunk_bound_and_restore_slot_order():
    query = _query("cpu")
    first = torch.randn(2, 70000)
    second = torch.randn_like(first)
    arena = m.FormulaProgramArena(query.slot_ids, (first, None, second, None, first))
    torch.testing.assert_close(query._summarize_values(arena), _reference(query, arena), rtol=0, atol=0)
