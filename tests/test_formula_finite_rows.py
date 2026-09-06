import torch

from arti._formula_finite_rows import _FiniteTensorRows


def test_finite_rows_deduplicate_regions_not_storage_or_stride():
    value = torch.tensor([[1., 2.], [3., torch.inf]])
    rows = _FiniteTensorRows()
    rows.add((value[:1], value[:1]))
    rows.add((value[1:],))
    rows.add((value[:, :1],))
    rows.add((value.t(),))
    rows.add((torch.ones((), dtype=torch.float64), torch.empty(0)))
    rows.add(())
    assert len(rows.tensors) == 6
    assert rows.evaluate(device="cpu").tolist() == [True, False, True, False, True, True]


def test_finite_rows_do_not_retain_verdicts_between_invocations():
    value = torch.ones(4)
    first = _FiniteTensorRows()
    first.add((value,))
    assert first.evaluate(device="cpu").tolist() == [True]
    value[0] = torch.nan
    second = _FiniteTensorRows()
    second.add((value,))
    assert second.evaluate(device="cpu").tolist() == [False]
    assert _FiniteTensorRows().evaluate(device="cpu").shape == (0,)


def test_repeated_retained_objects_prepare_address_once(monkeypatch):
    value = torch.ones(4)
    rows = _FiniteTensorRows()
    original = torch.Tensor.data_ptr
    calls = []

    def address(tensor):
        calls.append(tensor)
        return original(tensor)

    monkeypatch.setattr(torch.Tensor, "data_ptr", address)
    for _ in range(20):
        rows.add((value, value))
    assert len(calls) == 1
    assert len(rows.tensors) == 1
    assert rows.evaluate(device="cpu").tolist() == [True] * 20
