from __future__ import annotations

import weakref

import pytest
import torch

from arti import formula_v2 as formula


TYPE = formula.TensorType(("D",), (4,), dtype="float32")


def _count_scans(monkeypatch):
    scans = []
    original = torch.isfinite

    def counted(value):
        scans.append(id(value))
        return original(value)

    monkeypatch.setattr(torch, "isfinite", counted)
    return scans


def _check(value, value_type=TYPE):
    formula._validate_tensor_against_type(value, value_type, name="test")


@pytest.mark.parametrize("parameter", [False, True])
def test_scope_reuses_only_unchanged_successful_scans(monkeypatch, parameter):
    scans = _count_scans(monkeypatch)
    value = torch.ones(4)
    if parameter:
        value = torch.nn.Parameter(value)
    with formula._finite_validation_scope():
        _check(value)
        _check(value)
        assert len(scans) == 1
        with torch.no_grad():
            value.add_(1)
        _check(value)
        assert len(scans) == 2
        with torch.no_grad():
            value[0] = torch.inf
        for _ in range(2):
            with pytest.raises(formula.FormulaBindingError, match="finite"):
                _check(value)
        assert len(scans) == 4
    assert formula._FINITE_VALIDATION_MEMO.get() is None


def test_metadata_is_always_checked_even_after_a_finite_hit(monkeypatch):
    scans = _count_scans(monkeypatch)
    value = torch.ones(4)
    with formula._finite_validation_scope():
        _check(value)
        for wrong_type in (
            formula.TensorType(("D",), (3,), dtype="float32"),
            formula.TensorType(("D",), (4,), dtype="int64"),
        ):
            with pytest.raises(formula.FormulaBindingError):
                _check(value, wrong_type)
        assert len(scans) == 1


def test_nested_scope_exception_restores_parent_and_releases_tensors(monkeypatch):
    scans = _count_scans(monkeypatch)
    with formula._finite_validation_scope():
        value = torch.ones(4)
        ref = weakref.ref(value)
        _check(value)
        with pytest.raises(ValueError, match="nested"):
            with formula._finite_validation_scope():
                _check(value)
                raise ValueError("nested")
        _check(value)
        assert len(scans) == 2
        del value
        assert ref() is not None
    assert ref() is None
    assert formula._FINITE_VALIDATION_MEMO.get() is None


def test_separate_views_do_not_share_finite_facts(monkeypatch):
    scans = _count_scans(monkeypatch)
    value = torch.ones(8)
    left, right = value[:4], value[4:]
    with formula._finite_validation_scope():
        _check(left)
        _check(right)
        _check(left)
        assert len(scans) == 2
        value[4] = torch.nan
        _check(left)
        with pytest.raises(formula.FormulaBindingError, match="finite"):
            _check(right)
        assert len(scans) == 4


def test_inference_and_subclass_tensors_keep_original_checks(monkeypatch):
    class CustomTensor(torch.Tensor):
        pass

    scans = _count_scans(monkeypatch)
    with torch.inference_mode():
        inference = torch.ones(4)
    subclass = torch.ones(4).as_subclass(CustomTensor)
    with formula._finite_validation_scope():
        for value in (inference, subclass):
            _check(value)
            _check(value)
    assert len(scans) == 4


def test_recreated_broadcast_views_share_only_versioned_same_region(monkeypatch):
    scans = _count_scans(monkeypatch)
    value = torch.nn.Parameter(torch.ones(1, 4))
    kind = formula.TensorType(("B", "D"), ("B", 4), dtype="float32")
    with formula._finite_validation_scope():
        _check(value.expand(8, 4), kind)
        _check(value.expand(8, 4), kind)
        assert len(scans) == 1
        _check(value.expand(3, 4), kind)
        assert len(scans) == 2
        with torch.no_grad():
            value.add_(1)
        _check(value.expand(8, 4), kind)
        assert len(scans) == 3
        with torch.no_grad():
            value[0, 0] = torch.nan
        with pytest.raises(formula.FormulaBindingError, match="finite"):
            _check(value.expand(8, 4), kind)


def test_shared_base_with_different_strides_does_not_share_finite_fact(monkeypatch):
    scans = _count_scans(monkeypatch)
    base = torch.tensor([1., 2., 3., 4., torch.nan, 6., 7., 8.])
    with formula._finite_validation_scope():
        _check(base[:4])
        with pytest.raises(formula.FormulaBindingError, match="finite"):
            _check(base[::2])
        assert len(scans) == 2


@pytest.mark.parametrize("dense_base", [False, True])
def test_finite_base_does_not_admit_unscanned_storage(dense_base):
    storage = torch.tensor([1., 2., torch.nan, 4.])
    # set_ produces a base with no _base, backed by a larger storage.
    base = torch.empty(0).set_(storage.untyped_storage(), 0, (2,), (1,) if dense_base else (3,))
    expanded_region = base.as_strided((3,), (1,))
    kind = formula.TensorType(("D",), (3,), dtype="float32")
    with formula._finite_validation_scope():
        _check(base, formula.TensorType(("D",), (2,), dtype="float32"))
        with pytest.raises(formula.FormulaBindingError, match="finite"):
            _check(expanded_region, kind)


def test_default_is_uncached_and_ranking_scores_gradients_remain_equal(monkeypatch):
    from benchmarks import train_federated_branch_visible_federation as training
    from benchmarks._federated_v4_federation import build_autonomous_effect_federation

    scans = _count_scans(monkeypatch)
    value = torch.ones(4)
    _check(value)
    _check(value)
    assert len(scans) == 2
    federation = build_autonomous_effect_federation(
        hidden_dim=4, rank=4, seed=421, device=torch.device("cpu"),
        plastic_branches=8, min_operations=1, max_operations=2,
    )
    results = []
    batched_mask = training.candidate_mask

    def reference_mask(query, arenas, candidates, *, steps, include_stop):
        return torch.tensor([
            [query._candidate_eligible(c, arena, steps=steps) for c in candidates]
            + ([query._stop_eligible(arena, steps=steps)] if include_stop else [])
            for arena in arenas
        ], dtype=torch.bool, device=arenas[0].device)

    for batched in (False, True):
        monkeypatch.setattr(training, "candidate_mask", batched_mask if batched else reference_mask)
        x = torch.ones(1, 2, 4, requires_grad=True)
        arena = federation.query._arena({"x": x})
        branches = tuple(training._SearchBranch(arena, x.new_zeros(()), ()) for _ in range(4))
        scans.clear()
        ranked = training._ranked_candidates_many(
            federation, branches, federation.transition_layers[0], steps=0, width=4,
        )
        score = sum(prob for row in ranked for _, prob in row)
        params = tuple(federation.query.network.parameters())
        grads = torch.autograd.grad(score, (x, *params), allow_unused=True)
        results.append((len(scans), ranked, grads))
    assert results[1][0] < results[0][0]
    for old_row, new_row in zip(results[0][1], results[1][1], strict=True):
        assert [candidate.candidate_id for candidate, _ in old_row] == [candidate.candidate_id for candidate, _ in new_row]
        for (_, old_score), (_, new_score) in zip(old_row, new_row, strict=True):
            torch.testing.assert_close(old_score, new_score, rtol=0, atol=0)
    for old_grad, new_grad in zip(results[0][2], results[1][2], strict=True):
        assert (old_grad is None) == (new_grad is None)
        if old_grad is not None:
            torch.testing.assert_close(old_grad, new_grad, rtol=0, atol=0)
