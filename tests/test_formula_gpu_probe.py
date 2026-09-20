"""CPU checks for the GPU probe's numerical acceptance criteria."""

import pytest
import torch

from benchmarks.probe_formula_expression_gpu import _compare
from benchmarks.summarize_formula_gpu_probe import kernel_counts


def test_score_metadata_allows_only_matching_negative_infinity():
    score = torch.tensor([0.5, -torch.inf])
    assert _compare(score, score.clone(), score_metadata=True)[0]["max_abs"] == 0
    with pytest.raises(AssertionError):
        _compare(score, score.clone())
    for invalid in (torch.inf, torch.nan):
        value = torch.tensor([invalid])
        with pytest.raises(AssertionError):
            _compare(value, value.clone(), score_metadata=True)


def test_gradient_comparison_preserves_unused_structure():
    value = torch.zeros(2)
    _compare((None, value), (None, value.clone()), gradient=True)
    with pytest.raises(AssertionError):
        _compare((None, value), (value, None), gradient=True)


def test_discrete_results_are_exact_and_dtype_is_preserved():
    with pytest.raises(AssertionError):
        _compare(torch.tensor([1]), torch.tensor([2]))
    with pytest.raises(AssertionError):
        _compare(torch.tensor([1.]), torch.tensor([1.], dtype=torch.float64))


def test_physical_kernel_count_excludes_gpu_annotation_regions_and_copies():
    trace = {"traceEvents": [
        {"cat": "kernel", "name": "fused"},
        {"cat": "kernel", "name": "fused"},
        {"cat": "gpu_user_annotation", "name": "probe_call"},
        {"cat": "gpu_memcpy", "name": "Memcpy DtoD"},
        {"cat": "cuda_runtime", "name": "cudaLaunchKernel"},
    ]}
    assert kernel_counts(trace) == {"fused": 2}
