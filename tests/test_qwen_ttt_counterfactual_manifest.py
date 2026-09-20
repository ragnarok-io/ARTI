from __future__ import annotations

import torch

from benchmarks.build_qwen_ttt_counterfactual_manifest import _tensor_hash


def test_tensor_hash_is_shape_and_value_stable() -> None:
    value = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    assert _tensor_hash(value) == _tensor_hash(value.clone())
    assert _tensor_hash(value) != _tensor_hash(value + 1)
