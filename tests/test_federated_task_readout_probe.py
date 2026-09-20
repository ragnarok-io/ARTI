import pytest
import torch

from benchmarks.probe_qwen_federated_causal_readout import compare_logits


def test_task_readout_comparison_detects_answer_position_changes():
    reference = torch.tensor([[6.0, 1.0, 0.0], [0.0, 5.0, 1.0]])
    labels = torch.tensor([[0, 1]])
    same = compare_logits(reference, reference.clone(), labels)
    assert same["maximum_absolute_error"] == 0
    assert same["argmax_agreement"] == 1
    assert same["actual_ce"] == same["reference_ce"]
    changed = compare_logits(reference, reference.flip(0), labels)
    assert changed["argmax_agreement"] == 0
    assert changed["actual_ce"] > changed["reference_ce"]
    with pytest.raises(ValueError, match="aligned"):
        compare_logits(reference[:1], reference, labels)
