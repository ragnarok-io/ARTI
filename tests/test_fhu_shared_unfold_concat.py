from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "fhu_shared_unfold_concat.py"
SPEC = importlib.util.spec_from_file_location("fhu_shared_unfold_concat", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
ConcatPulseModel = MODULE.ConcatPulseModel


def _model(mode: str) -> torch.nn.Module:
    return ConcatPulseModel(
        mode=mode,
        pulse_count=3,
        compact_k=2,
        exposed_per_pulse=1,
        dim=9,
        payload_dim=5,
        value_operators=2,
    )


def test_concat_pulse_modes_share_one_output_contract_and_backpropagate() -> None:
    streams = torch.randn(4, 3, 5, 9)
    q = torch.rand(4, 3, 5)
    input_sizes = set()
    for mode in ("concat_only", "independent", "shared"):
        model = _model(mode)
        prediction = model(streams, q)
        assert prediction.shape == (4, 5)
        input_sizes.add(model.readout.in_features)
        prediction.square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert len(input_sizes) == 1


def test_shared_unfold_reuses_the_common_operator_and_layout_bank() -> None:
    shared = _model("shared")
    independent = _model("independent")
    assert shared.shared_unfold is not None
    assert shared.independent_unfolds is None
    assert independent.shared_unfold is None
    assert len(independent.independent_unfolds) == 3
    assert shared.unfold_parameter_count < independent.unfold_parameter_count


def test_one_shared_unfold_adapts_to_single_and_concatenated_pulse_sizes() -> None:
    model = _model("shared")
    assert model.shared_unfold is not None
    streams = torch.randn(2, 3, 5, 9)
    q = torch.rand(2, 3, 5)
    compact = [
        core(streams[:, index], q[:, index])
        for index, core in enumerate(model.cores)
    ]

    single = model.shared_unfold(compact[0], target_length=3)
    concatenated = model.shared_unfold(
        torch.cat(compact, dim=1),
        target_length=9,
    )

    assert single.shape == (2, 3, 9)
    assert concatenated.shape == (2, 9, 9)
