from __future__ import annotations

import json

import pytest
import torch

import arti


def test_inspect_executes_forward_and_restores_training_mode() -> None:
    layer = arti.nn.Layer(8, profile="recall")
    layer.train()
    report = arti.inspect(layer, torch.randn(2, 5, 8))

    assert layer.training is True
    assert report.input_shapes == [2, 5, 8]
    assert report.output_shapes["y"] == [2, 5, 8]
    assert report.output_shapes["pooled"] == [2, 8]
    assert report.mechanisms["recall"] is True
    assert report.parameter_groups["recall"] > 0
    assert report.parameter_groups["virtual_recall"] > 0
    assert report.latency_seconds is not None and report.latency_seconds >= 0
    json.dumps(report.to_dict())
    assert "Enabled mechanisms" in report.to_markdown()


def test_inspect_reports_multisource_input_contract_and_workspace() -> None:
    layer = arti.nn.Layer(8, profile="multisource", coord_dim=2)
    report = arti.inspect(layer)

    assert report.required_inputs["coord"] is True
    assert report.required_inputs["visibility"] is True
    assert report.required_inputs["frame_operators"] is True
    assert report.workspace["interface_slots"] == 8
    assert report.synthetic_context is False


def test_inspect_mapping_example_and_tensor_memory() -> None:
    layer = arti.nn.Layer(8)
    x = torch.randn(2, 3, 8)
    report = arti.inspect(layer, {"x": x})

    assert report.input_shapes == {"x": [2, 3, 8]}
    assert report.input_bytes == x.numel() * x.element_size()
    assert report.output_bytes > 0


def test_inspect_rejects_ambiguous_mapping_call() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        arti.inspect(arti.nn.Layer(4), {"x": torch.randn(1, 4)}, mask=torch.ones(1, dtype=torch.bool))


def test_inspect_visual_scan_reports_carrier_and_pulse_separately() -> None:
    config = arti.VisualScanConfig(low_size=(8, 8), patch_size=(4, 4), pulse_count=4)
    report = arti.inspect(arti.VisualScan(config))

    assert report.mechanisms["pulse"] is True
    assert report.mechanisms["half"] is True
    assert report.mechanisms["carrier"] is True
    assert report.workspace["pulse_slots"] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_inspect_reports_cuda_peak_memory_and_amp_dtype() -> None:
    layer = arti.nn.Layer(16, profile="recall").cuda()
    x = torch.randn(8, 32, 16, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        report = arti.inspect(layer, x)

    assert report.devices == ("cuda:0",)
    assert report.peak_cuda_memory_bytes is not None and report.peak_cuda_memory_bytes > 0
