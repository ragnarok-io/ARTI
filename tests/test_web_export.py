import hashlib
import json

import pytest
import torch

from arti.nn import Fold, Half, LearnedPulse
from arti.web import export


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "module,inputs,output_shape",
    [
        (Half().eval(), {"x": torch.randn(2, 5, 4)}, ["batch", "tokens", 4]),
        (Fold(k=3, dim=4).eval(), {"x": torch.randn(2, 5, 4), "mask": torch.ones(2, 5)}, ["batch", 3, 4]),
        (LearnedPulse(k=3, dim=4).eval(), {"x": torch.randn(2, 5, 4), "q": torch.rand(2, 5, 1), "mask": torch.ones(2, 5)}, ["batch", 3, 4]),
    ],
)
def test_web_export_writes_hashed_artifact(tmp_path, module, inputs, output_shape):
    result = export(module, tmp_path / "layer-web", example_inputs=inputs)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    lock = json.loads(result.lock_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "arti.web"
    assert manifest["module"]["type"] == type(module).__name__
    assert manifest["output"]["shape"] == output_shape
    assert [item["name"] for item in manifest["inputs"]] == list(inputs)
    assert manifest["files"]["model.onnx"]["sha256"] == _sha256(result.model_path)
    assert lock["manifest"]["sha256"] == _sha256(result.manifest_path)


@pytest.mark.parametrize(
    "module,message",
    [
        (Half(stochastic=True).eval(), "stochastic Half"),
        (Fold(k=2, dim=4, mode="attention").eval(), "mode='soft'"),
        (Fold(k=2).eval(), "explicit dim"),
        (LearnedPulse(k=2, dim=4, fold_topk=2).eval(), "topk"),
    ],
)
def test_web_export_rejects_unsupported_modes(tmp_path, module, message):
    with pytest.raises(ValueError, match=message):
        export(module, tmp_path / "bad", example_inputs={"x": torch.randn(1, 3, 4)})


def test_web_export_requires_eval_float32_and_declared_inputs(tmp_path):
    with pytest.raises(ValueError, match="module.eval"):
        export(Half(), tmp_path / "training", example_inputs={"x": torch.randn(1, 3, 4)})
    with pytest.raises(ValueError, match="float32"):
        export(Half().eval(), tmp_path / "dtype", example_inputs={"x": torch.randn(1, 3, 4).double()})
    with pytest.raises(ValueError, match="unsupported"):
        export(Half().eval(), tmp_path / "q", example_inputs={"x": torch.randn(1, 3, 4), "q": torch.ones(1, 3)})
