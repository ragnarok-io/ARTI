import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from arti.nn import Fold, Half, LearnedPulse, StatefulRecall
from arti.web import artifact_schema, export, export_stateful_recall, render_artifact_typescript, render_typescript_contract, stateful_artifact_schema


ROOT = Path(__file__).resolve().parents[1]


class GenericAffine(nn.Module):
    def forward(self, signal, gate):
        return {"result": signal * (1.0 + gate), "salience": gate.expand_as(signal)}


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
    assert manifest["format_version"] == 2
    assert manifest["module"]["type"].endswith(type(module).__name__)
    assert manifest["outputs"][0]["shape"] == output_shape
    assert [item["name"] for item in manifest["inputs"]] == list(inputs)
    assert manifest["files"]["model.onnx"]["sha256"] == _sha256(result.model_path)
    assert lock["manifest"]["sha256"] == _sha256(result.manifest_path)
    assert manifest["files"]["artifact.ts"]["sha256"] == _sha256(result.typescript_path)
    assert lock["files"]["artifact.ts"] == manifest["files"]["artifact.ts"]


def test_web_export_accepts_generic_named_inputs_and_outputs(tmp_path):
    signal = torch.randn(2, 5, 4)
    gate = torch.rand(2, 5, 1)
    result = export(GenericAffine().eval(), tmp_path / "generic", example_inputs={"signal": signal, "gate": gate})
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert [item["name"] for item in manifest["inputs"]] == ["signal", "gate"]
    assert [item["name"] for item in manifest["outputs"]] == ["result", "salience"]
    assert manifest["module"]["type"].endswith("GenericAffine")
    generated = result.typescript_path.read_text(encoding="utf-8")
    assert '"signal": Tensor' in generated and '"gate": Tensor' in generated
    assert '"result": Tensor' in generated and '"salience": Tensor' in generated
    assert "async run(inputs: ArtifactInputs): Promise<ArtifactOutputs>" in generated
    assert "async forward(" not in generated


def test_artifact_typescript_is_deterministic_typed_and_mechanism_free():
    manifest = {
        "format": "arti.web",
        "format_version": 2,
        "module": {"type": "anything.Half", "config": {"threshold": 0.5}},
        "inputs": [{"name": "input-value", "dtype": "float32", "shape": [1]}],
        "outputs": [{"name": "result", "dtype": "float32", "shape": [1]}],
    }
    first = render_artifact_typescript(manifest)
    assert first == render_artifact_typescript(dict(reversed(list(manifest.items()))))
    assert '"input-value": Tensor' in first
    assert "async forward(value: Tensor): Promise<Tensor>" in first
    assert "module.forward(value)" in first
    assert "Half" not in first and "Fold" not in first and "Recall" not in first
    assert "module.type" not in first


def test_generated_artifact_typescript_fixture_is_current_and_compiled_by_web_build():
    manifest = {
        "format": "arti.web",
        "format_version": 2,
        "inputs": [{"name": "signal"}, {"name": "gate"}],
        "outputs": [{"name": "result"}, {"name": "salience"}],
    }
    generated = render_artifact_typescript(manifest)
    fixture = (Path(__file__).parents[1] / "packages" / "web" / "tests" / "generated-artifact.ts").read_text(encoding="utf-8")
    assert fixture.split("\n", 1)[1] == generated.split("\n", 1)[1]


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
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        export(Half().eval(), tmp_path / "q", example_inputs={"x": torch.randn(1, 3, 4), "q": torch.ones(1, 3)})


def test_web_contract_schema_and_generated_types_are_python_owned():
    schema = artifact_schema()
    assert schema["properties"]["manifest"]["properties"]["format_version"] == {"const": 2}
    generated = render_typescript_contract()
    assert "Generated by arti.web.contract" in generated
    assert "format_version: 2" in generated
    assert (ROOT / "packages" / "web" / "src" / "generated" / "contract.ts").read_text(encoding="utf-8") == generated


def test_stateful_web_export_writes_read_update_and_fixed_state_contract(tmp_path):
    module = StatefulRecall(4, slots=3).eval()
    result = export_stateful_recall(
        module,
        tmp_path / "stateful",
        example_x=torch.randn(2, 5, 4),
        example_mask=torch.ones(2, 5),
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    lock = json.loads(result.lock_path.read_text(encoding="utf-8"))
    assert manifest["format_version"] == 3
    assert manifest["artifact_kind"] == "stateful"
    assert set(manifest["entrypoints"]) == {"read", "update"}
    assert [item["name"] for item in manifest["state"]] == ["keys", "values", "strengths"]
    assert all(item["initializer"] == "zeros" for item in manifest["state"])
    assert manifest["persistence"] == "explicit"
    assert manifest["limits"]["max_state_bytes_per_batch"] == (3 * 4 + 3 * 4 + 3) * 4
    assert stateful_artifact_schema()["properties"]["manifest"]["properties"]["format_version"] == {"const": 3}
    for name, path in (("read.onnx", result.read_model_path), ("update.onnx", result.update_model_path)):
        assert manifest["files"][name]["sha256"] == _sha256(path)
        assert lock["files"][name]["sha256"] == _sha256(path)


def test_generated_stateful_contract_enforces_resource_and_path_boundaries():
    generated = render_typescript_contract()
    assert "MAX_STATEFUL_FILES" in generated
    assert "MAX_STATEFUL_ENTRYPOINTS" in generated
    assert "MAX_STATEFUL_ARTIFACT_BYTES" in generated
    assert "safeArtifactFileName" in generated
    assert "state budget does not match declared state shapes" in generated
