import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from arti.nn import Fold, Half, LearnedPulse, StatefulRecall
from arti.web import ARTIWebTensorMetadata, artifact_schema, export, export_stateful_recall, render_artifact_typescript, render_typescript_contract, stateful_artifact_schema


ROOT = Path(__file__).resolve().parents[1]


class GenericAffine(nn.Module):
    def forward(self, signal, gate):
        return {"result": signal * (1.0 + gate), "salience": gate.expand_as(signal)}


class NestedInfo(nn.Module):
    def forward(self, x, *, return_info=False):
        if not return_info:
            return x
        return x + 1, {
            "workspace": x * 2,
            "mask": x[..., 0] > 0,
            "index": torch.arange(x.shape[1], dtype=torch.int64),
            "score": x.mean(),
        }


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


def test_web_export_flattens_nested_tensors_with_python_owned_metadata(tmp_path):
    x = torch.randn(2, 5, 4)
    names = ["y", "workspace", "mask", "index", "score"]
    result = export(
        NestedInfo().eval(),
        tmp_path / "inspectable",
        example_inputs={"x": x},
        forward_kwargs={"return_info": True},
        output_names=names,
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}, "workspace": {0: "batch"}, "mask": {0: "batch"}},
        output_metadata={
            "workspace": ARTIWebTensorMetadata(role="workspace", atol=1e-5, rtol=1e-5),
            "mask": ARTIWebTensorMetadata(logical_type="mask"),
            "index": ARTIWebTensorMetadata(logical_type="index"),
        },
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    contracts = {item["name"]: item for item in manifest["outputs"]}
    assert manifest["module"]["forward_kwargs"] == {"return_info": True}
    assert contracts["y"]["role"] == "primary"
    assert contracts["workspace"]["role"] == "workspace"
    assert contracts["mask"]["dtype"] == "bool"
    assert contracts["mask"]["logical_type"] == "mask"
    assert contracts["index"]["dtype"] == "int64"
    assert contracts["index"]["tolerance"] == {"atol": 0.0, "rtol": 0.0}
    assert contracts["score"]["shape"] == []
    assert contracts["workspace"]["dynamic_axes"] == {"0": "batch"}
    assert all(item["max_bytes"] > 0 and "tolerance" in item for item in manifest["inputs"] + manifest["outputs"])


def test_web_export_selects_python_named_nested_outputs(tmp_path):
    result = export(
        NestedInfo().eval(),
        tmp_path / "selected-inspectable",
        example_inputs={"x": torch.randn(2, 5, 4)},
        forward_kwargs={"return_info": True},
        output_names=["y", "workspace", "mask", "index", "score"],
        include_outputs=["y", "workspace", "index"],
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert [item["name"] for item in manifest["outputs"]] == ["y", "workspace", "index"]


@pytest.mark.parametrize("include_outputs", [[], ["missing"], ["y", "y"]])
def test_web_export_rejects_invalid_selected_outputs(tmp_path, include_outputs):
    with pytest.raises(ValueError, match="include_outputs"):
        export(
            NestedInfo().eval(),
            tmp_path / "invalid-selected-inspectable",
            example_inputs={"x": torch.randn(2, 5, 4)},
            forward_kwargs={"return_info": True},
            output_names=["y", "workspace", "mask", "index", "score"],
            include_outputs=include_outputs,
        )


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


def test_web_runtime_source_contains_no_arti_mechanism_implementation():
    source_root = ROOT / "packages" / "web" / "src"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in source_root.rglob("*.ts")
        if "generated" not in path.parts
    )
    for mechanism in ("Half", "Fold", "Pulse", "UnFold", "FusionPulse"):
        assert f"class {mechanism}" not in source
        assert f"function {mechanism.lower()}" not in source.lower()
