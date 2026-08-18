from __future__ import annotations

import json

import pytest
import torch
from torch import nn

import arti
from arti.component_graph import ComponentGraphError, component_graph, validate_component_graph


class Composite(nn.Module):
    def __init__(self, *, shared: bool = False) -> None:
        super().__init__()
        self.recall = arti.Recall(dim=4, slots=2)
        self.half = arti.Half()
        projection = nn.Linear(4, 4)
        self.left = projection
        self.right = projection if shared else nn.Linear(4, 4)
        self.scale = nn.Parameter(torch.ones(()))

    def component_bindings(self):
        return [("recall.output", "half.input")]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.right(self.left(x)) * self.scale


def test_component_graph_describes_nested_components_and_bindings() -> None:
    graph = component_graph(Composite(shared=False))

    assert graph["format"] == "arti.component.graph"
    assert graph["root"] == "node-0000"
    assert any(node["ref"] == "arti/recall@1" for node in graph["nodes"])
    assert any(node["ref"] == "arti/half@1" for node in graph["nodes"])
    assert graph["bindings"] == [{"kind": "data", "from": "recall.output", "to": "half.input"}]
    assert graph["closure_fingerprint"] == arti.component_closure_fingerprint(graph)
    assert validate_component_graph(graph) == graph


def test_component_graph_reuses_shared_module_and_records_shared_parameter() -> None:
    graph = component_graph(Composite(shared=True))

    left_mount = next(item for item in graph["mounts"] if item["path"] == "$.left")
    right_mount = next(item for item in graph["mounts"] if item["path"] == "$.right")
    assert left_mount["node"] == right_mount["node"]
    assert any(group["kind"] == "same_parameter" for group in graph["parameter_groups"])


def test_component_graph_round_trip_is_stored_and_checked(tmp_path) -> None:
    source = Composite(shared=True).eval()
    saved = arti.save(source, tmp_path / "composite.st")
    manifest = json.loads(saved.manifest_path.read_text(encoding="utf-8"))
    graph = manifest["architecture"]["component_graph"]

    restored = Composite(shared=True).eval()
    loaded = arti.load(saved.weights_path, model=restored)
    assert loaded.manifest["architecture"]["component_graph"] == graph


def test_component_graph_rejects_shared_parameter_mismatch(tmp_path) -> None:
    saved = arti.save(Composite(shared=True).eval(), tmp_path / "shared.st")

    with pytest.raises(ValueError, match="component graph"):
        arti.load(saved.weights_path, model=Composite(shared=False).eval())


def test_component_graph_rejects_cycles() -> None:
    graph = component_graph(Composite())
    graph["edges"].append({"kind": "contains", "from": graph["root"], "to": graph["root"], "mount": "$"})
    graph["closure_fingerprint"] = arti.component_closure_fingerprint(graph)

    with pytest.raises(ComponentGraphError, match="cycle"):
        validate_component_graph(graph)


def test_legacy_manifest_without_component_graph_remains_loadable(tmp_path) -> None:
    saved = arti.save(arti.Half(learnable=True).eval(), tmp_path / "legacy-shape.st")
    manifest = json.loads(saved.manifest_path.read_text(encoding="utf-8"))
    manifest["architecture"].pop("component_graph")
    saved.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_hash = __import__("hashlib").sha256(saved.manifest_path.read_bytes()).hexdigest()
    lock = json.loads(saved.lock_path.read_text(encoding="utf-8"))
    lock["manifest_sha256"] = manifest_hash
    lock["files"]["manifest"]["sha256"] = manifest_hash
    saved.lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    loaded = arti.load(saved.weights_path, model=arti.Half(learnable=True).eval())
    assert "component_graph" not in loaded.manifest["architecture"]
