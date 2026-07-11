from __future__ import annotations

from pathlib import Path

import pytest
import torch

import arti
import arti.serialization as serialization
from arti.fit.artifacts import load_artifact
from arti.providers import ARTIProviderError, TransformersProvider
from arti.cli import parse_sample_shape, sample_tensor_from_spec


class _WriteMarker:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return Path.write_text, (self.marker, "executed", "utf-8")


def test_artifact_loader_rejects_pickle_code_execution(tmp_path: Path) -> None:
    marker = tmp_path / "executed.txt"
    payload = tmp_path / "malicious.pt"
    torch.save({"payload": _WriteMarker(marker)}, payload)

    with pytest.raises((RuntimeError, pickle_error_type())):
        load_artifact(payload)

    assert not marker.exists()


def pickle_error_type() -> type[Exception]:
    error = getattr(torch.serialization, "pickle", None)
    return getattr(error, "UnpicklingError", RuntimeError)


def test_public_source_has_no_unrestricted_torch_load() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "arti"
    offenders = []
    for path in root.rglob("*.py"):
        if "weights_only=" + "False" in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(root).as_posix())
    assert offenders == []


def test_declarative_provider_rejects_remote_code_before_import(monkeypatch) -> None:
    provider = TransformersProvider()
    monkeypatch.setattr(type(provider), "available", property(lambda self: True))

    with pytest.raises(ARTIProviderError, match="trust_remote_code"):
        provider.load("untrusted/model", task="causal-lm", revision=None, kwargs={"trust_remote_code": True})


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: arti.Half(threshold=float("nan")), "threshold"),
        (lambda: arti.Half(base=float("nan")), "base"),
        (lambda: arti.Half(scale=float("inf")), "scale"),
        (lambda: arti.Fold(k=2, temperature=float("nan")), "temperature"),
    ],
)
def test_nonfinite_activation_configuration_is_rejected(factory, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory()


def test_cli_rejects_tensor_dimension_bombs_before_allocation() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        parse_sample_shape("1,1073741824,1073741824")
    with pytest.raises(ValueError, match="cannot exceed"):
        sample_tensor_from_spec({"shape": [1, 1073741824, 1073741824], "kind": "zeros"})


def test_arti_st_rejects_oversized_json_sidecars(tmp_path: Path, monkeypatch) -> None:
    saved = arti.save(torch.nn.Linear(2, 2), tmp_path / "model.st")
    monkeypatch.setattr(serialization, "MAX_JSON_BYTES", 32)

    with pytest.raises(ValueError, match="exceeds"):
        arti.load(saved.weights_path)


def test_checkpoint_decoder_rejects_excessive_nesting() -> None:
    tree: object = None
    for _ in range(serialization.MAX_CHECKPOINT_TREE_DEPTH + 2):
        tree = {"__arti_list__": [tree]}

    with pytest.raises(ValueError, match="maximum depth"):
        serialization._decode_tree(tree, {})
