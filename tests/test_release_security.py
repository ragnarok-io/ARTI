from __future__ import annotations

import pytest
import torch

import arti.cli as cli
from arti.providers import ARTIProviderError, _reject_remote_code
from arti.cli import parse_sample_shape, sample_tensor_from_spec
from arti.serialization import (
    MAX_CHECKPOINT_TREE_DEPTH,
    MAX_JSON_BYTES,
    _check_version_compatibility,
    _decode_tree,
    _load_json,
)


def test_cli_state_summary_requests_restricted_torch_loader(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def restricted_load(path, **kwargs):
        captured.update(kwargs)
        return {"weight": torch.ones(2)}

    monkeypatch.setattr(cli.torch, "load", restricted_load)
    report = cli.summarize_state_dict(tmp_path / "state.pt")

    assert report["ok"] is True
    assert captured["weights_only"] is True


def test_declarative_pretrained_loading_rejects_remote_code() -> None:
    with pytest.raises(ARTIProviderError, match="trust_remote_code=True"):
        _reject_remote_code({"trust_remote_code": True})
    _reject_remote_code({"trust_remote_code": False})
    _reject_remote_code({})


def test_json_sidecar_size_limit_is_enforced(tmp_path) -> None:
    path = tmp_path / "large.json"
    path.write_bytes(b" " * (MAX_JSON_BYTES + 1))

    with pytest.raises(ValueError, match="exceeds"):
        _load_json(path)


def test_checkpoint_tree_depth_limit_is_enforced() -> None:
    value: object = "leaf"
    for _ in range(MAX_CHECKPOINT_TREE_DEPTH + 1):
        value = {"__arti_list__": [value]}

    with pytest.raises(ValueError, match="maximum depth"):
        _decode_tree(value, {})


def test_legacy_zero_major_artifact_remains_readable_by_one_major() -> None:
    _check_version_compatibility("0.2.0", "1.0.0")
    with pytest.raises(ValueError, match="major version"):
        _check_version_compatibility("2.0.0", "1.0.0")


def test_public_one_major_artifact_remains_readable_by_two_major() -> None:
    _check_version_compatibility("0.2.0", "2.0.0")
    _check_version_compatibility("1.9.0", "2.0.0")
    with pytest.raises(ValueError, match="major version"):
        _check_version_compatibility("3.0.0", "2.0.0")
    with pytest.raises(ValueError, match="newer than ARTI"):
        _check_version_compatibility("0.99.0", "2.0.0")
    with pytest.raises(ValueError, match="newer than ARTI"):
        _check_version_compatibility("1.10.0", "2.0.0")


def test_cli_rejects_tensor_dimension_bombs_before_allocation() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        parse_sample_shape("1,1073741824,1073741824")
    with pytest.raises(ValueError, match="cannot exceed"):
        sample_tensor_from_spec({"shape": [1, 1073741824, 1073741824], "kind": "zeros"})
