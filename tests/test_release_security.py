from __future__ import annotations

import base64
import hashlib
import re

import pytest
import torch
from scripts.check_package import (
    archive_inventory,
    archive_member_findings,
    content_findings,
    wheel_record_findings,
)

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


def test_package_content_audit_rejects_absolute_user_paths_without_echoing_them() -> None:
    separator = bytes([92])
    path = (
        b"C:"
        + separator
        + b"Users"
        + separator
        + b"alice"
        + separator
        + b".cache"
        + separator
        + b"run.json"
    )
    findings = content_findings("sample.py", path)

    assert findings == ["sample.py: contains an absolute user-directory path"]
    assert "alice" not in findings[0]


@pytest.mark.parametrize(
    "path",
    [
        b"C:"
        + bytes([92])
        + b"Users"
        + bytes([92])
        + b"alice"
        + bytes([92])
        + b".cache"
        + bytes([92])
        + b"model.bin",
        b"/home/" + b"alice" + b"/.cache/model.bin",
        b"/Users/" + b"alice" + b"/Library/Caches/model.bin",
    ],
)
def test_package_content_audit_rejects_home_paths_regardless_of_subdirectory(path: bytes) -> None:
    findings = content_findings("metadata.json", path)

    assert findings == ["metadata.json: contains an absolute user-directory path"]
    assert "alice" not in findings[0]


def test_package_content_audit_rejects_common_secret_token_shapes_without_echoing_them() -> None:
    findings = content_findings("metadata.txt", b"ghp_" + b"A" * 36)

    assert findings == ["metadata.txt: contains a possible GitHub token"]
    assert "A" * 20 not in findings[0]


@pytest.mark.parametrize(
    ("prefix", "suffix"),
    [
        (b"glpat-", b"A" * 24),
        (b"xoxb-", b"A" * 24),
        (b"AIza", b"A" * 35),
        (b"Bearer ", b"A" * 28),
    ],
)
def test_package_content_audit_rejects_additional_provider_credentials(
    prefix: bytes, suffix: bytes
) -> None:
    findings = content_findings("config.txt", prefix + suffix)

    assert findings == [
        "config.txt: contains a possible "
        + {
            b"glpat-": "GitLab token",
            b"xoxb-": "Slack token",
            b"AIza": "Google API key",
            b"Bearer ": "Bearer token",
        }[prefix]
    ]


def test_wheel_record_accepts_credential_shaped_digest_but_scans_paths() -> None:
    record_name = "arti_fit-3.1.0a2.dist-info/RECORD"
    payload = b"payload-61220"
    checksum = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("=")
    assert re.search(rb"hf_[A-Za-z0-9]{30,}", checksum.encode())
    entry = f"arti/module.py,sha256={checksum},{len(payload)}\n"
    content = f"{entry}{record_name},,\n".encode("utf-8")
    members = {"arti/module.py": payload, record_name: content}

    assert wheel_record_findings(record_name, content, members) == []

    leaked_path = "arti/hf_" + "A" * 32 + ".py"
    entry = f"{leaked_path},sha256={checksum},{len(payload)}\n"
    content = f"{entry}{record_name},,\n".encode("utf-8")
    members = {leaked_path: payload, record_name: content}
    assert wheel_record_findings(record_name, content, members) == [
        f"{record_name} path: contains a possible Hugging Face token"
    ]


def test_wheel_record_rejects_malformed_digest_instead_of_ignoring_it() -> None:
    record_name = "arti_fit-3.1.0a2.dist-info/RECORD"
    content = f"arti/module.py,hf_{'A' * 40},123\n{record_name},,\n".encode("utf-8")

    assert wheel_record_findings(record_name, content, {record_name: content}) == [
        f"{record_name}: malformed wheel RECORD"
    ]


def test_archive_inventory_rejects_unreviewed_members() -> None:
    expected_names = ["arti/__init__.py", "arti/_version.py"]
    count, digest = archive_inventory(expected_names)
    expected = {"file_count": count, "paths_sha256": digest}

    assert archive_member_findings("wheel", expected_names, expected) == []
    findings = archive_member_findings(
        "wheel", expected_names + ["arti/private_notes.md"], expected
    )
    assert findings == ["wheel: file inventory differs from the reviewed release manifest"]


def test_archive_inventory_rejects_workspace_paths_without_echoing_them() -> None:
    count, digest = archive_inventory(["arti/__init__.py"])
    expected = {"file_count": count, "paths_sha256": digest}

    findings = archive_member_findings("wheel", ["arti/.cache/private.bin"], expected)

    assert findings == [
        "wheel: contains a disallowed archive path",
        "wheel: file inventory differs from the reviewed release manifest",
    ]
