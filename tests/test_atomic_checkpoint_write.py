from __future__ import annotations

import json

import pytest
import torch

from arti import serialization


def sharing_error(code=32):
    error = PermissionError("Windows file sharing violation")
    error.winerror = code
    return error


def test_atomic_json_retries_temporary_windows_sharing_lock(tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    serialization._atomic_json(path, {"step": 1})
    replace = serialization.os.replace
    calls, sleeps = [], []

    def locked_then_ready(source, destination):
        calls.append((source, destination))
        if len(calls) <= 2:
            raise sharing_error(5)
        replace(source, destination)

    monkeypatch.setattr(serialization.os, "replace", locked_then_ready)
    monkeypatch.setattr(serialization.time, "sleep", sleeps.append)
    serialization._atomic_json(path, {"step": 2})
    assert json.loads(path.read_text())["step"] == 2
    assert len(calls) == 3 and sleeps == [0.01, 0.02]
    assert not (tmp_path / ".run.json.tmp").exists()


def test_permanent_windows_lock_is_bounded_and_preserves_previous_json(tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    serialization._atomic_json(path, {"step": 1})
    calls, sleeps = [], []

    def locked(*args):
        calls.append(args)
        raise sharing_error(32)

    monkeypatch.setattr(serialization.os, "replace", locked)
    monkeypatch.setattr(serialization.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError):
        serialization._atomic_json(path, {"step": 2})
    assert len(calls) == 6 and sum(sleeps) == pytest.approx(0.31)
    assert json.loads(path.read_text())["step"] == 1
    assert not (tmp_path / ".run.json.tmp").exists()


def test_non_windows_permission_error_is_not_retried(tmp_path, monkeypatch):
    calls = []

    def denied(*args):
        calls.append(args)
        raise PermissionError("permanent permissions")

    monkeypatch.setattr(serialization.os, "replace", denied)
    monkeypatch.setattr(serialization.time, "sleep", lambda _: pytest.fail("must not retry"))
    with pytest.raises(PermissionError):
        serialization._atomic_json(tmp_path / "run.json", {"step": 1})
    assert len(calls) == 1


def test_atomic_tensor_save_uses_the_same_bounded_replace(tmp_path, monkeypatch):
    path = tmp_path / "weights.st"
    replace = serialization.os.replace
    calls = []

    def locked_once(source, destination):
        calls.append((source, destination))
        if len(calls) == 1:
            raise sharing_error(33)
        replace(source, destination)

    monkeypatch.setattr(serialization.os, "replace", locked_once)
    monkeypatch.setattr(serialization.time, "sleep", lambda _: None)
    serialization._atomic_safetensors({"weight": torch.ones(2)}, path, metadata={"kind": "test"})
    assert len(calls) == 2
    assert path.exists() and not (tmp_path / ".weights.st.tmp").exists()
