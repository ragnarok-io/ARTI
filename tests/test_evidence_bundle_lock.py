from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

LOCK_SPEC = importlib.util.spec_from_file_location("lock_evidence_bundle", ROOT / "benchmarks" / "lock_evidence_bundle.py")
assert LOCK_SPEC is not None
lock_evidence_bundle = importlib.util.module_from_spec(LOCK_SPEC)
assert LOCK_SPEC.loader is not None
LOCK_SPEC.loader.exec_module(lock_evidence_bundle)

VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_evidence_bundle_lock", ROOT / "benchmarks" / "verify_evidence_bundle_lock.py"
)
assert VERIFY_SPEC is not None
verify_evidence_bundle_lock = importlib.util.module_from_spec(VERIFY_SPEC)
assert VERIFY_SPEC.loader is not None
VERIFY_SPEC.loader.exec_module(verify_evidence_bundle_lock)


def test_summarize_records_seed_count_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text('{"provenance": {"seed_values": [0, 1, 2]}}', encoding="utf-8")

    summary = lock_evidence_bundle.summarize(path)

    assert summary["seed_count"] == 3
    assert summary["bytes"] > 0
    assert len(summary["sha256"]) == 64


def test_verify_lock_detects_hash_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "benchmarks" / "results" / "high_power_nature_results.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"provenance": {"seed_values": [0, 1, 2, 3, 4, 5]}}', encoding="utf-8")
    lock = {"artifacts": [{"path": artifact.relative_to(tmp_path).as_posix(), "sha256": "bad", "seed_count": 6}]}

    failures = verify_evidence_bundle_lock.verify(tmp_path, lock)

    assert any("sha256 mismatch" in failure for failure in failures)


def test_verify_lock_detects_low_seed_count(tmp_path: Path) -> None:
    artifact = tmp_path / "benchmarks" / "results" / "high_power_proxy_results.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"provenance": {"seed_values": [0, 1, 2]}}', encoding="utf-8")
    digest = verify_evidence_bundle_lock.sha256_file(artifact)
    lock = {"artifacts": [{"path": artifact.relative_to(tmp_path).as_posix(), "sha256": digest, "seed_count": 3}]}

    failures = verify_evidence_bundle_lock.verify(tmp_path, lock)

    assert any("seed count below" in failure for failure in failures)
