from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import uuid

import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"


def load_supervisor():
    if str(BENCHMARKS) not in sys.path:
        sys.path.insert(0, str(BENCHMARKS))
    spec = importlib.util.spec_from_file_location(
        "formula_topology_v4_arm_supervisor_test",
        BENCHMARKS / "run_formula_topology_v4_arm_characterization.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def receipt(**updates):
    value = {
        "cause": "COMPLETED",
        "returncode": 0,
        "job_ownership_established": True,
        "job_close_failed": False,
        "cleanup_confirmed": True,
    }
    value.update(updates)
    return value


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (receipt(cause="TIMEOUT"), "INVALID_TIMEOUT"),
        (receipt(cause="JOB_SETUP_FAILURE"), "INVALID_PROCESS_LIFECYCLE"),
        (receipt(job_ownership_established=False), "INVALID_PROCESS_LIFECYCLE"),
        (receipt(cleanup_confirmed=False), "INVALID_CLEANUP"),
        (receipt(job_close_failed=True), "INVALID_CLEANUP"),
        (receipt(returncode=3), "INVALID_ARTIFACT"),
    ],
)
def test_failure_receipts_never_classify_as_success(
    tmp_path: Path, candidate: dict, expected: str
) -> None:
    supervisor = load_supervisor()
    assert supervisor._classification_from_receipt(candidate, tmp_path / "missing") == expected


def test_only_clean_receipt_with_manifest_can_succeed(tmp_path: Path) -> None:
    supervisor = load_supervisor()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="ascii")
    assert supervisor._classification_from_receipt(receipt(), manifest) == "VALID_CHARACTERIZED"


def test_output_must_be_new_and_outside_checkout(tmp_path: Path) -> None:
    supervisor = load_supervisor()
    external = ROOT.parent / f"ARTI-test-output-{uuid.uuid4().hex}"
    assert supervisor._validate_new_external_output(external) == external.absolute()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="must not already exist"):
        supervisor._validate_new_external_output(existing)
    with pytest.raises(ValueError, match="outside"):
        supervisor._validate_new_external_output(ROOT / ".tmp" / "inside-checkout")


def test_sha_lines_rejects_missing_publication_file(tmp_path: Path) -> None:
    supervisor = load_supervisor()
    (tmp_path / "present.txt").write_text("x", encoding="ascii")
    with pytest.raises(RuntimeError, match="missing"):
        supervisor._sha_lines(
            tmp_path,
            ["present.txt", "missing.txt"],
            deadline=float("inf"),
        )


def test_supervisor_and_verifier_reconstruct_the_same_source_identity() -> None:
    supervisor = load_supervisor()
    spec = importlib.util.spec_from_file_location(
        "formula_topology_v4_arm_verifier_identity_test",
        BENCHMARKS / "verify_formula_topology_v4_arm_characterization.py",
    )
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = verifier
    spec.loader.exec_module(verifier)
    assert supervisor._git_source_identity() == verifier._current_source_identity()
