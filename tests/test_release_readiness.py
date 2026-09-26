from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_release_readiness", ROOT / "scripts" / "check_release_readiness.py"
)
check_release_readiness = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(check_release_readiness)


def test_public_release_identity_and_versions_match() -> None:
    assert check_release_readiness.check_version_consistency() == []
    assert check_release_readiness.check_public_identity() == []


def test_release_workflow_runs_the_same_ci_before_build_and_publish() -> None:
    assert check_release_readiness.check_workflows() == []


def test_release_workflow_rejects_publish_without_ci_dependency() -> None:
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    release = release.replace("build:\n    needs: quality", "build:")
    failures = check_release_readiness.check_workflows(release_text=release)
    assert any(
        "release workflow missing required gate" in item and "build:" in item for item in failures
    )


def test_readiness_uses_only_public_release_artifacts() -> None:
    payload = check_release_readiness.check_readiness()
    assert payload["ok"] is True
    assert payload["distribution"] == "arti-fit"
    assert payload["required_gates"] == [
        "ci",
        "version-consistency",
        "package-build",
        "release-workflow",
    ]


def test_readiness_has_no_private_benchmark_report_dependency() -> None:
    source = (ROOT / "scripts" / "check_release_readiness.py").read_text(encoding="utf-8")
    assert "benchmarks/results" not in source
    assert "verify_core_goal" not in source
