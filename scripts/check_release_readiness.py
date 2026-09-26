"""Validate public package identity and release workflow wiring."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _value(relative: str, pattern: str, label: str) -> str:
    match = re.search(pattern, read(relative), flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"{label} not found in {relative}")
    return match.group(1)


def project_version() -> str:
    return _value("pyproject.toml", r'^version = "([^\"]+)"$', "project version")


def package_version() -> str:
    return _value("src/arti/_version.py", r'^__version__ = "([^\"]+)"$', "package version")


def check_version_consistency() -> list[str]:
    version = project_version()
    failures = []
    package = package_version()
    if package != version:
        failures.append(f"version mismatch: pyproject={version}, package={package}")
    citation = _value("CITATION.cff", r"^version:\s*([^\s]+)\s*$", "citation version")
    if citation != version:
        failures.append(f"version mismatch: pyproject={version}, CITATION.cff={citation}")
    if (
        re.search(rf"^##\s+{re.escape(version)}\s*$", read("CHANGELOG.md"), flags=re.MULTILINE)
        is None
    ):
        failures.append(f"CHANGELOG.md is missing an exact heading for {version}")
    pinned_versions = re.findall(r'arti-fit(?:\[[^\]]+\])?==([^"\s]+)', read("README.md"))
    if not pinned_versions or any(item != version for item in pinned_versions):
        failures.append(f"README install pins must all match {version}")
    return failures


def check_public_identity() -> list[str]:
    failures = []
    if re.search(r'^name = "arti-fit"$', read("pyproject.toml"), flags=re.MULTILINE) is None:
        failures.append("public distribution name must remain arti-fit")
    if "The PyPI distribution is `arti-fit`" not in read("README.md"):
        failures.append("README must identify the public arti-fit distribution")
    return failures


def check_workflows(ci_text: str | None = None, release_text: str | None = None) -> list[str]:
    ci = read(".github/workflows/ci.yml") if ci_text is None else ci_text
    release = read(".github/workflows/release.yml") if release_text is None else release_text
    failures = []
    for snippet in (
        "name: CI",
        "workflow_call:",
        "uv sync --locked --extra dev",
        "tests/test_component_registry.py",
        "tests/test_release_security.py",
        "tests/test_release_readiness.py",
        "scripts/check_package.py",
        "mkdocs build --strict",
    ):
        if snippet not in ci:
            failures.append(f"CI workflow missing release gate: {snippet}")
    for snippet in (
        "quality:\n    uses: ./.github/workflows/ci.yml",
        "build:\n    needs: quality",
        "scripts/check_release_readiness.py",
        "scripts/check_package.py",
        "publish:\n    needs: build",
        "pypa/gh-action-pypi-publish@release/v1",
    ):
        if snippet not in release:
            failures.append(f"release workflow missing required gate: {snippet}")
    return failures


def check_release_documentation() -> list[str]:
    release = read("docs/guides/release.md")
    required = (
        "arti-fit",
        "workflow_call",
        "check_release_readiness.py",
        "check_package.py",
        "PyPI Trusted Publishing",
    )
    return [
        f"release guide missing current public release detail: {item}"
        for item in required
        if item not in release
    ]


def check_readiness() -> dict[str, object]:
    failures = []
    failures.extend(check_version_consistency())
    failures.extend(check_public_identity())
    failures.extend(check_workflows())
    failures.extend(check_release_documentation())
    return {
        "ok": not failures,
        "kind": "public-release-readiness",
        "version": project_version(),
        "distribution": "arti-fit",
        "required_gates": ["ci", "version-consistency", "package-build", "release-workflow"],
        "failures": failures,
    }


def main() -> None:
    payload = check_readiness()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
