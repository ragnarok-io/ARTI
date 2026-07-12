from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import arti
import arti.jax as arti_jax


ROOT = Path(__file__).resolve().parents[1]


def project_version() -> str:
    for line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"')
    raise AssertionError("project version not found in pyproject.toml")


def test_public_version_matches_pyproject():
    assert arti.__version__ == project_version()


def test_release_identity_and_citation_are_consistent():
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    version = project_version()

    assert payload["project"]["name"] == "arti-fit"
    assert payload["project"]["authors"] == [{"name": "Thiocy"}]
    assert f"version: {version}" in citation
    assert "name: Thiocy" in citation
    assert "uv add arti-fit" in readme


def test_package_declares_pep561_type_marker():
    assert (ROOT / "src" / "arti" / "py.typed").exists()


def test_package_declares_safetensors_as_core_dependency():
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = payload["project"]["dependencies"]
    assert any(dependency.startswith("safetensors") for dependency in dependencies)


def test_package_declares_pretrained_ecosystem_extras():
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = payload["project"]["optional-dependencies"]
    assert {"qwen", "peft", "sd"}.issubset(extras)
    assert any(dependency.startswith("peft") for dependency in extras["peft"])


def test_package_declares_web_export_extra():
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = payload["project"]["optional-dependencies"]
    assert {"onnx>=1.16", "onnxscript>=0.3"}.issubset(extras["web"])


def test_backend_status_is_explicit():
    assert "torch" in arti.available_backends()
    assert arti_jax.backend_status() in {"available", "unavailable"}
    if arti_jax.backend_status() == "available":
        assert "jax" in arti.available_backends()
        assert "jax" not in arti.planned_backends()
    else:
        assert "jax" in arti.planned_backends()
