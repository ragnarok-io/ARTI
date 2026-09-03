from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_uses_current_stable_surface() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for retired_name in (
        "expert_contract",
        "freeze_expert_banks",
        "save_expert",
        "migrate_pt",
    ):
        assert retired_name not in readme
    assert "`arti.ARTILayer` is now `arti/layer@2`" in readme
    assert "`AdaptivePulse`" in readme
    assert "`arti.mechanisms`" in readme
    assert "`arti.legacy`" in readme


def test_legacy_recall_docs_use_legacy_namespace() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    webgpu = (ROOT / "docs" / "webgpu-alpha.md").read_text(encoding="utf-8")

    assert "`arti.experimental`" in readme
    assert "from arti.legacy import StatefulRecall" in webgpu
    assert "from arti.experimental.web import export_stateful_recall" in webgpu
    assert "from arti.nn import StatefulRecall" not in webgpu
    assert "from arti.experimental import StatefulRecall" not in webgpu
    assert "from arti.web import export_stateful_recall" not in webgpu


def test_current_install_version_and_plasticity_correction() -> None:
    import arti
    from arti import mechanisms

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "formula-fabric.md").read_text(encoding="utf-8")
    stability = (ROOT / "STABILITY.md").read_text(encoding="utf-8")
    assert f'arti-fit=={arti.__version__}' in readme
    assert "FormulaProgramQueryV4" in readme
    assert "`arti.mechanisms.FormulaProgramQueryV2`" not in readme
    assert "predecessor" in guide and "branch-local" in guide
    assert "3.0.13a2 correction removes" in stability
    assert not hasattr(mechanisms, "FormulaProgramQueryV2")
    assert callable(mechanisms.FormulaProgramQueryV3)
    assert callable(mechanisms.FormulaProgramQueryV4)
