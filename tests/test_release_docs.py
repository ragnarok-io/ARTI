from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_uses_current_recall_bank_api() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for retired_name in (
        "expert_contract",
        "freeze_expert_banks",
        "save_expert",
        "migrate_pt",
    ):
        assert retired_name not in readme
    assert ".bank_contract(" in readme
    assert ".freeze_banks()" in readme
    assert ".save_bank(" in readme


def test_experimental_recall_docs_use_experimental_namespace() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    webgpu = (ROOT / "docs" / "webgpu-alpha.md").read_text(encoding="utf-8")

    assert "`arti.experimental`" in readme
    assert "from arti.experimental import StatefulRecall" in webgpu
    assert "from arti.experimental.web import export_stateful_recall" in webgpu
    assert "from arti.nn import StatefulRecall" not in webgpu
    assert "from arti.web import export_stateful_recall" not in webgpu
