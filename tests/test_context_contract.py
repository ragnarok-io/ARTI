import json

import arti
from arti.experimental.web import artifact_schema, render_typescript_contract


def test_arti_config_exposes_context_contract() -> None:
    config = arti.ARTIConfig(
        input_dim=4,
        hidden_dim=6,
        coord_dim=2,
        coord_frame_mode="paired_rotation",
        require_coord=True,
    )

    contract = config.context_contract()

    assert contract["valid_mask"] == {
        "shape": ["B", "N"],
        "dtype": "bool",
        "required": False,
    }
    assert contract["visibility"]["shape"] == ["B", "N", "N"]
    assert contract["frame"]["mode"] == "paired_rotation"
    assert contract["frame"]["coord"]["required"] is True


def test_arti_st_manifest_records_context_contract(tmp_path) -> None:
    layer = arti.legacy.ARTILayer(
        input_dim=4,
        hidden_dim=4,
        coord_dim=2,
        coord_frame_mode="paired_rotation",
        use_recall=False,
        use_virtual_recall=False,
        use_virtual_interface=False,
        use_pairwise_context=False,
        use_phase_mixer=False,
        operator_count=1,
        interface_slots=1,
        recall_slots=1,
    ).eval()

    saved = arti.save(layer, tmp_path / "context.st")
    manifest = json.loads(saved.manifest_path.read_text(encoding="utf-8"))

    assert manifest["architecture"]["context_contract"] == layer.config.context_contract()
    assert (
        manifest["architecture"]["component_provenance"]["schema_version"]
        == arti.COMPONENT_PROVENANCE_VERSION
    )


def test_web_contract_accepts_context_metadata_without_assigning_semantics() -> None:
    module_schema = artifact_schema()["properties"]["manifest"]["properties"]["module"]
    assert "context_contract" in module_schema["properties"]
    assert "component_provenance" in module_schema["properties"]
    generated = render_typescript_contract()
    assert "context_contract?: Record<string, unknown>" in generated
    assert "component_provenance?: Record<string, unknown>" in generated
    assert "module context contract" in generated
