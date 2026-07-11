from __future__ import annotations

import json

import pytest
import torch

import arti
import arti.nn as ann


def test_minimal_layer_has_no_optional_mechanism_parameters() -> None:
    layer = ann.Layer(8)
    output = layer(torch.randn(2, 4, 8))
    names = tuple(name for name, _ in layer.named_parameters())

    assert output.y.shape == (2, 4, 8)
    assert output.virtual_y is None
    assert not any(name.startswith("state.phase") for name in names)
    assert not any(name.startswith("state.interface") for name in names)
    assert not any(name.startswith("state.recall") for name in names)
    assert layer.config.explain()["synthetic_context"] is False


def test_recall_profile_enables_only_recall_specific_mechanisms() -> None:
    layer = ann.Layer(8, profile="recall")
    explanation = layer.config.explain()

    assert explanation["mechanisms"]["recall"] is True
    assert explanation["mechanisms"]["virtual_recall"] is True
    assert explanation["mechanisms"]["half"] is True
    assert explanation["mechanisms"]["phase"] is False
    assert explanation["mechanisms"]["virtual_interface"] is False


def test_multisource_profile_requires_visibility_and_accepts_operator_frames() -> None:
    layer = ann.Layer(8, profile="multisource", coord_dim=2)
    x = torch.randn(1, 3, 8)
    coord = torch.nn.functional.one_hot(torch.tensor([[0, 1, 0]]), num_classes=2).float()
    operators = torch.eye(8).repeat(2, 1, 1)

    with pytest.raises(ValueError, match="visibility is required"):
        layer(x, coord=coord, frame_operators=operators)

    with pytest.raises(ValueError, match="coord is required"):
        layer(x, visibility=torch.ones(1, 3, 3, dtype=torch.bool), frame_operators=operators)

    visibility = torch.ones(1, 3, 3, dtype=torch.bool)
    output = layer(x, coord=coord, visibility=visibility, frame_operators=operators)
    assert output.y.shape == x.shape


def test_features_and_profiles_are_json_roundtrip_safe_and_transparent() -> None:
    selected = arti.features(recall=True, recall_slots=7, virtual_recall=False)
    restored = arti.FeatureConfig.from_dict(json.loads(json.dumps(selected.to_dict())))
    config = restored.compile(input_dim=16)
    roundtrip = arti.ARTIConfig.from_dict(json.loads(json.dumps(config.to_dict())))

    assert restored == selected
    assert roundtrip == config
    assert config.explain()["capacities"]["recall_slots"] == 7
    assert arti.layer_profiles() == ("minimal", "recall", "multisource")


def test_config_diff_reports_only_changed_execution_fields() -> None:
    minimal = arti.profile("minimal").compile(input_dim=8)
    recall = arti.profile("recall").compile(input_dim=8)
    difference = minimal.diff(recall)

    assert set(difference) == {"recall_steps", "recall_slots", "use_recall", "use_virtual_recall"}
    assert difference["use_recall"] == {"self": False, "other": True}


def test_layer_rejects_ambiguous_or_semantically_empty_feature_configuration() -> None:
    with pytest.raises(ValueError, match="either features or profile"):
        ann.Layer(8, features=arti.features(), profile="minimal")
    with pytest.raises(ValueError, match="requires pairwise_context or virtual_interface"):
        arti.features(visibility=True)
    with pytest.raises(ValueError, match="requires phase=True"):
        arti.features(coord_dim=2, coord_frame_mode="operator_bank")


def test_legacy_arti_layer_defaults_remain_available() -> None:
    legacy = arti.ARTILayer(input_dim=8, hidden_dim=8)
    output = legacy(torch.randn(2, 3, 8))

    assert output.y.shape == (2, 3, 8)
    assert legacy.config.use_phase_mixer is True
    assert legacy.config.use_virtual_interface is True
    assert legacy.config.use_recall is True


def test_progressive_layer_arti_st_roundtrip(tmp_path) -> None:
    selected = arti.profile("recall", recall_slots=6)
    source = ann.Layer(8, features=selected).eval()
    x = torch.randn(2, 4, 8)
    expected = source(x).y
    path = tmp_path / "progressive.arti.st"

    arti.save(source, path, config={"features": selected.to_dict(), "dim": 8})
    restored = ann.Layer(8, features=selected).eval()
    result = arti.load(path, model=restored)

    assert result.manifest["architecture"]["config"]["features"] == selected.to_dict()
    assert torch.allclose(restored(x).y, expected)
