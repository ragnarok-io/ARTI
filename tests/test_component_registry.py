from __future__ import annotations

from copy import deepcopy
import importlib
import json

import pytest
import torch
from safetensors import safe_open

import arti
from arti.component_registry import (
    ComponentCompatibilityError,
    component_graph_fingerprint,
    component_provenance,
    resolve_component,
    register_component,
    validate_component_provenance,
)
from arti.pulse import PulseCompressor


class _RuntimeOnlyArtifactFixture(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))


register_component(
    "tests/runtime-only-artifact@1",
    component_type=_RuntimeOnlyArtifactFixture,
    lifecycle="alpha",
    variant="runtime-only-test-fixture",
    constructible=False,
    artifact_policy="runtime_only",
)


def test_arti_st_explicitly_rejects_runtime_only_component(tmp_path) -> None:
    with pytest.raises(ValueError, match="cannot persist non-portable"):
        arti.save(_RuntimeOnlyArtifactFixture(), tmp_path / "runtime-only.st")


def test_concat_rejects_global_recall_before_mutating_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Host(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.recall = arti.ARTILatentRecallField(
                hidden_dim=4,
                slots=4,
                routing="grouped",
                group_size=2,
                group_topk=1,
                routing_normalizer="global",
            )

    model = Host()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    payloads = {
        "first.st": {
            "manifest": {"adapter_state_sha256": "1" * 64},
            "adapter_state_dict": deepcopy(before),
        },
        "second.st": {
            "manifest": {"adapter_state_sha256": "2" * 64},
            "adapter_state_dict": deepcopy(before),
        },
    }
    fit_project = importlib.import_module("arti.fit.project")
    monkeypatch.setattr(
        fit_project,
        "validate_artifact",
        lambda path, **_kwargs: payloads[path.name],
    )

    with pytest.raises(ValueError, match="routing_normalizer='per_bank'"):
        arti.concatenate_adapter_banks(
            model,
            ("first.st", "second.st"),
        )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_artifact_scope_rejects_runtime_only_provenance() -> None:
    provenance = component_provenance(_RuntimeOnlyArtifactFixture())
    assert validate_component_provenance(provenance) == provenance
    with pytest.raises(ComponentCompatibilityError, match="artifact_policy"):
        validate_component_provenance(provenance, artifact_scope=True)


def test_public_components_have_canonical_refs_and_aliases() -> None:
    half = arti.Half()
    pulse = resolve_component("Pulse", k=2, dim=4)
    unfold = resolve_component("arti/unfold@1", dim=4, exposed=2)

    assert arti.component_ref(half) == "arti/half@1"
    assert arti.component_ref(pulse) == "arti/pulse@1"
    assert arti.component_ref(resolve_component("arti/learned-pulse@1", k=2, dim=4)) == "arti/pulse@1"
    assert arti.component_ref(unfold) == "arti/unfold@1"
    assert all("@" not in key for key in pulse.state_dict())


def test_half_provenance_records_sampling_and_learning_options() -> None:
    half = arti.Half(stochastic=False, learnable=True)
    root = next(item for item in arti.component_provenance(half)["components"] if item["path"] == "$")
    assert root["config_schema_version"] == 2
    assert root["config"] == {
        "threshold": 1.0,
        "base": 0.5,
        "scale": 1.0,
        "stochastic": False,
        "learnable": True,
        "survival": {
            "ref": "arti/survival@1",
            "origin": "builtin",
            "portable": True,
            "runtime_only": False,
            "config": {
                "threshold": 1.0,
                "base": 0.5,
                "scale": 1.0,
                "learnable": True,
            },
        },
    }


def test_contextual_half_has_a_distinct_mechanism_reference() -> None:
    half = arti.Half(stochastic=False, context_mode="contextual", context_axes=(-1,))
    root = next(item for item in component_provenance(half)["components"] if item["path"] == "$")

    assert arti.component_ref(half) == "arti/half@2"
    assert root["variant"] == "contextual"
    assert root["config_schema_version"] == 3
    assert root["config"]["context_mode"] == "contextual"
    resolved = resolve_component("arti/half@2", stochastic=False)
    assert isinstance(resolved, arti.Half)
    assert resolved.context_mode == "contextual"


def test_half_version_resolution_rejects_cross_version_modes() -> None:
    scalar = resolve_component("arti/half@1", stochastic=False)
    contextual = resolve_component("arti/half@2", stochastic=False)

    assert arti.component_ref(scalar) == "arti/half@1"
    assert arti.component_ref(contextual) == "arti/half@2"
    with pytest.raises(ValueError, match="half@1 requires context_mode='none'"):
        resolve_component("arti/half@1", context_mode="contextual")
    with pytest.raises(ValueError, match="half@2 requires context_mode='contextual'"):
        resolve_component("arti/half@2", context_mode="none")


def test_pulse_provenance_records_recursive_dependencies() -> None:
    provenance = component_provenance(arti.Pulse(k=2, dim=4))
    refs = {item["ref"] for item in provenance["components"]}
    root = next(item for item in provenance["components"] if item["path"] == "$")

    assert "arti/pulse@1" in refs
    assert "arti/fold@1" in refs
    assert "arti/half@1" in refs
    assert set(root["dependencies"]) >= {"arti/fold@1", "arti/half@1"}
    assert validate_component_provenance(provenance) == provenance


def test_version_one_provenance_is_strictly_migrated_on_read() -> None:
    current = component_provenance(arti.Half(stochastic=False))
    legacy_components = []
    for item in current["components"]:
        legacy = dict(item)
        legacy.pop("capabilities")
        legacy_components.append(legacy)
    legacy = {
        "schema_version": 1,
        "components": legacy_components,
        "fingerprint": component_graph_fingerprint(legacy_components),
    }

    assert validate_component_provenance(legacy) == current


def test_reversible_runtime_record_identities_remain_registered() -> None:
    topology = arti.alpha.ReversibleTopology(
        2,
        policy=arti.alpha.FixedTopologyPolicy(),
    )
    folded = topology.fold(torch.randn(1, 4, 3))

    assert arti.component_ref(folded.record) == "arti/fold-record@1"
    assert arti.component_ref(folded) == "arti/fold-state@1"


def test_vhyper_runtime_types_have_nonportable_canonical_identities() -> None:
    catalog = {entry["ref"]: entry for entry in arti.component_catalog()}
    expected = {
        "arti/fixed-resident-bucket@1": "runtime_only",
        "arti/fixed-page-refs@1": "runtime_only",
        "arti/hot-page-pool@1": "host_bound",
        "arti/bound-hot-page-pool@1": "host_bound",
        "arti/captured-hot-step@1": "host_bound",
        "arti/resident-latency-receipt@1": "runtime_only",
        "arti/cuda-activity-receipt@1": "runtime_only",
        "arti/runtime-checkpoint-receipt@1": "runtime_only",
        "arti/restored-runtime-checkpoint@1": "host_bound",
    }

    for reference, artifact_policy in expected.items():
        assert catalog[reference]["constructible"] is False
        assert catalog[reference]["artifact_policy"] == artifact_policy


def test_disabled_optional_paths_are_not_recorded_as_dependencies() -> None:
    pulse = arti.Pulse(k=2, dim=4, use_half=False)
    pulse_root = next(item for item in component_provenance(pulse)["components"] if item["path"] == "$")
    recall = arti.Recall(dim=4, slots=2, activation="none")
    recall_root = next(item for item in component_provenance(recall)["components"] if item["path"] == "$")

    assert isinstance(pulse.half_act, torch.nn.Identity)
    assert "arti/half@1" not in pulse_root["dependencies"]
    assert "arti/half@1" not in recall_root["dependencies"]


def test_formula_and_refiner_dependencies_are_versioned() -> None:
    recall = arti.Recall(dim=4, slots=2)
    refiner = arti.RecallRefiner(recall)
    root = next(item for item in component_provenance(recall)["components"] if item["path"] == "$")
    refiner_root = next(
        item for item in component_provenance(refiner)["components"] if item["path"] == "$"
    )

    assert "arti/delta@1" in root["dependencies"]
    assert "arti/recall@4" in refiner_root["dependencies"]


def test_per_bank_recall_has_versioned_partition_provenance() -> None:
    recall = arti.Recall(
        dim=4,
        slots=8,
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=4,
        routing_normalizer="per_bank",
    )
    recall.state.recall.configure_expert_routes(
        ("game", "animal"),
        ((0, 2), (2, 4)),
        member_fingerprints=("1" * 64, "2" * 64),
    )
    recall.state.recall.set_expert_weights((1.0, 0.25))
    recall.state.recall.set_expert_influences((1.0, -0.5))

    spec = arti.component_spec(recall)

    assert arti.component_ref(recall) == "arti/recall@4"
    assert spec.config_schema_version == 1
    assert spec.config["routing_normalizer"] == "per_bank"
    assert spec.config["expert_names"] == ["game", "animal"]
    assert spec.config["expert_route_ranges"] == [[0, 2], [2, 4]]
    assert spec.config["expert_member_fingerprints"] == ["1" * 64, "2" * 64]
    assert spec.config["expert_weights"] == [1.0, 0.25]
    assert spec.config["expert_influences"] == [1.0, -0.5]
    reconstructed = arti.resolve_component(spec.reference, **spec.config)
    assert arti.component_spec(reconstructed).config == spec.config


def test_global_recall_config_schema_tracks_expert_asset_identity() -> None:
    recall = arti.Recall(
        dim=4,
        slots=8,
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=4,
    )
    recall.state.recall.configure_expert_routes(
        ("one",),
        ((0, 4),),
        member_fingerprints=("1" * 64,),
    )

    spec = arti.component_spec(recall)

    assert spec.reference == "arti/recall@4"
    assert spec.config_schema_version == 1
    assert spec.config["expert_member_fingerprints"] == ["1" * 64]


def test_per_bank_recall_rejects_orphan_member_fingerprints() -> None:
    with pytest.raises(ValueError, match="expert assembly config is incomplete"):
        arti.resolve_component(
            "arti/recall@3",
            dim=4,
            slots=8,
            activation="none",
            routing="grouped",
            group_size=2,
            group_topk=4,
            routing_normalizer="per_bank",
            expert_member_fingerprints=["1" * 64],
        )


def test_per_bank_recall_artifact_round_trip_binds_partition_assembly(tmp_path) -> None:
    def build(
        *,
        weights: tuple[float, float],
        influences: tuple[float, float] = (1.0, -0.25),
        fingerprints: tuple[str, str] = ("1" * 64, "2" * 64),
    ) -> arti.Recall:
        module = arti.Recall(
            dim=4,
            slots=8,
            activation="none",
            routing="grouped",
            group_size=2,
            group_topk=4,
            routing_normalizer="per_bank",
        )
        module.state.recall.configure_expert_routes(
            ("game", "animal"),
            ((0, 2), (2, 4)),
            member_fingerprints=fingerprints,
        )
        module.state.recall.set_expert_weights(weights)
        module.state.recall.set_expert_influences(influences)
        return module

    model = build(weights=(1.0, 0.5)).eval()
    value = torch.randn(2, 3, 4)
    expected = model(value)
    saved = arti.save(model, tmp_path / "recall-per-bank.st")

    restored = build(weights=(1.0, 0.5)).eval()
    arti.load(saved.weights_path, model=restored)
    torch.testing.assert_close(restored(value), expected)
    expected_candidates = arti.alpha.query_recall_branches(model, value, max_k=4)
    restored_candidates = arti.alpha.query_recall_branches(restored, value, max_k=4)
    assert restored_candidates.partition_names == expected_candidates.partition_names
    assert (
        restored_candidates.partition_member_fingerprints
        == expected_candidates.partition_member_fingerprints
    )
    assert (
        restored_candidates.partition_layout_fingerprint
        == expected_candidates.partition_layout_fingerprint
    )
    torch.testing.assert_close(
        restored_candidates.route_mass,
        expected_candidates.route_mass,
    )
    assert torch.equal(
        restored_candidates.candidate_partition_index,
        expected_candidates.candidate_partition_index,
    )

    incompatible = build(weights=(1.0, 0.25)).eval()
    with pytest.raises(ValueError, match="provenance|architecture"):
        arti.load(saved.weights_path, model=incompatible)

    wrong_asset = build(
        weights=(1.0, 0.5),
        fingerprints=("1" * 64, "3" * 64),
    ).eval()
    with pytest.raises(ValueError, match="provenance|architecture"):
        arti.load(saved.weights_path, model=wrong_asset)

    global_recall = arti.Recall(
        dim=4,
        slots=8,
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=4,
    ).eval()
    with pytest.raises(ValueError, match="provenance|architecture"):
        arti.load(saved.weights_path, model=global_recall)

    global_saved = arti.save(
        global_recall,
        tmp_path / "recall-global.st",
    )
    with pytest.raises(ValueError, match="provenance|architecture"):
        arti.load(global_saved.weights_path, model=build(weights=(1.0, 0.5)))

    zero_weight = build(
        weights=(1.0, 0.0),
        influences=(1.0, 1.0),
    ).eval()
    zero_expected = zero_weight(value)
    zero_saved = arti.save(zero_weight, tmp_path / "recall-zero-weight.st")
    zero_restored = build(
        weights=(1.0, 0.0),
        influences=(1.0, 1.0),
    ).eval()
    arti.load(zero_saved.weights_path, model=zero_restored)
    torch.testing.assert_close(zero_restored(value), zero_expected)
    zero_candidates = arti.alpha.query_recall_branches(
        zero_restored,
        value,
        max_k=4,
    )
    assert zero_candidates.active_partition_mask().tolist() == [
        [True, False],
        [True, False],
    ]
    assert zero_candidates.active_branch_count_by_partition().tolist() == [
        [2, 0],
        [2, 0],
    ]

    if torch.cuda.is_available():
        cuda_model = build(weights=(1.0, 0.5)).eval()
        arti.load(saved.weights_path, model=cuda_model, map_location="cuda")
        cuda_value = value.to("cuda")
        torch.testing.assert_close(cuda_model(cuda_value).cpu(), expected)


def test_config_fingerprint_is_order_independent() -> None:
    provenance = component_provenance(arti.Half())
    components = deepcopy(provenance["components"])
    config = components[0]["config"]
    components[0]["config"] = dict(reversed(tuple(config.items())))

    assert component_graph_fingerprint(components) == component_graph_fingerprint(
        provenance["components"]
    )


def test_unknown_version_and_legacy_require_explicit_admission() -> None:
    unknown = deepcopy(component_provenance(arti.Half()))
    unknown["components"][0]["ref"] = "arti/half@3"
    unknown["components"][0]["mechanism_version"] = 3
    unknown["fingerprint"] = component_graph_fingerprint(unknown["components"])
    with pytest.raises(ComponentCompatibilityError, match="unknown component"):
        validate_component_provenance(unknown)

    legacy = component_provenance(PulseCompressor())
    with pytest.raises(ComponentCompatibilityError, match="legacy"):
        validate_component_provenance(legacy)
    assert validate_component_provenance(legacy, allow_legacy=True) == legacy


def test_legacy_artifact_admission_is_explicit(tmp_path) -> None:
    class LegacyContainer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pulse = PulseCompressor()
            self.scale = torch.nn.Parameter(torch.ones(()))

        def forward(self, x: torch.Tensor, pulse_ids: torch.Tensor):
            result = self.pulse(x, pulse_ids)
            return result.pulse * self.scale

    model = LegacyContainer().eval()
    x = torch.randn(1, 2, 3)
    pulse_ids = torch.tensor([[0, 1]])
    saved = arti.save(model, tmp_path / "legacy.st")

    with pytest.raises(ValueError, match="legacy"):
        arti.load(saved.weights_path, model=LegacyContainer().eval())

    loaded = arti.load(
        saved.weights_path,
        model=LegacyContainer().eval(),
        allow_legacy=True,
    )
    assert loaded.manifest["architecture"]["component_provenance"]["components"][0]["lifecycle"] == "legacy"
    assert torch.equal(model(x, pulse_ids), loaded.model(x, pulse_ids) if loaded.model is not None else model(x, pulse_ids))


def test_arti_st_component_graph_round_trip_and_strict_shape_rejection(tmp_path) -> None:
    torch.manual_seed(41)
    model = arti.Pulse(k=2, dim=4).eval()
    x = torch.randn(2, 3, 4)
    torch.manual_seed(42)
    expected = model(x).detach()
    saved = arti.save(model, tmp_path / "pulse.st")
    manifest = json.loads(saved.manifest_path.read_text(encoding="utf-8"))
    provenance = manifest["architecture"]["component_provenance"]

    with safe_open(saved.weights_path, framework="pt", device="cpu") as handle:
        assert handle.metadata()["component_graph_sha256"] == provenance["fingerprint"]

    restored = arti.Pulse(k=2, dim=4).eval()
    loaded = arti.load(saved.weights_path, model=restored)
    assert loaded.manifest["architecture"]["component_provenance"] == provenance
    torch.manual_seed(42)
    assert torch.equal(restored(x), expected)

    wrong_shape = arti.Pulse(k=3, dim=4).eval()
    with pytest.raises(ValueError, match="component provenance"):
        arti.load(saved.weights_path, model=wrong_shape)
