from __future__ import annotations

from copy import deepcopy

import pytest
import torch

import arti
from arti.component_registry import (
    ComponentCompatibilityError,
    component_graph_fingerprint,
    component_provenance,
    resolve_component,
    validate_component_provenance,
)
from arti.pulse import PulseCompressor


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


def test_exact_version_resolution_cannot_drift_between_half_or_recall_versions() -> None:
    scalar = resolve_component("arti/half@1", stochastic=False)
    contextual = resolve_component("arti/half@2", stochastic=False)
    legacy_recall = resolve_component("arti/recall@1", dim=4, slots=2)
    current_recall = resolve_component("arti/recall@2", dim=4, slots=2)

    assert arti.component_ref(scalar) == "arti/half@1"
    assert arti.component_ref(contextual) == "arti/half@2"
    assert arti.component_ref(legacy_recall) == "arti/recall@1"
    assert arti.component_ref(current_recall) == "arti/recall@2"
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
    assert "arti/recall@2" in refiner_root["dependencies"]


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
