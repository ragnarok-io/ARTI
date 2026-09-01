from __future__ import annotations

from copy import deepcopy

import pytest
import torch

import arti
from arti.component_registry import ComponentCompatibilityError


def test_component_catalog_is_canonical_and_explicit_about_deprecation() -> None:
    catalog = arti.component_catalog()
    refs = [item["ref"] for item in catalog]
    assert refs == sorted(refs)
    assert len(refs) == len(set(refs))
    pulse = next(item for item in catalog if item["ref"] == "arti/pulse@1")
    assert "Pulse" in pulse["aliases"]
    assert "arti/learned-pulse@1" in pulse["deprecated_aliases"]
    assert arti.component_ref(arti.resolve_component("arti/learned-pulse@1", k=2, dim=4)) == "arti/pulse@1"
    assert next(item for item in catalog if item["ref"] == "arti/recall-state@1")["variant"] == "values-only"
    assert next(item for item in catalog if item["ref"] == "arti/survival@1")["kind"] == "survival"
    assert next(
        item for item in catalog if item["ref"] == "arti/recall-branch-batch@3"
    )["constructible"] is False
    assert all("constructible" in item for item in catalog)
    assert all("artifact_policy" in item for item in catalog)
    assert next(
        item for item in catalog if item["ref"] == "arti/recall-branch-batch@3"
    )["artifact_policy"] == "runtime_only"
    assert next(
        item for item in catalog if item["ref"] == "arti/recall@2"
    )["config_schema_version"] == 2
    assert next(
        item for item in catalog if item["ref"] == "arti/recall@3"
    )["config_schema_version"] == 4
    assert next(
        item for item in catalog if item["ref"] == "arti/batched-refine-result@1"
    )["config_schema_version"] == 4
    executor = next(
        item for item in catalog if item["ref"] == "arti/batched-refine@1"
    )
    assert executor["constructible"] is False
    assert executor["artifact_policy"] == "runtime_only"
    assert executor["config_schema_version"] == 2
    assert next(item for item in catalog if item["ref"] == "arti/layer@2")[
        "lifecycle"
    ] == "stable"
    assert next(item for item in catalog if item["ref"] == "arti/pulse@2")[
        "lifecycle"
    ] == "stable"
    assert next(item for item in catalog if item["ref"] == "arti/layer@1")[
        "lifecycle"
    ] == "legacy"


def test_component_schema_does_not_change_when_trainability_changes() -> None:
    model = arti.Pulse(k=2, dim=4).eval()
    before = arti.component_provenance(model)
    model.requires_grad_(False)
    assert arti.component_provenance(model) == before


def test_recall_state_has_a_canonical_component_identity() -> None:
    state = arti.mechanisms.RecallState.zeros(1, 3, 4, dtype=torch.float32)
    assert arti.component_ref(state) == "arti/recall-state@1"
    provenance = arti.component_spec(state).to_dict()
    assert provenance["variant"] == "values-only"
    assert provenance["state_schema_version"] == arti.mechanisms.RECALL_STATE_SCHEMA_VERSION


def test_state_contract_binds_model_graph_and_tensor_schema() -> None:
    model = arti.Pulse(k=2, dim=4).eval()
    contract = arti.component_state_contract(model, model.state_dict(), scope="all")
    assert contract["state_schema"]["fingerprint"]
    assert arti.validate_component_state_contract(contract, state_dict=model.state_dict(), model=model) == contract

    tampered = deepcopy(contract)
    tampered["state_schema"]["tensors"][0]["shape"] = [999]
    with pytest.raises(ComponentCompatibilityError, match="state schema fingerprint"):
        arti.validate_component_state_contract(tampered)


def test_state_contract_rejects_wrong_tensor_schema() -> None:
    model = arti.Pulse(k=2, dim=4).eval()
    contract = arti.component_state_contract(model, model.state_dict(), scope="all")
    wrong = dict(model.state_dict())
    name = next(iter(wrong))
    wrong[name] = torch.zeros(1)
    with pytest.raises(ComponentCompatibilityError, match="does not match supplied state_dict"):
        arti.validate_component_state_contract(contract, state_dict=wrong)
