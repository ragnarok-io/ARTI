from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from torch import Tensor, nn

from arti.recall_registry import (
    DuplicateRecallFormulaError,
    FrozenRecallFormulaRegistryError,
    InvalidRecallFormulaIdError,
    RecallFormulaId,
    RecallFormulaRegistry,
    RecallFormulaRegistryError,
    UnknownRecallFormulaError,
    describe_formula,
    list_formulas,
)


class Formula(nn.Module):
    def __init__(self, label: str = "formula") -> None:
        super().__init__()
        self.label = label

    def forward(self, state: Tensor, factors: Tensor) -> Tensor:
        return state


def test_formula_id_requires_canonical_exact_version() -> None:
    identity = RecallFormulaId.parse("acme-lab/signed_gate@12")
    assert identity.namespace == "acme-lab"
    assert identity.name == "signed_gate"
    assert identity.version == 12
    assert identity.base_id == "acme-lab/signed_gate"
    assert identity.reference == "acme-lab/signed_gate@12"

    invalid = (
        "state",
        "state@1",
        "ARTI/state@1",
        "arti/State@1",
        "arti/state@0",
        "arti/state@01",
        "arti//state@1",
        "arti/state@1.0",
        "../state@1",
    )
    for reference in invalid:
        with pytest.raises(InvalidRecallFormulaIdError):
            RecallFormulaId.parse(reference)


def test_builtin_registration_uses_reserved_namespace_and_is_portable() -> None:
    registry = RecallFormulaRegistry()
    registration = registry.register_builtin(
        "arti/state@1",
        factory=lambda: Formula("builtin"),
        description="State formula",
    )

    assert registration.origin == "builtin"
    assert registration.portable is True
    assert registration.instantiate().label == "builtin"
    assert registry.describe("arti/state@1").to_dict() == {
        "reference": "arti/state@1",
        "namespace": "arti",
        "name": "state",
        "version": 1,
        "origin": "builtin",
        "provider_kind": "factory",
        "portable": True,
        "description": "State formula",
    }

    with pytest.raises(InvalidRecallFormulaIdError, match="must use"):
        registry.register_builtin("acme/state@1", factory=Formula)
    with pytest.raises(InvalidRecallFormulaIdError, match="reserved"):
        registry.register("arti/custom@1", factory=Formula)


def test_registered_factory_is_explicit_and_creates_fresh_instances() -> None:
    registry = RecallFormulaRegistry()
    registration = registry.register(
        "acme/signed-gate@2",
        factory=Formula,
    )

    first = registry.resolve("acme/signed-gate@2").instantiate()
    second = registration.instantiate()
    assert registration.origin == "registered"
    assert registration.provider_kind == "factory"
    assert registration.portable is False
    assert isinstance(first, Formula)
    assert isinstance(second, Formula)
    assert first is not second


def test_registered_instances_are_rejected_to_avoid_parameter_sharing() -> None:
    registry = RecallFormulaRegistry()
    formula = Formula("local")
    with pytest.raises(RecallFormulaRegistryError, match="silently share"):
        registry.register("lab/experiment@1", instance=formula)
    with pytest.raises(RecallFormulaRegistryError, match="silently share"):
        registry.register("lab/unsafe@1", instance=Formula(), portable=True)
    with pytest.raises(RecallFormulaRegistryError, match="portable=true"):
        registry.register("lab/unsafe@1", factory=Formula, portable=True)


def test_registration_rejects_ambiguous_provider_and_duplicates() -> None:
    registry = RecallFormulaRegistry()
    with pytest.raises(RecallFormulaRegistryError, match="requires factory"):
        registry.register("acme/formula@1")
    with pytest.raises(RecallFormulaRegistryError, match="silently share"):
        registry.register(
            "acme/formula@1",
            factory=Formula,
            instance=Formula(),
        )
    with pytest.raises(RecallFormulaRegistryError, match="callable"):
        registry.register("acme/formula@1", factory=object())  # type: ignore[arg-type]

    registry.register("acme/formula@1", factory=Formula)
    with pytest.raises(DuplicateRecallFormulaError, match="already registered"):
        registry.register("acme/formula@1", factory=Formula)


def test_resolve_requires_exact_registered_version_without_discovery() -> None:
    registry = RecallFormulaRegistry()
    registry.register("acme/formula@1", factory=Formula)

    assert registry.resolve("acme/formula@1").reference == "acme/formula@1"
    with pytest.raises(UnknownRecallFormulaError, match="not explicitly registered"):
        registry.resolve("acme/formula@2")
    with pytest.raises(InvalidRecallFormulaIdError):
        registry.resolve("acme/formula")


def test_list_is_sorted_and_descriptions_do_not_expose_provider_objects() -> None:
    registry = RecallFormulaRegistry()
    registry.register("zed/formula@2", factory=Formula)
    registry.register("acme/formula@3", factory=Formula)
    registry.register_builtin("arti/state@1", factory=Formula)

    descriptions = registry.list()
    assert tuple(item.reference for item in descriptions) == (
        "acme/formula@3",
        "arti/state@1",
        "zed/formula@2",
    )
    assert all("_provider" not in item.to_dict() for item in descriptions)


def test_frozen_snapshot_is_immutable_and_isolated_from_source_registry() -> None:
    registry = RecallFormulaRegistry()
    registry.register("acme/first@1", factory=Formula)
    snapshot = registry.freeze()

    assert snapshot.is_frozen
    assert snapshot.resolve("acme/first@1").reference == "acme/first@1"
    registry.register("acme/second@1", factory=Formula)

    with pytest.raises(UnknownRecallFormulaError):
        snapshot.resolve("acme/second@1")
    with pytest.raises(FrozenRecallFormulaRegistryError):
        snapshot.register("acme/third@1", factory=Formula)


def test_duplicate_registration_is_atomic_across_threads() -> None:
    registry = RecallFormulaRegistry()

    def register() -> str:
        try:
            registry.register("acme/concurrent@1", factory=Formula)
        except DuplicateRecallFormulaError:
            return "duplicate"
        return "registered"

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(lambda _: register(), range(16)))

    assert results.count("registered") == 1
    assert results.count("duplicate") == 15
    assert len(registry.list()) == 1


def test_factory_returning_none_fails_at_instantiation() -> None:
    registry = RecallFormulaRegistry()
    registration = registry.register(
        "acme/broken@1",
        factory=lambda: None,  # type: ignore[arg-type,return-value]
    )
    with pytest.raises(RecallFormulaRegistryError, match="torch.nn.Module"):
        registration.instantiate()


def test_factory_must_not_return_the_same_module_instance_twice() -> None:
    registry = RecallFormulaRegistry()
    shared = Formula()
    registration = registry.register(
        "acme/shared@1",
        factory=lambda: shared,
    )

    assert registration.instantiate() is shared
    with pytest.raises(RecallFormulaRegistryError, match="shared module instance"):
        registration.instantiate()


def test_public_discovery_includes_builtin_formula_ids() -> None:
    references = {description.reference for description in list_formulas()}

    assert {"delta-v1", "affine-v1", "state-v1"} <= references
    assert describe_formula("single").reference == "delta-v1"
