from __future__ import annotations

import hashlib
import json

import pytest

from arti.recall_formula import resolve_builtin_formula
from arti.recall_registry import InvalidRecallFormulaIdError
from arti.recall_manifest import (
    RecallFormulaManifest,
    RecallLayoutManifest,
    canonical_json,
    canonical_sha256,
)


def _manifest(*, origin: str = "builtin", portable: bool = True) -> RecallFormulaManifest:
    formula = resolve_builtin_formula("arti/state@1")
    factors = formula.contract.factor_names
    return RecallFormulaManifest(
        formula_ref=formula.contract.identity.reference,
        factor_names=factors,
        origin=origin,
        portable=portable,
        layout=RecallLayoutManifest(factor_order=factors),
        capabilities=("torch.compile", "torch.eager"),
    )


def test_manifest_is_canonical_json_safe_and_roundtrips() -> None:
    manifest = _manifest()
    payload = manifest.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert RecallFormulaManifest.from_dict(json.loads(json.dumps(payload))) == manifest
    assert manifest.canonical_json() == canonical_json(payload)
    assert manifest.sha256 == hashlib.sha256(
        manifest.canonical_json().encode("utf-8")
    ).hexdigest()
    assert manifest.sha256 == canonical_sha256(payload)
    assert payload["layout_fingerprint"] == payload["layout"]["fingerprint"]


def test_canonical_hash_does_not_depend_on_mapping_key_order() -> None:
    payload = _manifest().to_dict()
    reversed_payload = dict(reversed(tuple(payload.items())))

    assert canonical_json(payload) == canonical_json(reversed_payload)
    assert canonical_sha256(payload) == canonical_sha256(reversed_payload)


def test_layout_fingerprint_binds_factor_order() -> None:
    payload = _manifest().to_dict()
    payload["layout"]["factor_order"] = ["opacity", "gain", "content"]

    with pytest.raises(ValueError, match="layout fingerprint"):
        RecallFormulaManifest.from_dict(payload)


def test_formula_rejects_mismatched_factor_order_and_duplicates() -> None:
    layout = RecallLayoutManifest(factor_order=("content", "gain"))
    with pytest.raises(ValueError, match="exactly match"):
        RecallFormulaManifest(
            formula_ref=resolve_builtin_formula("arti/state@1").contract.identity.reference,
            factor_names=("gain", "content"),
            origin="builtin",
            portable=True,
            layout=layout,
        )
    with pytest.raises(ValueError, match="duplicate"):
        RecallLayoutManifest(factor_order=("content", "content"))


def test_custom_formula_cannot_claim_portability() -> None:
    with pytest.raises(ValueError, match="only builtin"):
        _manifest(origin="custom", portable=True)

    local = _manifest(origin="custom", portable=False)
    assert RecallFormulaManifest.from_dict(local.to_dict()) == local


def test_only_builtin_formulas_can_claim_portability() -> None:
    manifest = _manifest(origin="builtin", portable=True)
    assert manifest.portable

    with pytest.raises(ValueError, match="only builtin"):
        _manifest(origin="registered", portable=True)

    registered = _manifest(origin="registered", portable=False)
    assert not registered.portable


def test_portable_builtin_manifest_must_match_immutable_builtin_contract() -> None:
    factors = ("content",)
    with pytest.raises(ValueError, match="immutable builtin"):
        RecallFormulaManifest(
            formula_ref="arti/forged@sha256:" + "0" * 64,
            factor_names=factors,
            origin="builtin",
            portable=True,
            layout=RecallLayoutManifest(factor_order=factors),
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("api_version", True, "integer"),
        ("formula_ref", "package.module:Formula", "full SHA-256"),
        ("origin", "plugin", "origin"),
        ("portable", 1, "boolean"),
        ("factor_names", ("content",), "JSON array"),
        ("capabilities", ["torch.eager", object()], "only strings"),
    ],
)
def test_from_dict_strictly_validates_field_types_and_values(
    field: str, value: object, error: str
) -> None:
    payload = _manifest().to_dict()
    payload[field] = value

    with pytest.raises((TypeError, ValueError), match=error):
        RecallFormulaManifest.from_dict(payload)


def test_from_dict_rejects_missing_unknown_and_executable_reference_fields() -> None:
    missing = _manifest().to_dict()
    missing.pop("formula_ref")
    with pytest.raises(ValueError, match="missing required fields"):
        RecallFormulaManifest.from_dict(missing)

    for field in ("callable", "class_path", "module", "entry_point"):
        payload = _manifest().to_dict()
        payload[field] = "untrusted.package:Formula"
        with pytest.raises(ValueError, match="unknown fields"):
            RecallFormulaManifest.from_dict(payload)


def test_from_dict_rejects_precanonical_api_version() -> None:
    payload = _manifest().to_dict()
    payload["api_version"] = 1

    with pytest.raises(ValueError, match="unsupported"):
        RecallFormulaManifest.from_dict(payload)


def test_manifest_rejects_source_declarations() -> None:
    factors = ("content",)
    with pytest.raises(ValueError, match="full SHA-256"):
        RecallFormulaManifest(
            formula_ref="arti/delta@1",
            factor_names=factors,
            origin="builtin",
            portable=True,
            layout=RecallLayoutManifest(factor_order=factors),
        )


def test_constructor_rejects_callables_and_noncanonical_capabilities() -> None:
    factors = ("content",)
    layout = RecallLayoutManifest(factor_order=factors)
    with pytest.raises(InvalidRecallFormulaIdError, match="must be a string"):
        RecallFormulaManifest(
            formula_ref=lambda: None,  # type: ignore[arg-type]
            factor_names=factors,
            origin="custom",
            portable=False,
            layout=layout,
        )
    with pytest.raises(ValueError, match="sorted"):
        RecallFormulaManifest(
            formula_ref=resolve_builtin_formula("arti/state@1").contract.identity.reference,
            factor_names=factors,
            origin="builtin",
            portable=True,
            layout=layout,
            capabilities=("torch.eager", "torch.compile"),
        )


def test_layout_rejects_unknown_fields_and_fingerprint_tampering() -> None:
    layout = RecallLayoutManifest(factor_order=("content",))
    unknown = layout.to_dict()
    unknown["class_path"] = "package.Formula"
    with pytest.raises(ValueError, match="unknown fields"):
        RecallLayoutManifest.from_dict(unknown)

    tampered = layout.to_dict()
    tampered["fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        RecallLayoutManifest.from_dict(tampered)


def test_canonical_json_rejects_non_json_and_nonfinite_values() -> None:
    with pytest.raises(TypeError, match="finite JSON"):
        canonical_json({"callable": lambda: None})
    with pytest.raises(TypeError, match="finite JSON"):
        canonical_json({"value": float("nan")})
