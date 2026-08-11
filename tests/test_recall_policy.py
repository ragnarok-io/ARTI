from __future__ import annotations

import json

import pytest

from arti.recall_policy import (
    FormulaOptimizationHints,
    FormulaOptimizationRequirements,
    RecallParameterTag,
    ResolvedFormulaOptimizationPolicy,
    resolve_formula_optimization_policy,
    validate_parameter_tags,
)


def test_policy_resolution_uses_priority_and_tracks_leaf_sources() -> None:
    hints = FormulaOptimizationHints(
        {
            "compute_dtype": "bfloat16",
            "lr_scale_by_role": {"router": 0.2, "bank": 1.0},
        }
    )
    resolved = resolve_formula_optimization_policy(
        framework={
            "compute_dtype": "float32",
            "lr_scale_by_role": {"router": 1.0, "reader": 0.5},
            "weight_decay": 0.0,
        },
        formula_hints=hints,
        accept_formula_hints=True,
        project={"lr_scale_by_role": {"bank": 0.8}, "weight_decay": 0.01},
        user={"compute_dtype": "float16", "lr_scale_by_role": {"router": 0.1}},
    )

    assert resolved.values["compute_dtype"] == "float16"
    assert resolved.values["weight_decay"] == 0.01
    assert dict(resolved.values["lr_scale_by_role"]) == {
        "router": 0.1,
        "reader": 0.5,
        "bank": 0.8,
    }
    assert resolved.sources == {
        "compute_dtype": "user",
        "lr_scale_by_role.router": "user",
        "lr_scale_by_role.reader": "framework",
        "lr_scale_by_role.bank": "project",
        "weight_decay": "project",
    }
    assert resolved.formula_hints_accepted is True


def test_formula_hints_require_explicit_acceptance() -> None:
    hints = FormulaOptimizationHints({"compute_dtype": "bfloat16"})

    rejected = resolve_formula_optimization_policy(
        framework={"compute_dtype": "float32"},
        formula_hints=hints,
    )
    accepted = resolve_formula_optimization_policy(
        framework={"compute_dtype": "float32"},
        formula_hints=hints,
        accept_formula_hints=True,
    )

    assert rejected.values["compute_dtype"] == "float32"
    assert rejected.sources["compute_dtype"] == "framework"
    assert rejected.formula_hints_accepted is False
    assert accepted.values["compute_dtype"] == "bfloat16"
    assert accepted.sources["compute_dtype"] == "formula"


def test_policy_resolution_replaces_incompatible_nested_shapes_cleanly() -> None:
    scalar = resolve_formula_optimization_policy(
        framework={"schedule": {"warmup": 10}},
        user={"schedule": "constant"},
    )
    nested = resolve_formula_optimization_policy(
        framework={"schedule": "constant"},
        user={"schedule": {"warmup": 20}},
    )

    assert scalar.to_dict() == {
        "values": {"schedule": "constant"},
        "sources": {"schedule": "user"},
        "formula_hints_accepted": False,
    }
    assert nested.to_dict() == {
        "values": {"schedule": {"warmup": 20}},
        "sources": {"schedule.warmup": "user"},
        "formula_hints_accepted": False,
    }


def test_requirements_validate_without_supplying_policy_values() -> None:
    requirements = FormulaOptimizationRequirements(
        {
            "fp32_master_weights": True,
            "compute_dtype": "bfloat16",
        }
    )

    with pytest.raises(ValueError, match="missing required field 'fp32_master_weights'"):
        resolve_formula_optimization_policy(
            framework={"compute_dtype": "bfloat16"},
            requirements=requirements,
        )
    with pytest.raises(ValueError, match="must equal 'bfloat16'"):
        resolve_formula_optimization_policy(
            framework={"fp32_master_weights": True, "compute_dtype": "float16"},
            requirements=requirements,
        )

    resolved = resolve_formula_optimization_policy(
        project={"fp32_master_weights": True},
        user={"compute_dtype": "bfloat16"},
        requirements=requirements,
    )
    assert resolved.values == {
        "fp32_master_weights": True,
        "compute_dtype": "bfloat16",
    }


@pytest.mark.parametrize(
    "field",
    [
        "optimizer",
        "optimizer_factory",
        "register_hook",
        "gradient_hook",
        "requires_grad",
        "grad_mutation",
        "custom_backward",
    ],
)
def test_formula_metadata_rejects_training_behavior_injection(field: str) -> None:
    with pytest.raises(ValueError, match="cannot inject"):
        FormulaOptimizationHints({field: True})
    with pytest.raises(ValueError, match="cannot inject"):
        FormulaOptimizationRequirements({field: True})


def test_formula_metadata_rejects_unknown_and_non_json_values() -> None:
    with pytest.raises(ValueError, match="unsupported formula optimization field"):
        FormulaOptimizationHints({"mystery_schedule": "fast"})
    with pytest.raises(TypeError, match="JSON-safe"):
        FormulaOptimizationHints({"clip_scope": object()})
    with pytest.raises(ValueError, match="NaN"):
        FormulaOptimizationHints({"clip_grad_norm": float("nan")})
    with pytest.raises(TypeError, match="string-keyed mapping"):
        FormulaOptimizationRequirements.from_dict({"required_values": ["float32"]})


def test_formula_metadata_round_trips_as_json_and_is_immutable() -> None:
    hints = FormulaOptimizationHints(
        {
            "lr_scale_by_role": {"router": 0.1},
            "compute_dtype": "bfloat16",
        }
    )
    requirements = FormulaOptimizationRequirements({"fp32_master_weights": True})

    encoded_hints = json.loads(json.dumps(hints.to_dict()))
    encoded_requirements = json.loads(json.dumps(requirements.to_dict()))
    assert FormulaOptimizationHints.from_dict(encoded_hints) == hints
    assert FormulaOptimizationRequirements.from_dict(encoded_requirements) == requirements
    with pytest.raises(TypeError):
        hints.values["compute_dtype"] = "float32"


def test_parameter_tags_validate_identity_layout_and_exact_coverage() -> None:
    tags = (
        RecallParameterTag(
            "recall.bank",
            role="value_bank",
            storage_group="content",
            factor="content",
            gradient_layout="row_sparse",
        ),
        RecallParameterTag(
            "recall.router.weight",
            role="router",
            storage_group="router",
        ),
    )

    assert (
        validate_parameter_tags(
            tags,
            parameter_names=("recall.bank", "recall.router.weight"),
        )
        == tags
    )
    assert RecallParameterTag.from_dict(tags[0].to_dict()) == tags[0]

    with pytest.raises(ValueError, match="exactly one"):
        validate_parameter_tags((tags[0], tags[0]))
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_parameter_tags(tags[:1], parameter_names=("recall.bank", "missing"))
    with pytest.raises(ValueError, match="gradient_layout"):
        RecallParameterTag("x", role="bank", storage_group="bank", gradient_layout="block_sparse")


def test_resolved_policy_rejects_incomplete_or_invalid_provenance() -> None:
    with pytest.raises(ValueError, match="every resolved policy leaf"):
        ResolvedFormulaOptimizationPolicy(
            values={"compute_dtype": "float32"},
            sources={},
            formula_hints_accepted=False,
        )
    with pytest.raises(ValueError, match="policy sources"):
        ResolvedFormulaOptimizationPolicy(
            values={"compute_dtype": "float32"},
            sources={"compute_dtype": "artifact"},
            formula_hints_accepted=False,
        )


def test_policy_result_is_json_safe_and_detached_from_inputs() -> None:
    user = {"schedule": {"milestones": [10, 20]}}
    resolved = resolve_formula_optimization_policy(user=user)
    user["schedule"]["milestones"].append(30)

    payload = resolved.to_dict()
    assert payload["values"] == {"schedule": {"milestones": [10, 20]}}
    assert payload["sources"] == {"schedule.milestones": "user"}
    json.dumps(payload)
