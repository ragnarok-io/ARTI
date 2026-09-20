from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from benchmarks._formula_topology_v4_arm_contract import (
    ArmContractError,
    build_c0_binding,
    build_lineage,
    build_shared_workload_projection,
    build_workload_identity_document,
    canonical_digest,
    canonical_json,
    load_contract_documents,
    read_json_pointer,
    require_exact_keys,
    validate_c0_binding,
    validate_lineage,
    validate_schema_instance,
    validate_workload_identity_document,
)


ROOT = Path(__file__).resolve().parents[1]
C0_ROOT = ROOT.parent / "ARTI-formula-topology-v4-c0-005-cbeadc3"
HEX = "a" * 64


def workload_identity(arm: str = "sham") -> dict[str, object]:
    return {
        "arm": arm,
        "batch_size": 128,
        "seed": 19031,
        "depth": 2,
        "formula_scale": 1.0,
        "dtype": "torch.float32",
        "device": "cuda:0",
        "episode_sha256": "0" * 64,
        "target_sha256": "1" * 64,
        "task_transform_sha256": "2" * 64,
        "formula_contract_sha256": "3" * 64,
        "topology_contract_sha256": "4" * 64,
        "initial_model_state_sha256": "5" * 64,
        "initial_optimizer_state_sha256": "6" * 64,
        "optimizer_config_sha256": "7" * 64,
        "parameter_layout_sha256": "8" * 64,
        "optimizer_layout_sha256": "9" * 64,
        "operator_graph_sha256": "a" * 64,
        "source_identity_sha256": "b" * 64,
        "environment_identity_sha256": "c" * 64,
    }


def c0_inputs() -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
    documents = load_contract_documents()
    binding = documents.prereg["calibration_binding"]
    completion = {
        "classification": binding["completion_classification"],
        "completed": True,
        "formal_authorization": False,
        "formal_seeds_consumed": False,
        "reusable_as_preflight": False,
        "scientific_score": False,
        "prereg_sha256": binding["prereg_sha256"],
        "worker_manifest_sha256": binding["manifest_sha256"],
    }
    metric_names = {**binding["primary_metrics"], **binding["secondary_metrics"]}
    units = {**binding["primary_units"], **binding["secondary_units"]}
    metrics = {
        logical: {"metric": metric, "unit": units[logical], "sum": 1.0}
        for logical, metric in metric_names.items()
    }
    manifest = {
        "classification": binding["manifest_classification"],
        "formal_authorization": False,
        "formal_seeds_consumed": False,
        "reusable_as_preflight": False,
        "scientific_score": False,
        "candidate_metrics": metric_names,
        "primary_metrics": {"read": "dram_read", "write": "dram_write"},
        "gpu_name": "test gpu",
        "compute_capability": [12, 0],
        "cuda_runtime": "12.8",
        "torch": "2.11.0+cu128",
        "ncu_launcher_sha256": "d" * 64,
        "ncu_executable_sha256": "e" * 64,
        "observations": {
            "first": {"metrics": copy.deepcopy(metrics)},
            "second": {"metrics": copy.deepcopy(metrics)},
        },
    }
    hashes = {
        "prereg": binding["prereg_sha256"],
        "completion": binding["completion_sha256"],
        "manifest": binding["manifest_sha256"],
        "outer_sha256s": binding["outer_sha256s_sha256"],
        "worker_sha256s": binding["worker_sha256s_sha256"],
    }
    return completion, manifest, hashes


def lineage_ref(arm: str, attempt: int, classification: str) -> dict[str, object]:
    return {
        "arm": arm,
        "attempt_index": attempt,
        "classification": classification,
        "completion_sha256": canonical_digest([arm, attempt, classification]),
    }


def test_frozen_documents_are_loaded_and_bound() -> None:
    documents = load_contract_documents()
    assert documents.prereg["experiment_id"] == "formula-topology-v4-arm-characterization-001"
    assert "c0-binding.json" in documents.artifact_contract["artifact_files"]
    assert documents.schema["$schema"].endswith("draft/2020-12/schema")
    assert sum(documents.prereg["budgets_seconds"]["phase_limits"].values()) == 540


def test_canonical_json_is_order_independent_ascii_and_compact() -> None:
    left = {"z": "\u96ea", "a": [1, True, None]}
    right = {"a": [1, True, None], "z": "\u96ea"}
    assert canonical_json(left) == canonical_json(right)
    assert canonical_json(left) == '{"a":[1,true,null],"z":"\\u96ea"}'
    assert canonical_digest(left) == canonical_digest(right)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_nonfinite_numbers(bad: float) -> None:
    with pytest.raises(ArmContractError, match="non-finite"):
        canonical_json({"nested": [bad]})


def test_exact_keys_reject_missing_and_extra() -> None:
    with pytest.raises(ArmContractError, match="missing=.*b.*extra=.*c"):
        require_exact_keys({"a": 1, "c": 2}, {"a", "b"}, name="sample")


def test_json_pointer_supports_root_arrays_and_rfc6901_escapes() -> None:
    payload = {"a/b": {"~key": ["zero", "one"]}}
    assert read_json_pointer(payload, "") is payload
    assert read_json_pointer(payload, "/a~1b/~0key/1") == "one"


@pytest.mark.parametrize("pointer", ["missing-slash", "/bad~2escape", "/items/01", "/items/-"])
def test_json_pointer_rejects_invalid_or_missing_paths(pointer: str) -> None:
    with pytest.raises(ArmContractError):
        read_json_pointer({"items": [1]}, pointer)


def test_workload_projection_removes_only_arm() -> None:
    full = workload_identity("dynamic")
    shared = build_shared_workload_projection(full)
    assert "arm" not in shared
    assert shared == {key: value for key, value in full.items() if key != "arm"}
    assert full["arm"] == "dynamic"


def test_workload_projection_rejects_extra_key() -> None:
    full = workload_identity()
    full["unexpected"] = 1
    with pytest.raises(ArmContractError, match="extra=.*unexpected"):
        build_shared_workload_projection(full)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("batch_size", 127),
        ("depth", 3),
        ("device", "cpu"),
        ("episode_sha256", "not-a-sha256"),
    ],
)
def test_workload_projection_enforces_schema_values(field: str, bad_value: object) -> None:
    full = workload_identity()
    full[field] = bad_value
    with pytest.raises(ArmContractError, match="violates schema"):
        build_shared_workload_projection(full)


def test_public_schema_validation_entrypoint_enforces_draft_2020_12() -> None:
    with pytest.raises(ArmContractError, match="violates schema.*batch_size"):
        validate_schema_instance(
            {**workload_identity(), "batch_size": True},
            "workloadIdentity",
            name="workload",
        )


def test_workload_identity_document_round_trips() -> None:
    document = build_workload_identity_document(
        workload_identity("static"),
        attempt_index=2,
        arm_semantics_sha256=HEX,
    )
    validate_workload_identity_document(document)
    assert document["shared_workload_identity_sha256"] == canonical_digest(
        document["shared_workload_identity"]
    )


@pytest.mark.parametrize("field", ["shared_workload_identity", "full_workload_identity_sha256"])
def test_workload_identity_document_rejects_projection_or_digest_tampering(field: str) -> None:
    document = build_workload_identity_document(
        workload_identity(),
        attempt_index=1,
        arm_semantics_sha256=HEX,
    )
    if field == "shared_workload_identity":
        document[field]["seed"] = 7
    else:
        document[field] = "f" * 64
    with pytest.raises(ArmContractError):
        validate_workload_identity_document(document)


def test_c0_binding_constructs_exact_projection_and_digest() -> None:
    completion, manifest, hashes = c0_inputs()
    binding = build_c0_binding(
        completion=completion,
        manifest=manifest,
        artifact_hashes=hashes,
    )
    validate_c0_binding(binding)
    assert binding["canonical_projection_sha256"] == canonical_digest(binding["projection"])
    assert set(binding) == {
        "format",
        "experiment_id",
        "projection",
        "canonical_projection_sha256",
    }


def test_c0_binding_rejects_extra_projection_key() -> None:
    completion, manifest, hashes = c0_inputs()
    binding = build_c0_binding(completion=completion, manifest=manifest, artifact_hashes=hashes)
    binding["projection"]["extra"] = True
    with pytest.raises(ArmContractError, match="Additional properties|keys mismatch"):
        validate_c0_binding(binding)


def test_c0_binding_rejects_digest_tampering() -> None:
    completion, manifest, hashes = c0_inputs()
    binding = build_c0_binding(completion=completion, manifest=manifest, artifact_hashes=hashes)
    binding["canonical_projection_sha256"] = "0" * 64
    with pytest.raises(ArmContractError, match="digest mismatch"):
        validate_c0_binding(binding)


@pytest.mark.parametrize("mutation", ["missing", "wrong"])
def test_c0_projection_requires_every_metric_unit_in_every_observation(mutation: str) -> None:
    completion, manifest, hashes = c0_inputs()
    metric = manifest["observations"]["second"]["metrics"]["dram_read"]
    if mutation == "missing":
        del metric["unit"]
    else:
        metric["unit"] = "sector"
    with pytest.raises(ArmContractError, match="unit"):
        build_c0_binding(completion=completion, manifest=manifest, artifact_hashes=hashes)


def test_c0_projection_rejects_unregistered_artifact_hashes() -> None:
    completion, manifest, hashes = c0_inputs()
    hashes["manifest"] = "f" * 64
    with pytest.raises(ArmContractError, match="do not match preregistration"):
        build_c0_binding(completion=completion, manifest=manifest, artifact_hashes=hashes)


def test_c0_projection_constructor_enforces_nested_schema_types() -> None:
    completion, manifest, hashes = c0_inputs()
    manifest["compute_capability"] = [12, "zero"]
    with pytest.raises(ArmContractError, match="violates schema.*compute_capability"):
        build_c0_binding(completion=completion, manifest=manifest, artifact_hashes=hashes)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("artifact_hash", "artifact hashes"),
        ("completion_authority", "authority"),
        ("manifest_authority", "authority"),
        ("units", "observed_units|observed units"),
    ],
)
def test_c0_validation_rejects_resealed_constant_tampering(
    mutation: str, message: str
) -> None:
    completion, manifest, hashes = c0_inputs()
    document = build_c0_binding(
        completion=completion,
        manifest=manifest,
        artifact_hashes=hashes,
    )
    projection = document["projection"]
    if mutation == "artifact_hash":
        projection["artifact_hashes"]["manifest"] = "f" * 64
    elif mutation == "completion_authority":
        projection["completion"]["authority"]["formal_authorization"] = True
    elif mutation == "manifest_authority":
        projection["manifest"]["authority"]["scientific_score"] = True
    else:
        projection["observed_units"]["dram_read"] = "sector"
    document["canonical_projection_sha256"] = canonical_digest(projection)
    with pytest.raises(ArmContractError, match=message):
        validate_c0_binding(document)


def test_lineage_first_sham_attempt_is_valid() -> None:
    document = build_lineage(
        arm="sham",
        attempt_index=1,
        same_arm_predecessor=None,
        locked_prior_arms=[],
    )
    validate_lineage(document)


def test_lineage_retry_binds_immediate_invalid_predecessor() -> None:
    document = build_lineage(
        arm="sham",
        attempt_index=2,
        same_arm_predecessor=lineage_ref("sham", 1, "INVALID_TIMEOUT"),
        locked_prior_arms=[],
    )
    validate_lineage(document)


def test_lineage_dynamic_binds_prior_valid_arms_in_order() -> None:
    document = build_lineage(
        arm="dynamic",
        attempt_index=1,
        same_arm_predecessor=None,
        locked_prior_arms=[
            lineage_ref("sham", 1, "VALID_CHARACTERIZED"),
            lineage_ref("static", 2, "VALID_CHARACTERIZED"),
        ],
    )
    validate_lineage(document)


@pytest.mark.parametrize(
    ("attempt", "predecessor"),
    [
        (1, lineage_ref("sham", 1, "INVALID_TIMEOUT")),
        (2, None),
        (3, lineage_ref("sham", 1, "INVALID_TIMEOUT")),
        (2, lineage_ref("static", 1, "INVALID_TIMEOUT")),
        (2, lineage_ref("sham", 1, "VALID_CHARACTERIZED")),
    ],
)
def test_lineage_rejects_invalid_predecessor_rules(
    attempt: int, predecessor: dict[str, object] | None
) -> None:
    with pytest.raises(ArmContractError):
        build_lineage(
            arm="sham",
            attempt_index=attempt,
            same_arm_predecessor=predecessor,
            locked_prior_arms=[],
        )


def test_lineage_rejects_missing_or_reordered_locked_arms() -> None:
    with pytest.raises(ArmContractError, match="every earlier"):
        build_lineage(
            arm="dynamic",
            attempt_index=1,
            same_arm_predecessor=None,
            locked_prior_arms=[lineage_ref("sham", 1, "VALID_CHARACTERIZED")],
        )
    with pytest.raises(ArmContractError, match="order"):
        build_lineage(
            arm="dynamic",
            attempt_index=1,
            same_arm_predecessor=None,
            locked_prior_arms=[
                lineage_ref("static", 1, "VALID_CHARACTERIZED"),
                lineage_ref("sham", 1, "VALID_CHARACTERIZED"),
            ],
        )


def test_only_requested_files_are_new_or_modified() -> None:
    assert (ROOT / "benchmarks" / "_formula_topology_v4_arm_contract.py").is_file()
    assert (ROOT / "tests" / "test_formula_topology_v4_arm_contract.py").is_file()
