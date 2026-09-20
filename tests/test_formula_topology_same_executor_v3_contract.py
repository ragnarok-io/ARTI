from __future__ import annotations

import importlib.util
import array
import base64
import hashlib
import json
import math
import struct
import sys
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
VERIFY_PATH = BENCHMARKS / "verify_formula_topology_same_executor_v3.py"
CONTRACT_PATH = BENCHMARKS / "formula_topology_same_executor_v3_artifact_contract.json"


def load_verifier():
    name = "formula_topology_same_executor_v3_verifier_test"
    spec = importlib.util.spec_from_file_location(name, VERIFY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_ncu(path: Path, *, process_id: int = 1) -> None:
    rows = ["ID,Process ID,Kernel Name,Metric Name,Metric Unit,Metric Value"]
    for index in range(20):
        rows.append(
            f"{index},{process_id},kernel_{index},dram__bytes_read.sum,byte,50"
        )
        rows.append(
            f"{index},{process_id},kernel_{index},dram__bytes_write.sum,byte,25"
        )
    path.write_bytes(("\r\n".join(rows) + "\r\n").encode("utf-8"))


def write_ncu_master(path: Path, process_ids: dict[str, int]) -> None:
    rows = ['"ID","Process ID","Kernel Name","Metric Name","Metric Unit","Metric Value"']
    for process_id in process_ids.values():
        for index in range(20):
            rows.append(
                f'"{index}","{process_id}","kernel_{index}","dram__bytes_read.sum","byte","50"'
            )
            rows.append(
                f'"{index}","{process_id}","kernel_{index}","dram__bytes_write.sum","byte","25"'
            )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _compressed_f32(values: list[float]) -> str:
    payload = array.array("f", values).tobytes()
    return base64.b64encode(zlib.compress(payload)).decode("ascii")


def metric(
    value: float,
    *,
    max_absolute_error: float | None = None,
    target_values: list[float] | None = None,
) -> dict[str, Any]:
    count = 512 * 8 * 8
    if target_values is None:
        target = [0.0] * count
        target[0] = math.sqrt(50.0)
        target[8] = -math.sqrt(50.0)
    else:
        target = target_values
    valid_count = 4096
    means = [sum(target[index::8]) / valid_count for index in range(8)]
    denominator = sum(
        (item - means[index % 8]) ** 2 for index, item in enumerate(target)
    )
    prediction = list(target)
    prediction[0] += math.sqrt(value * denominator)
    target = list(array.array("f", target))
    prediction = list(array.array("f", prediction))
    means = [sum(target[index::8]) / valid_count for index in range(8)]
    denominator = sum(
        (item - means[index % 8]) ** 2 for index, item in enumerate(target)
    )
    squared_error = sum(
        (left - right) ** 2
        for left, right in zip(prediction, target, strict=True)
    )
    actual_max = max(
        abs(left - right)
        for left, right in zip(prediction, target, strict=True)
    )
    return {
        "feature_count": 8,
        "normalized_mse": squared_error / denominator,
        "squared_error_sum": squared_error,
        "target_centered_sum_squares": denominator,
        "valid_count": 4096,
        "max_absolute_error": actual_max if max_absolute_error is None else max_absolute_error,
        "raw_evidence": {
            "shape": [512, 8, 8],
            "prediction_f32_zlib_base64": _compressed_f32(prediction),
            "target_f32_zlib_base64": _compressed_f32(target),
            "mask_u8_zlib_base64": base64.b64encode(zlib.compress(bytes([1]) * 4096)).decode("ascii"),
        },
    }


def float32_tensor_hash(value: list[list[float]]) -> str:
    header = json.dumps(
        {"dtype": "torch.float32", "shape": [8, 8]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw = b"".join(struct.pack("<f", item) for row in value for item in row)
    return hashlib.sha256(header + b"\n" + raw).hexdigest()


TASK_VALUE = [[1.0 if row == column else 0.0 for column in range(8)] for row in range(8)]
TASK_HASH = float32_tensor_hash(TASK_VALUE)


@lru_cache(maxsize=None)
def replay_receipts(seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inserted = str(BENCHMARKS) not in sys.path
    if inserted:
        sys.path.insert(0, str(BENCHMARKS))
    try:
        from _formula_topology_same_executor_v3 import EpisodeGenerator

        train = [
            {
                **EpisodeGenerator(seed, split="train").generate(
                    128, batch_index=index
                ).receipt(),
                "depth": 1 + (index % 2),
            }
            for index in range(220)
        ]
        evaluation = EpisodeGenerator(seed, split="eval_paired").generate(
            512, batch_index=0
        ).receipt()
        return train, evaluation
    finally:
        if inserted:
            sys.path.remove(str(BENCHMARKS))


@lru_cache(maxsize=None)
def replay_target(seed: int, depth: int) -> list[float]:
    inserted = str(BENCHMARKS) not in sys.path
    if inserted:
        sys.path.insert(0, str(BENCHMARKS))
    try:
        from _formula_topology_same_executor_v3 import EpisodeGenerator

        episode = EpisodeGenerator(seed, split="eval_paired").generate(
            512, batch_index=0
        )
        return episode.oracle_target(depth).reshape(-1).tolist()
    finally:
        if inserted:
            sys.path.remove(str(BENCHMARKS))


def provenance(verifier) -> dict[str, str]:
    segments = [f"{depth + 20:064x}" for depth in (1, 2, 3)]
    programs = {
        str(depth): verifier.canonical_json_sha256(
            {
                "component_ref": "arti.experiment/formula-commit-blend@1",
                "segments": segments[:depth],
            }
        )
        for depth in (1, 2, 3)
    }
    return {
        **verifier.expected_bindings(),
        "environment_sha256": "1" * 64,
        "formula_program_sha256": verifier.canonical_json_sha256(programs),
        "git_head": "test-clean-commit",
        "task_transform_sha256": TASK_HASH,
    }


def execution_receipts(verifier, seed: int) -> dict[str, Any]:
    arms = ("dynamic", "static", "sham")
    common_hash = f"{seed + 100:064x}"
    initial_states = {arm: common_hash for arm in arms}
    final_states = {
        "dynamic": f"{seed + 101:064x}",
        "static": f"{seed + 102:064x}",
        "sham": common_hash,
    }
    taint = {}
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    for index, field in enumerate(contract["taint_fields"]):
        output = f"{seed + 200 + index:064x}"
        taint[field] = {
            "baseline_input_sha256": f"{seed + 300 + index:064x}",
            "replacement_input_sha256": f"{seed + 400 + index:064x}",
            "baseline_branch_output_sha256": f"{seed + 500 + index:064x}",
            "replacement_branch_output_sha256": f"{seed + 600 + index:064x}",
            "baseline_priority_sha256": output,
            "replacement_priority_sha256": output,
            "baseline_permutation_sha256": output,
            "replacement_permutation_sha256": output,
            "baseline_fold_record_sha256": output,
            "replacement_fold_record_sha256": output,
            "branch_owner_ref": "arti.experiment/formula-topology-same-executor@3",
        }
    segments = [f"{depth + 20:064x}" for depth in (1, 2, 3)]
    programs = {
        str(depth): verifier.canonical_json_sha256(
            {
                "component_ref": "arti.experiment/formula-commit-blend@1",
                "segments": segments[:depth],
            }
        )
        for depth in (1, 2, 3)
    }
    namespaces = contract["rng_namespaces"]
    namespace_receipts: dict[str, Any] = {}
    train_episode_receipts, eval_episode_receipt = replay_receipts(seed)
    for name in namespaces[:5]:
        namespace_receipts[name] = [
            *[
                {
                    "master_seed": seed,
                    "split": "train",
                    "batch_index": batch_index,
                    "stream": name,
                    "derived_seed": verifier.expected_namespace_seed(seed, f"train/{batch_index}/{name}"),
                }
                for batch_index in range(220)
            ],
            {
                "master_seed": seed,
                "split": "eval_paired",
                "batch_index": 0,
                "stream": name,
                "derived_seed": verifier.expected_namespace_seed(seed, f"eval_paired/0/{name}"),
            },
        ]
    prereg = json.loads(verifier.PREREG_PATH.read_text(encoding="utf-8"))
    for name in namespaces[5:]:
        master_seed = prereg["metric_contract"]["bootstrap_seed"] if name == "bootstrap" else seed
        namespace_receipts[name] = {
            "master_seed": master_seed,
            "namespace": name,
            "derived_seed": master_seed if name == "bootstrap" else verifier.expected_namespace_seed(master_seed, name),
        }
    route_slots = [[[0, 1], [2, 3]], [[0, 2]], [[1]]]
    route_receipts: dict[str, list[dict[str, Any]]] = {}
    for depth in (1, 2, 3):
        rows = []
        for step in range(depth):
            slots_for_step = route_slots[step]
            max_arity = max(len(slots) for slots in slots_for_step)
            weights = [
                [[
                    [
                        [
                            1.0
                            if slot
                            == (
                                operand_slots[operand_index]
                                if operand_index < len(operand_slots)
                                else 0
                            )
                            else 0.0
                            for slot in range(4)
                        ]
                        for operand_index in range(max_arity)
                    ]
                    for operand_slots in slots_for_step
                ]]
                for _ in range(8)
            ]
            masks = [[[True] * len(slots_for_step)] for _ in range(8)]
            rows.append(
                {
                    "step": step,
                    "batch_size": 8,
                    "program_fingerprint": segments[step],
                    "operand_slots": slots_for_step,
                    "weights_value": weights,
                    "valid_value": masks,
                    "fire_value": masks,
                    "commit_value": masks,
                    "weights": verifier.replay_tensor_sha256(weights, boolean=False),
                    "valid": verifier.replay_tensor_sha256(masks, boolean=True),
                    "fire": verifier.replay_tensor_sha256(masks, boolean=True),
                    "commit": verifier.replay_tensor_sha256(masks, boolean=True),
                }
            )
        route_receipts[str(depth)] = rows
    return {
        "same_executor": {
            "arm_initial_state_sha256": initial_states,
            "arm_final_state_sha256": final_states,
            "arm_initial_optimizer_sha256": {arm: "8" * 64 for arm in arms},
            "arm_parameter_layout_sha256": {arm: "9" * 64 for arm in arms},
            "arm_optimizer_layout_sha256": {arm: "a" * 64 for arm in arms},
            "arm_operator_signature_sha256": {arm: "b" * 64 for arm in arms},
            "both_priority_paths_executed": {arm: True for arm in arms},
            "gradient_multiplier": {"dynamic": 1.0, "static": 1.0, "sham": 0.0},
        },
        "topology_source": {
            "source_component_ref": "arti.experiment/formula-topology-policy@1",
            "declared_source_inputs": ["keys", "role_queries"],
            "source_input_sha256": {"keys": "c" * 64, "role_queries": "d" * 64},
            "topology_contract_sha256": "e" * 64,
            "operator_contract_sha256": "f" * 64,
            "surrogate_component_ref": "arti/topology-surrogate@2",
            "surrogate_contract_sha256": "1" * 64,
            "producer_fingerprint_sha256": "2" * 64,
            "proposal_call_count": 1,
            "payload_crossed_policy_api": False,
            "hard_forward_only": True,
            "surrogate_backward_only": True,
            "surrogate_enabled_source_grad_norm": 1.0,
            "surrogate_disabled_source_grad_norm": 0.0,
            "surrogate_hard_output_max_abs_difference": 0.0,
            "surrogate_payload_grad_max_abs_difference": 0.0,
        },
        "taint_replacements": taint,
        "formula_commit_blend": {
            "component_ref": "arti.experiment/formula-commit-blend@1",
            "formula_fabric_ref": "arti/formula-fabric@1",
            "program_fingerprints": programs,
            "route_receipts": route_receipts,
            "segment_program_fingerprints": segments,
            "segment_commit_counts": [16, 8, 8],
            "segment_version_sha256": ["5" * 64, "6" * 64, "7" * 64],
            "expected_final_version_sha256": "8" * 64,
            "actual_final_version_sha256": "8" * 64,
            "normal_scale_sha256": "3" * 64,
            "zero_scale_sha256": "4" * 64,
            "candidate_delta_norm": 1.0,
            "zero_commit_max_abs_error": 0.0,
            "version_increment_count": 4,
            "commit_mask_all_true": True,
            "formula_executed": True,
            "per_ssa_step_blend": True,
        },
        "rng": {
            "algorithm": "sha256-counter-torch-generator",
            "stream_version": "arti.formula-topology-rng.v1",
            "namespaces": namespaces,
            "namespace_receipts": namespace_receipts,
            "namespace_receipt_sha256": {
                name: verifier.canonical_json_sha256(namespace_receipts[name])
                for name in namespaces
            },
            "train_eval_disjoint": True,
            "global_rng_unchanged": True,
            "episode_receipts": {
                "train": train_episode_receipts,
                "eval": eval_episode_receipt,
            },
        },
        "task_transform": {
            "dtype": "float32",
            "shape": [8, 8],
            "value": TASK_VALUE,
            "value_sha256": TASK_HASH,
        },
    }


def seed_artifact(verifier, seed: int, *, static: float = 0.10, dynamic: float = 0.05) -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    condition_values = {
        "dynamic_full": dynamic,
        "static_full": static,
        "sham_full": 0.11,
        "oracle_topology": 0.0,
        "identity_topology": 0.09,
        "seeded_random_topology": 0.09,
        "fresh_initialization_topology": 0.09,
        "within_example_topology_shuffle": 0.09,
        "target_key_shuffle": 0.09,
        "payload_only_policy_probe": 0.09,
        "wrong_formula": 0.09,
        "formula_zero_real_commit": 0.09,
        "slot_permutation_equivariance": dynamic,
    }
    conditions: dict[str, Any] = {}
    for index, condition in enumerate(contract["required_conditions"]):
        depths = contract["condition_depths"][condition]
        by_depth = {
            str(depth): metric(
                condition_values[condition],
                max_absolute_error=0.0 if condition == "oracle_topology" else None,
                target_values=replay_target(seed, depth),
            )
            for depth in depths
        }
        if condition == "dynamic_full":
            by_depth["2"] = metric(0.045, target_values=replay_target(seed, 2))
        trace_receipt = [{"condition": condition, "index": index}]
        conditions[condition] = {
            "metrics_by_depth": by_depth,
            "trace_receipt": trace_receipt,
            "trace_sha256": verifier.canonical_json_sha256(trace_receipt),
        }
    receipts = execution_receipts(verifier, seed)
    train_hash = verifier.canonical_json_sha256(receipts["rng"]["episode_receipts"]["train"])
    eval_hash = verifier.canonical_json_sha256(receipts["rng"]["episode_receipts"]["eval"])
    return {
        "format": contract["formats"]["seed"],
        "execution_mode": "formal",
        "seed": seed,
        "provenance": provenance(verifier),
        "episode_hashes": {
            "train": train_hash,
            "eval_depth_2": eval_hash,
            "eval_depth_3": eval_hash,
        },
        "conditions": conditions,
        "execution_receipts": receipts,
        "contract_checks": {
            "formula_reference_parity": True,
            "fold_unfold_bit_exact_round_trip": True,
            "hard_forward_original_payload_only": True,
            "surrogate_absent_from_reported_forward": True,
            "mask_version_finite_gradient_checks": True,
            "policy_input_taint_replacement_tests": True,
            "formula_zero_real_commit_versions_incremented": True,
            "slot_permutation_equivariance_error": 0.0,
        },
        "wall_seconds": 1.0,
    }


def raw_cost(verifier, arm: str, *, forward_flops: float = 100.0) -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    prereg = json.loads(verifier.PREREG_PATH.read_text(encoding="utf-8"))
    expected = verifier.replay_cost_workload_identity(
        prereg["formal_run"]["cost_profile_seed"],
        prereg["formal_run"]["batch_size"],
        arm,
    )
    workload_identity = {
        "format": "arti.formula-topology-same-executor-cost-workload.v3",
        **expected,
        "depth": 2,
        "dtype": "torch.float32",
        "device_type": "cuda",
        "batch_size": prereg["formal_run"]["batch_size"],
    }
    return {
        "format": contract["formats"]["raw_cost"],
        "arm": arm,
        "scope": "complete_forward_backward_gradient_postprocess_optimizer_step",
        "forward_flops": forward_flops,
        "backward_flops": 200.0,
        "kernel_launches": 20.0,
        "dram_read_bytes": 1000.0,
        "dram_write_bytes": 500.0,
        "warm_latency_samples_seconds": [0.01] * 50,
        "cold_latency_seconds": 0.02,
        "peak_allocated_memory_bytes": 768,
        "peak_reserved_memory_bytes": 1024,
        "optimizer_parameter_elements": 1024,
        "optimizer_parameter_tensors": 3,
        "optimizer_parameter_layout_sha256": "5" * 64,
        "optimizer_state_layout_sha256": "6" * 64,
        "optimizer_state_tensor_count": 9,
        "operator_signature": {"aten::mm": 4, "aten::sort": 1},
        "workload_identity": workload_identity,
        "ncu_metric_names": {
            "dram_read_bytes": "dram__bytes_read.sum",
            "dram_write_bytes": "dram__bytes_write.sum",
        },
        "ncu_version": "NVIDIA Nsight Compute test",
    }


def build_bundle(
    tmp_path: Path,
    *,
    seed_count: int = 8,
    decision: str = "ALL_SEEDS_COMPLETED",
    cost_failure: bool = False,
    futility: bool = False,
) -> tuple[Any, Path]:
    verifier = load_verifier()
    prereg = json.loads(verifier.PREREG_PATH.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    common = provenance(verifier)

    preflight = {
        "format": contract["formats"]["preflight"],
        "valid": True,
        "mode": "formal",
        "counter_status": "AVAILABLE",
        "git_head": common["git_head"],
        "git_clean": True,
        "provenance": common,
        "duration_seconds": 1.0,
        "cost_preflight_passed": not cost_failure,
        "orchestrator_capability_sha256": "7" * 64,
    }
    write_json(tmp_path / "preflight.json", preflight)

    raw_references: dict[str, dict[str, Any]] = {}
    reported: dict[str, dict[str, float]] = {}
    costs: dict[str, dict[str, Any]] = {}
    process_ids = {
        arm: 101 + index for index, arm in enumerate(contract["cost_arms"])
    }
    ncu_master_path = tmp_path / "ncu-all-arms.csv"
    process_map_path = tmp_path / "ncu-process-map.json"
    write_ncu_master(ncu_master_path, process_ids)
    process_entries: list[dict[str, Any]] = []
    for arm in contract["cost_arms"]:
        receipt_path = tmp_path / f"ncu-child-{arm}.json"
        stdout_path = tmp_path / f"ncu-child-{arm}.stdout.txt"
        stderr_path = tmp_path / f"ncu-child-{arm}.stderr.txt"
        argv = [
            "--arm", arm,
            "--batch-size", str(prereg["formal_run"]["batch_size"]),
            "--seed", str(prereg["formal_run"]["cost_profile_seed"]),
            "--nvtx-step", "--nvtx-receipt", str(receipt_path),
        ]
        write_json(
            receipt_path,
            {
                "format": "arti.formula-topology-same-executor-ncu-child.v3",
                "process_id": process_ids[arm],
                "arm": arm,
                "batch_size": prereg["formal_run"]["batch_size"],
                "seed": prereg["formal_run"]["cost_profile_seed"],
                "scope": "complete_forward_backward_gradient_postprocess_optimizer_step",
                "argv_sha256": hashlib.sha256(
                    json.dumps(
                        argv, ensure_ascii=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest(),
                "workload_identity": raw_cost(verifier, arm)["workload_identity"],
                "completed": True,
            },
        )
        stdout_path.write_bytes(b"0\n")
        stderr_path.write_bytes(b"")
        process_entries.append(
            {
                "arm": arm,
                "process_id": process_ids[arm],
                "returncode": 0,
                "receipt_file": receipt_path.name,
                "receipt_sha256": verifier.sha256_file(receipt_path),
                "stdout_file": stdout_path.name,
                "stdout_sha256": verifier.sha256_file(stdout_path),
                "stderr_file": stderr_path.name,
                "stderr_sha256": verifier.sha256_file(stderr_path),
            }
        )
    write_json(
        process_map_path,
        {
            "format": "arti.formula-topology-same-executor-ncu-process-map.v3",
            "batch_size": prereg["formal_run"]["batch_size"],
            "seed": prereg["formal_run"]["cost_profile_seed"],
            "processes": process_entries,
        },
    )
    for arm in contract["cost_arms"]:
        forward = 200.0 if cost_failure and arm == "dynamic" else 100.0
        item = raw_cost(verifier, arm, forward_flops=forward)
        path = tmp_path / f"cost-{arm}.json"
        write_json(path, item)
        ncu_path = tmp_path / f"ncu-{arm}.csv"
        write_ncu(ncu_path, process_id=process_ids[arm])
        attributed_path = tmp_path / f"attributed-{arm}.json"
        latency_path = tmp_path / f"latency-{arm}.json"
        write_json(
            attributed_path,
            {
                "format": "arti.formula-topology-same-executor-attributed-cost.v3",
                "arm": arm,
                "batch_size": prereg["formal_run"]["batch_size"],
                "seed": prereg["formal_run"]["cost_profile_seed"],
                "scope": "complete_forward_backward_gradient_postprocess_optimizer_step",
                **{
                key: item[key]
                for key in (
                    "forward_flops", "backward_flops", "operator_signature",
                    "optimizer_parameter_elements", "optimizer_parameter_tensors",
                    "optimizer_parameter_layout_sha256", "optimizer_state_layout_sha256",
                    "optimizer_state_tensor_count",
                    "workload_identity",
                )
                },
            },
        )
        write_json(
            latency_path,
            {
                "format": "arti.formula-topology-same-executor-latency-cost.v3",
                "arm": arm,
                "batch_size": prereg["formal_run"]["batch_size"],
                "seed": prereg["formal_run"]["cost_profile_seed"],
                **{
                key: item[key]
                for key in (
                    "cold_latency_seconds", "warm_latency_samples_seconds",
                    "peak_allocated_memory_bytes", "peak_reserved_memory_bytes",
                    "workload_identity",
                )
                },
            },
        )
        costs[arm] = item
        raw_references[arm] = {
            "file": path.name,
            "sha256": verifier.sha256_file(path),
            "ncu_file": ncu_path.name,
            "ncu_sha256": verifier.sha256_file(ncu_path),
            "ncu_master_file": ncu_master_path.name,
            "ncu_master_sha256": verifier.sha256_file(ncu_master_path),
            "ncu_process_map_file": process_map_path.name,
            "ncu_process_map_sha256": verifier.sha256_file(process_map_path),
            "ncu_process_id": process_ids[arm],
            "attributed_file": attributed_path.name,
            "attributed_sha256": verifier.sha256_file(attributed_path),
            "latency_file": latency_path.name,
            "latency_sha256": verifier.sha256_file(latency_path),
        }
    pairs = {
        "dynamic_vs_static": ("dynamic", "static"),
        "dynamic_vs_sham": ("dynamic", "sham"),
        "static_vs_sham": ("static", "sham"),
    }
    for name, (left, right) in pairs.items():
        reported[name] = {
            "forward_flops": verifier.symmetric_ratio(costs[left]["forward_flops"], costs[right]["forward_flops"]),
            "backward_flops": 1.0,
            "kernel_launches": 1.0,
            "dram_read_bytes": 1.0,
            "dram_write_bytes": 1.0,
            "warm_p95_latency": 1.0,
            "peak_allocated_memory_bytes": 1.0,
            "peak_reserved_memory_bytes": 1.0,
        }
    hardware = {
        "format": contract["formats"]["hardware"],
        "physical_counter_status": "AVAILABLE",
        "finalized": True,
        "provenance": common,
        "raw_receipts": raw_references,
        "reported_ratios": reported,
    }
    write_json(tmp_path / "hardware-cost.json", hardware)

    artifact_hashes: dict[str, str] = {}
    receipts: list[dict[str, Any]] = []
    if not cost_failure:
        seeds = prereg["formal_run"]["seeds"][:seed_count]
        for index, seed in enumerate(seeds):
            static = 0.04 if futility and index < 2 else 0.10
            artifact = seed_artifact(verifier, seed, static=static)
            path = tmp_path / f"seed-{seed}.json"
            write_json(path, artifact)
            artifact_hashes[path.name] = verifier.sha256_file(path)
            stdout_path = tmp_path / f"seed-{seed}.stdout.txt"
            stderr_path = tmp_path / f"seed-{seed}.stderr.txt"
            stdout_path.write_bytes(b"")
            stderr_path.write_bytes(b"")
            receipts.append(
                {
                    "seed": seed,
                    "cause": "COMPLETED",
                    "returncode": 0,
                    "duration_seconds": 1.0,
                    "stdout_file": stdout_path.name,
                    "stdout_sha256": verifier.sha256_file(stdout_path),
                    "stderr_file": stderr_path.name,
                    "stderr_sha256": verifier.sha256_file(stderr_path),
                }
            )
    run = {
        "format": contract["formats"]["run_manifest"],
        "decision": decision,
        "provenance": common,
        "preflight_sha256": verifier.sha256_file(tmp_path / "preflight.json"),
        "hardware_sha256": verifier.sha256_file(tmp_path / "hardware-cost.json"),
        "artifact_sha256": artifact_hashes,
        "process_receipts": receipts,
        "gate_wall_seconds": float(max(1, len(receipts))),
    }
    write_json(tmp_path / "run-manifest.json", run)
    return verifier, tmp_path


def refresh_seed_hash(verifier, directory: Path, seed_path: Path) -> None:
    run_path = directory / "run-manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["artifact_sha256"][seed_path.name] = verifier.sha256_file(seed_path)
    write_json(run_path, run)


def refresh_hardware_hash(verifier, directory: Path) -> None:
    run_path = directory / "run-manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["hardware_sha256"] = verifier.sha256_file(directory / "hardware-cost.json")
    write_json(run_path, run)


def test_contract_declares_all_terminal_classes_and_frozen_conditions() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    prereg = json.loads(
        (BENCHMARKS / "formula_topology_same_executor_v3_prereg.json").read_text(encoding="utf-8")
    )
    assert contract["terminal_classes"] == [
        "INVALID_COST_PRECHECK",
        "INVALID_CONTRACT",
        "SCIENTIFIC_NO_GO_FUTILITY",
        "SCIENTIFIC_NO_GO",
        "GO",
    ]
    assert contract["required_conditions"] == prereg["required_conditions"]
    assert contract["condition_depths"]["dynamic_full"] == [2, 3]


def test_valid_complete_bundle_is_go(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "GO"
    assert result["valid"] is True
    assert result["errors"] == []


def test_unknown_seed_key_fails_closed(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    seed_path = next(directory.glob("seed-*.json"))
    artifact = json.loads(seed_path.read_text(encoding="utf-8"))
    artifact["producer_note"] = "must not be accepted"
    write_json(seed_path, artifact)
    refresh_seed_hash(verifier, directory, seed_path)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "unknown=['producer_note']" in result["errors"][0]


def test_seed_file_tamper_breaks_manifest_hash(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    seed_path = next(directory.glob("seed-*.json"))
    seed_path.write_text(seed_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "seed artifact hash mismatch" in result["errors"][0]


def test_amendment_hash_missing_or_tampered_fails_closed(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    seed_path = next(directory.glob("seed-*.json"))
    artifact = json.loads(seed_path.read_text(encoding="utf-8"))
    artifact["provenance"]["amendment_sha256"] = "f" * 64
    write_json(seed_path, artifact)
    refresh_seed_hash(verifier, directory, seed_path)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "amendment_sha256 does not bind current frozen source" in result["errors"][0]

    artifact["provenance"].pop("amendment_sha256")
    write_json(seed_path, artifact)
    refresh_seed_hash(verifier, directory, seed_path)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "missing=['amendment_sha256']" in result["errors"][0]


def test_implementation_source_hash_tamper_fails_closed(tmp_path: Path) -> None:
    for key in (
        "runner_sha256",
        "helper_sha256",
        "cost_target_sha256",
        "cost_collector_sha256",
        "preflight_sha256",
        "orchestrator_sha256",
        "reversible_topology_sha256",
        "topology_sha256",
        "formula_fabric_sha256",
        "component_registry_sha256",
    ):
        directory = tmp_path / key
        directory.mkdir()
        verifier, _ = build_bundle(directory)
        seed_path = next(directory.glob("seed-*.json"))
        artifact = json.loads(seed_path.read_text(encoding="utf-8"))
        artifact["provenance"][key] = "f" * 64
        write_json(seed_path, artifact)
        refresh_seed_hash(verifier, directory, seed_path)
        result = verifier.verify_bundle(directory)
        assert result["terminal_class"] == "INVALID_CONTRACT"
        assert f"{key} does not bind current frozen source" in result["errors"][0]


def test_amendment_execution_receipt_tamper_fails_closed(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    seed_path = next(directory.glob("seed-*.json"))
    artifact = json.loads(seed_path.read_text(encoding="utf-8"))
    artifact["execution_receipts"]["topology_source"][
        "payload_crossed_policy_api"
    ] = True
    write_json(seed_path, artifact)
    refresh_seed_hash(verifier, directory, seed_path)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "payload crossed the policy API" in result["errors"][0]


def test_metric_summary_is_recomputed_not_trusted(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    seed_path = next(directory.glob("seed-*.json"))
    artifact = json.loads(seed_path.read_text(encoding="utf-8"))
    artifact["conditions"]["dynamic_full"]["metrics_by_depth"]["3"]["normalized_mse"] = 0.001
    write_json(seed_path, artifact)
    refresh_seed_hash(verifier, directory, seed_path)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "does not reconstruct" in result["errors"][0]


def test_raw_prediction_tamper_is_rejected(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    seed_path = next(directory.glob("seed-*.json"))
    artifact = json.loads(seed_path.read_text(encoding="utf-8"))
    evidence = artifact["conditions"]["dynamic_full"]["metrics_by_depth"]["3"]["raw_evidence"]
    evidence["prediction_f32_zlib_base64"] = evidence["target_f32_zlib_base64"]
    write_json(seed_path, artifact)
    refresh_seed_hash(verifier, directory, seed_path)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "squared_error_sum does not reconstruct" in result["errors"][0]


def test_rng_derived_seed_tamper_is_rejected(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    seed_path = next(directory.glob("seed-*.json"))
    artifact = json.loads(seed_path.read_text(encoding="utf-8"))
    receipt = artifact["execution_receipts"]["rng"]
    receipt["namespace_receipts"]["keys"][0]["derived_seed"] += 1
    receipt["namespace_receipt_sha256"]["keys"] = verifier.canonical_json_sha256(
        receipt["namespace_receipts"]["keys"]
    )
    write_json(seed_path, artifact)
    refresh_seed_hash(verifier, directory, seed_path)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "derived seed differs" in result["errors"][0]


def test_rng_master_seed_cannot_be_rebased(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    seed_path = next(directory.glob("seed-*.json"))
    artifact = json.loads(seed_path.read_text(encoding="utf-8"))
    receipt = artifact["execution_receipts"]["rng"]
    row = receipt["namespace_receipts"]["keys"][0]
    row["master_seed"] += 1
    row["derived_seed"] = verifier.expected_namespace_seed(
        row["master_seed"], f"{row['split']}/{row['batch_index']}/{row['stream']}"
    )
    receipt["namespace_receipt_sha256"]["keys"] = verifier.canonical_json_sha256(
        receipt["namespace_receipts"]["keys"]
    )
    write_json(seed_path, artifact)
    refresh_seed_hash(verifier, directory, seed_path)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "master seed differs from artifact" in result["errors"][0]


def test_condition_target_must_remain_paired(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    seed_path = next(directory.glob("seed-*.json"))
    artifact = json.loads(seed_path.read_text(encoding="utf-8"))
    changed = metric(0.09)
    target_raw = zlib.decompress(
        base64.b64decode(changed["raw_evidence"]["target_f32_zlib_base64"])
    )
    target_values = array.array("f")
    target_values.frombytes(target_raw)
    target_values[16] = 1.0
    changed["raw_evidence"]["target_f32_zlib_base64"] = base64.b64encode(
        zlib.compress(target_values.tobytes())
    ).decode("ascii")
    target = list(target_values)
    prediction = list(target)
    prediction[0] += math.sqrt(9.0)
    changed["raw_evidence"]["prediction_f32_zlib_base64"] = _compressed_f32(prediction)
    valid_count = 4096
    means = [sum(target[index::8]) / valid_count for index in range(8)]
    changed["target_centered_sum_squares"] = sum(
        (value - means[index % 8]) ** 2 for index, value in enumerate(target)
    )
    changed["squared_error_sum"] = 9.0
    changed["normalized_mse"] = 9.0 / changed["target_centered_sum_squares"]
    changed["max_absolute_error"] = 3.0
    artifact["conditions"]["identity_topology"]["metrics_by_depth"]["3"] = changed
    write_json(seed_path, artifact)
    refresh_seed_hash(verifier, directory, seed_path)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "do not match the replayed oracle episode" in result["errors"][0]


def test_all_conditions_cannot_share_a_forged_oracle(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    seed_path = next(directory.glob("seed-*.json"))
    artifact = json.loads(seed_path.read_text(encoding="utf-8"))
    for condition in artifact["conditions"].values():
        if "3" in condition["metrics_by_depth"]:
            current = condition["metrics_by_depth"]["3"]
            condition["metrics_by_depth"]["3"] = metric(
                float(current["normalized_mse"])
            )
    write_json(seed_path, artifact)
    refresh_seed_hash(verifier, directory, seed_path)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "do not match the replayed oracle episode" in result["errors"][0]


def test_route_structure_tamper_is_rejected(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    seed_path = next(directory.glob("seed-*.json"))
    artifact = json.loads(seed_path.read_text(encoding="utf-8"))
    artifact["execution_receipts"]["formula_commit_blend"]["route_receipts"]["3"][0]["operand_slots"] = [[0, 2], [1, 3]]
    write_json(seed_path, artifact)
    refresh_seed_hash(verifier, directory, seed_path)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "operand slots differ" in result["errors"][0]


def test_payload_gradient_contract_tamper_is_rejected(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    seed_path = next(directory.glob("seed-*.json"))
    artifact = json.loads(seed_path.read_text(encoding="utf-8"))
    artifact["execution_receipts"]["topology_source"]["surrogate_payload_grad_max_abs_difference"] = 1e-3
    write_json(seed_path, artifact)
    refresh_seed_hash(verifier, directory, seed_path)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "surrogate causal probe failed" in result["errors"][0]


def test_unpaired_depth_evaluation_fails_closed(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    seed_path = next(directory.glob("seed-*.json"))
    artifact = json.loads(seed_path.read_text(encoding="utf-8"))
    artifact["episode_hashes"]["eval_depth_3"] = "f" * 64
    write_json(seed_path, artifact)
    refresh_seed_hash(verifier, directory, seed_path)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "evaluation is not paired" in result["errors"][0]


def test_taint_without_live_non_policy_branch_fails_closed(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    seed_path = next(directory.glob("seed-*.json"))
    artifact = json.loads(seed_path.read_text(encoding="utf-8"))
    receipt = artifact["execution_receipts"]["taint_replacements"]["target"]
    receipt["replacement_branch_output_sha256"] = receipt[
        "baseline_branch_output_sha256"
    ]
    write_json(seed_path, artifact)
    refresh_seed_hash(verifier, directory, seed_path)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "did not affect its non-policy branch" in result["errors"][0]


def test_missing_raw_cost_receipt_is_invalid(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    (directory / "cost-sham.json").unlink()
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "raw cost receipt mismatch" in result["errors"][0]


def test_nsight_raw_csv_tamper_is_invalid(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    path = directory / "ncu-dynamic.csv"
    path.write_text(path.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "Nsight raw receipt mismatch" in result["errors"][0]


def test_attributed_raw_receipt_tamper_is_invalid(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    path = directory / "attributed-dynamic.json"
    item = json.loads(path.read_text(encoding="utf-8"))
    item["forward_flops"] += 1
    write_json(path, item)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "raw profiler receipt mismatch" in result["errors"][0]


def test_workload_identity_cannot_claim_a_different_cost_episode(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    path = directory / "attributed-dynamic.json"
    item = json.loads(path.read_text(encoding="utf-8"))
    item["workload_identity"]["target_sha256"] = "c" * 64
    write_json(path, item)
    hardware_path = directory / "hardware-cost.json"
    hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    hardware["raw_receipts"]["dynamic"]["attributed_sha256"] = verifier.sha256_file(path)
    write_json(hardware_path, hardware)
    refresh_hardware_hash(verifier, directory)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "differs from replay" in result["errors"][0]


def test_nsight_master_rejects_unmapped_metric_process(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    path = directory / "ncu-all-arms.csv"
    path.write_text(
        path.read_text(encoding="utf-8")
        + '"99","999","unknown","dram__bytes_read.sum","byte","1"\n'
        + '"99","999","unknown","dram__bytes_write.sum","byte","1"\n',
        encoding="utf-8",
    )
    hardware_path = directory / "hardware-cost.json"
    hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    for reference in hardware["raw_receipts"].values():
        reference["ncu_master_sha256"] = verifier.sha256_file(path)
    write_json(hardware_path, hardware)
    refresh_hardware_hash(verifier, directory)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "process set differs" in result["errors"][0]


def test_nsight_process_map_pid_must_match_master(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    path = directory / "ncu-process-map.json"
    process_map = json.loads(path.read_text(encoding="utf-8"))
    process_map["processes"][0]["process_id"] = 999
    write_json(path, process_map)
    hardware_path = directory / "hardware-cost.json"
    hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    for reference in hardware["raw_receipts"].values():
        reference["ncu_process_map_sha256"] = verifier.sha256_file(path)
    write_json(hardware_path, hardware)
    refresh_hardware_hash(verifier, directory)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "child invocation identity is invalid" in result["errors"][0]


def test_nsight_per_arm_csv_must_be_exact_master_partition(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    path = directory / "ncu-dynamic.csv"
    value = path.read_text(encoding="utf-8").replace("kernel_0", "forged_kernel")
    path.write_text(value, encoding="utf-8", newline="")
    hardware_path = directory / "hardware-cost.json"
    hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    hardware["raw_receipts"]["dynamic"]["ncu_sha256"] = verifier.sha256_file(path)
    write_json(hardware_path, hardware)
    refresh_hardware_hash(verifier, directory)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "exact master partition" in result["errors"][0]


def test_nsight_child_receipt_is_bound_to_replayed_workload(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    receipt_path = directory / "ncu-child-dynamic.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["workload_identity"]["initial_model_state_sha256"] = "c" * 64
    write_json(receipt_path, receipt)
    map_path = directory / "ncu-process-map.json"
    process_map = json.loads(map_path.read_text(encoding="utf-8"))
    process_map["processes"][0]["receipt_sha256"] = verifier.sha256_file(
        receipt_path
    )
    write_json(map_path, process_map)
    hardware_path = directory / "hardware-cost.json"
    hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    for reference in hardware["raw_receipts"].values():
        reference["ncu_process_map_sha256"] = verifier.sha256_file(map_path)
    write_json(hardware_path, hardware)
    refresh_hardware_hash(verifier, directory)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "differs from replay" in result["errors"][0]


def test_unsafe_profiler_sidecar_path_is_invalid(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    hardware_path = directory / "hardware-cost.json"
    hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    hardware["raw_receipts"]["dynamic"]["attributed_file"] = str(
        (directory / "attributed-dynamic.json").resolve()
    )
    write_json(hardware_path, hardware)
    run_path = directory / "run-manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["hardware_sha256"] = verifier.sha256_file(hardware_path)
    write_json(run_path, run)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "raw profiler filename" in result["errors"][0]


def test_profiler_sidecar_requires_canonical_filename(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    original = directory / "latency-dynamic.json"
    renamed = directory / "latency-copy.json"
    renamed.write_bytes(original.read_bytes())
    hardware_path = directory / "hardware-cost.json"
    hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    hardware["raw_receipts"]["dynamic"]["latency_file"] = renamed.name
    hardware["raw_receipts"]["dynamic"]["latency_sha256"] = verifier.sha256_file(renamed)
    write_json(hardware_path, hardware)
    run_path = directory / "run-manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["hardware_sha256"] = verifier.sha256_file(hardware_path)
    write_json(run_path, run)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "raw profiler filename" in result["errors"][0]


def test_process_output_tamper_is_invalid(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    path = next(directory.glob("seed-*.stdout.txt"))
    path.write_text("tamper", encoding="utf-8")
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "stdout does not reconstruct" in result["errors"][0]


def test_reported_cost_ratio_is_reconstructed(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    hardware_path = directory / "hardware-cost.json"
    hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    hardware["reported_ratios"]["dynamic_vs_static"]["dram_read_bytes"] = 1.01
    write_json(hardware_path, hardware)
    run_path = directory / "run-manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["hardware_sha256"] = verifier.sha256_file(hardware_path)
    write_json(run_path, run)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "reported cost ratio does not reconstruct" in result["errors"][0]


def test_cost_precheck_failure_has_distinct_terminal_class(tmp_path: Path) -> None:
    verifier, directory = build_bundle(
        tmp_path,
        seed_count=0,
        decision="COST_PRECHECK_FAILED",
        cost_failure=True,
    )
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_COST_PRECHECK"
    assert result["valid"] is False
    assert result["cost_failures"]


def test_four_seed_futility_is_verified(tmp_path: Path) -> None:
    verifier, directory = build_bundle(
        tmp_path,
        seed_count=4,
        decision="FUTILITY_STOP",
        futility=True,
    )
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "SCIENTIFIC_NO_GO_FUTILITY"
    assert result["valid"] is True


def test_unjustified_futility_is_invalid(tmp_path: Path) -> None:
    verifier, directory = build_bundle(
        tmp_path,
        seed_count=4,
        decision="FUTILITY_STOP",
        futility=False,
    )
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "futility closure" in result["errors"][0]


def test_complete_scientific_failure_is_no_go_not_invalid(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    for seed_path in directory.glob("seed-*.json"):
        artifact = json.loads(seed_path.read_text(encoding="utf-8"))
        artifact["conditions"]["dynamic_full"]["metrics_by_depth"]["3"] = metric(
            0.20, target_values=replay_target(artifact["seed"], 3)
        )
        write_json(seed_path, artifact)
        refresh_seed_hash(verifier, directory, seed_path)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "SCIENTIFIC_NO_GO"
    assert result["valid"] is True
    assert result["scientific_failures"]


def test_budget_overrun_is_invalid_contract(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    run_path = directory / "run-manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["process_receipts"][0]["duration_seconds"] = 50.0001
    run["gate_wall_seconds"] = 51.0
    write_json(run_path, run)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "per-seed wall-time budget exceeded" in result["errors"][0]


def test_nonfinite_json_is_rejected(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    seed_path = next(directory.glob("seed-*.json"))
    artifact = json.loads(seed_path.read_text(encoding="utf-8"))
    artifact["wall_seconds"] = float("nan")
    seed_path.write_text(json.dumps(artifact), encoding="utf-8")
    refresh_seed_hash(verifier, directory, seed_path)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "non-finite JSON constant" in result["errors"][0]


def test_duplicate_json_key_is_rejected(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    preflight_path = directory / "preflight.json"
    text = preflight_path.read_text(encoding="utf-8")
    preflight_path.write_text(
        text.replace('"valid": true', '"valid": true,\n  "valid": true', 1),
        encoding="utf-8",
    )
    run_path = directory / "run-manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["preflight_sha256"] = verifier.sha256_file(preflight_path)
    write_json(run_path, run)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "duplicate JSON key" in result["errors"][0]


def test_string_cost_decision_is_rejected(tmp_path: Path) -> None:
    verifier, directory = build_bundle(tmp_path)
    preflight_path = directory / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["cost_preflight_passed"] = "true"
    write_json(preflight_path, preflight)
    run_path = directory / "run-manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["preflight_sha256"] = verifier.sha256_file(preflight_path)
    write_json(run_path, run)
    result = verifier.verify_bundle(directory)
    assert result["terminal_class"] == "INVALID_CONTRACT"
    assert "must be boolean" in result["errors"][0]


def test_verifier_source_does_not_import_or_execute_runner() -> None:
    source = VERIFY_PATH.read_text(encoding="utf-8")
    assert "train_formula_topology_same_executor" not in source
    assert "importlib" not in source
    assert "RUNNER_PATH" in source
    assert "subprocess.run((sys.executable" not in source
