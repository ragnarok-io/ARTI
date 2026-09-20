"""Profile the BR6.5 GPU-resident K-wide Branch Search authority window.

This is an inference-only mechanism and transfer receipt. It does not train a
Bank and does not claim downstream quality or packed active-K execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from arti import Recall, RefinePolicy, alpha


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


def _source_receipt() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    relative_files = (
        "scripts/profile_branch_search_resident_authority.py",
        "src/arti/branch_search.py",
        "src/arti/branch_search_runtime.py",
        "src/arti/gpu_resident.py",
    )
    files = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in relative_files
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", *relative_files],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "git_revision": revision,
        "selected_source_files": files,
        "selected_diff_fingerprint": hashlib.sha256(diff).hexdigest(),
        "source_fingerprint": _fingerprint(
            {
                "git_revision": revision,
                "selected_source_files": files,
                "selected_diff_fingerprint": hashlib.sha256(diff).hexdigest(),
            }
        ),
    }


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _resident_pool(value: torch.Tensor) -> alpha.BoundHotPagePool:
    batch, tokens, dim = value.shape
    page_id = torch.arange(batch, dtype=torch.int64).unsqueeze(1).expand(-1, tokens)
    offset = torch.arange(tokens, dtype=torch.int64).unsqueeze(0).expand(batch, -1)
    support = torch.ones(batch, tokens, dtype=torch.bool)
    refs = alpha.FixedPageRefs(
        logical_slot=offset.contiguous(),
        page_id=page_id.contiguous(),
        offset=offset.contiguous(),
        expected_generation=torch.zeros(batch, tokens, dtype=torch.int64),
        read_mask=support,
        write_mask=support,
        commit_mask=support,
    )
    pool = alpha.HotPagePool(value.detach().clone())
    bucket = alpha.FixedResidentBucket(
        batch_size=batch,
        workset_slots=tokens,
        feature_dim=dim,
        dtype=value.dtype,
        device=value.device,
    )
    return alpha.bind_hot_page_pool(pool, bucket, refs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--slots", type=int, default=128)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=28082027)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(
        args.batch,
        args.tokens,
        args.dim,
        args.slots,
        args.k,
        args.refine_steps,
        args.samples,
    ) <= 0:
        raise ValueError("all numeric arguments must be positive")

    device = torch.device("cuda", torch.cuda.current_device())
    source_receipt = _source_receipt()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    value = torch.randn(args.batch, args.tokens, args.dim, device=device)
    recall = Recall(
        dim=args.dim,
        slots=args.slots,
        formula="arti/delta@1",
        activation="none",
        routing="grouped",
        group_size=4,
        group_topk=args.k,
        key_dim=args.dim,
    ).to(device)
    recall.eval()
    with torch.no_grad():
        recall.state.recall.bank.normal_(std=0.2)
        recall.state.recall.group_bank.normal_(std=0.2)
        result = alpha.run_branch_search(
            recall,
            value,
            max_k=args.k,
            refine_policy=RefinePolicy.fixed(
                args.refine_steps,
                trace_level="routes",
            ),
        )
    executor = alpha.BranchSearchExecutor.from_resident_result(result)
    future = result.value[:, min(1, args.k - 1)].detach().cpu().contiguous()
    runtime = alpha.VolatileTensorRuntime(
        {"state": value.detach().cpu().contiguous()},
        world_id="resident-profile-world",
        store_instance_id="resident-profile-store",
        abi_fingerprint="3" * 64,
        provenance_fingerprint=str(source_receipt["source_fingerprint"]),
    )
    snapshot = runtime.snapshot()
    branch_ids = tuple(f"route-{index}" for index in range(args.k))
    spec = alpha.BranchBatchSpecV2.from_branch_search(
        snapshot,
        executor,
        run_id="resident-profile-1",
        branch_ids=branch_ids,
        input_fingerprint="6" * 64,
        future_tape_fingerprint=alpha.tensor_content_fingerprint(future),
        scorer_ref=alpha.FROZEN_MSE_SCORER_REF,
        scorer_config_fingerprint=alpha.frozen_mse_scorer_config_fingerprint(),
        allowed_write_keys=("state",),
        budgets=tuple(alpha.BranchBudget(1, args.refine_steps) for _ in branch_ids),
    )

    profiled_chain: dict[str, object] = {}

    def execute_once() -> alpha.ResidentBranchCommitReceipt:
        bound = _resident_pool(value)
        try:
            run = alpha.bind_resident_branch_run(
                result,
                executor,
                spec,
                future,
                bound,
            )
            score = run.score()
            decision = run.decide(score, idempotency_key="resident-profile-commit")
            receipt = run.commit(decision)
            profiled_chain.clear()
            profiled_chain.update(
                {
                    "result_manifest_fingerprint": (
                        executor.candidate_manifest_fingerprint
                    ),
                    "executor_config_fingerprint": executor.config_fingerprint,
                    "executor_state_fingerprint": executor.state_fingerprint,
                    "execution_context_fingerprint": (
                        executor.execution_context.fingerprint
                    ),
                    "spec_fingerprint": spec.fingerprint,
                    "run_instance_token": score.run_instance_token,
                    "score_receipt_fingerprint": score.receipt_fingerprint,
                    "decision_fingerprint": decision.decision_fingerprint,
                    "commit_receipt_fingerprint": receipt.receipt_fingerprint,
                    "pool_layout_fingerprint": receipt.pool_layout_fingerprint,
                    "pool_pointer_layout_fingerprint": _fingerprint(
                        bound.pointer_layout_receipt().__dict__
                    ),
                }
            )
            return receipt
        finally:
            bound.close(close_pool=True)

    timings: list[float] = []
    for _ in range(args.samples):
        start = time.perf_counter()
        execute_once()
        torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - start) * 1000.0)

    torch.cuda.synchronize(device)
    allocated_before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    activity = alpha.profile_cuda_callable_activity(
        execute_once,
        device=device,
    )
    peak_allocated = torch.cuda.max_memory_allocated(device)
    element_size = result.value.element_size()
    one_branch_bytes = (
        result.value.shape[0]
        * result.value.shape[2]
        * result.value.shape[3]
        * element_size
    )
    full_k_result_bytes = result.value.numel() * element_size
    full_k_delta_bytes = result.delta.numel() * result.delta.element_size()
    activity_payload = asdict(activity)
    run_binding = {
        **profiled_chain,
        "cuda_trace_fingerprint": activity.trace_fingerprint,
        "source_fingerprint": source_receipt["source_fingerprint"],
    }
    run_binding["binding_fingerprint"] = _fingerprint(run_binding)
    payload = {
        "schema": "arti.branch-search.resident-authority-physical-receipt@1",
        "claim_boundary": (
            "provenance-bound inference-only resident authority transfer receipt; "
            "no training, downstream quality, HBM, or packed active-K claim"
        ),
        "source": source_receipt,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
        },
        "run_binding": run_binding,
        "device": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "shape": list(value.shape),
        "k": args.k,
        "refine_steps": args.refine_steps,
        "samples": args.samples,
        "latency_ms": {
            "p50": statistics.median(timings),
            "p95": _percentile(timings, 0.95),
            "p99": _percentile(timings, 0.99),
        },
        "payload_bytes": {
            "one_branch_value": one_branch_bytes,
            "full_k_value": full_k_result_bytes,
            "full_k_delta": full_k_delta_bytes,
            "future_h2d_expected": future.numel() * future.element_size(),
            "k_score_d2h_expected": args.k * torch.tensor([], dtype=torch.float32).element_size(),
        },
        "cuda_activity": activity_payload,
        "no_full_branch_d2h_observed": activity.d2h_bytes < one_branch_bytes,
        "allocator": {
            "allocated_before": allocated_before,
            "allocated_after": torch.cuda.memory_allocated(device),
            "peak_allocated": peak_allocated,
            "peak_increment": max(0, peak_allocated - allocated_before),
        },
        "hbm_read_bytes": {
            "available": False,
            "value": None,
            "reason": "requires an available Nsight/CUPTI hardware counter",
        },
        "hbm_write_bytes": {
            "available": False,
            "value": None,
            "reason": "requires an available Nsight/CUPTI hardware counter",
        },
    }
    if not payload["no_full_branch_d2h_observed"]:
        raise RuntimeError("resident authority window copied at least one full branch to host")
    payload["receipt_fingerprint"] = _fingerprint(payload)
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
