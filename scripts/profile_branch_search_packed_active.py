"""Profile static-capacity versus initial eligible-K packed K-wide Branch Search.

This is an inference-only execution-layout profile. It does not train Recall,
compact branches that stop during refine, or claim downstream quality.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path

import torch

from arti import Recall, RefinePolicy, alpha


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (
        position - lower
    )


def _latency(
    operation: object,
    *,
    device: torch.device,
    warmups: int,
    samples: int,
) -> dict[str, float]:
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize(device)
    timings: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return {
        "p50_ms": statistics.median(timings),
        "p95_ms": _percentile(timings, 0.95),
        "p99_ms": _percentile(timings, 0.99),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--slots", type=int, default=128)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--active-k", type=int, default=2)
    parser.add_argument("--refine-steps", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=28082028)
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
        args.active_k,
        args.refine_steps,
        args.warmups,
        args.samples,
    ) <= 0:
        raise ValueError("all numeric arguments must be positive")
    if args.active_k > args.k:
        raise ValueError("active-k must not exceed k")

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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
    value = torch.randn(args.batch, args.tokens, args.dim, device=device)
    active_k = torch.full(
        (args.batch,), args.active_k, device=device, dtype=torch.int64
    )
    policy = RefinePolicy.fixed(args.refine_steps, trace_level="routes")
    with torch.inference_mode():
        candidates = alpha.query_recall_branches(
            recall,
            value,
            max_k=args.k,
            active_k=active_k,
        )

    static_plan = alpha.BranchSearchPlan.recall_only()
    packed_plan = alpha.BranchSearchPlan.recall_only(
        execution_layout="packed_active"
    )

    def run(plan: alpha.BranchSearchPlan) -> alpha.BranchSearchResult:
        with torch.inference_mode():
            return alpha.run_branch_search(
                recall,
                value,
                candidates=candidates,
                refine_policy=policy,
                plan=plan,
            )

    static_result = run(static_plan)
    packed_result = run(packed_plan)
    parity = {
        "value_max_abs": float(
            (static_result.value - packed_result.value).abs().max().item()
        ),
        "delta_max_abs": float(
            (static_result.delta - packed_result.delta).abs().max().item()
        ),
    }

    def profile(plan: alpha.BranchSearchPlan) -> dict[str, object]:
        def operation() -> alpha.BranchSearchResult:
            return run(plan)

        torch.cuda.synchronize(device)
        allocated_before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        latency = _latency(
            operation,
            device=device,
            warmups=args.warmups,
            samples=args.samples,
        )
        activity = alpha.profile_cuda_callable_activity(
            operation,
            device=device,
        )
        peak = torch.cuda.max_memory_allocated(device)
        observed = operation()
        return {
            "latency": latency,
            "cuda_activity": asdict(activity),
            "allocator": {
                "allocated_before": allocated_before,
                "allocated_after": torch.cuda.memory_allocated(device),
                "peak_increment_bytes": max(0, peak - allocated_before),
            },
            "physical_branch_rows": int(
                observed.global_diagnostics[
                    "batched_physical_branch_rows"
                ].item()
            ),
            "static_capacity_rows": int(
                observed.global_diagnostics[
                    "batched_static_capacity_rows"
                ].item()
            ),
            "eligible_branch_rows": int(
                observed.global_diagnostics[
                    "batched_eligible_branch_rows"
                ].item()
            ),
        }

    profile_payload = {
        "schema": "arti.branch-search.packed-active-profile@1",
        "claim_boundary": (
            "inference-only initial eligible-K layout profile; no training, "
            "downstream quality, stopped-branch compaction, compile, CUDA Graph, "
            "or HBM claim"
        ),
        "device": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "shape": [args.batch, args.tokens, args.dim],
        "slots": args.slots,
        "capacity_k": args.k,
        "active_k": args.active_k,
        "refine_steps": args.refine_steps,
        "parity": parity,
        "static_capacity": profile(static_plan),
        "packed_active": profile(packed_plan),
        "hbm_counters": {
            "available": False,
            "reason": "requires an available Nsight/CUPTI hardware counter",
        },
    }
    encoded = json.dumps(profile_payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
