"""Profile the fixed all-hot volatile tensor runtime S4 CUDA Graph replay window."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from arti import alpha


def _program(*, slots: int, dim: int) -> alpha.FormulaFabricProgram:
    return alpha.FormulaFabricProgram(
        arena_capacity=slots,
        feature_dim=dim,
        steps=((alpha.FormulaInvocation(alpha.FormulaPrimitive.ADD, slots - 1),),),
    )


def _route(*, slots: int, device: torch.device) -> alpha.FormulaRoutePlan:
    weights = torch.zeros(1, 1, 1, 2, slots, device=device)
    weights[..., 0, 0] = 1
    weights[..., 1, 1] = 1
    enabled = torch.ones(1, 1, 1, dtype=torch.bool, device=device)
    return alpha.FormulaRoutePlan(weights, enabled, enabled, enabled)


def _build(
    *,
    workset_slots: int,
    active_count: int,
    feature_dim: int,
    refine_steps: int,
    device: torch.device,
) -> tuple[alpha.BoundHotPagePool, torch.nn.Module]:
    if active_count < 3 or active_count > workset_slots:
        raise ValueError("active_count must be within [3, workset_slots]")
    value = torch.randn(1, workset_slots, feature_dim, device=device)
    pool = alpha.HotPagePool(value)
    bucket = alpha.FixedResidentBucket(
        batch_size=1,
        workset_slots=workset_slots,
        feature_dim=feature_dim,
        dtype=value.dtype,
        device=device,
        refine_steps=refine_steps,
    )
    support = torch.ones(1, workset_slots, dtype=torch.bool)
    refs = alpha.FixedPageRefs(
        logical_slot=torch.arange(workset_slots, dtype=torch.int64).unsqueeze(0),
        page_id=torch.zeros(1, workset_slots, dtype=torch.int64),
        offset=torch.arange(workset_slots, dtype=torch.int64).unsqueeze(0),
        expected_generation=torch.zeros(1, workset_slots, dtype=torch.int64),
        read_mask=support,
        write_mask=support,
        commit_mask=support,
    )
    bound = alpha.bind_hot_page_pool(pool, bucket, refs)
    topology = alpha.ReversibleTopology(
        active_count=active_count,
        policy=alpha.FixedTopologyPolicy(order=list(reversed(range(workset_slots)))),
    ).to(device)
    fold, unfold = topology.operations()
    compute = alpha.FormulaFabricCompute(
        alpha.FormulaFabric(_program(slots=active_count, dim=feature_dim)).to(device),
        active_count=active_count,
    )
    operation = alpha.TopologyFormulaResidentOperation(
        fold,
        unfold,
        compute,
        _route(slots=active_count, device=device),
        refine_steps=refine_steps,
    )
    return bound, torch.compile(operation, fullgraph=True, dynamic=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workset-slots", type=int, default=64)
    parser.add_argument("--active-count", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--refine-steps", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--collect-software-receipt", action="store_true")
    parser.add_argument("--enable-cuda-profiler", action="store_true")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", torch.cuda.current_device())
    bound, operation = _build(
        workset_slots=args.workset_slots,
        active_count=args.active_count,
        feature_dim=args.feature_dim,
        refine_steps=args.refine_steps,
        device=device,
    )

    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup_stream):
        for _ in range(args.warmups):
            bound.eager_step(operation, commit=args.commit)
    torch.cuda.current_stream(device).wait_stream(warmup_stream)
    torch.cuda.synchronize(device)
    captured = bound.capture(operation, commit=args.commit)
    captured.replay()
    torch.cuda.synchronize(device)

    latency = None
    activity = None
    if args.collect_software_receipt:
        latency = alpha.measure_captured_replays(captured, warmups=3, samples=10)
        activity = alpha.profile_captured_cuda_activity(captured, replays=args.replays)

    before = bound.pointer_layout_receipt()
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    if args.enable_cuda_profiler:
        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.replays):
        captured.replay()
    if args.enable_cuda_profiler:
        torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)
    reserved_after = torch.cuda.memory_reserved(device)
    after = bound.pointer_layout_receipt()
    payload = {
        "schema": "arti.volatile_runtime.s4.hot-path-profile@1",
        "bucket": {
            "batch_size": 1,
            "workset_slots": args.workset_slots,
            "active_count": args.active_count,
            "feature_dim": args.feature_dim,
            "dtype": "float32",
            "refine_steps": args.refine_steps,
        },
        "replays": args.replays,
        "pointer_stable": before == after,
        "allocated_before": allocated_before,
        "allocated_after": allocated_after,
        "reserved_before": reserved_before,
        "reserved_after": reserved_after,
        "layout": {
            "shape": list(before.shape),
            "stride": list(before.stride),
            "dtype": before.dtype,
            "device_type": device.type,
        },
        "pool_version_sum": int(bound.pool.version.sum().item()),
        "commit_mode": "commit" if args.commit else "sham-no-write",
        "operation_ref": operation._orig_mod._component_reference,
        "cuda_profiler_region_enabled": args.enable_cuda_profiler,
    }
    if latency is not None and activity is not None:
        payload["latency"] = asdict(latency)
        payload["cuda_activity"] = asdict(activity)
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
