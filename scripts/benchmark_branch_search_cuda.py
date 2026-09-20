"""Controlled CUDA receipt for true K-wide ARTI K-wide Branch Search.

This is a mechanism/cost experiment over a frozen synthetic teacher. It does
not train Recall and does not claim downstream quality.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from arti import Recall, RefinePolicy, alpha


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _profile(
    operation: Callable[[], object],
    *,
    device: torch.device,
    warmups: int,
    samples: int,
) -> dict[str, object]:
    def invoke() -> object:
        with torch.inference_mode():
            return operation()

    for _ in range(warmups):
        invoke()
    torch.cuda.synchronize(device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        invoke()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    peak = torch.cuda.max_memory_allocated(device)

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        invoke()
        torch.cuda.synchronize(device)
    cuda_events = [
        event for event in profiler.events() if str(event.device_type).lower().endswith("cuda")
    ]
    return {
        "warmups": warmups,
        "samples": samples,
        "p50_ms": statistics.median(timings),
        "p95_ms": _percentile(timings, 0.95),
        "p99_ms": _percentile(timings, 0.99),
        "allocated_before": allocated_before,
        "allocated_after": torch.cuda.memory_allocated(device),
        "reserved_before": reserved_before,
        "reserved_after": torch.cuda.memory_reserved(device),
        "peak_increment_bytes": max(0, peak - allocated_before),
        "kernel_events_one_call": len(cuda_events),
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


def _candidate_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    # prediction: [B,K,N,D], target: [B,N,D]
    tokens = prediction.shape[2]
    split = max(1, tokens // 2)
    validation = (prediction[:, :, :split] - target[:, None, :split]).square().mean((2, 3))
    heldout = (prediction[:, :, split:] - target[:, None, split:]).square().mean((2, 3))
    if split == tokens:
        heldout = validation
    winner = validation.argmin(dim=1)
    selected = heldout.gather(1, winner[:, None]).squeeze(1)
    oracle = heldout.min(dim=1).values
    if prediction.shape[1] == 1:
        diversity = prediction.new_zeros(())
    else:
        pairwise = prediction[:, :, None] - prediction[:, None, :]
        upper = torch.triu(
            torch.ones(
                prediction.shape[1],
                prediction.shape[1],
                dtype=torch.bool,
                device=prediction.device,
            ),
            diagonal=1,
        )
        diversity = pairwise.square().mean((3, 4))[:, upper].mean()
    return {
        "validation_selected_heldout_mse": float(selected.mean().item()),
        "heldout_oracle_mse": float(oracle.mean().item()),
        "winner_oracle_gap": float((selected - oracle).mean().item()),
        "candidate_diversity_mse": float(diversity.item()),
    }


def _recall(*, dim: int, slots: int, k: int, device: torch.device) -> Recall:
    recall = Recall(
        dim=dim,
        slots=slots,
        formula="arti/delta@1",
        activation="none",
        routing="grouped",
        group_size=4,
        group_topk=k,
        key_dim=dim,
    ).to(device)
    recall.eval()
    with torch.no_grad():
        recall.state.recall.bank.normal_(std=0.2)
        recall.state.recall.group_bank.normal_(std=0.2)
    return recall


def _formula_operation(
    *,
    batch: int,
    dim: int,
    topology: bool,
    device: torch.device,
) -> alpha.FormulaResidentOperation | alpha.TopologyFormulaResidentOperation:
    """Build one fixed, non-commutative Formula/Topology control operation."""

    program = alpha.FormulaFabricProgram(
        arena_capacity=3,
        feature_dim=dim,
        steps=((alpha.FormulaInvocation(alpha.FormulaPrimitive.ADD, 2),),),
    )
    weights = torch.zeros(batch, 1, 1, 2, 3, device=device)
    weights[..., 0, 0] = 1
    weights[..., 1, 1] = 1
    enabled = torch.ones(batch, 1, 1, dtype=torch.bool, device=device)
    route = alpha.FormulaRoutePlan(weights, enabled, enabled, enabled)
    compute = alpha.FormulaFabricCompute(
        alpha.FormulaFabric(program),
        active_count=3,
    ).to(device)
    if not topology:
        return alpha.FormulaResidentOperation(compute, route).to(device)
    fold, unfold = alpha.ReversibleTopology(
        active_count=3,
        policy=alpha.FixedTopologyPolicy(order=[2, 0, 1]),
    ).operations()
    return alpha.TopologyFormulaResidentOperation(
        fold,
        unfold,
        compute,
        route,
    ).to(device)


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--slots", type=int, default=128)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=28082026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.k < 1 or args.refine_steps < 1:
        raise ValueError("k and refine_steps must be positive")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    recall = _recall(dim=args.dim, slots=args.slots, k=args.k, device=device)
    value = torch.randn(args.batch, args.tokens, args.dim, device=device)
    teacher_matrix = torch.randn(args.dim, args.dim, device=device) / math.sqrt(args.dim)
    target = value + 0.2 * torch.tanh(value @ teacher_matrix)
    policy = RefinePolicy.fixed(args.refine_steps, trace_level="routes")
    deep_policy = RefinePolicy.fixed(
        args.k * args.refine_steps,
        trace_level="routes",
    )

    with torch.no_grad():
        candidates = alpha.query_recall_branches(recall, value, max_k=args.k)
        wide = alpha.run_branch_search(
            recall,
            value,
            candidates=candidates,
            refine_policy=policy,
        )
        score_only_prediction = value[:, None] + candidates.candidate_context.permute(0, 2, 1, 3)
        single_candidates = alpha.query_recall_branches(recall, value, max_k=1)
        single_deep = alpha.run_branch_search(
            recall,
            value,
            candidates=single_candidates,
            refine_policy=deep_policy,
        )

        k1_policy = RefinePolicy.fixed(args.refine_steps, trace_level="routes")
        k1 = alpha.run_branch_search(
            recall,
            value,
            candidates=single_candidates,
            refine_policy=k1_policy,
        )
        direct, _, _ = recall.state(
            value,
            single_candidates.token_mask,
            refine_policy=k1_policy,
            selected_groups=single_candidates.flattened_execution_groups(),
        )
        k1_max_abs = float((k1.value[:, 0] - direct).abs().max().item())

        order = torch.arange(args.k - 1, -1, -1, device=device)
        permuted = alpha.run_branch_search(
            recall,
            value,
            candidates=candidates.permute_branches(order),
            refine_policy=policy,
        )
        permutation_max_abs = float(
            (permuted.value - wide.value.index_select(1, order)).abs().max().item()
        )

        shuffled_recall = copy.deepcopy(recall)
        shuffled_order = torch.randperm(shuffled_recall.state.recall.group_bank.shape[0], device=device)
        shuffled_recall.state.recall.group_bank.copy_(
            shuffled_recall.state.recall.group_bank.index_select(0, shuffled_order)
        )
        shuffled = alpha.run_branch_search(
            shuffled_recall,
            value,
            max_k=args.k,
            refine_policy=policy,
        )
        reset_recall = copy.deepcopy(recall)
        reset_recall.state.recall.bank.zero_()
        reset = alpha.run_branch_search(
            reset_recall,
            value,
            max_k=args.k,
            refine_policy=policy,
        )

        # A separate three-token control makes topology observable: ADD writes
        # slot 0 + slot 1 into slot 2, while Fold@2 changes which original
        # positions occupy those slots. All paths reuse the same candidates and
        # BranchSearch executor.
        control_batch = min(args.batch, 4)
        control_value = torch.randn(control_batch, 3, args.dim, device=device)
        control_recall = _recall(
            dim=args.dim,
            slots=args.slots,
            k=args.k,
            device=device,
        )
        control_candidates = alpha.query_recall_branches(
            control_recall,
            control_value,
            max_k=args.k,
        )
        control_policy = RefinePolicy.fixed(1, trace_level="routes")
        formula_operation = _formula_operation(
            batch=control_batch * args.k,
            dim=args.dim,
            topology=False,
            device=device,
        )
        topology_operation = _formula_operation(
            batch=control_batch * args.k,
            dim=args.dim,
            topology=True,
            device=device,
        )
        control_recall_only = alpha.run_branch_search(
            control_recall,
            control_value,
            candidates=control_candidates,
            refine_policy=control_policy,
        )
        control_formula = alpha.run_branch_search(
            control_recall,
            control_value,
            candidates=control_candidates,
            refine_policy=control_policy,
            plan=alpha.BranchSearchPlan.compose(formula_operation),
        )
        control_topology = alpha.run_branch_search(
            control_recall,
            control_value,
            candidates=control_candidates,
            refine_policy=control_policy,
            plan=alpha.BranchSearchPlan.compose(topology_operation),
        )
        control_topology_repeat = alpha.run_branch_search(
            control_recall,
            control_value,
            candidates=control_candidates,
            refine_policy=control_policy,
            plan=alpha.BranchSearchPlan.compose(topology_operation),
        )

    def formula_control_call() -> object:
        return alpha.run_branch_search(
            control_recall,
            control_value,
            candidates=control_candidates,
            refine_policy=control_policy,
            plan=alpha.BranchSearchPlan.compose(formula_operation),
        )

    def topology_control_call() -> object:
        return alpha.run_branch_search(
            control_recall,
            control_value,
            candidates=control_candidates,
            refine_policy=control_policy,
            plan=alpha.BranchSearchPlan.compose(topology_operation),
        )

    def score_only_call() -> object:
        return alpha.query_recall_branches(recall, value, max_k=args.k)

    def wide_call() -> object:
        return alpha.run_branch_search(
            recall,
            value,
            candidates=candidates,
            refine_policy=policy,
        )

    def deep_call() -> object:
        return alpha.run_branch_search(
            recall,
            value,
            candidates=single_candidates,
            refine_policy=deep_policy,
        )

    receipt = {
        "schema": "arti.branch-search.br6.cuda@2",
        "claim_boundary": (
            "frozen synthetic mechanism/cost receipt; no training and no downstream efficacy claim"
        ),
        "device": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "shape": [args.batch, args.tokens, args.dim],
        "slots": args.slots,
        "k": args.k,
        "branch_search_runtime_steps": args.refine_steps,
        "single_deep_steps": args.k * args.refine_steps,
        "logical_refine_token_work": {
            "wide": args.batch * args.k * args.tokens * args.refine_steps,
            "single_deep": args.batch * args.tokens * args.k * args.refine_steps,
        },
        "score_only": _candidate_metrics(score_only_prediction, target),
        "wide": _candidate_metrics(wide.value, target),
        "single_deep": _candidate_metrics(single_deep.value, target),
        "shuffled_candidate": _candidate_metrics(shuffled.value, target),
        "reset_candidate": _candidate_metrics(reset.value, target),
        "active_branches": int(candidates.branch_mask.sum().item()),
        "k1_max_abs_error": k1_max_abs,
        "candidate_permutation_max_abs_error": permutation_max_abs,
        "formula_topology_control": {
            "shape": [control_batch, 3, args.dim],
            "k": args.k,
            "teacher": "fixed TopologyFold@2 -> ADD Formula -> TopologyUnFold@2",
            "recall_only": _candidate_metrics(
                control_recall_only.value,
                control_topology.value[:, 0],
            ),
            "formula_only": _candidate_metrics(
                control_formula.value,
                control_topology.value[:, 0],
            ),
            "topology_formula": _candidate_metrics(
                control_topology.value,
                control_topology.value[:, 0],
            ),
            "formula_vs_topology_mse": float(
                (control_formula.value - control_topology.value).square().mean().item()
            ),
            "deterministic_repeat_max_abs_error": float(
                (control_topology.value - control_topology_repeat.value).abs().max().item()
            ),
            "operation_ref": control_topology.operation_ref,
            "formula_ref": control_topology.candidates.formula_ref,
            "topology_ref": list(control_topology.topology_refs),
            "topology_unfold_verified": bool(
                control_topology.branch_diagnostics[
                    "batched_topology_unfold_verified"
                ].all()
            ),
        },
        "physical": {
            "score_only": _profile(
                score_only_call,
                device=device,
                warmups=args.warmups,
                samples=args.samples,
            ),
            "wide": _profile(
                wide_call,
                device=device,
                warmups=args.warmups,
                samples=args.samples,
            ),
            "single_deep": _profile(
                deep_call,
                device=device,
                warmups=args.warmups,
                samples=args.samples,
            ),
            "formula_control": _profile(
                formula_control_call,
                device=device,
                warmups=args.warmups,
                samples=args.samples,
            ),
            "topology_formula_control": _profile(
                topology_control_call,
                device=device,
                warmups=args.warmups,
                samples=args.samples,
            ),
        },
        "provenance": {
            "script_sha256": _source_sha256(),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
    }
    encoded = json.dumps(receipt, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
