"""Controlled CUDA task for K-wide Recall breadth search.

One Recall query exposes two mutually exclusive latent hypotheses.  Each
hypothesis becomes its own refine trajectory.  A calibration prefix chooses a
trajectory and a held-out suffix measures whether the same hypothesis
continues to solve the task.  The task is analytic and does not train a Bank.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from torch import Tensor

from arti import Recall, RefinePolicy, alpha


def _branch_errors(
    prediction: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    split = prediction.shape[2] // 2
    calibration = (
        prediction[:, :, :split] - target[:, None, :split]
    ).square().mean((2, 3))
    heldout = (
        prediction[:, :, split:] - target[:, None, split:]
    ).square().mean((2, 3))
    selected = calibration.argmin(dim=1)
    selected_error = heldout.gather(1, selected[:, None]).squeeze(1)
    return selected, selected_error, heldout.min(dim=1).values


def _metrics(prediction: Tensor, target: Tensor) -> dict[str, object]:
    selected, selected_error, oracle_error = _branch_errors(prediction, target)
    diversity = (
        prediction[:, 0] - prediction[:, 1]
    ).square().mean() if prediction.shape[1] > 1 else prediction.new_zeros(())
    return {
        "selected_branch": selected.cpu().tolist(),
        "heldout_selected_mse": float(selected_error.mean().item()),
        "heldout_oracle_mse": float(oracle_error.mean().item()),
        "winner_oracle_gap": float((selected_error - oracle_error).mean().item()),
        "candidate_diversity_mse": float(diversity.item()),
    }


def _latency(operation, *, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    ordered = sorted(values)
    return {
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "p99_ms": ordered[-1],
    }


def _make_hypothesis_bank(
    dim: int,
    device: torch.device,
    *,
    group_topk: int = 2,
) -> tuple[Recall, Tensor]:
    recall = Recall(
        dim=dim,
        slots=2,
        formula="arti/delta@1",
        activation="none",
        routing="grouped",
        key_dim=dim,
        group_size=1,
        group_topk=group_topk,
    ).to(device).eval()
    direction = torch.zeros(dim, device=device)
    direction[0] = 1.0
    field = recall.state.recall
    with torch.no_grad():
        field.query.weight.copy_(torch.eye(dim, device=device))
        field.group_bank.copy_(torch.stack((direction, -direction)))
        field.key_bank.copy_(torch.stack((direction, -direction)))
        field.bank.copy_(torch.stack((direction, -direction)))
    return recall, direction


def _single_path_copy(source: Recall, device: torch.device) -> Recall:
    single, _ = _make_hypothesis_bank(source.dim, device, group_topk=1)
    with torch.no_grad():
        source_field = source.state.recall
        target_field = single.state.recall
        target_field.query.weight.copy_(source_field.query.weight)
        target_field.group_bank.copy_(source_field.group_bank)
        target_field.key_bank.copy_(source_field.key_bank)
        target_field.bank.copy_(source_field.bank)
    return single


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--refine-steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=28082026)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".tmp/branch-search-br4-breadth-task-v2.json"),
    )
    args = parser.parse_args()
    if args.batch <= 1 or args.tokens < 2 or args.tokens % 2:
        raise ValueError("batch must exceed one and tokens must be positive and even")
    if args.dim <= 0 or args.refine_steps <= 1:
        raise ValueError("dim must be positive and refine-steps must exceed one")
    if not torch.cuda.is_available():
        raise RuntimeError("the breadth task requires CUDA")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    recall, direction = _make_hypothesis_bank(args.dim, device)
    value = torch.randn(args.batch, args.tokens, args.dim, device=device) * 0.1
    value[..., 0] = 0.0
    signs = torch.where(
        torch.arange(args.batch, device=device) % 2 == 0,
        1.0,
        -1.0,
    )
    target = value + (
        signs[:, None, None]
        * float(args.refine_steps)
        * direction[None, None, :]
    )
    wide_policy = RefinePolicy.fixed(args.refine_steps, trace_level="routes")
    deep_policy = RefinePolicy.fixed(2 * args.refine_steps, trace_level="routes")

    with torch.inference_mode():
        candidates = alpha.query_recall_branches(recall, value, max_k=2)
        wide = alpha.run_branch_search(
            recall,
            value,
            candidates=candidates,
            refine_policy=wide_policy,
        )
        score_only = value[:, None] + candidates.candidate_context.permute(0, 2, 1, 3)
        single_recall = _single_path_copy(recall, device)
        single_candidates = alpha.query_recall_branches(
            single_recall,
            value,
            max_k=1,
        )
        single = alpha.run_branch_search(
            single_recall,
            value,
            candidates=single_candidates,
            refine_policy=deep_policy,
        )

        reset = Recall(
            dim=args.dim,
            slots=2,
            formula="arti/delta@1",
            activation="none",
            routing="grouped",
            key_dim=args.dim,
            group_size=1,
            group_topk=2,
        ).to(device).eval()
        with torch.no_grad():
            reset.state.recall.query.weight.copy_(recall.state.recall.query.weight)
            reset.state.recall.group_bank.copy_(recall.state.recall.group_bank)
            reset.state.recall.key_bank.copy_(recall.state.recall.key_bank)
            reset.state.recall.bank.zero_()
        reset_wide = alpha.run_branch_search(
            reset,
            value,
            max_k=2,
            refine_policy=wide_policy,
        )
        split = args.tokens // 2
        shuffled_lineage = torch.cat(
            (
                wide.value[:, :, :split],
                wide.value[:, :, split:].flip(1),
            ),
            dim=2,
        )

        def wide_call():
            return alpha.run_branch_search(
                recall,
                value,
                candidates=candidates,
                refine_policy=wide_policy,
            )

        def deep_call():
            return alpha.run_branch_search(
                single_recall,
                value,
                candidates=single_candidates,
                refine_policy=deep_policy,
            )
        latency = {
            "wide": _latency(wide_call, warmup=args.warmup, repeats=args.repeats),
            "single_deep": _latency(deep_call, warmup=args.warmup, repeats=args.repeats),
        }

    receipt = {
        "schema": "arti.branch-search.breadth-task@2",
        "claim_boundary": (
            "analytic controlled task for query breadth; no training or downstream model claim"
        ),
        "device": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "shape": [args.batch, args.tokens, args.dim],
        "k": 2,
        "refine_steps": args.refine_steps,
        "single_deep_steps": 2 * args.refine_steps,
        "logical_token_refine_work": {
            "wide": args.batch * 2 * args.tokens * args.refine_steps,
            "single_deep": args.batch * args.tokens * 2 * args.refine_steps,
        },
        "wide": _metrics(wide.value, target),
        "score_only_topk": _metrics(score_only, target),
        "single_deep": _metrics(single.value, target),
        "shuffled_candidate_lineage": _metrics(shuffled_lineage, target),
        "reset_bank": _metrics(reset_wide.value, target),
        "latency": latency,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
