"""Replay K-wide Branch Search with a frozen trained Z-Image Recall asset.

This is an inference-only engineering receipt.  It uses real Z-Image hidden
traces and a previously trained grouped Recall Bank, but its frozen target is
the canonical full-Top-K Recall output.  The result therefore measures
preservation of trained Reader behavior, not image quality or training gain.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from arti import alpha
from arti.nn import Half, Recall
from arti.recall_refine import RefinePolicy


DEFAULT_OUTPUT = Path(".tmp/branch-search-br4-trained-v1.json")
BOUNDARIES = (6, 14, 22, 29)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_recall(device: torch.device) -> Recall:
    recall = Recall(
        3840,
        1280,
        activation="half",
        routing="grouped",
        key_dim=32,
        group_size=128,
        group_topk=2,
    ).to(device=device, dtype=torch.float32)
    # Half does not follow train/eval mode.  Use its deterministic expected
    # survival path so branch shape cannot change the random-number stream.
    recall.state.recall_activation = Half(stochastic=False).to(device)
    return recall.eval()


def _load_boundary(
    tensors: dict[str, torch.Tensor],
    boundary: int,
    device: torch.device,
) -> Recall:
    prefix = f"layers.{boundary}.attention.to_out.0.adapter.layer."
    recall = _make_recall(device)
    expected = recall.state_dict()
    state: dict[str, torch.Tensor] = {}
    for name in expected:
        source = prefix + name
        if source in tensors:
            state[name] = tensors[source].to(device=device, dtype=expected[name].dtype)
    loaded = recall.load_state_dict(state, strict=False)
    trainable = set(dict(recall.named_parameters()))
    missing_trainable = sorted(trainable.intersection(loaded.missing_keys))
    if missing_trainable:
        raise RuntimeError(
            f"trained asset is missing trainable Recall parameters: {missing_trainable}"
        )
    if loaded.unexpected_keys:
        raise RuntimeError(
            f"trained asset has unexpected Recall parameters: {loaded.unexpected_keys}"
        )
    return recall


def _mse_per_token(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (value - target).float().square().mean(dim=-1)


def _branch_metrics(
    value: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, object]:
    per_token = _mse_per_token(value, target.unsqueeze(1))
    split = max(1, per_token.shape[-1] // 2)
    calibration = per_token[..., :split].mean(dim=-1)
    heldout = per_token[..., split:].mean(dim=-1)
    selected = calibration.argmin(dim=-1)
    selected_error = heldout.gather(1, selected.unsqueeze(-1)).squeeze(-1)
    oracle_error = heldout.min(dim=-1).values
    return {
        "calibration_selected_branch": selected.cpu().tolist(),
        "heldout_selected_mse": float(selected_error.mean().item()),
        "heldout_oracle_mse": float(oracle_error.mean().item()),
        "winner_oracle_gap": float((selected_error - oracle_error).mean().item()),
        "per_branch_heldout_mse": heldout.mean(dim=0).cpu().tolist(),
    }


def _value_metrics(value: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    split = max(1, value.shape[-2] // 2)
    return {
        "full_mse": float(F.mse_loss(value.float(), target.float()).item()),
        "heldout_mse": float(
            F.mse_loss(value[..., split:, :].float(), target[..., split:, :].float()).item()
        ),
    }


def _time_cuda(function, *, warmup: int = 5, repeats: int = 20) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    ordered = sorted(samples)
    return {
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "p99_ms": ordered[-1],
    }


def _evaluate_boundary(
    recall: Recall,
    value: torch.Tensor,
    *,
    refine_steps: int,
) -> dict[str, object]:
    wide_policy = RefinePolicy.fixed(refine_steps, trace_level="routes")
    deep_policy = RefinePolicy.fixed(2 * refine_steps, trace_level="routes")
    one_policy = RefinePolicy.fixed(1, trace_level="routes")
    shuffled = copy.deepcopy(recall)
    reset = copy.deepcopy(recall)
    with torch.no_grad():
        shuffled.state.recall.group_bank.copy_(
            shuffled.state.recall.group_bank.roll(1, dims=0)
        )
        reset.state.recall.bank.zero_()
    with torch.inference_mode():
        canonical = recall(value, refine_policy=wide_policy)
        score_only = recall(value, refine_policy=one_policy)
        candidates = alpha.query_recall_branches(recall, value, max_k=2)
        wide = alpha.run_branch_search(
            recall,
            value,
            candidates=candidates,
            refine_policy=wide_policy,
        )
        single_equal_depth = alpha.run_branch_search(
            recall,
            value,
            max_k=1,
            refine_policy=wide_policy,
        )
        single_deep = alpha.run_branch_search(
            recall,
            value,
            max_k=1,
            refine_policy=deep_policy,
        )

        weights = candidates.selection_weight
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        route_mix = (wide.value * weights.transpose(1, 2).unsqueeze(-1)).sum(dim=1)

        shuffled_wide = alpha.run_branch_search(
            shuffled,
            value,
            max_k=2,
            refine_policy=wide_policy,
        )

        reset_wide = alpha.run_branch_search(
            reset,
            value,
            max_k=2,
            refine_policy=wide_policy,
        )

        wide_diversity = float(
            (wide.value[:, 0] - wide.value[:, 1]).float().square().mean().item()
        )
        route_changed = wide.branch_diagnostics.get("recall_route_history")
        route_change = 0.0
        if route_changed is not None and route_changed.shape[2] > 1:
            route_change = float(
                (route_changed[:, :, 1:] - route_changed[:, :, :-1])
                .abs()
                .float()
                .mean()
                .item()
            )

        latency = {
            "wide": _time_cuda(
                lambda: alpha.run_branch_search(
                    recall,
                    value,
                    candidates=candidates,
                    refine_policy=wide_policy,
                )
            ),
            "single_deep": _time_cuda(
                lambda: alpha.run_branch_search(
                    recall,
                    value,
                    max_k=1,
                    refine_policy=deep_policy,
                )
            ),
        }

    return {
        "canonical_target": "trained full-Top-2 Recall at equal refine depth",
        "input": _value_metrics(value, canonical),
        "score_only_topk": _value_metrics(score_only, canonical),
        "wide": {
            **_branch_metrics(wide.value, canonical),
            "candidate_diversity_mse": wide_diversity,
            "route_change_mean_abs": route_change,
        },
        "wide_route_mix": _value_metrics(route_mix, canonical),
        "single_equal_depth": _branch_metrics(single_equal_depth.value, canonical),
        "single_deep_equal_logical_steps": _branch_metrics(single_deep.value, canonical),
        "shuffled_group_keys": _branch_metrics(shuffled_wide.value, canonical),
        "reset_value_bank": _branch_metrics(reset_wide.value, canonical),
        "latency": latency,
        "logical_token_refine_work": {
            "wide": int(value.shape[0] * 2 * value.shape[1] * refine_steps),
            "single_deep": int(value.shape[0] * value.shape[1] * 2 * refine_steps),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refine-steps", type=int, default=3)
    args = parser.parse_args()
    if args.refine_steps <= 0:
        raise ValueError("refine-steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("trained-asset K-wide Branch Search benchmark requires CUDA")
    device = torch.device("cuda")
    started = time.perf_counter()
    tensors = load_file(args.artifact, device="cpu")
    trace_payload = load_file(args.trace, device="cpu")
    trace = trace_payload["trace"]
    if trace.shape != (len(BOUNDARIES), 108, 3840):
        raise RuntimeError(f"unexpected trace shape: {tuple(trace.shape)}")

    boundaries: dict[str, object] = {}
    for index, boundary in enumerate(BOUNDARIES):
        recall = _load_boundary(tensors, boundary, device)
        value = trace[index].unsqueeze(0).to(device=device, dtype=torch.float32)
        boundaries[str(boundary)] = _evaluate_boundary(
            recall,
            value,
            refine_steps=args.refine_steps,
        )
        del recall, value
        torch.cuda.empty_cache()

    wide = [
        float(item["wide"]["heldout_selected_mse"])
        for item in boundaries.values()
    ]
    deep = [
        float(item["single_deep_equal_logical_steps"]["heldout_selected_mse"])
        for item in boundaries.values()
    ]
    receipt = {
        "format": "arti.branch-search.trained-asset.v1",
        "device": torch.cuda.get_device_name(),
        "artifact_name": args.artifact.name,
        "artifact_sha256": _sha256(args.artifact),
        "trace_name": args.trace.name,
        "trace_sha256": _sha256(args.trace),
        "boundaries": boundaries,
        "aggregate": {
            "wide_selected_heldout_mse_mean": statistics.mean(wide),
            "single_deep_heldout_mse_mean": statistics.mean(deep),
            "wide_minus_single_deep": statistics.mean(wide) - statistics.mean(deep),
            "wide_wins_boundaries": sum(w < d for w, d in zip(wide, deep, strict=True)),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "frozen trained-Reader behavior replay on real Z-Image hidden traces; "
            "not image quality, training gain, or end-to-end downstream efficacy"
        ),
        "half": "deterministic expected survival",
        "hbm_bytes": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
