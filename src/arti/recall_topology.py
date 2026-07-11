"""Open-topology planning and cached-trace screening for Layered Recall."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from .layered_recall import LayerRecall, LayerRecallSpec, LayerRecallStack, LayeredRecallConfig


TRACE_KINDS = ("clean", "corrupt_single", "corrupt_combined", "unseen")


@dataclass(frozen=True)
class LayeredRecallCandidate:
    """One arbitrary topology and its training-side hyperparameters."""

    name: str
    config: LayeredRecallConfig
    abstention_weight: float = 1.0
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if self.abstention_weight < 0:
            raise ValueError("abstention_weight must be non-negative")


@dataclass(frozen=True)
class LayeredRecallBudget:
    """Hard resource limits for topology screening or confirmation batches."""

    max_parameters: int
    max_steps: int
    max_runtime_seconds: float = 120.0
    max_candidates: int | None = None

    def __post_init__(self) -> None:
        if self.max_parameters <= 0 or self.max_steps <= 0 or self.max_runtime_seconds <= 0:
            raise ValueError("parameter, step, and runtime budgets must be positive")
        if self.max_candidates is not None and self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive when provided")


@dataclass(frozen=True)
class LayeredRecallCost:
    """Static cost estimate for one topology."""

    parameters: int
    token_multiply_adds: int
    layer_count: int


@dataclass(frozen=True)
class LayeredRecallScore:
    """Observed offline score; every field is minimized by Pareto selection."""

    candidate: str
    normalized_repair_mse: float
    normalized_unseen_delta_mse: float
    parameters: int
    runtime_seconds: float
    per_layer: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    completed_steps: int = 0
    status: str = "completed"

    @property
    def repair_gain_per_million_parameters(self) -> float:
        return (1.0 - self.normalized_repair_mse) / max(self.parameters / 1_000_000, 1e-12)


@dataclass(frozen=True)
class LayeredRecallTraceCache:
    """Frozen layer-local hidden traces, independent of the source model runtime."""

    tensors: Mapping[str, Tensor]
    layer_dims: Mapping[str, int]
    source: Mapping[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if not self.layer_dims:
            raise ValueError("layer_dims must not be empty")
        expected = {trace_key(path, kind) for path in self.layer_dims for kind in TRACE_KINDS}
        missing = expected - set(self.tensors)
        if missing:
            raise ValueError(f"trace cache is missing tensors: {sorted(missing)}")
        for path, dim in self.layer_dims.items():
            if dim <= 0:
                raise ValueError(f"layer dimension for {path!r} must be positive")
            for kind in TRACE_KINDS:
                tensor = self.tensors[trace_key(path, kind)]
                if tensor.ndim not in {2, 3} or tensor.shape[-1] != dim:
                    raise ValueError(f"invalid {kind} trace shape for {path!r}")

    @property
    def fingerprint(self) -> str:
        metadata = {
            "schema_version": self.schema_version,
            "layer_dims": dict(self.layer_dims),
            "source": dict(self.source),
            "shapes": {key: list(value.shape) for key, value in sorted(self.tensors.items())},
        }
        return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> tuple[Path, Path]:
        weights = Path(path)
        if weights.suffix != ".safetensors":
            weights = weights.with_suffix(".safetensors")
        manifest = weights.with_suffix(".json")
        weights.parent.mkdir(parents=True, exist_ok=True)
        contiguous = {key: value.detach().cpu().contiguous() for key, value in self.tensors.items()}
        save_file(contiguous, str(weights))
        payload = {
            "schema_version": self.schema_version,
            "layer_dims": dict(self.layer_dims),
            "source": dict(self.source),
            "fingerprint": self.fingerprint,
        }
        manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return weights, manifest

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cpu") -> "LayeredRecallTraceCache":
        weights = Path(path)
        if weights.suffix != ".safetensors":
            weights = weights.with_suffix(".safetensors")
        manifest = weights.with_suffix(".json")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        cache = cls(
            tensors=load_file(str(weights), device=str(device)),
            layer_dims={key: int(value) for key, value in payload["layer_dims"].items()},
            source=payload["source"],
            schema_version=int(payload["schema_version"]),
        )
        if cache.fingerprint != payload["fingerprint"]:
            raise ValueError("trace cache fingerprint mismatch")
        return cache


def trace_key(path: str, kind: str) -> str:
    if kind not in TRACE_KINDS:
        raise ValueError(f"unknown trace kind {kind!r}")
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    return f"layer_{digest}.{kind}"


def estimate_layered_recall_cost(
    candidate: LayeredRecallCandidate,
    *,
    tokens: int = 1,
    layer_dims: Mapping[str, int] | None = None,
) -> LayeredRecallCost:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    specs = resolved_specs(candidate.config)
    parameters = 0
    per_token = 0
    for spec in specs:
        dim = spec.dim if spec.dim is not None else (layer_dims or {}).get(spec.path)
        if dim is None:
            raise ValueError(f"candidate layer {spec.path!r} needs dim for static cost estimation")
        recognizer = 2 * spec.rank + 1 if spec.recognition_mode == "alignment" else 0
        parameters += spec.copies * (spec.slots * spec.rank + 2 * dim * spec.rank + (2 * spec.rank + 1) + recognizer)
        per_token += spec.copies * (2 * dim * spec.rank + 2 * spec.slots * spec.rank + 4 * spec.rank)
    return LayeredRecallCost(parameters=parameters, token_multiply_adds=per_token * tokens, layer_count=len(specs))


def candidates_within_budget(
    candidates: Iterable[LayeredRecallCandidate],
    budget: LayeredRecallBudget,
    *,
    tokens: int = 1,
) -> tuple[LayeredRecallCandidate, ...]:
    accepted = []
    for candidate in candidates:
        if estimate_layered_recall_cost(candidate, tokens=tokens).parameters <= budget.max_parameters:
            accepted.append(candidate)
    if budget.max_candidates is not None:
        accepted = accepted[: budget.max_candidates]
    return tuple(accepted)


def pareto_layered_recall(scores: Iterable[LayeredRecallScore]) -> tuple[LayeredRecallScore, ...]:
    values = tuple(scores)
    frontier = []
    for candidate in values:
        dominated = any(
            other is not candidate
            and other.normalized_repair_mse <= candidate.normalized_repair_mse
            and other.normalized_unseen_delta_mse <= candidate.normalized_unseen_delta_mse
            and other.parameters <= candidate.parameters
            and other.runtime_seconds <= candidate.runtime_seconds
            and (
                other.normalized_repair_mse < candidate.normalized_repair_mse
                or other.normalized_unseen_delta_mse < candidate.normalized_unseen_delta_mse
                or other.parameters < candidate.parameters
                or other.runtime_seconds < candidate.runtime_seconds
            )
            for other in values
        )
        if not dominated:
            frontier.append(candidate)
    return tuple(sorted(frontier, key=lambda score: (score.normalized_repair_mse, score.normalized_unseen_delta_mse, score.parameters)))


def screen_layered_recall_candidate(
    candidate: LayeredRecallCandidate,
    cache: LayeredRecallTraceCache,
    budget: LayeredRecallBudget,
    *,
    learning_rate: float = 3e-4,
    device: str | torch.device = "cpu",
) -> LayeredRecallScore:
    """Screen independent layer-local branches without executing the source model.

    This proxy measures whether each proposed branch can repair its cached local
    trace under a fixed budget. It deliberately does not claim to model
    cross-layer propagation; Pareto candidates still require full-model
    confirmation.
    """

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    cost = estimate_layered_recall_cost(candidate, layer_dims=cache.layer_dims)
    if cost.parameters > budget.max_parameters:
        raise ValueError("candidate exceeds max_parameters")
    target_device = torch.device(device)
    branches: dict[str, LayerRecall | LayerRecallStack] = {}
    optimizer_parameters = []
    layer_data = {}
    for spec in resolved_specs(candidate.config):
        dim = spec.dim or cache.layer_dims.get(spec.path)
        if dim is None:
            raise ValueError(f"cache does not provide a dimension for {spec.path!r}")
        if spec.path not in cache.layer_dims:
            raise ValueError(f"cache does not contain layer {spec.path!r}")
        lines = [
            LayerRecall(
                dim,
                rank=spec.rank,
                slots=spec.slots,
                use_half=spec.use_half,
                recognition_mode=spec.recognition_mode,
                recognition_threshold=spec.recognition_threshold,
                recognition_temperature=spec.recognition_temperature,
            )
            for _ in range(spec.copies)
        ]
        with torch.no_grad():
            for line in lines:
                line.emit.weight.zero_()
                if line.recognizer is not None:
                    line.recognizer.bias.fill_(-4.0)
        branch = (lines[0] if spec.copies == 1 else LayerRecallStack(lines, combine=spec.combine)).to(target_device)
        branches[spec.path] = branch
        optimizer_parameters.extend(branch.parameters())
        clean = cache.tensors[trace_key(spec.path, "clean")].to(device=target_device, dtype=torch.float32)
        single = cache.tensors[trace_key(spec.path, "corrupt_single")].to(device=target_device, dtype=torch.float32)
        combined = cache.tensors[trace_key(spec.path, "corrupt_combined")].to(device=target_device, dtype=torch.float32)
        unseen = cache.tensors[trace_key(spec.path, "unseen")].to(device=target_device, dtype=torch.float32)
        layer_data[spec.path] = {
            "clean": clean,
            "single": single,
            "combined": combined,
            "unseen": unseen,
            "single_scale": (single - clean).square().mean().detach().clamp_min(1e-8),
            "combined_scale": (combined - clean).square().mean().detach().clamp_min(1e-8),
        }
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=learning_rate)
    started = time.perf_counter()
    completed_steps = 0
    status = "completed"
    for step in range(budget.max_steps):
        if time.perf_counter() - started >= budget.max_runtime_seconds:
            status = "runtime_budget"
            break
        kind = "single" if step % 2 == 0 else "combined"
        optimizer.zero_grad(set_to_none=True)
        terms = []
        for path, branch in branches.items():
            data = layer_data[path]
            delta = branch(data[kind])
            repair = (data[kind] + delta - data["clean"]).square().mean() / data[f"{kind}_scale"]
            unseen_delta = branch(data["unseen"]).square().mean() / data[f"{kind}_scale"]
            terms.append(repair + candidate.abstention_weight * unseen_delta)
        torch.stack(terms).mean().backward()
        torch.nn.utils.clip_grad_norm_(optimizer_parameters, 1.0)
        optimizer.step()
        completed_steps += 1
    per_layer = {}
    repair_values, unseen_values = [], []
    with torch.no_grad():
        for path, branch in branches.items():
            data = layer_data[path]
            repairs = []
            unseens = []
            for kind in ("single", "combined"):
                delta = branch(data[kind])
                repairs.append(float((data[kind] + delta - data["clean"]).square().mean() / data[f"{kind}_scale"]))
                unseens.append(float(branch(data["unseen"]).square().mean() / data[f"{kind}_scale"]))
            per_layer[path] = {"normalized_repair_mse": sum(repairs) / 2, "normalized_unseen_delta_mse": sum(unseens) / 2}
            repair_values.append(per_layer[path]["normalized_repair_mse"])
            unseen_values.append(per_layer[path]["normalized_unseen_delta_mse"])
    return LayeredRecallScore(
        candidate=candidate.name,
        normalized_repair_mse=sum(repair_values) / len(repair_values),
        normalized_unseen_delta_mse=sum(unseen_values) / len(unseen_values),
        parameters=cost.parameters,
        runtime_seconds=time.perf_counter() - started,
        per_layer=per_layer,
        completed_steps=completed_steps,
        status=status,
    )


def resolved_specs(config: LayeredRecallConfig) -> tuple[LayerRecallSpec, ...]:
    if config.layers:
        return config.layers
    specs = []
    for path in config.layer_paths:
        rank = config.rank[path] if isinstance(config.rank, Mapping) else config.rank
        slots = config.slots[path] if isinstance(config.slots, Mapping) else config.slots
        specs.append(
            LayerRecallSpec(
                path=path,
                rank=int(rank),
                slots=int(slots),
                use_half=config.use_half,
                recognition_mode=config.recognition_mode,
                copies=1,
            )
        )
    return tuple(specs)


__all__ = [
    "TRACE_KINDS",
    "LayeredRecallBudget",
    "LayeredRecallCandidate",
    "LayeredRecallCost",
    "LayeredRecallScore",
    "LayeredRecallTraceCache",
    "candidates_within_budget",
    "estimate_layered_recall_cost",
    "pareto_layered_recall",
    "screen_layered_recall_candidate",
    "resolved_specs",
    "trace_key",
]
