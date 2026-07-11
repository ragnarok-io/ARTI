"""Scoped Recall artifacts and label-free episodic test-time adaptation."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
import torch.nn as nn
from torch import Tensor

from .serialization import ARTILoadResult, ARTISaveResult, load, save


RECALL_ARTIFACT_KIND = "arti.recall-expert"
RECALL_ARTIFACT_VERSION = 1
DEFAULT_ALLOWED_SIGNALS = ("masked_reconstruction", "next_token", "consistency")
_FORBIDDEN_SUPPORT_FIELDS = ("label", "target", "answer", "query")
_OVERFLOW_POLICIES = ("abstain", "shard", "allow")


@dataclass(frozen=True)
class RecallCapacityDecision:
    """A deterministic routing decision for one Recall support episode."""

    requested_items: int
    accepted_items: int
    dropped_items: int
    expert_item_counts: tuple[int, ...]
    overflowed: bool
    protected: bool


@dataclass(frozen=True)
class RecallCapacityPlan:
    """Declare bounded Recall capacity and a deterministic overflow policy.

    ``abstain`` rejects an episode that does not fit. ``shard`` assigns the
    fitting prefix across the declared experts and rejects the remainder.
    ``allow`` is intentionally unprotected and is useful only as a control
    control for measuring interference beyond the declared capacity.
    """

    slots_per_expert: int
    experts: int = 1
    overflow_policy: str = "abstain"

    def validate(self) -> "RecallCapacityPlan":
        if self.slots_per_expert <= 0 or self.experts <= 0:
            raise ValueError("Recall capacity slots_per_expert and experts must be positive")
        if self.overflow_policy not in _OVERFLOW_POLICIES:
            raise ValueError(f"overflow_policy must be one of {_OVERFLOW_POLICIES}")
        return self

    @property
    def total_capacity(self) -> int:
        return self.slots_per_expert * self.experts

    def decide(self, item_count: int) -> RecallCapacityDecision:
        """Route a support count without inspecting labels or query state."""

        self.validate()
        if item_count < 0:
            raise ValueError("item_count must be non-negative")
        overflowed = item_count > self.total_capacity
        if overflowed and self.overflow_policy == "abstain":
            return RecallCapacityDecision(item_count, 0, item_count, (), True, True)
        accepted = item_count if self.overflow_policy == "allow" else min(item_count, self.total_capacity)
        counts = []
        remaining = accepted
        for _ in range(self.experts):
            count = min(self.slots_per_expert, remaining)
            counts.append(count)
            remaining -= count
        return RecallCapacityDecision(
            requested_items=item_count,
            accepted_items=accepted,
            dropped_items=item_count - accepted,
            expert_item_counts=tuple(counts),
            overflowed=overflowed,
            protected=self.overflow_policy != "allow",
        )


@dataclass(frozen=True)
class RecallArtifactSpec:
    """Declarative compatibility and provenance data for a Recall expert."""

    capability: str
    base_model_fingerprint: str
    injection_fingerprint: str
    allowed_signals: tuple[str, ...] = DEFAULT_ALLOWED_SIGNALS
    visibility_policy: str = "caller_supplied"
    capacity_plan: RecallCapacityPlan | None = None
    training_metadata: Mapping[str, Any] | None = None

    def validate(self) -> "RecallArtifactSpec":
        if not self.capability or not self.capability.strip():
            raise ValueError("Recall artifact capability must be non-empty")
        if not _is_sha256(self.base_model_fingerprint):
            raise ValueError("Recall artifact base_model_fingerprint must be a SHA-256 value")
        if not _is_sha256(self.injection_fingerprint):
            raise ValueError("Recall artifact injection_fingerprint must be a SHA-256 value")
        if not self.allowed_signals or any(signal not in DEFAULT_ALLOWED_SIGNALS for signal in self.allowed_signals):
            raise ValueError(f"Recall artifact allowed_signals must be chosen from {DEFAULT_ALLOWED_SIGNALS}")
        if not self.visibility_policy:
            raise ValueError("Recall artifact visibility_policy must be non-empty")
        if self.capacity_plan is not None:
            self.capacity_plan.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["allowed_signals"] = list(self.allowed_signals)
        if self.capacity_plan is not None:
            payload["capacity_plan"] = asdict(self.capacity_plan)
        return payload


@dataclass(frozen=True)
class RecallTTTRecord:
    """Auditable result of one label-free Recall adaptation transaction."""

    steps: int
    allowed_signal: str
    support_examples: int
    initial_weights_sha256: str
    final_weights_sha256: str
    backbone_sha256_before: str
    backbone_sha256_after: str
    final_loss: float
    elapsed_seconds: float
    trainable_parameters: tuple[str, ...] = ()
    rolled_back: bool = False


def module_structure_fingerprint(module: nn.Module) -> str:
    """Fingerprint module topology and state shapes without learned values."""

    payload = {
        "class": f"{module.__class__.__module__}.{module.__class__.__qualname__}",
        "state": [
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in module.state_dict().items()
        ],
    }
    return _sha256_json(payload)


def recall_artifact_path(path: str | Path) -> Path:
    target = Path(path)
    if not target.name.endswith(".recall.arti.st"):
        raise ValueError("Recall expert artifacts must end in '.recall.arti.st' (for example coder.recall.arti.st)")
    return target


def export_recall_artifact(expert: nn.Module, path: str | Path, spec: RecallArtifactSpec) -> ARTISaveResult:
    """Export a dedicated Recall expert package with strict compatibility metadata."""

    target = recall_artifact_path(path)
    normalized = spec.validate()
    return save(
        expert,
        target,
        scope="all",
        config={
            "artifact_kind": RECALL_ARTIFACT_KIND,
            "artifact_version": RECALL_ARTIFACT_VERSION,
            "recall_expert": normalized.to_dict(),
        },
    )


def load_recall_artifact(
    path: str | Path,
    expert: nn.Module,
    *,
    base_model: nn.Module,
    injection_module: nn.Module | None = None,
    map_location: str | torch.device = "cpu",
) -> ARTILoadResult:
    """Validate and restore a Recall expert only when its host is compatible."""

    target = recall_artifact_path(path)
    # Validate package and compatibility before mutating the active expert.
    loaded = load(target, map_location=map_location)
    config = loaded.manifest.get("architecture", {}).get("config", {})
    if config.get("artifact_kind") != RECALL_ARTIFACT_KIND or config.get("artifact_version") != RECALL_ARTIFACT_VERSION:
        raise ValueError("artifact is not a Recall expert package")
    recorded = config.get("recall_expert")
    if not isinstance(recorded, Mapping):
        raise ValueError("Recall expert artifact is missing recall_expert metadata")
    spec = RecallArtifactSpec(
        capability=str(recorded.get("capability", "")),
        base_model_fingerprint=str(recorded.get("base_model_fingerprint", "")),
        injection_fingerprint=str(recorded.get("injection_fingerprint", "")),
        allowed_signals=tuple(recorded.get("allowed_signals", ())),
        visibility_policy=str(recorded.get("visibility_policy", "")),
        capacity_plan=_capacity_plan_from_payload(recorded.get("capacity_plan")),
        training_metadata=recorded.get("training_metadata"),
    ).validate()
    if spec.base_model_fingerprint != module_structure_fingerprint(base_model):
        raise ValueError("Recall artifact base model fingerprint does not match the active model")
    injection = expert if injection_module is None else injection_module
    if spec.injection_fingerprint != module_structure_fingerprint(injection):
        raise ValueError("Recall artifact injection fingerprint does not match the active Recall module")
    return load(target, model=expert, map_location=map_location, strict=True)


class RecallExpertRegistry:
    """Thread-safe active Recall expert switcher with rollback to a known state."""

    def __init__(self, expert: nn.Module, *, base_model: nn.Module, injection_module: nn.Module | None = None) -> None:
        self.expert = expert
        self.base_model = base_model
        self.injection_module = expert if injection_module is None else injection_module
        self._baseline = _clone_state(expert)
        self._active_path: Path | None = None
        self._lock = threading.RLock()

    @property
    def active_path(self) -> Path | None:
        return self._active_path

    def activate(self, path: str | Path, *, map_location: str | torch.device = "cpu") -> ARTILoadResult:
        with self._lock:
            previous = _clone_state(self.expert)
            try:
                loaded = load_recall_artifact(path, self.expert, base_model=self.base_model, injection_module=self.injection_module, map_location=map_location)
            except Exception:
                self.expert.load_state_dict(previous, strict=True)
                raise
            self._active_path = recall_artifact_path(path)
            return loaded

    def rollback(self) -> None:
        with self._lock:
            self.expert.load_state_dict(self._baseline, strict=True)
            self._active_path = None


class RecallExpertPool(nn.Module):
    """Keep multiple compatible Recall artifacts resident at the same time.

    A pool can route directly to one named expert or mix all loaded experts
    with global ``[E]`` or per-batch ``[B, E]`` weights. Mixing is recursive
    for Tensor, tuple/list, and mapping outputs so low-level Recall modules can
    preserve their diagnostics alongside the recalled tensor.
    """

    def __init__(self, template: nn.Module, *, base_model: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "_base_model_ref", weakref.ref(base_model))
        self.experts = nn.ModuleDict()
        self._paths: dict[str, Path] = {}
        self._factory = lambda: copy.deepcopy(template)
        self._template_fingerprint = module_structure_fingerprint(template)
        self._lock = threading.RLock()

    @property
    def loaded_experts(self) -> tuple[str, ...]:
        return tuple(self.experts.keys())

    def artifact_path(self, name: str) -> Path:
        if name not in self._paths:
            raise KeyError(f"unknown Recall expert {name!r}")
        return self._paths[name]

    def load_expert(
        self,
        name: str,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> ARTILoadResult:
        """Validate and add one artifact without replacing resident experts."""

        _validate_expert_name(name)
        with self._lock:
            if name in self.experts:
                raise ValueError(f"Recall expert {name!r} is already loaded")
            expert = self._factory()
            if module_structure_fingerprint(expert) != self._template_fingerprint:
                raise RuntimeError("Recall expert factory produced an incompatible module structure")
            if str(map_location) != "cpu":
                expert = expert.to(map_location)
            loaded = load_recall_artifact(
                path,
                expert,
                base_model=self._require_base_model(),
                injection_module=expert,
                map_location=map_location,
            )
            self.experts[name] = expert
            self._paths[name] = recall_artifact_path(path)
            return loaded

    def _require_base_model(self) -> nn.Module:
        base_model = self._base_model_ref()
        if base_model is None:
            raise RuntimeError("RecallExpertPool base model is no longer available")
        return base_model

    def unload(self, name: str) -> nn.Module:
        """Remove and return a resident expert."""

        with self._lock:
            if name not in self.experts:
                raise KeyError(f"unknown Recall expert {name!r}")
            expert = self.experts[name]
            del self.experts[name]
            del self._paths[name]
            return expert

    def concatenate(self, parameter: str = "bank", *, dim: int = 0) -> nn.Module:
        """Build one Recall module by concatenating resident trace banks.

        All non-concatenated state must be byte-identical. This keeps query,
        gate, and recognition operators shared while expanding the native
        Recall address space from ``K`` to ``sum(K_i)`` slots. The returned
        module performs one retrieval pass; it does not remix expert outputs.
        """

        if not self.experts:
            raise RuntimeError("RecallExpertPool has no loaded experts")
        modules = list(self.experts.values())
        states = [module.state_dict() for module in modules]
        if parameter not in states[0]:
            raise ValueError(f"Recall concat parameter {parameter!r} does not exist")
        keys = tuple(states[0].keys())
        if any(tuple(state.keys()) != keys for state in states[1:]):
            raise ValueError("Recall experts do not have matching state structure")
        for key in keys:
            if key == parameter:
                continue
            reference = states[0][key]
            if any(not torch.equal(reference, state[key].to(reference)) for state in states[1:]):
                raise ValueError(
                    f"Recall experts differ at shared state {key!r}; direct concat requires shared operators and bank-only specialization"
                )
        banks = [state[parameter] for state in states]
        if any(bank.ndim != banks[0].ndim for bank in banks[1:]):
            raise ValueError("Recall banks must have matching rank")
        if dim < -banks[0].ndim or dim >= banks[0].ndim:
            raise ValueError(f"concat dim {dim} is out of range for Recall bank rank {banks[0].ndim}")
        axis = dim % banks[0].ndim
        expected = list(banks[0].shape)
        for bank in banks[1:]:
            for index, size in enumerate(bank.shape):
                if index != axis and size != expected[index]:
                    raise ValueError("Recall banks must match on every non-concatenated dimension")

        merged = copy.deepcopy(modules[0])
        parent, leaf = _resolve_module_state_parent(merged, parameter)
        original = getattr(parent, leaf)
        if not isinstance(original, nn.Parameter):
            raise ValueError("Recall concat currently requires a Parameter bank")
        joined = torch.cat([bank.to(original) for bank in banks], dim=axis)
        setattr(parent, leaf, nn.Parameter(joined, requires_grad=original.requires_grad))
        return merged

    def forward(
        self,
        *args: Any,
        expert: str | None = None,
        mixture_weights: Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Route to one expert or blend every resident expert."""

        if not self.experts:
            raise RuntimeError("RecallExpertPool has no loaded experts")
        if expert is not None:
            if mixture_weights is not None:
                raise ValueError("expert and mixture_weights are mutually exclusive")
            if expert not in self.experts:
                raise KeyError(f"unknown Recall expert {expert!r}")
            return self.experts[expert](*args, **kwargs)
        if mixture_weights is None:
            raise ValueError("choose a named expert, pass explicit mixture_weights, or call pool.concatenate() for native Recall bank scaling")
        outputs = [module(*args, **kwargs) for module in self.experts.values()]
        weights = _normalize_mixture_weights(mixture_weights, len(outputs), outputs)
        return _mix_outputs(outputs, weights)


class RecallTTTSession:
    """An episodic transaction that updates only a Recall expert from support input.

    The public API has no query argument. Support mappings containing label,
    target, answer, or query fields are rejected before the user-supplied
    self-supervised objective runs.
    """

    def __init__(
        self,
        model: nn.Module,
        expert: nn.Module,
        support_loss: Callable[[nn.Module, Mapping[str, Any]], torch.Tensor],
        *,
        allowed_signals: Iterable[str] = DEFAULT_ALLOWED_SIGNALS,
        trainable_parameters: Iterable[str] | None = None,
    ) -> None:
        self.model = model
        self.expert = expert
        self.support_loss = support_loss
        self.allowed_signals = tuple(allowed_signals)
        if not self.allowed_signals or any(signal not in DEFAULT_ALLOWED_SIGNALS for signal in self.allowed_signals):
            raise ValueError(f"allowed_signals must be chosen from {DEFAULT_ALLOWED_SIGNALS}")
        named_expert_parameters = dict(expert.named_parameters())
        if not named_expert_parameters:
            raise ValueError("Recall TTT expert must have trainable parameters")
        if trainable_parameters is None:
            self.trainable_parameter_names = tuple(named_expert_parameters)
        else:
            self.trainable_parameter_names = tuple(trainable_parameters)
            unknown = set(self.trainable_parameter_names) - set(named_expert_parameters)
            if unknown:
                raise ValueError(f"unknown Recall TTT trainable parameters: {sorted(unknown)}")
            if not self.trainable_parameter_names:
                raise ValueError("Recall TTT trainable_parameters must not be empty")
        self._trainable_parameters = [named_expert_parameters[name] for name in self.trainable_parameter_names]
        expert_ids = {id(parameter) for parameter in self._trainable_parameters}
        for parameter in model.parameters():
            parameter.requires_grad_(id(parameter) in expert_ids)
        self._initial_expert = _clone_state(expert)
        self._backbone_before = _state_hash(model, exclude_parameter_ids=expert_ids)
        self._last_record: RecallTTTRecord | None = None

    @property
    def last_record(self) -> RecallTTTRecord | None:
        return self._last_record

    def adapt(
        self,
        support_batches: Iterable[Mapping[str, Any]],
        *,
        steps: int,
        learning_rate: float,
        allowed_signal: str = "consistency",
        reset: bool = True,
    ) -> RecallTTTRecord:
        if steps <= 0 or learning_rate <= 0:
            raise ValueError("steps and learning_rate must be positive")
        if allowed_signal not in self.allowed_signals:
            raise ValueError(f"allowed_signal {allowed_signal!r} is not permitted by this Recall session")
        batches = list(support_batches)
        if not batches:
            raise ValueError("Recall TTT requires at least one support batch")
        for batch in batches:
            _validate_support_batch(batch)
        if reset:
            self.expert.load_state_dict(self._initial_expert, strict=True)
        initial_hash = _state_hash(self.expert)
        optimizer = torch.optim.AdamW(self._trainable_parameters, lr=learning_rate)
        started = time.perf_counter()
        final_loss = 0.0
        self.model.train()
        for index in range(steps):
            loss = self.support_loss(self.model, batches[index % len(batches)])
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise ValueError("Recall TTT support_loss must return a scalar Tensor")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
        self.model.eval()
        backbone_after = _state_hash(self.model, exclude_parameter_ids={id(parameter) for parameter in self._trainable_parameters})
        if backbone_after != self._backbone_before:
            self.rollback()
            raise RuntimeError("Recall TTT changed frozen backbone state; transaction rolled back")
        record = RecallTTTRecord(
            steps=steps,
            allowed_signal=allowed_signal,
            support_examples=_support_example_count(batches),
            initial_weights_sha256=initial_hash,
            final_weights_sha256=_state_hash(self.expert),
            backbone_sha256_before=self._backbone_before,
            backbone_sha256_after=backbone_after,
            final_loss=final_loss,
            elapsed_seconds=time.perf_counter() - started,
            trainable_parameters=self.trainable_parameter_names,
        )
        self._last_record = record
        return record

    def rollback(self) -> RecallTTTRecord | None:
        self.expert.load_state_dict(self._initial_expert, strict=True)
        if self._last_record is None:
            return None
        self._last_record = RecallTTTRecord(**{**asdict(self._last_record), "rolled_back": True})
        return self._last_record


def _validate_support_batch(batch: Mapping[str, Any]) -> None:
    for key, value in batch.items():
        lowered = str(key).lower()
        if any(marker in lowered for marker in _FORBIDDEN_SUPPORT_FIELDS):
            raise ValueError(f"Recall TTT support batch may not contain label-bearing field {key!r}")
        if isinstance(value, Mapping):
            _validate_support_batch(value)


def _support_example_count(batches: list[Mapping[str, Any]]) -> int:
    total = 0
    for batch in batches:
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                total += int(value.shape[0])
                break
    return total


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


def _state_hash(module: nn.Module, *, exclude_parameter_ids: set[int] | None = None) -> str:
    excluded = exclude_parameter_ids or set()
    parameter_ids = {name: id(parameter) for name, parameter in module.named_parameters()}
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        if parameter_ids.get(name) in excluded:
            continue
        contiguous = tensor.detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _validate_expert_name(name: str) -> None:
    if not name or not name.strip() or "." in name:
        raise ValueError("Recall expert name must be non-empty and may not contain '.'")


def _resolve_module_state_parent(module: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = module
    for part in parts[:-1]:
        if part not in parent._modules:
            raise ValueError(f"Recall concat parameter path {name!r} is not a module Parameter")
        parent = parent._modules[part]
    return parent, parts[-1]


def _normalize_mixture_weights(weights: Tensor | None, count: int, outputs: list[Any]) -> Tensor:
    reference = _first_tensor(outputs[0])
    if weights is None:
        raise ValueError("mixture_weights are required for output mixing")
    weights = weights.to(device=reference.device, dtype=reference.dtype)
    if weights.ndim not in {1, 2} or weights.shape[-1] != count:
        raise ValueError(f"mixture_weights must have shape [{count}] or [B, {count}]")
    if bool((weights < 0).any()):
        raise ValueError("mixture_weights must be non-negative")
    normalizer = weights.sum(dim=-1, keepdim=True)
    if bool((normalizer <= 0).any()):
        raise ValueError("mixture_weights must contain positive mass")
    return weights / normalizer


def _first_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)) and value:
        return _first_tensor(value[0])
    if isinstance(value, Mapping) and value:
        return _first_tensor(next(iter(value.values())))
    raise TypeError("Recall expert output must contain at least one Tensor")


def _mix_outputs(outputs: list[Any], weights: Tensor) -> Any:
    first = outputs[0]
    if isinstance(first, Tensor):
        if any(not isinstance(output, Tensor) or output.shape != first.shape for output in outputs):
            raise TypeError("all Recall expert Tensor outputs must have matching shapes")
        stacked = torch.stack(outputs, dim=-1)
        if weights.ndim == 1:
            return (stacked * weights).sum(dim=-1)
        if first.ndim == 0 or first.shape[0] != weights.shape[0]:
            raise ValueError("per-batch mixture weights require Tensor outputs with matching leading batch size")
        shaped = weights.view(weights.shape[0], *([1] * (first.ndim - 1)), weights.shape[1])
        return (stacked * shaped).sum(dim=-1)
    if isinstance(first, tuple):
        if any(not isinstance(output, tuple) or len(output) != len(first) for output in outputs):
            raise TypeError("all Recall expert tuple outputs must have matching structure")
        return tuple(_mix_outputs([output[index] for output in outputs], weights) for index in range(len(first)))
    if isinstance(first, list):
        if any(not isinstance(output, list) or len(output) != len(first) for output in outputs):
            raise TypeError("all Recall expert list outputs must have matching structure")
        return [_mix_outputs([output[index] for output in outputs], weights) for index in range(len(first))]
    if isinstance(first, Mapping):
        keys = tuple(first.keys())
        if any(not isinstance(output, Mapping) or tuple(output.keys()) != keys for output in outputs):
            raise TypeError("all Recall expert mapping outputs must have matching keys")
        return {key: _mix_outputs([output[key] for output in outputs], weights) for key in keys}
    raise TypeError("Recall expert outputs must be Tensor, tuple/list, or mapping structures")


def _capacity_plan_from_payload(value: Any) -> RecallCapacityPlan | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Recall artifact capacity_plan must be a mapping")
    return RecallCapacityPlan(
        slots_per_expert=int(value.get("slots_per_expert", 0)),
        experts=int(value.get("experts", 1)),
        overflow_policy=str(value.get("overflow_policy", "abstain")),
    )
