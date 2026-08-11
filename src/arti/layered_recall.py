"""Layer-addressed Recall branches for frozen Transformer-style backbones."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .fit.insertion import get_parent_module, set_child_module
from .fit.scanner import run_model, scan_model
from .nn import Half
from .recall_workspace import RecallWorkspace
from .recall_artifacts import (
    RecallArtifactSpec,
    RecallExpertPool,
    export_recall_artifact,
    load_recall_artifact,
    module_structure_fingerprint,
)
from .serialization import load as load_arti
from .tensor_boundary import TensorLayout, find_primary_tensor, replace_tensor_at_path


class LayerRecall(nn.Module):
    """Low-rank candidate-trace Recall used at one named hidden layer."""

    def __init__(
        self,
        dim: int,
        *,
        rank: int = 16,
        slots: int = 8,
        use_half: bool = True,
        recognition_mode: str = "none",
        recognition_threshold: float = 0.5,
        recognition_temperature: float = 0.1,
        workspace: RecallWorkspace | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0 or rank <= 0 or slots <= 0:
            raise ValueError("dim, rank, and slots must be positive")
        if recognition_mode not in {"explicit", "alignment", "none"}:
            raise ValueError("recognition_mode must be 'explicit', 'alignment', or 'none'")
        if recognition_temperature <= 0:
            raise ValueError("recognition_temperature must be positive")
        self.dim = int(dim)
        self.rank = int(rank)
        self.slots = int(slots)
        self.use_half = bool(use_half)
        self.recognition_mode = recognition_mode
        if workspace is not None and workspace.dim != rank:
            raise ValueError("workspace feature dimension must match rank")
        self.workspace = workspace
        self.bank = nn.Parameter(torch.randn(slots, rank) * rank**-0.5)
        self.query = nn.Linear(dim, rank, bias=False)
        self.emit = nn.Linear(rank, dim, bias=False)
        self.recognizer = nn.Linear(rank * 2, 1) if recognition_mode == "alignment" else None
        self.survival = Half() if use_half else nn.Identity()
        self.register_buffer("recognition_threshold", torch.tensor(float(recognition_threshold)), persistent=False)
        self.register_buffer("recognition_temperature", torch.tensor(float(recognition_temperature)), persistent=False)

    def forward(
        self,
        hidden: Tensor,
        *,
        mask: Tensor | None = None,
        return_info: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if hidden.ndim not in {2, 3} or hidden.shape[-1] != self.dim:
            raise ValueError(f"hidden must have shape [B, {self.dim}] or [B, N, {self.dim}]")
        sequence = hidden.unsqueeze(1) if hidden.ndim == 2 else hidden
        valid = _mask_for(sequence, mask)
        query = self.query(sequence)
        bank = self.bank.to(device=query.device, dtype=query.dtype)
        logits = torch.einsum("bnr,kr->bnk", query, bank) * self.rank**-0.5
        weights = torch.softmax(logits, dim=-1)
        direct_context = torch.einsum("bnk,kr->bnr", weights, bank)
        workspace_info: dict[str, Tensor] | None = None
        if self.workspace is None:
            context = direct_context
        else:
            batch, tokens, slots = weights.shape
            candidates = weights.unsqueeze(-1) * bank.reshape(1, 1, slots, self.rank)
            candidates = candidates.reshape(batch, tokens * slots, self.rank)
            candidate_mask = valid.unsqueeze(-1).expand(-1, -1, slots).reshape(batch, tokens * slots)
            context, workspace_info = self.workspace(
                query,
                candidates,
                query_mask=valid,
                candidate_mask=candidate_mask,
                return_info=True,
            )
        features = torch.cat([query, context], dim=-1)
        if self.recognition_mode == "alignment":
            assert self.recognizer is not None
            recognition = torch.sigmoid(self.recognizer(features)).squeeze(-1)
        elif self.recognition_mode == "explicit":
            similarity = F.cosine_similarity(query, context, dim=-1, eps=1e-6)
            temperature = self.recognition_temperature.to(sequence).clamp_min(torch.finfo(sequence.dtype).eps)
            recognition = torch.sigmoid((similarity - self.recognition_threshold.to(sequence)) / temperature)
        else:
            recognition = torch.ones_like(logits[..., 0])
        recognition = recognition * valid.to(sequence.dtype)
        raw_delta = self.emit(context) * recognition.unsqueeze(-1)
        delta = self.survival(raw_delta) * valid.unsqueeze(-1).to(sequence.dtype)
        if hidden.ndim == 2:
            delta = delta[:, 0]
            raw_delta = raw_delta[:, 0]
            recognition = recognition[:, 0]
            weights = weights[:, 0]
        if not return_info:
            return delta
        info = {
            "raw_delta": raw_delta,
            "recognition": recognition,
            "weights": weights,
            "delta_norm": delta.norm(dim=-1),
        }
        if workspace_info is not None:
            info.update(
                {
                    "workspace_survival": workspace_info["survival"],
                    "workspace_slot_usage": workspace_info["slot_usage"],
                    "workspace_read_weights": workspace_info["read_weights"],
                    "workspace_mask": workspace_info["workspace_mask"],
                    "workspace_exposed_mask": workspace_info["exposed_mask"],
                    "workspace_source_index": workspace_info["source_index"],
                    "workspace_context_norm": workspace_info["context_norm"],
                }
            )
        return delta, info


class LayerRecallStack(nn.Module):
    """Multiple independent Recall lines attached to the same physical layer."""

    def __init__(self, branches: Iterable[LayerRecall], *, combine: str = "sum") -> None:
        super().__init__()
        values = tuple(branches)
        if not values:
            raise ValueError("branches must not be empty")
        if combine not in {"sum", "mean"}:
            raise ValueError("combine must be 'sum' or 'mean'")
        if len({branch.dim for branch in values}) != 1:
            raise ValueError("all Recall branches must use the same hidden dimension")
        self.branches = nn.ModuleList(values)
        self.combine = combine
        self.dim = values[0].dim
        self._line_enabled = [True] * len(values)

    def set_line_enabled(self, index: int, enabled: bool) -> None:
        if index < 0 or index >= len(self.branches):
            raise IndexError("Recall line index out of range")
        self._line_enabled[index] = bool(enabled)

    @contextmanager
    def enabled_lines(self, indices: Iterable[int]):
        selected = set(indices)
        if any(index < 0 or index >= len(self.branches) for index in selected):
            raise IndexError("Recall line index out of range")
        previous = list(self._line_enabled)
        try:
            self._line_enabled = [index in selected for index in range(len(self.branches))]
            yield self
        finally:
            self._line_enabled = previous

    def forward(
        self,
        hidden: Tensor,
        *,
        mask: Tensor | None = None,
        return_info: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        outputs = [
            branch(hidden, mask=mask, return_info=True)
            for index, branch in enumerate(self.branches)
            if self._line_enabled[index]
        ]
        if not outputs:
            delta = torch.zeros_like(hidden)
            if not return_info:
                return delta
            recognition = torch.zeros(hidden.shape[:-1], device=hidden.device, dtype=hidden.dtype)
            return delta, {
                "raw_delta": delta,
                "recognition": recognition,
                "weights": recognition.unsqueeze(-1),
                "delta_norm": delta.norm(dim=-1),
                "branch_delta_norm": delta.new_zeros((0, *hidden.shape[:-1])),
                "branch_recognition": delta.new_zeros((0, *hidden.shape[:-1])),
            }
        deltas = torch.stack([output[0] for output in outputs], dim=0)
        raw = torch.stack([output[1]["raw_delta"] for output in outputs], dim=0)
        recognition = torch.stack([output[1]["recognition"] for output in outputs], dim=0)
        weights = torch.cat([output[1]["weights"] for output in outputs], dim=-1)
        delta = deltas.sum(dim=0) if self.combine == "sum" else deltas.mean(dim=0)
        raw_delta = raw.sum(dim=0) if self.combine == "sum" else raw.mean(dim=0)
        if not return_info:
            return delta
        return delta, {
            "raw_delta": raw_delta,
            "recognition": recognition.mean(dim=0),
            "weights": weights,
            "delta_norm": delta.norm(dim=-1),
            "branch_delta_norm": deltas.norm(dim=-1),
            "branch_recognition": recognition,
        }


class LayerRecallWrapper(nn.Module):
    """Preserve a base layer's output contract while adding one Recall delta."""

    def __init__(
        self,
        base: nn.Module,
        recall: LayerRecall | LayerRecallStack,
        *,
        layer_path: str,
        batch_axis: int | None = None,
        feature_axis: int | None = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.recall = recall
        self.layer_path = layer_path
        self.batch_axis = batch_axis
        self.feature_axis = feature_axis
        self.enabled = True
        self.capture = False
        self._delta_gate: Callable[[Tensor], Tensor] | None = None
        self.last_pre: Tensor | None = None
        self.last_delta: Tensor | None = None
        self.last_raw_delta: Tensor | None = None
        self.last_post: Tensor | None = None
        self.last_recognition: Tensor | None = None
        self.last_survival: Tensor | None = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        output = self.base(*args, **kwargs)
        found = find_primary_tensor(output)
        if found is None:
            return output
        tensor, output_path = found
        inferred = TensorLayout.infer(tensor, self.base)
        layout = TensorLayout(
            tuple(int(value) for value in tensor.shape),
            inferred.batch_axis if self.batch_axis is None else self.batch_axis,
            inferred.feature_axis if self.feature_axis is None else self.feature_axis,
        )
        sequence = layout.pack(tensor)
        delta_sequence = torch.zeros_like(sequence)
        recognition = torch.zeros(
            sequence.shape[:-1], device=sequence.device, dtype=sequence.dtype
        )
        if self.enabled:
            delta_sequence, info = self.recall(sequence, return_info=True)
            recognition = info["recognition"]
            raw_delta_sequence = info.get("raw_delta", delta_sequence)
            if self._delta_gate is not None:
                gate = _normalize_delta_gate(self._delta_gate(sequence), sequence)
                scale = gate.unsqueeze(-1).to(delta_sequence)
                delta_sequence = delta_sequence * scale
                raw_delta_sequence = raw_delta_sequence * scale
                recognition = recognition * gate.to(recognition)
            delta_scale = delta_sequence.float().abs().mean(dim=-1)
            raw_scale = raw_delta_sequence.float().abs().mean(dim=-1)
            survival = (delta_scale / raw_scale.clamp_min(torch.finfo(torch.float32).tiny)).to(tensor.dtype)
        else:
            raw_delta_sequence = torch.zeros_like(sequence)
            survival = torch.zeros(
                sequence.shape[:-1], device=sequence.device, dtype=sequence.dtype
            )
        delta = layout.restore(delta_sequence)
        raw_delta = layout.restore(raw_delta_sequence)
        post = layout.restore(sequence + delta_sequence)
        if self.capture:
            self.last_pre = tensor
            self.last_delta = delta
            self.last_raw_delta = raw_delta
            self.last_post = post
            self.last_recognition = recognition
            self.last_survival = survival
        return replace_tensor_at_path(output, output_path, post)

    @property
    def has_delta_gate(self) -> bool:
        """Whether an external tensor gate currently modulates this Recall branch."""

        return self._delta_gate is not None

    def set_delta_gate(self, gate: Callable[[Tensor], Tensor] | None) -> None:
        """Install a non-serialized survival gate for the Recall delta."""

        if gate is not None and not callable(gate):
            raise TypeError("delta gate must be callable or None")
        self._delta_gate = gate

    def clear_trace(self) -> None:
        self.last_pre = None
        self.last_delta = None
        self.last_raw_delta = None
        self.last_post = None
        self.last_recognition = None
        self.last_survival = None


def _normalize_delta_gate(gate: Tensor, hidden: Tensor) -> Tensor:
    if not isinstance(gate, Tensor):
        raise TypeError("delta gate must return a Tensor")
    expected = hidden.shape[:-1]
    if hidden.ndim == 3 and tuple(gate.shape) == (hidden.shape[0],):
        gate = gate.unsqueeze(-1).expand(expected)
    if tuple(gate.shape) != tuple(expected):
        raise ValueError(f"delta gate returned shape {tuple(gate.shape)}; expected {tuple(expected)}")
    if not gate.is_floating_point():
        raise TypeError("delta gate must return a floating-point Tensor")
    return gate


@dataclass(frozen=True)
class LayerRecallSpec:
    """Independent declaration for one Recall branch at an arbitrary path."""

    path: str
    dim: int | None = None
    rank: int = 16
    slots: int = 8
    use_half: bool = True
    recognition_mode: str = "none"
    recognition_threshold: float = 0.5
    recognition_temperature: float = 0.1
    copies: int = 1
    combine: str = "sum"
    batch_axis: int | None = None
    feature_axis: int | None = None

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("path must not be empty")
        if self.dim is not None and self.dim <= 0:
            raise ValueError("dim must be positive when provided")
        if self.rank <= 0 or self.slots <= 0:
            raise ValueError("rank and slots must be positive")
        if self.recognition_mode not in {"explicit", "alignment", "none"}:
            raise ValueError("recognition_mode must be 'explicit', 'alignment', or 'none'")
        if self.recognition_temperature <= 0:
            raise ValueError("recognition_temperature must be positive")
        if self.copies <= 0:
            raise ValueError("copies must be positive")
        if self.combine not in {"sum", "mean"}:
            raise ValueError("combine must be 'sum' or 'mean'")
        if self.batch_axis is not None and (
            isinstance(self.batch_axis, bool) or not isinstance(self.batch_axis, int)
        ):
            raise ValueError("batch_axis must be an integer or None")
        if self.feature_axis is not None and (
            isinstance(self.feature_axis, bool) or not isinstance(self.feature_axis, int)
        ):
            raise ValueError("feature_axis must be an integer or None")


@dataclass(frozen=True)
class LayeredRecallConfig:
    """Open-ended declaration for any number of independently sized branches."""

    layer_paths: tuple[str, ...] = ()
    rank: int | Mapping[str, int] = 16
    slots: int | Mapping[str, int] = 8
    use_half: bool = True
    recognition_mode: str = "none"
    freeze_backbone: bool = True
    batch_axis: int | Mapping[str, int] | None = None
    feature_axis: int | Mapping[str, int] | None = None
    layers: tuple[LayerRecallSpec, ...] = ()

    def __post_init__(self) -> None:
        if bool(self.layer_paths) == bool(self.layers):
            raise ValueError("provide exactly one of layer_paths or layers")
        paths = self.layer_paths or tuple(layer.path for layer in self.layers)
        if len(set(paths)) != len(paths):
            raise ValueError("layer_paths must contain unique layer names")
        if self.recognition_mode not in {"explicit", "alignment", "none"}:
            raise ValueError("recognition_mode must be 'explicit', 'alignment', or 'none'")

    @property
    def paths(self) -> tuple[str, ...]:
        """Resolved ordered paths, independent of declaration style."""

        return self.layer_paths or tuple(layer.path for layer in self.layers)


@dataclass(frozen=True)
class LayeredRecallCalibration:
    """Detached per-layer corruption scales used by normalized local losses."""

    scales: dict[str, Tensor]
    method: str = "baseline_corruption_mse"
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.method != "baseline_corruption_mse":
            raise ValueError("method must be 'baseline_corruption_mse'")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not self.scales:
            raise ValueError("scales must not be empty")


@dataclass(frozen=True)
class LayeredRecallLoss:
    """Label-free local trajectory repair objective and diagnostics."""

    loss: Tensor
    repair_loss: Tensor
    unseen_loss: Tensor
    per_layer_mse: dict[str, Tensor]
    per_layer_delta_norm: dict[str, Tensor]
    per_layer_raw_delta_norm: dict[str, Tensor]
    per_layer_survival: dict[str, Tensor]
    per_layer_recognition: dict[str, Tensor]
    clean_output: Any
    corrupt_output: Any


class LayeredRecallModel(nn.Module):
    """Frozen backbone with independently addressable Recall branches."""

    def __init__(self, model: nn.Module, layer_paths: Iterable[str]) -> None:
        super().__init__()
        self.model = model
        self.layer_paths = tuple(layer_paths)
        if not self.layer_paths or len(set(self.layer_paths)) != len(self.layer_paths):
            raise ValueError("layer_paths must contain unique layer names")
        for path in self.layer_paths:
            if not isinstance(self._module_at(path), LayerRecallWrapper):
                raise ValueError(f"layer path {path!r} is not wrapped with LayerRecall")

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        config: LayeredRecallConfig,
        *,
        sample_batch: Any | None = None,
        dims: Mapping[str, int] | None = None,
    ) -> "LayeredRecallModel":
        """Attach a stable, serializable multi-layer Recall configuration."""

        resolved_dims = dict(dims or {})
        if config.layers:
            resolved_dims.update({layer.path: layer.dim for layer in config.layers if layer.dim is not None})
            rank: int | Mapping[str, int] = {layer.path: layer.rank for layer in config.layers}
            slots: int | Mapping[str, int] = {layer.path: layer.slots for layer in config.layers}
            use_half: bool | Mapping[str, bool] = {layer.path: layer.use_half for layer in config.layers}
            recognition_mode: str | Mapping[str, str] = {layer.path: layer.recognition_mode for layer in config.layers}
            recognition_threshold: float | Mapping[str, float] = {
                layer.path: layer.recognition_threshold for layer in config.layers
            }
            recognition_temperature: float | Mapping[str, float] = {
                layer.path: layer.recognition_temperature for layer in config.layers
            }
            copies: int | Mapping[str, int] = {layer.path: layer.copies for layer in config.layers}
            combine: str | Mapping[str, str] = {layer.path: layer.combine for layer in config.layers}
            batch_axis: int | Mapping[str, int] | None = {
                layer.path: layer.batch_axis for layer in config.layers if layer.batch_axis is not None
            }
            feature_axis: int | Mapping[str, int] | None = {
                layer.path: layer.feature_axis
                for layer in config.layers
                if layer.feature_axis is not None
            }
        else:
            rank = config.rank
            slots = config.slots
            use_half = config.use_half
            recognition_mode = config.recognition_mode
            recognition_threshold = 0.5
            recognition_temperature = 0.1
            copies = 1
            combine = "sum"
            batch_axis = config.batch_axis
            feature_axis = config.feature_axis
        return cls.attach(
            model,
            config.paths,
            sample_batch=sample_batch,
            dims=resolved_dims,
            rank=rank,
            slots=slots,
            use_half=use_half,
            recognition_mode=recognition_mode,
            recognition_threshold=recognition_threshold,
            recognition_temperature=recognition_temperature,
            copies=copies,
            combine=combine,
            batch_axis=batch_axis,
            feature_axis=feature_axis,
            freeze_backbone=config.freeze_backbone,
        )

    @classmethod
    def attach(
        cls,
        model: nn.Module,
        layer_paths: Iterable[str],
        *,
        sample_batch: Any | None = None,
        dims: Mapping[str, int] | None = None,
        rank: int | Mapping[str, int] = 16,
        slots: int | Mapping[str, int] = 8,
        use_half: bool | Mapping[str, bool] = True,
        recognition_mode: str | Mapping[str, str] = "none",
        recognition_threshold: float | Mapping[str, float] = 0.5,
        recognition_temperature: float | Mapping[str, float] = 0.1,
        copies: int | Mapping[str, int] = 1,
        combine: str | Mapping[str, str] = "sum",
        batch_axis: int | Mapping[str, int] | None = None,
        feature_axis: int | Mapping[str, int] | None = None,
        freeze_backbone: bool = True,
    ) -> "LayeredRecallModel":
        paths = tuple(layer_paths)
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("layer_paths must contain unique layer names")
        resolved_dims = dict(dims or {})
        runtime_candidates = {}
        if sample_batch is not None:
            report = scan_model(
                model,
                sample_batch,
                batch_axis=batch_axis,
                feature_axis=feature_axis,
            )
            runtime_candidates = {
                candidate.name: candidate
                for candidate in report.candidates
                if candidate.name in paths
            }
            resolved_dims.update({candidate.name: candidate.dim for candidate in report.candidates if candidate.name in paths})
            unresolved = tuple(path for path in paths if path not in resolved_dims)
            if unresolved:
                resolved_dims.update(_runtime_path_dims(model, unresolved, sample_batch))
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False
        for path in paths:
            parent, leaf = get_parent_module(model, path)
            base = parent[int(leaf)] if leaf.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)) else getattr(parent, leaf)
            dim = resolved_dims.get(path) or _infer_module_dim(base)
            if dim is None:
                raise ValueError(f"cannot infer hidden dimension for {path!r}; pass sample_batch or dims")
            copy_count = int(_per_path_value(copies, path))
            if copy_count <= 0:
                raise ValueError(f"copies for layer {path!r} must be positive")
            branches = [
                LayerRecall(
                    dim,
                    rank=_per_path(rank, path),
                    slots=_per_path(slots, path),
                    use_half=bool(_per_path_value(use_half, path)),
                    recognition_mode=str(_per_path_value(recognition_mode, path)),
                    recognition_threshold=float(_per_path_value(recognition_threshold, path)),
                    recognition_temperature=float(_per_path_value(recognition_temperature, path)),
                )
                for _ in range(copy_count)
            ]
            branch: LayerRecall | LayerRecallStack = branches[0] if copy_count == 1 else LayerRecallStack(
                branches, combine=str(_per_path_value(combine, path))
            )
            reference = next((parameter for parameter in base.parameters() if parameter.is_floating_point()), None)
            if reference is not None:
                branch.to(device=reference.device, dtype=reference.dtype)
            else:
                candidate = runtime_candidates.get(path)
                dtype = (
                    None
                    if candidate is None or candidate.dtype is None
                    else getattr(torch, candidate.dtype.removeprefix("torch."), None)
                )
                if candidate is not None and candidate.device is not None and isinstance(dtype, torch.dtype):
                    branch.to(device=torch.device(candidate.device), dtype=dtype)
            batch_axis_value = _optional_path_value(batch_axis, path)
            feature_axis_value = _optional_path_value(feature_axis, path)
            resolved_batch_axis = None if batch_axis_value is None else int(batch_axis_value)
            resolved_feature_axis = (
                None if feature_axis_value is None else int(feature_axis_value)
            )
            set_child_module(
                parent,
                leaf,
                LayerRecallWrapper(
                    base,
                    branch,
                    layer_path=path,
                    batch_axis=resolved_batch_axis,
                    feature_axis=resolved_feature_axis,
                ),
            )
        return cls(model, paths)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate autoregressive generation when the wrapped model provides it."""

        generate = getattr(self.model, "generate", None)
        if not callable(generate):
            raise AttributeError(f"{type(self.model).__name__!s} does not provide generate()")
        return generate(*args, **kwargs)

    @property
    def wrappers(self) -> dict[str, LayerRecallWrapper]:
        return {path: self._wrapper_at(path) for path in self.layer_paths}

    def set_enabled(self, enabled: bool, *, paths: Iterable[str] | None = None) -> None:
        selected = self._validate_paths(self.layer_paths if paths is None else tuple(paths))
        for path in selected:
            self._wrapper_at(path).enabled = enabled

    @contextmanager
    def enabled_layers(self, paths: Iterable[str]):
        """Temporarily enable only selected Recall layers and restore state."""

        selected = set(self._validate_paths(tuple(paths)))
        previous = {path: wrapper.enabled for path, wrapper in self.wrappers.items()}
        try:
            for path, wrapper in self.wrappers.items():
                wrapper.enabled = path in selected
            yield self
        finally:
            for path, enabled in previous.items():
                self.wrappers[path].enabled = enabled

    @contextmanager
    def disabled(self):
        """Temporarily disable every Recall branch and restore prior state."""

        with self.enabled_layers(()):
            yield self

    def calibrate(
        self,
        clean_inputs: Any,
        corrupt_inputs: Any,
        *,
        mask: Tensor | None = None,
        causal: bool = False,
        epsilon: float = 1e-8,
    ) -> LayeredRecallCalibration:
        """Measure detached no-Recall corruption MSE at every attached layer."""

        return calibrate_layered_recall(self, clean_inputs, corrupt_inputs, mask=mask, causal=causal, epsilon=epsilon)

    def diagnostics(self) -> dict[str, dict[str, Tensor]]:
        """Return the latest captured layer diagnostics without changing state."""

        report: dict[str, dict[str, Tensor]] = {}
        for path, wrapper in self.wrappers.items():
            values = {
                "raw_delta": wrapper.last_raw_delta,
                "delta": wrapper.last_delta,
                "recognition": wrapper.last_recognition,
                "survival": wrapper.last_survival,
            }
            report[path] = {name: value for name, value in values.items() if value is not None}
        return report

    def clear_traces(self) -> None:
        for wrapper in self.wrappers.values():
            wrapper.clear_trace()

    def recall_parameters(self) -> Iterable[nn.Parameter]:
        for wrapper in self.wrappers.values():
            yield from wrapper.recall.parameters()

    def export_layer(
        self,
        layer_path: str,
        path: str | Path,
        *,
        capability: str = "layered-trajectory-repair",
        training_metadata: Mapping[str, Any] | None = None,
    ):
        wrapper = self._wrapper_at(layer_path)
        metadata = {"layer_path": layer_path, **dict(training_metadata or {})}
        spec = RecallArtifactSpec(
            capability=capability,
            base_model_fingerprint=module_structure_fingerprint(self.model),
            injection_fingerprint=module_structure_fingerprint(wrapper.recall),
            visibility_policy="layer-local-mask",
            training_metadata=metadata,
        )
        return export_recall_artifact(wrapper.recall, path, spec)

    def load_layer(self, layer_path: str, path: str | Path, *, map_location: str | torch.device = "cpu"):
        self._validate_artifact_layer(layer_path, path)
        wrapper = self._wrapper_at(layer_path)
        return load_recall_artifact(path, wrapper.recall, base_model=self.model, injection_module=wrapper.recall, map_location=map_location)

    def concat_layer(
        self,
        layer_path: str,
        artifacts: Mapping[str, str | Path],
        *,
        map_location: str | torch.device = "cpu",
    ) -> LayerRecall:
        wrapper = self._wrapper_at(layer_path)
        if isinstance(wrapper.recall, LayerRecallStack):
            raise ValueError("concat_layer requires a single Recall line; stacked layers must be exported or loaded as a whole")
        pool = RecallExpertPool(wrapper.recall, base_model=self.model)
        for name, path in artifacts.items():
            self._validate_artifact_layer(layer_path, path)
            pool.load_expert(name, path, map_location=map_location)
        merged = pool.concatenate(parameter="bank")
        if not isinstance(merged, LayerRecall):
            raise TypeError("concatenated expert is not a LayerRecall")
        merged.slots = int(merged.bank.shape[0])
        wrapper.recall = merged
        return merged

    def append_layer_artifacts(
        self,
        layer_path: str,
        artifacts: Mapping[str, str | Path],
        *,
        map_location: str | torch.device = "cpu",
        combine: str = "sum",
        include_current: bool = True,
    ) -> LayerRecallStack:
        """Append complete independent Recall lines without remixing old weights."""

        wrapper = self._wrapper_at(layer_path)
        current = list(wrapper.recall.branches) if isinstance(wrapper.recall, LayerRecallStack) else [wrapper.recall]
        if not artifacts:
            raise ValueError("artifacts must contain at least one independent Recall line")
        template = current[0]
        appended: list[LayerRecall] = []
        for name, path in artifacts.items():
            if not name:
                raise ValueError("artifact names must not be empty")
            self._validate_artifact_layer(layer_path, path)
            branch = copy.deepcopy(template)
            load_recall_artifact(
                path,
                branch,
                base_model=self.model,
                injection_module=branch,
                map_location=map_location,
            )
            reference = next(template.parameters(), None)
            if reference is not None:
                branch.to(device=reference.device, dtype=reference.dtype)
            appended.append(branch)
        lines = [*current, *appended] if include_current else appended
        stack = LayerRecallStack(lines, combine=combine)
        wrapper.recall = stack
        return stack

    def _validate_artifact_layer(self, layer_path: str, path: str | Path) -> None:
        loaded = load_arti(path)
        metadata = loaded.manifest.get("architecture", {}).get("config", {}).get("recall_expert", {}).get("training_metadata", {})
        if metadata.get("layer_path") != layer_path:
            raise ValueError(f"Recall artifact belongs to layer {metadata.get('layer_path')!r}, not {layer_path!r}")

    def _module_at(self, path: str) -> nn.Module:
        module: nn.Module = self.model
        for part in path.split("."):
            module = module[int(part)] if part.isdigit() and isinstance(module, (nn.Sequential, nn.ModuleList)) else getattr(module, part)
        return module

    def _wrapper_at(self, path: str) -> LayerRecallWrapper:
        module = self._module_at(path)
        if not isinstance(module, LayerRecallWrapper):
            raise ValueError(f"layer path {path!r} is not a LayerRecallWrapper")
        return module

    def _validate_paths(self, paths: Iterable[str]) -> tuple[str, ...]:
        selected = tuple(paths)
        unknown = tuple(path for path in selected if path not in self.layer_paths)
        if unknown:
            raise ValueError(f"unknown Recall layer paths: {unknown}")
        return selected


def calibrate_layered_recall(
    model: LayeredRecallModel,
    clean_inputs: Any,
    corrupt_inputs: Any,
    *,
    mask: Tensor | None = None,
    causal: bool = False,
    epsilon: float = 1e-8,
) -> LayeredRecallCalibration:
    """Calibrate per-layer corruption scales with all Recall branches disabled."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    wrappers = model.wrappers
    previous_enabled = {path: wrapper.enabled for path, wrapper in wrappers.items()}
    previous_capture = {path: wrapper.capture for path, wrapper in wrappers.items()}
    try:
        for wrapper in wrappers.values():
            wrapper.enabled = False
            wrapper.capture = True
            wrapper.clear_trace()
        _run_inputs(model, clean_inputs, causal=causal)
        clean = {path: _require_trace(wrapper.last_pre, path, "clean calibration").detach() for path, wrapper in wrappers.items()}
        for wrapper in wrappers.values():
            wrapper.clear_trace()
        _run_inputs(model, corrupt_inputs, causal=causal)
        scales: dict[str, Tensor] = {}
        for path, wrapper in wrappers.items():
            corrupt = _require_trace(wrapper.last_pre, path, "corrupt calibration")
            local_mask = _mask_for(corrupt if corrupt.ndim == 3 else corrupt.unsqueeze(1), mask)
            if corrupt.ndim == 2:
                local_mask = local_mask[:, 0]
            error = (corrupt - clean[path]).square().mean(dim=-1)
            scale = (error * local_mask.to(error.dtype)).sum() / local_mask.sum().clamp_min(1)
            scales[path] = scale.detach().clamp_min(epsilon)
        return LayeredRecallCalibration(scales=scales, epsilon=epsilon)
    finally:
        for path, wrapper in wrappers.items():
            wrapper.enabled = previous_enabled[path]
            wrapper.capture = previous_capture[path]


def layered_recall_trajectory_loss(
    model: LayeredRecallModel,
    clean_inputs: Any,
    corrupt_inputs: Any,
    *,
    mask: Tensor | None = None,
    unseen_inputs: Any | None = None,
    unseen_weight: float = 0.25,
    layer_scales: Mapping[str, Tensor | float] | None = None,
    calibration: LayeredRecallCalibration | None = None,
    causal: bool = False,
) -> LayeredRecallLoss:
    """Build local clean/corrupt repair targets without labels or future tokens."""

    if unseen_weight < 0:
        raise ValueError("unseen_weight must be non-negative")
    if calibration is not None and layer_scales is not None:
        raise ValueError("pass calibration or layer_scales, not both")
    if calibration is not None:
        layer_scales = calibration.scales
    wrappers = model.wrappers
    previous = {path: wrapper.enabled for path, wrapper in wrappers.items()}
    for wrapper in wrappers.values():
        wrapper.capture = True
        wrapper.enabled = False
        wrapper.clear_trace()
    try:
        clean_output = _run_inputs(model, clean_inputs, causal=causal)
        clean_traces = {path: _require_trace(wrapper.last_pre, path, "clean") for path, wrapper in wrappers.items()}
        for wrapper in wrappers.values():
            wrapper.enabled = True
            wrapper.clear_trace()
        corrupt_output = _run_inputs(model, corrupt_inputs, causal=causal)
        per_layer_mse: dict[str, Tensor] = {}
        per_layer_delta_norm: dict[str, Tensor] = {}
        per_layer_raw_delta_norm: dict[str, Tensor] = {}
        per_layer_survival: dict[str, Tensor] = {}
        per_layer_recognition: dict[str, Tensor] = {}
        losses = []
        for path, wrapper in wrappers.items():
            post = _require_trace(wrapper.last_post, path, "corrupt")
            delta = _require_trace(wrapper.last_delta, path, "corrupt")
            target = clean_traces[path].detach()
            local_mask = _mask_for(post if post.ndim == 3 else post.unsqueeze(1), mask)
            if post.ndim == 2:
                local_mask = local_mask[:, 0]
            error = (post - target).square().mean(dim=-1)
            mse = (error * local_mask.to(error.dtype)).sum() / local_mask.sum().clamp_min(1)
            if layer_scales is not None:
                if path not in layer_scales:
                    raise ValueError(f"layer_scales is missing {path!r}")
                scale = torch.as_tensor(layer_scales[path], device=mse.device, dtype=mse.dtype).detach().clamp_min(1e-8)
                mse = mse / scale
            per_layer_mse[path] = mse
            per_layer_delta_norm[path] = delta.norm(dim=-1).mean()
            raw_delta = _require_trace(wrapper.last_raw_delta, path, "raw corrupt delta")
            per_layer_raw_delta_norm[path] = raw_delta.norm(dim=-1).mean()
            survival = wrapper.last_survival
            per_layer_survival[path] = torch.zeros((), device=mse.device) if survival is None else survival.mean()
            recognition = wrapper.last_recognition
            per_layer_recognition[path] = torch.zeros((), device=mse.device) if recognition is None else recognition.mean()
            losses.append(mse)
        repair_loss = torch.stack(losses).mean()
        unseen_loss = torch.zeros_like(repair_loss)
        if unseen_inputs is not None:
            for wrapper in wrappers.values():
                wrapper.clear_trace()
            _run_inputs(model, unseen_inputs, causal=causal)
            unseen_terms = []
            for path, wrapper in wrappers.items():
                term = _require_trace(wrapper.last_delta, path, "unseen").square().mean()
                if layer_scales is not None:
                    scale = torch.as_tensor(layer_scales[path], device=term.device, dtype=term.dtype).detach().clamp_min(1e-8)
                    term = term / scale
                unseen_terms.append(term)
            unseen_loss = torch.stack(unseen_terms).mean()
        return LayeredRecallLoss(
            loss=repair_loss + unseen_weight * unseen_loss,
            repair_loss=repair_loss,
            unseen_loss=unseen_loss,
            per_layer_mse=per_layer_mse,
            per_layer_delta_norm=per_layer_delta_norm,
            per_layer_raw_delta_norm=per_layer_raw_delta_norm,
            per_layer_survival=per_layer_survival,
            per_layer_recognition=per_layer_recognition,
            clean_output=clean_output,
            corrupt_output=corrupt_output,
        )
    finally:
        for path, wrapper in wrappers.items():
            wrapper.enabled = previous[path]
            wrapper.capture = False


def _run_inputs(model: nn.Module, inputs: Any, *, causal: bool) -> Any:
    return run_model(model, inputs, causal=causal)


def _mask_for(sequence: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return torch.ones(sequence.shape[:2], device=sequence.device, dtype=torch.bool)
    if mask.shape != sequence.shape[:2]:
        raise ValueError(f"mask must have shape {tuple(sequence.shape[:2])}")
    return mask.to(device=sequence.device, dtype=torch.bool)


def _infer_module_dim(module: nn.Module) -> int | None:
    for attribute in (
        "out_features",
        "out_channels",
        "num_channels",
        "num_features",
        "normalized_shape",
        "hidden_size",
        "embed_dim",
    ):
        value = getattr(module, attribute, None)
        if isinstance(value, int):
            return value
        if isinstance(value, (tuple, list)) and len(value) == 1:
            return int(value[0])
    return None


def _runtime_path_dims(
    model: nn.Module,
    paths: Iterable[str],
    sample_batch: Any,
    *,
    batch_axis: int | Mapping[str, int] | None = None,
    feature_axis: int | Mapping[str, int] | None = None,
) -> dict[str, int]:
    observed: dict[str, int] = {}
    handles = []

    def capture(path: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            found = find_primary_tensor(output)
            if found is not None:
                tensor = found[0]
                inferred = TensorLayout.infer(tensor, _module)
                layout = TensorLayout(
                    tuple(int(value) for value in tensor.shape),
                    inferred.batch_axis
                    if _optional_path_value(batch_axis, path) is None
                    else int(_optional_path_value(batch_axis, path)),
                    inferred.feature_axis
                    if _optional_path_value(feature_axis, path) is None
                    else int(_optional_path_value(feature_axis, path)),
                )
                observed[path] = layout.feature_dim

        return hook

    for path in paths:
        module: nn.Module = model
        for part in path.split("."):
            module = module[int(part)] if part.isdigit() and isinstance(module, (nn.Sequential, nn.ModuleList)) else getattr(module, part)
        handles.append(module.register_forward_hook(capture(path)))
    training = model.training
    try:
        model.eval()
        with torch.no_grad():
            run_model(model, sample_batch)
    finally:
        model.train(training)
        for handle in handles:
            handle.remove()
    return observed


def _per_path(value: int | Mapping[str, int], path: str) -> int:
    resolved = value[path] if isinstance(value, Mapping) else value
    if int(resolved) <= 0:
        raise ValueError(f"value for layer {path!r} must be positive")
    return int(resolved)


def _per_path_value(value: Any | Mapping[str, Any], path: str) -> Any:
    if isinstance(value, Mapping):
        if path not in value:
            raise ValueError(f"mapping is missing layer path {path!r}")
        return value[path]
    return value


def _optional_path_value(value: Any | Mapping[str, Any] | None, path: str) -> Any:
    if value is None:
        return None
    return value.get(path) if isinstance(value, Mapping) else value


def _require_trace(value: Tensor | None, path: str, kind: str) -> Tensor:
    if value is None:
        raise RuntimeError(f"layer {path!r} did not capture a {kind} trace")
    return value


__all__ = [
    "LayerRecall",
    "LayerRecallStack",
    "LayerRecallWrapper",
    "LayerRecallSpec",
    "LayeredRecallConfig",
    "LayeredRecallCalibration",
    "LayeredRecallLoss",
    "LayeredRecallModel",
    "calibrate_layered_recall",
    "layered_recall_trajectory_loss",
]
