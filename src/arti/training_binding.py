"""Model, parameter and asset bindings for the shared training session."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from torch import nn

from .serialization import load, save
from .training_config import ARTITrainingConfig


class TrainingBinding:
    def __init__(
        self,
        model: nn.Module,
        *,
        owner: Any = None,
        kind: str = "model",
        parameters: Iterable[nn.Parameter] | Callable[[], Iterable[nn.Parameter]] | None = None,
        state_module: nn.Module | Callable[[], nn.Module] | None = None,
        summary: Callable[[], Mapping[str, Any]] | None = None,
        save_checkpoint: Callable[..., Any] | None = None,
        load_checkpoint: Callable[..., Any] | None = None,
        checkpoint_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.owner = model if owner is None else owner
        self.kind = kind
        self._parameters = tuple(p for p in model.parameters() if p.requires_grad) if parameters is None else parameters if callable(parameters) else tuple(parameters)
        self._state_module = model if state_module is None else state_module
        self._summary = summary
        self._save = save_checkpoint
        self._load = load_checkpoint
        self.checkpoint_metadata = dict(checkpoint_metadata or {})

    @classmethod
    def from_attachment(cls, attachment) -> TrainingBinding:
        return cls(
            attachment._model,
            owner=attachment,
            kind="attachment",
            parameters=attachment.parameters,
            state_module=attachment._bundle,
            summary=lambda: attachment.summary().to_dict(),
            save_checkpoint=attachment.save,
            load_checkpoint=attachment.load,
        )

    def parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self._parameters() if callable(self._parameters) else self._parameters)

    def state_module(self) -> nn.Module:
        return self._state_module if isinstance(self._state_module, nn.Module) else self._state_module()

    def describe(self) -> dict[str, Any]:
        if self._summary is not None:
            return dict(self._summary())
        selected = self.parameters()
        ids = {id(p) for p in selected}
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in selected)
        return {
            "kind": self.kind,
            "trainable_parameters": trainable,
            "backbone_parameters": total - trainable,
            "estimated_parameter_bytes": sum(p.numel() * p.element_size() for p in selected),
            "parameter_names": [name for name, p in self.model.named_parameters() if id(p) in ids],
        }

    def save(self, path, *, optimizer=None, scheduler=None, training_state=None):
        if self._save is not None:
            return self._save(path, optimizer=optimizer, scheduler=scheduler, training_state=training_state)
        return save(self.state_module(), path, config={"training_binding": self.describe()}, optimizer=optimizer, scheduler=scheduler, training_state=training_state)

    def load(self, path, *, optimizer=None, scheduler=None, map_location=None, load_checkpoint=True):
        if self._load is not None:
            return self._load(path, optimizer=optimizer, scheduler=scheduler, map_location=map_location, load_checkpoint=load_checkpoint)
        device = next(self.model.parameters()).device if map_location is None else map_location
        return load(path, model=self.state_module(), optimizer=optimizer, scheduler=scheduler, map_location=device, load_checkpoint=load_checkpoint)


def trainer(
    model: nn.Module | TrainingBinding,
    *,
    objective=None,
    parameters: Iterable[nn.Parameter] | None = None,
    config: ARTITrainingConfig | None = None,
    optimizer=None,
    scheduler=None,
    resume_from_checkpoint: str | Path | bool | None = None,
    **settings,
):
    """Create the same training session used by attachments and providers."""
    from .attachment_training import ARTITrainingSession

    binding = model if isinstance(model, TrainingBinding) else TrainingBinding(model, parameters=parameters)
    if isinstance(model, TrainingBinding) and parameters is not None:
        raise ValueError("parameters belong to the supplied TrainingBinding")
    resolved = replace(config or ARTITrainingConfig(), **settings)
    return ARTITrainingSession(binding, config=resolved, objective=objective, optimizer=optimizer, scheduler=scheduler, resume_from_checkpoint=resume_from_checkpoint)


__all__ = ["TrainingBinding", "trainer"]
