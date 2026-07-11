"""Training engines and objectives for unified ARTI attachments."""

from __future__ import annotations

import inspect
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TYPE_CHECKING

import torch
from torch import Tensor

from .attachment_config import ARTIAttachTrainingConfig
from .layered_recall import layered_recall_trajectory_loss

try:
    from transformers import TrainerCallback as _TrainerCallback
except ImportError:
    _TrainerCallback = object


TRANSFORMERS_ARTIFACT = "model.recall.arti.st"


class ARTICheckpointCallback(_TrainerCallback):
    """Record that a Transformers checkpoint contains an ARTI-only payload."""

    def on_save(self, args, state, control, **_kwargs):
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        artifact = checkpoint / TRANSFORMERS_ARTIFACT
        if artifact.exists():
            marker = {
                "format": "arti.transformers.checkpoint",
                "format_version": 1,
                "global_step": int(state.global_step),
                "artifact": artifact.name,
            }
            (checkpoint / "arti-checkpoint.json").write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return control

if TYPE_CHECKING:
    from .attachment import ARTIAttachment


AttachmentObjective = Callable[..., Tensor]


@dataclass(frozen=True)
class ARTITrainingResult:
    engine: str
    steps: int
    loss_history: tuple[float, ...]
    checkpoint_path: Path | None = None


class ARTITrainingSession:
    """Small engine-neutral trainer created by ``model.arti.trainer()``."""

    def __init__(
        self,
        attachment: "ARTIAttachment",
        *,
        config: ARTIAttachTrainingConfig,
        objective: str | AttachmentObjective | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        resume_from_checkpoint: str | Path | bool | None = None,
    ) -> None:
        self.attachment = attachment
        self.model = attachment._model
        self.config = config
        self.objective = resolve_attachment_objective(objective or config.objective, corruption_probability=config.corruption_probability)
        parameters = list(attachment.parameters())
        self.optimizer = optimizer or torch.optim.AdamW(parameters, lr=config.learning_rate)
        self.scheduler = scheduler
        self.global_step = 0
        self.loss_history: list[float] = []
        self.engine_object: Any | None = None
        self.resume_from_checkpoint = resume_from_checkpoint
        if resume_from_checkpoint and config.engine != "transformers":
            self.load_checkpoint(self._resolve_resume_path(resume_from_checkpoint))

    def fit(
        self,
        train_data: Iterable[Any] | Any,
        *,
        steps: int | None = None,
        checkpoint_path: str | Path | None = None,
        trainer_kwargs: Mapping[str, Any] | None = None,
    ) -> ARTITrainingResult:
        target_steps = self.config.steps if steps is None else int(steps)
        if target_steps <= 0:
            raise ValueError("steps must be positive")
        if self.config.engine == "transformers":
            self._fit_transformers(train_data, target_steps, trainer_kwargs)
        elif self.config.engine == "accelerate":
            self._fit_accelerate(train_data, target_steps)
        else:
            self._fit_torch(train_data, target_steps)
        saved = None
        if checkpoint_path is not None:
            saved = self.save_checkpoint(checkpoint_path)
        return ARTITrainingResult(
            engine=self.config.engine,
            steps=self.global_step,
            loss_history=tuple(self.loss_history),
            checkpoint_path=saved,
        )

    def save_checkpoint(self, path: str | Path) -> Path:
        saved = self.attachment.save(
            path,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            training_state={
                "engine": self.config.engine,
                "global_step": self.global_step,
                "loss_history": list(self.loss_history),
            },
        )
        return saved.weights_path

    def load_checkpoint(self, path: str | Path, *, map_location: str | torch.device | None = None) -> None:
        loaded = self.attachment.load(
            path,
            map_location=map_location,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            load_checkpoint=True,
        )
        state = loaded.training_state
        if not isinstance(state, Mapping):
            raise ValueError("attachment artifact has no training checkpoint state")
        if state.get("engine") != self.config.engine:
            raise ValueError("attachment checkpoint belongs to a different training engine")
        self.global_step = int(state.get("global_step", 0))
        self.loss_history = [float(value) for value in state.get("loss_history", ())]

    def _fit_torch(self, train_data: Iterable[Any], steps: int) -> None:
        iterator = _cycling(train_data)
        accumulation = self.config.gradient_accumulation_steps
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        for _ in range(steps):
            total = 0.0
            for _micro in range(accumulation):
                batch = next(iterator)
                with _autocast(self.model, self.config.mixed_precision):
                    loss = _call_objective(self.objective, self.model, batch, self.attachment)
                (loss / accumulation).backward()
                total += float(loss.detach())
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()
            self.global_step += 1
            self.loss_history.append(total / accumulation)

    def _fit_accelerate(self, train_data: Iterable[Any], steps: int) -> None:
        try:
            from accelerate import Accelerator
        except ImportError as exc:
            raise RuntimeError("Accelerate integration requires `uv sync --extra qwen`") from exc
        accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )
        if self.scheduler is None:
            prepared_model, prepared_optimizer, prepared_data = accelerator.prepare(self.model, self.optimizer, train_data)
            prepared_scheduler = None
        else:
            prepared_model, prepared_optimizer, prepared_data, prepared_scheduler = accelerator.prepare(
                self.model, self.optimizer, train_data, self.scheduler
            )
        self.engine_object = accelerator
        iterator = _cycling(prepared_data)
        prepared_model.train()
        target_step = self.global_step + steps
        while self.global_step < target_step:
            batch = next(iterator)
            with accelerator.accumulate(prepared_model):
                loss = _call_objective(self.objective, prepared_model, batch, self.attachment)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    prepared_optimizer.step()
                    prepared_optimizer.zero_grad(set_to_none=True)
                    if prepared_scheduler is not None:
                        prepared_scheduler.step()
                    self.global_step += 1
                    self.loss_history.append(float(loss.detach()))
        self.optimizer = prepared_optimizer
        self.scheduler = prepared_scheduler

    def _fit_transformers(self, train_data: Any, steps: int, trainer_kwargs: Mapping[str, Any] | None) -> None:
        try:
            from transformers import Trainer, TrainingArguments
        except ImportError as exc:
            raise RuntimeError("Transformers integration requires `uv sync --extra qwen`") from exc
        options = dict(trainer_kwargs or {})
        arguments = options.pop("args", None)
        if arguments is None:
            values = dict(options.pop("training_args", {}))
            values.setdefault("output_dir", ".arti-trainer")
            values.setdefault("max_steps", steps)
            values.setdefault("learning_rate", self.config.learning_rate)
            values.setdefault("gradient_accumulation_steps", self.config.gradient_accumulation_steps)
            values.setdefault("bf16", self.config.mixed_precision == "bf16")
            values.setdefault("fp16", self.config.mixed_precision == "fp16")
            values.setdefault("report_to", [])
            values.setdefault("save_strategy", "no")
            values.setdefault("save_only_model", True)
            arguments = TrainingArguments(**values)
        objective = self.objective
        attachment = self.attachment
        callbacks = list(options.pop("callbacks", ()))

        session = self

        class AttachmentTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **_kwargs):
                loss = _call_objective(objective, model, inputs, attachment)
                if return_outputs:
                    return loss, {"loss": loss}
                return loss

            def _save(self, output_dir=None, state_dict=None):
                del state_dict
                target = Path(output_dir or self.args.output_dir)
                target.mkdir(parents=True, exist_ok=True)
                history = [float(row["loss"]) for row in self.state.log_history if "loss" in row]
                attachment.save(
                    target / TRANSFORMERS_ARTIFACT,
                    optimizer=self.optimizer,
                    scheduler=self.lr_scheduler,
                    training_state={
                        "engine": "transformers",
                        "global_step": session.global_step + int(self.state.global_step),
                        "loss_history": [*session.loss_history, *history],
                    },
                )

        trainer = AttachmentTrainer(
            model=self.model,
            args=arguments,
            train_dataset=train_data,
            optimizers=(self.optimizer, self.scheduler),
            callbacks=[ARTICheckpointCallback(), *callbacks],
            **options,
        )
        if self.resume_from_checkpoint:
            resume_path = self._resolve_resume_path(self.resume_from_checkpoint, output_dir=Path(arguments.output_dir))
            trainer.create_optimizer_and_scheduler(num_training_steps=steps)
            self.optimizer = trainer.optimizer
            self.scheduler = trainer.lr_scheduler
            self.load_checkpoint(resume_path)
        output = trainer.train()
        self.engine_object = trainer
        self.optimizer = trainer.optimizer
        self.scheduler = trainer.lr_scheduler
        history = [float(row["loss"]) for row in trainer.state.log_history if "loss" in row]
        self.loss_history.extend(history or [float(output.training_loss)])
        self.global_step += int(output.global_step)

    def _resolve_resume_path(
        self,
        value: str | Path | bool,
        *,
        output_dir: Path | None = None,
    ) -> Path:
        if value is True:
            candidate = getattr(self.attachment, "_resume_artifact", None)
            if candidate is not None:
                return Path(candidate)
            if output_dir is not None:
                checkpoints = sorted(
                    (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
                    key=lambda path: int(path.name.rsplit("-", 1)[-1]),
                )
                if checkpoints:
                    return checkpoints[-1] / TRANSFORMERS_ARTIFACT
            raise ValueError("no ARTI checkpoint is available for automatic resume")
        path = Path(value)
        return path / TRANSFORMERS_ARTIFACT if path.is_dir() else path


def resolve_attachment_objective(
    objective: str | AttachmentObjective,
    *,
    corruption_probability: float = 0.15,
) -> AttachmentObjective:
    if callable(objective):
        return objective
    if objective == "recall_alignment":
        return lambda model, batch, attachment: recall_alignment_objective(
            attachment, batch, corruption_probability=corruption_probability
        )
    if objective == "model_loss":
        return model_loss_objective
    raise ValueError("objective must be 'recall_alignment', 'model_loss', or a callable")


def recall_alignment_objective(
    attachment: "ARTIAttachment",
    batch: Any,
    *,
    corruption_probability: float = 0.15,
) -> Tensor:
    if isinstance(batch, Mapping) and "clean_inputs" in batch:
        clean = batch["clean_inputs"]
        corrupt = batch.get("corrupt_inputs")
        unseen = batch.get("unseen_inputs")
        mask = batch.get("mask")
    else:
        clean = batch.get("inputs") if isinstance(batch, Mapping) and "inputs" in batch else batch
        corrupt = None
        unseen = None
        mask = None
    if corrupt is None:
        corrupt = _corrupt_floating(clean, corruption_probability)
    return layered_recall_trajectory_loss(
        attachment._layered,
        clean,
        corrupt,
        unseen_inputs=unseen,
        mask=mask,
    ).loss


def model_loss_objective(model: torch.nn.Module, batch: Any, _attachment: "ARTIAttachment" | None = None) -> Tensor:
    output = model(**batch) if isinstance(batch, Mapping) else model(batch)
    loss = output.get("loss") if isinstance(output, Mapping) else getattr(output, "loss", None)
    if not isinstance(loss, Tensor):
        raise ValueError("model_loss objective requires model output with a Tensor .loss or ['loss']")
    return loss


def _call_objective(objective: AttachmentObjective, model: torch.nn.Module, batch: Any, attachment: "ARTIAttachment") -> Tensor:
    parameters = inspect.signature(objective).parameters
    loss = objective(model, batch, attachment) if len(parameters) >= 3 else objective(model, batch)
    if not isinstance(loss, Tensor) or loss.ndim != 0:
        raise ValueError("attachment objective must return a scalar Tensor")
    return loss


def _corrupt_floating(value: Any, probability: float) -> Any:
    if isinstance(value, Tensor):
        if not value.is_floating_point():
            raise ValueError("automatic Recall corruption requires floating tensors; provide clean_inputs/corrupt_inputs")
        keep = torch.rand(value.shape[:-1], device=value.device) >= probability
        return value * keep.unsqueeze(-1).to(value.dtype)
    if isinstance(value, Mapping):
        return {key: _corrupt_floating(item, probability) if isinstance(item, Tensor) and item.is_floating_point() else item for key, item in value.items()}
    raise ValueError("automatic Recall corruption requires a Tensor or tensor mapping")


def _cycling(data: Iterable[Any]):
    while True:
        produced = False
        for item in data:
            produced = True
            yield item
        if not produced:
            raise ValueError("training data must not be empty")


def _autocast(model: torch.nn.Module, precision: str):
    if precision == "no":
        return nullcontext()
    parameter = next(model.parameters(), torch.empty(0))
    device_type = parameter.device.type
    if device_type not in {"cpu", "cuda"}:
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device_type, dtype=dtype)


__all__ = [
    "ARTITrainingResult",
    "ARTITrainingSession",
    "ARTICheckpointCallback",
    "AttachmentObjective",
    "model_loss_objective",
    "recall_alignment_objective",
    "resolve_attachment_objective",
]
