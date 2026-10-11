"""Thin Trainer integration; the selected Hugging Face class owns the loop."""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
import torch


ARTIFACT = "model.arti.st"


def _resume_location(value, output_dir, owner):
    if not value:
        return None
    if value is True:
        from transformers.trainer_utils import get_last_checkpoint

        value = get_last_checkpoint(str(output_dir)) or getattr(owner, "_resume_artifact", None)
        if value is None:
            raise ValueError("no Transformers checkpoint is available for automatic resume")
    path = Path(value)
    if path.is_file() and path.name == ARTIFACT and (path.parent / "trainer_state.json").exists():
        return path.parent
    return path


class _ARTITrainerMixin:
    def __init__(self, *args, arti_session, **kwargs):
        self.arti_session = arti_session
        self._arti_step_offset = arti_session.global_step
        self._arti_loss_prefix = list(arti_session.loss_history)
        self._arti_last_artifact = None
        self._arti_best_view = False
        self._arti_micro_batches_seen = arti_session.micro_batches_seen
        super().__init__(*args, **kwargs)
        if self.model is not arti_session.model:
            raise ValueError("Trainer must retain the model owned by the ARTI training binding")
        if self.is_deepspeed_enabled or self.is_fsdp_enabled:
            raise ValueError(
                "the ARTI-only Trainer checkpoint bridge does not yet support sharded engines"
            )
        if not self._arti_native_loss:
            self.model_accepts_loss_kwargs = False
            self.loss_is_scaled_for_ga = False

    def create_optimizer(self, *args, **kwargs):
        selected = {id(p) for p in self.arti_session.binding.parameters()}
        excluded = [p for p in self.model.parameters() if p.requires_grad and id(p) not in selected]
        # Scope the native optimizer factory, without changing the execution model.
        try:
            for parameter in excluded:
                parameter.requires_grad_(False)
            result = super().create_optimizer(*args, **kwargs)
        finally:
            for parameter in excluded:
                parameter.requires_grad_(True)
        actual = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
        if actual != selected:
            raise ValueError("Trainer optimizer must own exactly the selected ARTI parameter role")
        return result

    @property
    def _arti_native_loss(self):
        from .attachment_training import model_loss_objective

        objective = self.arti_session.objective
        return objective is model_loss_objective or getattr(
            objective, "_arti_native_model_loss", False
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self._arti_native_loss:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        loss = self.arti_session._objective(model, inputs, self.arti_session.attachment)
        if return_outputs:
            return loss, {"loss": loss}
        return loss

    def training_step(self, *args, **kwargs):
        result = super().training_step(*args, **kwargs)
        self._arti_micro_batches_seen += 1
        return result

    def _arti_training_state(self):
        history = [float(row["loss"]) for row in self.state.log_history if "loss" in row]
        return {
            **self.arti_session.binding.checkpoint_metadata,
            "engine": "transformers",
            "binding_kind": self.arti_session.binding.kind,
            "global_step": self._arti_step_offset + int(self.state.global_step),
            "loss_history": [*self._arti_loss_prefix, *history],
            "transformers_step_offset": self._arti_step_offset,
            "transformers_loss_prefix": self._arti_loss_prefix,
            "micro_batches_seen": self._arti_micro_batches_seen,
        }

    def _arti_optimizer_metadata(self):
        fields = (
            "learning_rate",
            "weight_decay",
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "optim",
            "optim_args",
            "lr_scheduler_type",
            "lr_scheduler_kwargs",
            "warmup_steps",
        )
        return {
            "transformers_optimizer": None
            if self.arti_session._optimizer_supplied
            else {
                field: getattr(getattr(self.args, field), "value", getattr(self.args, field))
                for field in fields
            },
            "transformers_max_steps": self.args.max_steps,
            "transformers_optimizer_supplied": self.arti_session._optimizer_supplied,
            "transformers_scheduler_supplied": self.arti_session._scheduler_supplied,
        }

    def _sync_session(self):
        session = self.arti_session
        state = self._arti_training_state()
        session.global_step = state["global_step"]
        session.loss_history = state["loss_history"]
        session.optimizer = self.optimizer
        session.scheduler = self.lr_scheduler
        session.micro_batches_seen = self._arti_micro_batches_seen
        session.engine_object = self

    def _save(self, output_dir=None, state_dict=None):
        del state_dict
        target = Path(output_dir or self.args.output_dir)
        target.mkdir(parents=True, exist_ok=True)
        self.arti_session.binding.save(
            target / ARTIFACT, training_state=self._arti_training_state()
        )
        processor = self.processing_class or getattr(self.data_collator, "tokenizer", None)
        if processor is not None:
            processor.save_pretrained(target)
        torch.save(self.args, target / "training_args.bin")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        artifact = Path(resume_from_checkpoint) / ARTIFACT
        if not artifact.exists():
            return super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        loaded = self.arti_session.binding.load(artifact, load_checkpoint=True)
        state = loaded.training_state or {}
        self._arti_step_offset = int(state.get("transformers_step_offset", 0))
        self._arti_loss_prefix = list(state.get("transformers_loss_prefix", ()))
        self._arti_micro_batches_seen = int(state.get("micro_batches_seen", 0))

    def _load_best_model(self):
        self._sync_session()
        # Preserve a coherent last-state asset before the native best-weight view.
        target = Path(self.args.output_dir) / "last" / ARTIFACT
        self._arti_last_artifact = self.arti_session.save_checkpoint(target)
        artifact = Path(self.state.best_model_checkpoint) / ARTIFACT
        if artifact.exists():
            self.arti_session.binding.load(artifact, load_checkpoint=False)
        else:
            super()._load_best_model()
        self._arti_best_view = True
        self.arti_session._transformers_selection = {
            "best_checkpoint": artifact,
            "selected_checkpoint": artifact,
            "receipt": {
                "selection": "best",
                "policy": "best",
                "source": "transformers",
                "monitor": self.args.metric_for_best_model,
                "best_metric": self.state.best_metric,
                "last_step": self.arti_session.global_step,
            },
        }

    def _restore_without_training(self):
        from transformers.trainer_callback import TrainerState

        session = self.arti_session
        location = _resume_location(
            session.resume_from_checkpoint, self.args.output_dir, session.attachment
        )
        if location.is_dir():
            self.state = TrainerState.load_from_json(str(location / "trainer_state.json"))
            self.create_optimizer_and_scheduler(num_training_steps=self.state.max_steps)
            self._load_from_checkpoint(str(location))
            self._load_optimizer_and_scheduler(str(location))
            self._load_rng_state(str(location))
            self._sync_session()
        else:
            self.create_optimizer_and_scheduler(num_training_steps=self.args.max_steps)
            session.optimizer, session.scheduler = self.optimizer, self.lr_scheduler
            session.engine_object = self
            session.load_checkpoint(location)

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        if self.arti_session._checkpoint_view:
            raise ValueError("cannot train inside a model-only checkpoint context")
        if self._arti_best_view:
            self.arti_session.binding.load(self._arti_last_artifact, load_checkpoint=False)
            self._arti_best_view = False
        self._arti_last_artifact = None
        self.arti_session._transformers_selection = {}
        self._arti_micro_batches_seen = self.arti_session.micro_batches_seen
        if resume_from_checkpoint is None:
            resume_from_checkpoint = self.arti_session.resume_from_checkpoint
        location = _resume_location(
            resume_from_checkpoint, self.args.output_dir, self.arti_session.attachment
        )
        if location is None:
            self._arti_step_offset = self.arti_session.global_step
            self._arti_loss_prefix = list(self.arti_session.loss_history)
        if location is not None and location.is_file():
            warnings.warn(
                "A portable ARTI file restores weights/optimizer, but starts a new Trainer data/RNG trajectory. "
                "Resume a checkpoint directory for native Trainer continuation.",
                UserWarning,
                stacklevel=2,
            )
            self.arti_session.binding.load(location, load_checkpoint=False)
            self.create_optimizer_and_scheduler(num_training_steps=self.args.max_steps)
            self.arti_session.optimizer = self.optimizer
            self.arti_session.scheduler = self.lr_scheduler
            self.arti_session.engine_object = self
            self.arti_session.load_checkpoint(location)
            self._arti_step_offset = self.arti_session.global_step
            self._arti_loss_prefix = list(self.arti_session.loss_history)
            self._arti_micro_batches_seen = self.arti_session.micro_batches_seen
            self._created_lr_scheduler = False
            location = None
        self.arti_session.resume_from_checkpoint = None
        try:
            return super().train(
                resume_from_checkpoint=None if location is None else str(location), *args, **kwargs
            )
        finally:
            self._sync_session()


def build_transformers_trainer(
    session, train_data, *, steps=None, trainer_class=None, trainer_kwargs=None, prepare_resume=True
):
    from transformers import Trainer, TrainingArguments

    from .attachment_training import ARTICheckpointCallback

    if getattr(session.engine_object, "_arti_best_view", False):
        session.binding.load(session.engine_object._arti_last_artifact, load_checkpoint=False)
        session.engine_object._arti_best_view = False

    base = Trainer if trainer_class is None else trainer_class
    if not isinstance(base, type) or not issubclass(base, Trainer):
        raise TypeError("trainer_class must be a Transformers Trainer subclass")
    options = dict(trainer_kwargs or {})
    arguments = options.pop("args", None)
    if arguments is None:
        values = dict(options.pop("training_args", {}))
        for key in tuple(options):
            if key in TrainingArguments.__dataclass_fields__:
                values[key] = options.pop(key)
        values.setdefault("output_dir", ".arti-trainer")
        values.setdefault("max_steps", session.config.steps)
        values.setdefault("learning_rate", session.config.learning_rate)
        values.setdefault("gradient_accumulation_steps", session.config.gradient_accumulation_steps)
        values.setdefault("bf16", session.config.mixed_precision == "bf16")
        values.setdefault("fp16", session.config.mixed_precision == "fp16")
        values.setdefault("report_to", [])
        values.setdefault("save_only_model", False)
        if session.config.device == "cpu":
            values.setdefault("use_cpu", True)
        arguments = TrainingArguments(**values)
    else:
        arguments = copy.deepcopy(arguments)
    location = (
        _resume_location(session.resume_from_checkpoint, arguments.output_dir, session.attachment)
        if prepare_resume
        else None
    )
    if location is not None and location.is_file():
        session.load_checkpoint(location)
    if arguments.gradient_checkpointing and not arguments.gradient_checkpointing_kwargs:
        arguments.gradient_checkpointing_kwargs = {"use_reentrant": False}
    if steps is not None:
        if type(steps) is not int or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        previous = 0
        if location is not None and location.is_dir():
            with (location / "trainer_state.json").open(encoding="utf-8") as stream:
                previous = int(json.load(stream)["global_step"])
        arguments.max_steps = previous + steps
    if any(
        key in options for key in ("model", "model_init", "optimizers", "optimizer_cls_and_kwargs")
    ):
        raise ValueError("model and optimizer ownership belong to the ARTI TrainingBinding")
    callbacks = list(options.pop("callbacks", ()))
    adapted = type("ARTITrainer", (_ARTITrainerMixin, base), {"__module__": __name__})
    trainer = adapted(
        arti_session=session,
        model=session.model,
        args=arguments,
        train_dataset=train_data,
        optimizers=(
            session.optimizer if session._optimizer_supplied or session.optimizer.state else None,
            session.scheduler,
        ),
        callbacks=[ARTICheckpointCallback(), *callbacks],
        **options,
    )
    if session.config.distributed and int(trainer.args.world_size) < 2:
        raise RuntimeError(
            "distributed=True requires torchrun or accelerate launch on at least two processes"
        )
    return trainer
