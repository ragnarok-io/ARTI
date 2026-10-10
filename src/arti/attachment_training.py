"""Training engines and objectives for unified ARTI attachments."""

from __future__ import annotations

import copy
import inspect
import json
import math
import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TYPE_CHECKING

import torch
from torch import Tensor

from .training_config import ARTITrainingConfig
from .training_binding import TrainingBinding
from .fit.scanner import run_model
from .training_run import ARTITrainingRunConfig, RunReporter, _TrainingRun

try:
    from transformers import TrainerCallback as _TrainerCallback
except ImportError:
    _TrainerCallback = object


TRANSFORMERS_ARTIFACT = "model.arti.st"


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
    run_directory: Path | None = None
    stop_reason: str = "steps_completed"


@dataclass(frozen=True)
class ARTIEvaluationResult:
    loss: float
    batches: int
    weight: float
    split: str


class ARTITrainingSession:
    """Shared trainer for attachment, pretrained and native-model bindings."""

    def __init__(
        self,
        attachment: "ARTIAttachment | TrainingBinding",
        *,
        config: ARTITrainingConfig,
        objective: str | AttachmentObjective | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        resume_from_checkpoint: str | Path | bool | None = None,
    ) -> None:
        self.binding = attachment if isinstance(attachment, TrainingBinding) else TrainingBinding.from_attachment(attachment)
        self.attachment = self.binding.owner
        self.model = self.binding.model
        self.config = config
        if config.device != "auto":
            self.model.to(config.device)
        if (objective or config.objective) == "tensor_alignment" and self.binding.kind != "attachment":
            raise ValueError("tensor_alignment requires an attachment; native/provider training uses its task objective")
        self.objective = resolve_attachment_objective(objective or config.objective, corruption_probability=config.corruption_probability)
        self._objective = _bind_objective(self.objective)
        parameters = list(self.binding.parameters())
        if not parameters:
            raise ValueError(
                "training binding has no trainable parameters; configure a trainable model or program graph"
            )
        owned = {id(p) for p in self.binding.state_module().parameters()}
        if len({id(p) for p in parameters}) != len(parameters) or any(id(p) not in owned or not p.requires_grad for p in parameters):
            raise ValueError("training parameters must be unique trainable parameters owned by the state module")
        if optimizer is not None and _optimizer_parameter_ids(optimizer) != {id(parameter) for parameter in parameters}:
            raise ValueError("optimizer parameters must exactly match the selected ARTI trainable role")
        self.optimizer = optimizer or torch.optim.AdamW(parameters, lr=config.learning_rate)
        self.scheduler = scheduler
        self.global_step = 0
        self.loss_history: list[float] = []
        self.engine_object: Any | None = None
        self.resume_from_checkpoint = resume_from_checkpoint
        self.micro_batches_seen = 0
        self._data_state: Any | None = None
        self._train_data: Any | None = None
        self._scaler: Any | None = None
        self._scaler_state: Any | None = None
        self._run: _TrainingRun | None = None
        self._pending_losses: list[Tensor] = []
        self._stop_reason = "steps_completed"
        self._evaluation_data: Iterable[Any] | None = None
        self._evaluation_weight: Callable[[Any], float] | None = None
        self._objective_weight: Callable[[Any], float] | None = None
        if resume_from_checkpoint and config.engine != "transformers":
            self.load_checkpoint(self._resolve_resume_path(resume_from_checkpoint))

    def fit(
        self,
        train_data: Iterable[Any] | Any,
        *,
        steps: int | None = None,
        checkpoint_path: str | Path | None = None,
        trainer_kwargs: Mapping[str, Any] | None = None,
        run: ARTITrainingRunConfig | None = None,
        eval_data: Iterable[Any] | None = None,
        evaluation_weight: Callable[[Any], float] | None = None,
        objective_weight: Callable[[Any], float] | None = None,
        max_grad_norm: float | None = None,
        reporter: RunReporter | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> ARTITrainingResult:
        target_steps = self.config.steps if steps is None else int(steps)
        if target_steps < 0:
            raise ValueError("steps must be non-negative")
        if self.config.distributed and self.config.engine == "torch":
            raise ValueError("distributed training requires engine='accelerate' or 'transformers'")
        if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be finite and positive")
        if self.config.engine != "torch" and (run is not None or eval_data is not None or reporter is not None or stop_requested is not None or objective_weight is not None or max_grad_norm is not None):
            raise ValueError("local run policy currently requires engine='torch'; other engines own their run lifecycle")
        self._stop_reason = "steps_completed"
        self._stop_requested = stop_requested
        self._train_data = train_data
        self._evaluation_data = eval_data
        self._evaluation_weight = evaluation_weight
        self._objective_weight = objective_weight
        self._max_grad_norm = max_grad_norm
        if self.config.engine == "torch":
            policy = run or ARTITrainingRunConfig()
            if policy.evaluation_steps is not None and eval_data is None:
                raise ValueError("evaluation_steps requires explicit eval_data")
            self._run = _TrainingRun(policy, self.plan(steps=target_steps), reporter)
        try:
            if target_steps == 0:
                pass
            elif self.config.engine == "transformers":
                self._fit_transformers(train_data, target_steps, trainer_kwargs)
            elif self.config.engine == "accelerate":
                self._fit_accelerate(train_data, target_steps)
            else:
                self._fit_torch(train_data, target_steps)
            saved = None
            if checkpoint_path is not None:
                saved = self.save_checkpoint(checkpoint_path)
            if self._run is not None:
                self._save_run_checkpoint()
                saved = saved or self._run.latest
                self._run.payload.update(status="completed", stop_reason=self._stop_reason, global_step=self.global_step)
                self._run.write()
        except BaseException as exc:
            if self._run is not None:
                self._run.payload.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error_type=type(exc).__name__, completed_optimizer_steps=self.global_step)
                self._run.write()
            self._pending_losses.clear()
            raise
        finally:
            self._train_data = None
        return ARTITrainingResult(
            engine=self.config.engine,
            steps=self.global_step,
            loss_history=tuple(self.loss_history),
            checkpoint_path=saved,
            run_directory=None if self._run is None else self._run.directory,
            stop_reason=self._stop_reason,
        )

    def plan(self, *, steps: int | None = None) -> dict[str, Any]:
        summary = self.binding.describe()
        return {
            "training": {**self.config.__dict__, "steps": self.config.steps if steps is None else steps},
            "objective": self.config.objective if self.objective is model_loss_objective else getattr(self.objective, "__qualname__", type(self.objective).__name__),
            "model": summary,
            **({"attachment": summary} if self.binding.kind == "attachment" else {}),
            "binding_kind": self.binding.kind,
            "global_step": self.global_step,
            "estimates_are_static": True,
        }

    def check(self, batch: Any) -> dict[str, Any]:
        """Check the actual task/gradient connection without an optimizer update."""
        with self._evaluation_snapshot(), torch.enable_grad(), _autocast(self.model, self.config.mixed_precision):
            loss = self._objective(self.model, batch, self.attachment)
            if not loss.requires_grad:
                raise ValueError("task loss is disconnected from ARTI trainable parameters")
            parameters = self.binding.parameters()
            gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
            connected = sum(gradient is not None for gradient in gradients)
            finite = bool(torch.isfinite(loss).all()) and all(bool(torch.isfinite(g).all()) for g in gradients if g is not None)
            return {"loss": float(loss.detach()), "connected_parameters": connected, "total_parameters": len(parameters), "finite": finite, "ok": finite and connected > 0}

    def evaluate(
        self,
        data: Iterable[Any],
        *,
        split: str = "validation",
        weight: Callable[[Any], float] | None = None,
    ) -> ARTIEvaluationResult:
        """Read-only evaluation; the caller defines sample/token weighting."""
        losses = []
        weights = []
        with self._evaluation_snapshot(), torch.no_grad():
            self.model.eval()
            for batch in data:
                with _autocast(self.model, self.config.mixed_precision):
                    losses.append(self._objective(self.model, batch, self.attachment).detach().float())
                local_weight = 1.0 if weight is None else float(weight(batch))
                if not math.isfinite(local_weight) or local_weight <= 0:
                    raise ValueError("evaluation weights must be finite and positive")
                weights.append(local_weight)
        if not losses:
            raise ValueError("evaluation data must not be empty")
        stacked = torch.stack(losses)
        weighted = (stacked * stacked.new_tensor(weights)).sum() / sum(weights)
        return ARTIEvaluationResult(float(weighted), len(losses), sum(weights), split)

    def evaluation_context(self):
        """Preserve model modes/RNG and ARTI state while computing custom metrics."""
        return self._evaluation_snapshot()

    @contextmanager
    def _evaluation_snapshot(self):
        from .resource_graph import ProgramGraph

        bundle = self.binding.state_module()
        state = copy.deepcopy(bundle.state_dict())
        graph_states = [(module, module.state()) for module in bundle.modules() if isinstance(module, ProgramGraph)]
        buffers = [(buffer, buffer.detach().clone()) for buffer in self.model.buffers()]
        modes = [(module, module.training) for module in self.model.modules()]
        rng = _rng_state()
        try:
            yield
        finally:
            bundle.load_state_dict(state)
            for graph, graph_state in graph_states:
                graph.restore_state(graph_state)
            with torch.no_grad():
                for buffer, value in buffers:
                    buffer.copy_(value)
            for module, training in modes:
                module.training = training
            _restore_rng(rng)

    def _checkpoint_state(self) -> dict[str, Any]:
        state = {
            **self.binding.checkpoint_metadata,
            "engine": self.config.engine,
            "binding_kind": self.binding.kind,
            "global_step": self.global_step,
            "steps": self.global_step,
            "loss_history": list(self.loss_history),
            "micro_batches_seen": self.micro_batches_seen,
            "rng": _rng_state(),
            "scaler": None if self._scaler is None else self._scaler.state_dict(),
        }
        if self._train_data is not None and callable(getattr(self._train_data, "state_dict", None)):
            state["data_state"] = self._train_data.state_dict()
        return state

    def _flush_losses(self) -> None:
        if not self._pending_losses:
            return
        values = torch.stack(self._pending_losses).cpu().tolist()
        self.loss_history.extend(values)
        self._pending_losses.clear()
        if self._run is not None:
            self._run.emit("metrics", step=self.global_step, loss=values[-1], mean_loss=sum(values) / len(values), optimizer_steps=len(values))

    def _save_run_checkpoint(self) -> None:
        assert self._run is not None
        self._flush_losses()
        path = self._run.checkpoint_path(self.global_step)
        if path != self._run.latest:
            saved = self.save_checkpoint(path)
            self._run.saved(saved, self.global_step)
    def save_checkpoint(self, path: str | Path) -> Path:
        saved = self.binding.save(
            path,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            training_state=self._checkpoint_state(),
        )
        return saved.weights_path

    def load_checkpoint(self, path: str | Path, *, map_location: str | torch.device | None = None) -> None:
        loaded = self.binding.load(
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
        if state.get("binding_kind", self.binding.kind) != self.binding.kind:
            raise ValueError("checkpoint belongs to a different training binding")
        self.global_step = int(state.get("global_step", state.get("steps", 0)))
        self.loss_history = [float(value) for value in state.get("loss_history", ())]
        self.micro_batches_seen = int(state.get("micro_batches_seen", 0))
        self._data_state = state.get("data_state")
        self._scaler_state = state.get("scaler")
        if state.get("rng") is not None:
            _restore_rng(state["rng"])

    def _fit_torch(self, train_data: Iterable[Any], steps: int) -> None:
        iterator = self._training_iterator(train_data)
        accumulation = self.config.gradient_accumulation_steps
        self.model.train()
        parameter = next(self.model.parameters(), torch.empty(0))
        use_scaler = self.config.mixed_precision == "fp16" and parameter.device.type == "cuda"
        scaler = (self._scaler or torch.amp.GradScaler("cuda", enabled=True)) if use_scaler else None
        self._scaler = scaler
        if scaler is not None and self._scaler_state is not None:
            scaler.load_state_dict(self._scaler_state)
            self._scaler_state = None
        self.optimizer.zero_grad(set_to_none=True)
        first_step = self.global_step + 1
        for _ in range(steps):
            if self._run is not None and self._run.expired():
                self._stop_reason = "time_budget"
                break
            if self._stop_requested is not None and self._stop_requested():
                self._stop_reason = "requested"
                break
            totals = []
            batches = [next(iterator) for _micro in range(accumulation)]
            weights = [1.0 if self._objective_weight is None else float(self._objective_weight(batch)) for batch in batches]
            if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
                raise ValueError("objective weights must be finite and positive")
            total_weight = sum(weights)
            for batch, weight in zip(batches, weights, strict=True):
                with _autocast(self.model, self.config.mixed_precision):
                    loss = self._objective(self.model, batch, self.attachment)
                scaled_loss = loss * (weight / total_weight)
                if scaler is None:
                    scaled_loss.backward()
                else:
                    scaler.scale(scaled_loss).backward()
                totals.append(loss.detach().float() * (weight / total_weight))
                self.micro_batches_seen += 1
            if self._max_grad_norm is not None:
                if scaler is not None:
                    scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.binding.parameters(), self._max_grad_norm)
            if scaler is None:
                self.optimizer.step()
            else:
                scaler.step(self.optimizer)
                scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()
            self.global_step += 1
            self._pending_losses.append(torch.stack(totals).sum())
            assert self._run is not None
            if self._run.checkpoint_due(self.global_step, first_step):
                self._save_run_checkpoint()
            elif self.global_step % self._run.config.log_steps == 0:
                self._flush_losses()
            interval = self._run.config.evaluation_steps
            if interval is not None and self.global_step % interval == 0:
                evaluation = self.evaluate(self._evaluation_data, weight=self._evaluation_weight)
                self._run.payload["evaluations"].append({"step": self.global_step, **evaluation.__dict__})
                self._run.emit("evaluation", step=self.global_step, **evaluation.__dict__)
                self._run.write()
        self._flush_losses()

    def _training_iterator(self, data: Iterable[Any]):
        if self._data_state is not None:
            loader = getattr(data, "load_state_dict", None)
            if not callable(loader):
                raise ValueError("checkpoint contains data_state; provide a resumable data loader")
            loader(self._data_state)
            self._data_state = None
        if isinstance(data, Sequence):
            if len(data) == 0:
                raise ValueError("training data must not be empty")
            if self._run is not None:
                self._run.payload["data_resume"] = "indexed_sequence"
            offset = self.micro_batches_seen
            while True:
                yield data[offset % len(data)]
                offset += 1
        else:
            if self._run is not None:
                self._run.payload["data_resume"] = "loader_state" if callable(getattr(data, "state_dict", None)) else "weights_optimizer_rng_only"
            yield from _cycling(data)

    def _fit_accelerate(self, train_data: Iterable[Any], steps: int) -> None:
        try:
            from accelerate import Accelerator, DistributedDataParallelKwargs
            from accelerate.utils import GradientAccumulationPlugin
        except ImportError as exc:
            raise RuntimeError("Accelerate integration requires `uv sync --extra qwen`") from exc
        accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=self.config.gradient_accumulation_steps, sync_with_dataloader=False),
            cpu=self.config.device == "cpu",
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)] if self.config.distributed else None,
        )
        if self.config.distributed and accelerator.num_processes < 2:
            raise RuntimeError("distributed=True requires torchrun or a multi-process accelerate launch")
        if self.scheduler is None:
            prepared_model, prepared_optimizer, prepared_data = accelerator.prepare(self.model, self.optimizer, train_data)
            prepared_scheduler = None
        else:
            prepared_model, prepared_optimizer, prepared_data, prepared_scheduler = accelerator.prepare(
                self.model, self.optimizer, train_data, self.scheduler
            )
        self.engine_object = accelerator
        self._scaler = accelerator.scaler
        if self._scaler is not None and self._scaler_state is not None:
            self._scaler.load_state_dict(self._scaler_state)
            self._scaler_state = None
        iterator = _cycling(prepared_data)
        prepared_model.train()
        target_step = self.global_step + steps
        totals = []
        prepared_optimizer.zero_grad(set_to_none=True)
        while self.global_step < target_step:
            batch = next(iterator)
            with accelerator.accumulate(prepared_model):
                loss = self._objective(prepared_model, batch, self.attachment)
                accelerator.backward(loss)
                self.micro_batches_seen += 1
                totals.append(loss.detach().float())
                if accelerator.sync_gradients:
                    prepared_optimizer.step()
                    prepared_optimizer.zero_grad(set_to_none=True)
                    if prepared_scheduler is not None:
                        prepared_scheduler.step()
                    self.global_step += 1
                    self._pending_losses.append(torch.stack(totals).mean())
                    totals.clear()
                    if len(self._pending_losses) >= 10:
                        self._flush_losses()
        self._flush_losses()
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
            for name in tuple(options):
                if name in getattr(TrainingArguments, "__dataclass_fields__", {}):
                    values[name] = options.pop(name)
            values.setdefault("output_dir", ".arti-trainer")
            values.setdefault("max_steps", steps)
            values.setdefault("learning_rate", self.config.learning_rate)
            values.setdefault("gradient_accumulation_steps", self.config.gradient_accumulation_steps)
            values.setdefault("bf16", self.config.mixed_precision == "bf16")
            values.setdefault("fp16", self.config.mixed_precision == "fp16")
            values.setdefault("report_to", [])
            values.setdefault("save_strategy", "no")
            values.setdefault("save_only_model", True)
            if self.config.device == "cpu":
                values.setdefault("use_cpu", True)
            arguments = TrainingArguments(**values)
        objective = self._objective
        attachment = self.attachment
        binding = self.binding
        callbacks = list(options.pop("callbacks", ()))

        session = self

        class ARTITrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **_kwargs):
                if session.objective is model_loss_objective or getattr(session.objective, "_arti_native_model_loss", False):
                    return super().compute_loss(model, inputs, return_outputs=return_outputs, **_kwargs)
                loss = objective(model, inputs, attachment)
                if return_outputs:
                    return loss, {"loss": loss}
                return loss

            def _save(self, output_dir=None, state_dict=None):
                del state_dict
                target = Path(output_dir or self.args.output_dir)
                target.mkdir(parents=True, exist_ok=True)
                history = [float(row["loss"]) for row in self.state.log_history if "loss" in row]
                binding.save(
                    target / TRANSFORMERS_ARTIFACT,
                    optimizer=self.optimizer,
                    scheduler=self.lr_scheduler,
                    training_state={
                        **binding.checkpoint_metadata,
                        "engine": "transformers",
                        "binding_kind": binding.kind,
                        "global_step": session.global_step + int(self.state.global_step),
                        "loss_history": [*session.loss_history, *history],
                    },
                )

        trainer = ARTITrainer(
            model=self.model,
            args=arguments,
            train_dataset=train_data,
            optimizers=(self.optimizer, self.scheduler),
            callbacks=[ARTICheckpointCallback(), *callbacks],
            **options,
        )
        if self.config.distributed and int(getattr(trainer.args, "world_size", 1)) < 2:
            raise RuntimeError("distributed=True requires torchrun or accelerate launch on at least two processes")
        if self.resume_from_checkpoint:
            resume_path = self._resolve_resume_path(self.resume_from_checkpoint, output_dir=Path(arguments.output_dir))
            trainer.create_optimizer_and_scheduler(num_training_steps=steps)
            self.optimizer = trainer.optimizer
            self.scheduler = trainer.lr_scheduler
            self.load_checkpoint(resume_path)
            self.resume_from_checkpoint = None
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
    if objective == "tensor_alignment":
        return lambda model, batch, attachment: tensor_alignment_objective(
            attachment, batch, corruption_probability=corruption_probability
        )
    if objective == "model_loss":
        return model_loss_objective
    raise ValueError("objective must be 'tensor_alignment', 'model_loss', or a callable")


def tensor_alignment_objective(
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
    wrappers = attachment._layers.wrappers
    previous = {
        path: (wrapper.enabled, wrapper.capture)
        for path, wrapper in wrappers.items()
    }
    try:
        for wrapper in wrappers.values():
            wrapper.enabled = False
            wrapper.capture = True
            wrapper.clear_trace()
        run_model(attachment._model, clean)
        clean_targets = {
            path: _required_trace(wrapper.last_pre, path, "clean").detach()
            for path, wrapper in wrappers.items()
        }
        for wrapper in wrappers.values():
            wrapper.enabled = True
            wrapper.clear_trace()
        run_model(attachment._model, corrupt)
        losses = []
        for path, wrapper in wrappers.items():
            post = _required_trace(wrapper.last_post, path, "corrupt")
            error = (post - clean_targets[path]).square().mean(dim=-1)
            local_mask = _alignment_mask(error, mask)
            losses.append(
                (error * local_mask.to(error.dtype)).sum()
                / local_mask.sum().clamp_min(1)
            )
        loss = torch.stack(losses).mean()
        if unseen is not None:
            for wrapper in wrappers.values():
                wrapper.clear_trace()
            run_model(attachment._model, unseen)
            unseen_loss = torch.stack(
                [
                    _required_trace(wrapper.last_delta, path, "unseen")
                    .square()
                    .mean()
                    for path, wrapper in wrappers.items()
                ]
            ).mean()
            loss = loss + 0.25 * unseen_loss
        return loss
    finally:
        for path, wrapper in wrappers.items():
            wrapper.enabled, wrapper.capture = previous[path]
            if not wrapper.capture:
                wrapper.clear_trace()


def model_loss_objective(model: torch.nn.Module, batch: Any, _attachment: "ARTIAttachment" | None = None) -> Tensor:
    output = model(**batch) if isinstance(batch, Mapping) else model(batch)
    loss = output.get("loss") if isinstance(output, Mapping) else getattr(output, "loss", None)
    if not isinstance(loss, Tensor):
        raise ValueError("model_loss objective requires model output with a Tensor .loss or ['loss']")
    return loss


def _call_objective(objective: AttachmentObjective, model: torch.nn.Module, batch: Any, attachment: "ARTIAttachment") -> Tensor:
    return _bind_objective(objective)(model, batch, attachment)


def _bind_objective(objective: AttachmentObjective) -> AttachmentObjective:
    takes_attachment = len(inspect.signature(objective).parameters) >= 3

    def call(model, batch, attachment):
        loss = objective(model, batch, attachment) if takes_attachment else objective(model, batch)
        if not isinstance(loss, Tensor) or loss.ndim != 0:
            raise ValueError("attachment objective must return a scalar Tensor")
        return loss

    return call


def _rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["torch"].cpu())
    version, internal_state, gaussian = state["python"]
    random.setstate((version, tuple(internal_state), gaussian))
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


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


def _optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    }


def _required_trace(value: Tensor | None, path: str, kind: str) -> Tensor:
    if value is None:
        raise RuntimeError(f"layer {path!r} did not capture a {kind} tensor")
    return value


def _alignment_mask(error: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return torch.ones_like(error, dtype=torch.bool)
    if tuple(mask.shape) != tuple(error.shape):
        raise ValueError(f"mask must have shape {tuple(error.shape)}")
    return mask.to(device=error.device, dtype=torch.bool)


__all__ = [
    "ARTITrainingResult",
    "ARTIEvaluationResult",
    "ARTITrainingSession",
    "ARTICheckpointCallback",
    "AttachmentObjective",
    "model_loss_objective",
    "tensor_alignment_objective",
    "resolve_attachment_objective",
]
