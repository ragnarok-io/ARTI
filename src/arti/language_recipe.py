"""Causal-LM SFT recipes with native task loss and portable ARTI deployment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
import math
import json
from pathlib import Path
from typing import Any

import torch

from .attachment import ARTI
from .attachment_hub import DEFAULT_ARTIFACT
from .recipe_adapters import FormulaAdapterConfig, formula_adapter_factory
from .recipe_chat import (
    ChatBatches,
    ChatDataset,
    ChatFields,
    ChatSplit,
    collate_chat,
    encode_final_answer,
    normalized_messages,
    target_token_weight,
    tokenizer_fingerprint,
)
from .recipe_data import _positive_int, json_fingerprint
from .recipe_workflow import fit_task_recipe, read_recipe_config, write_json, write_jsonl
from .serialization import load
from .training_binding import TrainingBinding, trainer
from .training_config import ARTITrainingConfig
from .training_run import ARTITrainingRunConfig


@dataclass(frozen=True)
class CausalLMModelConfig:
    id: str = "Qwen/Qwen3-0.6B"
    revision: str | None = None
    dtype: str = "auto"
    local_files_only: bool = False
    attn_implementation: str = "sdpa"

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("model.id is required")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("model.dtype must be auto, float32, float16 or bfloat16")
        if type(self.local_files_only) is not bool:
            raise ValueError("model.local_files_only must be boolean")


@dataclass(frozen=True)
class CausalLMGenerationConfig:
    max_new_tokens: int = 64
    batch_size: int = 8

    def __post_init__(self):
        _positive_int(self.max_new_tokens, "generation.max_new_tokens")
        _positive_int(self.batch_size, "generation.batch_size")


@dataclass(frozen=True)
class CausalLMOptimizerConfig:
    weight_decay: float = 0.01
    scheduler: str = "constant"
    warmup_steps: int = 0
    total_steps: int | None = None
    implementation: str = "auto"

    def __post_init__(self):
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("optimizer.weight_decay must be finite and nonnegative")
        if self.scheduler not in {"constant", "constant_with_warmup", "linear", "cosine"}:
            raise ValueError("unsupported SFT scheduler")
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise ValueError("optimizer.warmup_steps must be nonnegative")
        if self.scheduler in {"linear", "cosine"}:
            _positive_int(self.total_steps, "optimizer.total_steps")
        if self.implementation not in {"auto", "foreach", "fused"}:
            raise ValueError("optimizer implementation must be auto, foreach or fused")


@dataclass(frozen=True)
class CausalLMRecipeConfig:
    data_path: str | Path
    model: CausalLMModelConfig = field(default_factory=CausalLMModelConfig)
    fields: ChatFields = field(default_factory=ChatFields)
    adapter: FormulaAdapterConfig = field(default_factory=FormulaAdapterConfig)
    batch_size: int = 8
    max_length: int = 1024
    overlength: str = "error"
    pad_to_multiple_of: int = 8
    group_by_length: bool = True
    template_kwargs: Mapping[str, Any] = field(default_factory=lambda: {"enable_thinking": False})
    gradient_checkpointing: bool = False
    max_grad_norm: float | None = 1.0
    seed: int = 17
    training: ARTITrainingConfig = field(
        default_factory=lambda: ARTITrainingConfig(device="cpu", steps=200, learning_rate=1e-3)
    )
    optimizer: CausalLMOptimizerConfig = field(default_factory=CausalLMOptimizerConfig)
    generation: CausalLMGenerationConfig = field(default_factory=CausalLMGenerationConfig)
    run: ARTITrainingRunConfig = field(default_factory=ARTITrainingRunConfig)

    def __post_init__(self):
        object.__setattr__(self, "data_path", Path(self.data_path).resolve())
        _positive_int(self.batch_size, "data.batch_size")
        _positive_int(self.max_length, "data.max_length")
        _positive_int(self.pad_to_multiple_of, "data.pad_to_multiple_of")
        if self.overlength not in {"error", "drop"}:
            raise ValueError(
                "data.overlength must be error or drop; answer truncation is not implicit"
            )
        if type(self.group_by_length) is not bool or type(self.gradient_checkpointing) is not bool:
            raise ValueError("group_by_length/gradient_checkpointing must be boolean")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("recipe.seed must be nonnegative")
        if self.training.engine != "torch" or self.training.distributed:
            raise ValueError(
                "the qualified local SFT recipe currently uses single-process Torch; distributed engine state is a separate profile"
            )
        if self.training.objective != "model_loss":
            raise ValueError("SFT requires the real native causal-LM loss")
        json_fingerprint(dict(self.template_kwargs))
        if self.max_grad_norm is not None and (
            not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0
        ):
            raise ValueError("max_grad_norm must be finite and positive")

    @classmethod
    def read(cls, path: str | Path):
        path = Path(path)
        payload = read_recipe_config(path)
        if payload.get("task") != "causal-lm":
            raise ValueError("recipe.task must be causal-lm")
        allowed = {
            "task",
            "seed",
            "model",
            "data",
            "adapter",
            "training",
            "optimizer",
            "generation",
            "run",
            "gradient_checkpointing",
            "max_grad_norm",
        }
        if payload.keys() - allowed:
            raise ValueError(f"unknown SFT recipe settings: {sorted(payload.keys() - allowed)}")
        data = dict(payload.get("data", {}))
        allowed_data = {
            "path",
            "fields",
            "batch_size",
            "max_length",
            "overlength",
            "pad_to_multiple_of",
            "group_by_length",
            "template_kwargs",
        }
        if data.keys() - allowed_data or not isinstance(data.get("path"), str) or not data["path"]:
            raise ValueError("SFT data requires a JSONL path and supported preparation settings")
        data_path = Path(data.pop("path"))
        if not data_path.is_absolute():
            data_path = path.parent / data_path
        fields = ChatFields(**data.pop("fields", {}))
        adapter = dict(payload.get("adapter", {}))
        if adapter.get("layers") is not None:
            if not isinstance(adapter["layers"], list):
                raise ValueError("adapter.layers must be an array")
            adapter["layers"] = tuple(adapter["layers"])
        run = dict(payload.get("run", {}))
        if run.get("output_dir") is not None:
            directory = Path(run["output_dir"])
            run["output_dir"] = str(
                directory if directory.is_absolute() else path.parent / directory
            )
        return cls(
            data_path,
            fields=fields,
            model=CausalLMModelConfig(**payload.get("model", {})),
            adapter=FormulaAdapterConfig(**adapter),
            seed=payload.get("seed", 17),
            training=replace(
                ARTITrainingConfig(device="cpu", steps=200, learning_rate=1e-3),
                **payload.get("training", {}),
            ),
            optimizer=CausalLMOptimizerConfig(**payload.get("optimizer", {})),
            generation=CausalLMGenerationConfig(**payload.get("generation", {})),
            run=ARTITrainingRunConfig(**run),
            gradient_checkpointing=payload.get("gradient_checkpointing", False),
            max_grad_norm=payload.get("max_grad_norm", 1.0),
            **data,
        )

    def to_dict(self):
        return {
            "task": "causal-lm",
            "seed": self.seed,
            "model": asdict(self.model),
            "data": {
                "path": str(self.data_path),
                "fields": asdict(self.fields),
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "overlength": self.overlength,
                "pad_to_multiple_of": self.pad_to_multiple_of,
                "group_by_length": self.group_by_length,
                "template_kwargs": dict(self.template_kwargs),
            },
            "adapter": asdict(self.adapter),
            "training": asdict(self.training),
            "optimizer": asdict(self.optimizer),
            "generation": asdict(self.generation),
            "gradient_checkpointing": self.gradient_checkpointing,
            "max_grad_norm": self.max_grad_norm,
            "run": self.run.to_dict(),
        }


def _load_tokenizer(source, *, revision=None, local_files_only=False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source, revision=revision, local_files_only=local_files_only, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(config: CausalLMModelConfig, *, source_override=None):
    from transformers import AutoModelForCausalLM

    kwargs = {
        "local_files_only": config.local_files_only,
        "attn_implementation": config.attn_implementation,
        "torch_dtype": "auto" if config.dtype == "auto" else getattr(torch, config.dtype),
    }
    if source_override is None:
        kwargs["revision"] = config.revision
    return AutoModelForCausalLM.from_pretrained(source_override or config.id, **kwargs)


def plan_causal_lm_recipe(path):
    """Inspect native structure on meta without loading weights or touching CUDA."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = CausalLMRecipeConfig.read(path)
    native = AutoConfig.from_pretrained(
        config.model.id,
        revision=config.model.revision,
        local_files_only=config.model.local_files_only,
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(native)
    candidates = ARTI.discover(model, layers=config.adapter.layers or "model.layers.*.mlp")
    candidates = (
        candidates
        if config.adapter.layers is not None
        else candidates[-config.adapter.last_layers :]
    )
    if not candidates or any(item.dim is None for item in candidates):
        raise ValueError("no declared native SFT insertion dimensions; specify adapter.layers")
    tokenizer = _load_tokenizer(
        config.model.id,
        revision=config.model.revision,
        local_files_only=config.model.local_files_only,
    )
    data = ChatDataset(
        config.data_path,
        tokenizer,
        fields=config.fields,
        template_kwargs=config.template_kwargs,
        max_length=config.max_length,
        overlength=config.overlength,
        pad_to_multiple_of=config.pad_to_multiple_of,
    )
    return {
        "ok": True,
        "kind": "recipe-plan",
        "task": "causal-lm",
        "dry_run": True,
        "weights_loaded": False,
        "model_allocation": "meta",
        "estimates_are_static": True,
        "model": {
            "base_model": config.model.id,
            "revision": config.model.revision or getattr(native, "_commit_hash", None),
            "backbone_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(2 * item.dim * config.adapter.rank for item in candidates),
            "layers": [asdict(item) for item in candidates],
        },
        "task_recipe": {
            "task": "causal-lm",
            "supervision": "final-assistant-through-native-eos",
            "dataset": data.manifest(),
            "adapter": asdict(config.adapter),
        },
        "training": asdict(config.training),
        "run": config.run.to_dict(),
    }


def _device(model):
    return next(model.parameters()).device


def _forward_batch(model, batch):
    return model(
        **{key: value.to(_device(model), non_blocking=True) for key, value in batch.items()},
        use_cache=False,
    )


def causal_lm_model_loss(model, batch):
    """Preserve the host's shifted/masked causal loss, including true EOS labels."""
    return _forward_batch(model, batch).loss


causal_lm_model_loss._arti_native_model_loss = True


def _attach_program(model, config, *, layer_factory=None, paths=None):
    paths = paths or config.layers
    if paths is None:
        candidates = ARTI.discover(model, layers="model.layers.*.mlp")
        paths = tuple(item.path for item in candidates[-config.last_layers :])
    if not paths:
        raise ValueError(
            "no Qwen/causal-LM MLP attachment points; specify adapter.layers explicitly"
        )
    factory = formula_adapter_factory(config) if layer_factory is None else layer_factory
    model = ARTI.attach(model, factory, layers=paths, freeze_backbone=True)
    # FP32 trainable operands/master optimizer state, independent of host BF16.
    for layer in model.arti.layers.values():
        layer.float()
    return model


class CausalLMRecipe:
    task_name = "causal-lm"
    evaluation_weight = staticmethod(target_token_weight)
    training_weight = staticmethod(target_token_weight)

    def __init__(
        self, config: CausalLMRecipeConfig, *, model=None, tokenizer=None, layer_factory=None
    ):
        self.config = config
        self.tokenizer = tokenizer or _load_tokenizer(
            config.model.id,
            revision=config.model.revision,
            local_files_only=config.model.local_files_only,
        )
        self.data = ChatDataset(
            config.data_path,
            self.tokenizer,
            fields=config.fields,
            template_kwargs=config.template_kwargs,
            max_length=config.max_length,
            overlength=config.overlength,
            pad_to_multiple_of=config.pad_to_multiple_of,
        )
        self.model = model if model is not None else _load_model(config.model)
        if not config.model.revision:
            resolved = getattr(self.model.config, "_commit_hash", None)
            if not resolved:
                raise ValueError("provide an immutable model revision for this recipe")
            self.config = replace(config, model=replace(config.model, revision=resolved))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.model = _attach_program(self.model, config.adapter, layer_factory=layer_factory)
        self.model.config.use_cache = False
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.task_contract = {
            "format": "arti.causal-lm-recipe",
            "format_version": 1,
            "task": "causal-lm",
            "base_model": {
                key: value
                for key, value in asdict(self.config.model).items()
                if key != "local_files_only"
            },
            "fields": asdict(config.fields),
            "adapter": asdict(config.adapter),
            "adapter_factory": "builtin-formula" if layer_factory is None else "caller-supplied",
            "resolved_layers": list(self.model.arti.paths),
            "template_kwargs": dict(config.template_kwargs),
            "tokenizer_sha256": self.data.tokenizer_fingerprint,
            "supervision": "final-assistant-through-native-eos",
            "objective": "native-causal-lm-loss",
            "gradient_normalization": "supervised-tokens-per-optimizer-step",
            "max_length": config.max_length,
            "overlength": config.overlength,
            "pad_to_multiple_of": config.pad_to_multiple_of,
            "dataset": self.data.manifest(),
            "optimizer": asdict(config.optimizer),
            "max_grad_norm": config.max_grad_norm,
            "generation": asdict(config.generation),
            "execution": "torch-transformers-with-arti-program",
            "checkpoint_selection": "final",
        }
        self.task_contract = json.loads(json.dumps(self.task_contract, allow_nan=False))
        self.task_fingerprint = json_fingerprint(self.task_contract)
        self.task_contract["sha256"] = self.task_fingerprint
        binding = TrainingBinding(
            self.model,
            owner=self.model.arti,
            kind="causal-lm-recipe",
            parameters=self.model.arti.parameters,
            state_module=self.model.arti._bundle,
            summary=lambda: self.model.arti.summary().to_dict(),
            save_checkpoint=self._save_checkpoint,
            load_checkpoint=self._load_checkpoint,
            checkpoint_metadata={
                "task_recipe_sha256": self.task_fingerprint,
                "training_config": asdict(config.training),
            },
        )
        if config.training.device != "auto":
            self.model.to(config.training.device)
        implementation = config.optimizer.implementation
        if implementation == "auto":
            implementation = "fused" if _device(self.model).type == "cuda" else "foreach"
        self.optimizer_implementation = implementation
        optimizer = torch.optim.AdamW(
            list(binding.parameters()),
            lr=config.training.learning_rate,
            weight_decay=config.optimizer.weight_decay,
            **{implementation: True},
        )
        from transformers import get_scheduler

        scheduler = get_scheduler(
            config.optimizer.scheduler,
            optimizer=optimizer,
            num_warmup_steps=config.optimizer.warmup_steps,
            num_training_steps=config.optimizer.total_steps,
        )
        self.session = trainer(
            binding,
            config=config.training,
            objective=causal_lm_model_loss,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        self.train_data = ChatBatches(
            self.data.splits["train"],
            batch_size=config.batch_size,
            seed=config.seed,
            group_by_length=config.group_by_length,
        )

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(
            config
            if isinstance(config, CausalLMRecipeConfig)
            else CausalLMRecipeConfig.read(config),
            **kwargs,
        )

    def plan(self):
        return {
            **self.session.plan(),
            "task_recipe": self.task_contract,
            "batch_size": self.config.batch_size,
            "optimizer_implementation": self.optimizer_implementation,
            "data_resume": "epoch-length-order-cursor-and-token-counts",
            "backbone_frozen": True,
            "adapter_compute_dtype": "float32",
            "deployment": {
                "base_weights_included": False,
                "base_revision_pinned": True,
                "tokenizer_included": True,
                "backend": "torch-transformers",
                "adapter_factory": self.task_contract["adapter_factory"],
                "state_scope": "read-only-adapters"
                if self.task_contract["adapter_factory"] == "builtin-formula"
                else "caller-managed",
            },
        }

    def check(self):
        return {
            **self.session.check(self.data.splits["train"].batches(self.config.batch_size)[0]),
            "task_recipe_sha256": self.task_fingerprint,
        }

    def fit(
        self,
        *,
        steps=None,
        run=None,
        resume_from_checkpoint=None,
        reporter=None,
        stop_requested=None,
    ):
        result = fit_task_recipe(
            self,
            steps=steps,
            run=run,
            resume_from_checkpoint=resume_from_checkpoint,
            reporter=reporter,
            stop_requested=stop_requested,
        )
        result["data_cursor"] = self.train_data.state_dict()
        write_json(Path(result["run_directory"]) / "task_result.json", result)
        return result

    def write_run_artifacts(self, directory):
        self.tokenizer.save_pretrained(directory / "tokenizer")
        write_json(directory / "plan.json", self.plan())

    def evaluate(self, split="validation", *, output=None, generate_samples=0):
        if split not in self.data.splits or not len(self.data.splits[split]):
            raise ValueError(f"no SFT records for split {split!r}")
        with self.session.evaluation_context():
            report, rows = _evaluate_chat(
                self.model,
                self.tokenizer,
                self.data.splits[split],
                self.config.batch_size,
                dict(self.config.template_kwargs),
                self.config.generation,
                generate_samples,
            )
        report.update(
            split=split,
            task_recipe_sha256=self.task_fingerprint,
            dataset_sha256=self.data.splits[split].fingerprint,
        )
        _save_evaluation(output, report, rows)
        return report

    def export(self, directory):
        directory = Path(directory)
        self.tokenizer.save_pretrained(directory / "tokenizer")
        saved = self.model.arti.save_pretrained(
            directory,
            base_model=self.config.model.id,
            revision=self.config.model.revision,
            metadata={"task_recipe": self.task_contract},
        )
        write_json(
            directory / "deployment.json",
            {
                "format": "arti.causal-lm-deployment",
                "format_version": 1,
                "task_recipe_sha256": self.task_fingerprint,
                "artifact": saved.artifact_path.name,
                "tokenizer": "tokenizer",
                "base_model": {"id": self.config.model.id, "revision": self.config.model.revision},
                "backend": "torch-transformers",
                "adapter_factory": self.task_contract["adapter_factory"],
                "state_scope": "read-only-adapters"
                if self.task_contract["adapter_factory"] == "builtin-formula"
                else "caller-managed",
                "base_weights_included": False,
            },
        )
        return saved.artifact_path

    def load_model(self, path):
        self._load_checkpoint(path, load_checkpoint=False)

    def _save_checkpoint(self, path, **kwargs):
        return self.model.arti.save(path, metadata={"task_recipe": self.task_contract}, **kwargs)

    def _load_checkpoint(
        self, path, *, optimizer=None, scheduler=None, map_location=None, load_checkpoint=True
    ):
        path = _artifact_path(path)
        inspected = load(path, load_checkpoint=load_checkpoint)
        if inspected.manifest["architecture"]["config"].get("task_recipe") != self.task_contract:
            raise ValueError(
                "SFT task/model/template/data/optimizer contract differs from the checkpoint"
            )
        if load_checkpoint:
            state = inspected.training_state
            if (
                not isinstance(state, Mapping)
                or state.get("task_recipe_sha256") != self.task_fingerprint
            ):
                raise ValueError("inference asset is not a resumable SFT checkpoint")
            previous = state.get("training_config", {})
            for key in (
                "engine",
                "objective",
                "learning_rate",
                "mixed_precision",
                "gradient_accumulation_steps",
            ):
                if previous.get(key) != asdict(self.config.training)[key]:
                    raise ValueError(f"resume training.{key} differs from its checkpoint")
            if state.get("data_state") is not None:
                self.train_data.validate_state(state["data_state"])
        return self.model.arti.load(
            path,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location or _device(self.model),
            load_checkpoint=load_checkpoint,
        )


def _artifact_path(path):
    path = Path(path)
    return path / DEFAULT_ARTIFACT if path.is_dir() else path


class CausalLMPredictor:
    def __init__(self, model, tokenizer, contract):
        self.model, self.tokenizer, self.contract = model.eval(), tokenizer, dict(contract)
        self.fields = ChatFields(**contract["fields"])
        self.generation = CausalLMGenerationConfig(**contract["generation"])

    @classmethod
    def from_pretrained(
        cls,
        path,
        *,
        device="cpu",
        local_files_only=None,
        model=None,
        tokenizer=None,
        layer_factory=None,
        base_model=None,
    ):
        artifact = _artifact_path(path)
        declaration = (
            load(artifact, load_checkpoint=False)
            .manifest["architecture"]["config"]
            .get("task_recipe", {})
        )
        if (
            declaration.get("format") != "arti.causal-lm-recipe"
            or declaration.get("format_version") != 1
        ):
            raise ValueError("asset is not a supported causal-LM recipe")
        if declaration.get("sha256") != json_fingerprint(
            {key: value for key, value in declaration.items() if key != "sha256"}
        ):
            raise ValueError("SFT declaration fingerprint does not match")
        if declaration.get("adapter_factory") == "caller-supplied" and layer_factory is None:
            raise ValueError(
                "this asset requires its caller-supplied layer_factory; pass it to "
                "CausalLMPredictor.from_pretrained before loading the host"
            )
        if tokenizer is None:
            tokenizer = _load_tokenizer(artifact.parent / "tokenizer", local_files_only=True)
        if tokenizer_fingerprint(tokenizer) != declaration["tokenizer_sha256"]:
            raise ValueError("deployment tokenizer differs from the trained tokenizer")
        source = CausalLMModelConfig(**declaration["base_model"])
        if local_files_only is not None:
            source = replace(source, local_files_only=local_files_only)
        model = model if model is not None else _load_model(source, source_override=base_model)
        adapter = dict(declaration["adapter"])
        if adapter.get("layers") is not None:
            adapter["layers"] = tuple(adapter["layers"])
        config = FormulaAdapterConfig(**adapter)
        model = _attach_program(
            model, config, layer_factory=layer_factory, paths=tuple(declaration["resolved_layers"])
        )
        model.arti.load(artifact, map_location=device)
        model.to(device)
        return cls(model, tokenizer, declaration)

    def predict_records(self, records, *, batch_size=None, max_new_tokens=None):
        messages = [
            normalized_messages(row.get(self.fields.messages), completed=False) for row in records
        ]
        generated = _generate_chat(
            self.model,
            self.tokenizer,
            messages,
            self.contract["template_kwargs"],
            replace(
                self.generation,
                batch_size=self.generation.batch_size if batch_size is None else batch_size,
                max_new_tokens=self.generation.max_new_tokens
                if max_new_tokens is None
                else max_new_tokens,
            ),
        )
        for record, result in zip(records, generated, strict=True):
            if self.fields.id in record:
                result["id"] = record[self.fields.id]
        return generated

    def evaluate_records(self, records, *, batch_size=8, output=None, generate_samples=0):
        examples = []
        seen = set()
        for row in records:
            identifier = row.get(self.fields.id)
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise ValueError("evaluation requires distinct nonempty string ids")
            seen.add(identifier)
            examples.append(
                encode_final_answer(
                    identifier,
                    row.get(self.fields.messages),
                    self.tokenizer,
                    template_kwargs=self.contract["template_kwargs"],
                )
            )
        if not examples:
            raise ValueError("evaluation records must not be empty")
        split = ChatSplit(
            tuple(examples),
            json_fingerprint(
                [
                    {
                        "id": item.id,
                        "input_ids": item.input_ids.tolist(),
                        "labels": item.labels.tolist(),
                    }
                    for item in examples
                ]
            ),
            self.tokenizer.pad_token_id,
            self.contract["pad_to_multiple_of"],
        )
        report, rows = _evaluate_chat(
            self.model,
            self.tokenizer,
            split,
            batch_size,
            self.contract["template_kwargs"],
            self.generation,
            generate_samples,
        )
        report.update(
            split="evaluation",
            task_recipe_sha256=self.contract["sha256"],
            dataset_sha256=split.fingerprint,
        )
        _save_evaluation(output, report, rows)
        return report


def _generate_chat(model, tokenizer, conversations, template_kwargs, config):
    results = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(conversations), config.batch_size):
            rows = conversations[start : start + config.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    **dict(template_kwargs),
                )
                for messages in rows
            ]
            length = max(map(len, prompts))
            inputs = torch.full(
                (len(rows), length), tokenizer.pad_token_id, dtype=torch.long, device=_device(model)
            )
            mask = torch.zeros_like(inputs)
            for index, prompt in enumerate(prompts):
                inputs[index, -len(prompt) :] = torch.tensor(
                    prompt, dtype=torch.long, device=inputs.device
                )
                mask[index, -len(prompt) :] = 1
            tokens = (
                model.generate(
                    input_ids=inputs,
                    attention_mask=mask,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )[:, length:]
                .cpu()
                .tolist()
            )
            for row in tokens:
                stopped = tokenizer.eos_token_id in row
                if stopped:
                    row = row[: row.index(tokenizer.eos_token_id) + 1]
                results.append(
                    {
                        "text": tokenizer.decode(row, skip_special_tokens=True),
                        "token_ids": row,
                        "generated_tokens": len(row),
                        "finish_reason": "stop" if stopped else "length",
                    }
                )
    return results


def _evaluate_chat(
    model, tokenizer, split, batch_size, template_kwargs, generation, generate_samples
):
    _positive_int(batch_size, "evaluation.batch_size")
    if type(generate_samples) is not int or generate_samples < 0:
        raise ValueError("generate_samples must be nonnegative")
    modes = [(module, module.training) for module in model.modules()]
    totals = torch.zeros(2, device=_device(model))
    tokens = 0
    rows = []
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(split), batch_size):
                examples = split.examples[start : start + batch_size]
                batch = collate_chat(
                    examples,
                    pad_token_id=split.pad_token_id,
                    pad_to_multiple_of=split.pad_to_multiple_of,
                )
                output = _forward_batch(model, batch)
                count = target_token_weight(batch)
                targets = batch["labels"][:, 1:].to(output.logits.device)
                valid = targets.ne(-100)
                predicted = output.logits[:, :-1].argmax(-1)
                totals[0] += output.loss.detach().float() * count
                totals[1] += (predicted.eq(targets) & valid).sum()
                tokens += count
                per_row_correct = (predicted.eq(targets) & valid).sum(-1).cpu().tolist()
                rows.extend(
                    {
                        "id": example.id,
                        "target": example.messages[-1]["content"],
                        "target_tokens": example.target_tokens,
                        "teacher_forced_correct_tokens": correct,
                    }
                    for example, correct in zip(examples, per_row_correct, strict=True)
                )
            values = totals.cpu().tolist()
            loss = values[0] / tokens
            report = {
                "samples": len(split),
                "supervised_tokens": tokens,
                "loss": loss,
                "perplexity": math.exp(loss) if loss < 700 else None,
                "teacher_forced_token_accuracy": values[1] / tokens,
                "objective": "native-causal-lm-loss",
                "weighting": "supervised_tokens",
            }
        if generate_samples:
            selected = split.examples[:generate_samples]
            completions = _generate_chat(
                model,
                tokenizer,
                [list(example.messages[:-1]) for example in selected],
                template_kwargs,
                generation,
            )
            matches = []
            for result, row in zip(completions, rows, strict=False):
                exact = result["text"].strip() == row["target"].strip()
                row.update(generation=result, exact_match=exact)
                matches.append(exact)
            report.update(
                generated_samples=len(matches), generation_exact_match=sum(matches) / len(matches)
            )
        return report, rows
    finally:
        for module, training in modes:
            module.training = training


def _save_evaluation(output, report, rows):
    if output is not None:
        write_json(output, report)
        write_jsonl(Path(output).with_suffix(".predictions.jsonl"), rows)


__all__ = [
    "CausalLMRecipe",
    "CausalLMRecipeConfig",
    "CausalLMPredictor",
    "CausalLMModelConfig",
    "CausalLMGenerationConfig",
    "CausalLMOptimizerConfig",
]
