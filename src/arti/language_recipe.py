"""Causal-LM SFT recipes with native task loss and portable ARTI deployment."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
import math
import json
from pathlib import Path
from typing import Any

import torch

from .attachment import ARTI
from .attachment_hub import DEFAULT_ARTIFACT
from .recipe_adapters import FormulaAdapterConfig, LayerFactoryConfig, formula_adapter_factory, _resolve_layer_factory
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
    _chat_example_batches,
    _validate_token_budget,
)
from .recipe_data import _positive_int, json_fingerprint
from .recipe_evaluation import CausalLMEvaluationConfig, generation_evaluation_plan, score_causal_lm_predictions, select_generation_examples
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
    cache_dir: str | Path | None = None
    evaluation: CausalLMEvaluationConfig = field(default_factory=CausalLMEvaluationConfig)
    layer_factory: LayerFactoryConfig | None = None
    max_tokens_per_batch: int | None = None
    source_manifest: str | Path | None = None

    def __post_init__(self):
        object.__setattr__(self, "data_path", Path(self.data_path).resolve())
        if not isinstance(self.evaluation, CausalLMEvaluationConfig):
            raise ValueError("evaluation must be a CausalLMEvaluationConfig")
        if self.layer_factory is not None and not isinstance(self.layer_factory, LayerFactoryConfig):
            raise ValueError("layer_factory must be a LayerFactoryConfig or null")
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)) or not str(self.cache_dir).strip():
                raise ValueError("data.cache_dir must be a nonempty directory or null")
            object.__setattr__(self, "cache_dir", Path(self.cache_dir).resolve())
        if self.source_manifest is not None:
            if not isinstance(self.source_manifest, (str, Path)) or not str(self.source_manifest).strip():
                raise ValueError("data.source_manifest must be a nonempty path or null")
            object.__setattr__(self, "source_manifest", Path(self.source_manifest).resolve())
        _positive_int(self.batch_size, "data.batch_size")
        _positive_int(self.max_length, "data.max_length")
        _positive_int(self.pad_to_multiple_of, "data.pad_to_multiple_of")
        if self.max_tokens_per_batch is not None:
            _positive_int(self.max_tokens_per_batch, "data.max_tokens_per_batch")
        if self.overlength not in {"error", "drop"}:
            raise ValueError(
                "data.overlength must be error or drop; answer truncation is not implicit"
            )
        if type(self.group_by_length) is not bool or type(self.gradient_checkpointing) is not bool:
            raise ValueError("group_by_length/gradient_checkpointing must be boolean")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("recipe.seed must be nonnegative")
        if self.training.engine not in {"torch", "accelerate"} or self.training.distributed:
            raise ValueError(
                "the qualified local SFT recipe uses single-process Torch or Accelerate; distributed engine state is a separate profile"
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
            "evaluation",
            "run",
            "gradient_checkpointing",
            "max_grad_norm",
            "layer_factory",
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
            "cache_dir",
            "max_tokens_per_batch",
            "source_manifest",
        }
        if data.keys() - allowed_data or not isinstance(data.get("path"), str) or not data["path"]:
            raise ValueError("SFT data requires a JSONL path and supported preparation settings")
        data_path = Path(data.pop("path"))
        if not data_path.is_absolute():
            data_path = path.parent / data_path
        if data.get("cache_dir") is not None:
            if not isinstance(data["cache_dir"], str) or not data["cache_dir"].strip():
                raise ValueError("data.cache_dir must be a nonempty directory or null")
            cache = Path(data["cache_dir"])
            data["cache_dir"] = cache if cache.is_absolute() else path.parent / cache
        if data.get("source_manifest") is not None:
            if not isinstance(data["source_manifest"], str) or not data["source_manifest"].strip():
                raise ValueError("data.source_manifest must be a nonempty path or null")
            manifest = Path(data["source_manifest"])
            data["source_manifest"] = manifest if manifest.is_absolute() else path.parent / manifest
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
            evaluation=CausalLMEvaluationConfig(**payload.get("evaluation", {})),
            layer_factory=None if payload.get("layer_factory") is None else LayerFactoryConfig(**payload["layer_factory"]),
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
                "cache_dir": None if self.cache_dir is None else str(self.cache_dir),
                "fields": asdict(self.fields),
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "overlength": self.overlength,
                "pad_to_multiple_of": self.pad_to_multiple_of,
                "group_by_length": self.group_by_length,
                "template_kwargs": dict(self.template_kwargs),
                **({"max_tokens_per_batch": self.max_tokens_per_batch} if self.max_tokens_per_batch is not None else {}),
                **({"source_manifest": str(self.source_manifest)} if self.source_manifest is not None else {}),
            },
            "adapter": asdict(self.adapter),
            "training": asdict(self.training),
            "optimizer": asdict(self.optimizer),
            "generation": asdict(self.generation),
            "evaluation": asdict(self.evaluation),
            "gradient_checkpointing": self.gradient_checkpointing,
            "max_grad_norm": self.max_grad_norm,
            "run": self.run.to_dict(),
            **({"layer_factory": self.layer_factory.to_dict()} if self.layer_factory is not None else {}),
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


def prepare_causal_lm_data(config, *, tokenizer=None, output=None):
    """Prepare final-answer supervision without a host model or optimizer."""
    config = config if isinstance(config, CausalLMRecipeConfig) else CausalLMRecipeConfig.read(config)
    tokenizer = tokenizer if tokenizer is not None else _load_tokenizer(
        config.model.id, revision=config.model.revision, local_files_only=config.model.local_files_only
    )
    data = _prepare_chat_dataset(config, tokenizer)
    report = {
        "ok": True,
        "kind": "recipe-prepare-data",
        "task": "causal-lm",
        "weights_loaded": False,
        "model_allocation": "none",
        "optimizer_created": False,
        "data_preparation": dict(data.preparation),
        "dataset": data.manifest(include_ids=True),
        "batching_plan": _training_batches(data, config).plan(),
    }
    if output is not None:
        write_json(output, report)
    return report


def _prepare_chat_dataset(config, tokenizer, *, split=None):
    data = ChatDataset(
        config.data_path,
        tokenizer,
        fields=config.fields,
        template_kwargs=config.template_kwargs,
        max_length=config.max_length,
        overlength=config.overlength,
        pad_to_multiple_of=config.pad_to_multiple_of,
        cache_dir=config.cache_dir,
        source_manifest=config.source_manifest,
        splits=None if split is None else (split,),
    )
    for prepared_split in data.splits.values():
        _validate_token_budget(prepared_split.examples, prepared_split.pad_to_multiple_of, config.max_tokens_per_batch)
    return data


def _training_batches(data, config):
    return ChatBatches(data.splits["train"], batch_size=config.batch_size, seed=config.seed,
                       group_by_length=config.group_by_length, max_tokens_per_batch=config.max_tokens_per_batch)


def _batching_metadata(config):
    return {"batching": {"max_tokens_per_batch": config.max_tokens_per_batch}} if config.max_tokens_per_batch is not None else {}


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
    data = _prepare_chat_dataset(config, tokenizer)
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
            "trainable_parameters": sum(2 * item.dim * config.adapter.rank for item in candidates) if config.layer_factory is None else None,
            "layers": [asdict(item) for item in candidates],
        },
        "task_recipe": {
            "task": "causal-lm",
            "supervision": "final-assistant-through-native-eos",
            "dataset": data.manifest(),
            "adapter": asdict(config.adapter),
            **({"layer_factory": config.layer_factory.to_dict()} if config.layer_factory is not None else {}),
        },
        "training": asdict(config.training),
        "evaluation": asdict(config.evaluation),
        "evaluation_plan": generation_evaluation_plan(data.splits["validation"].examples, config.evaluation, config.generation),
        "data_preparation": dict(data.preparation),
        "batching_plan": _training_batches(data, config).plan(),
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

    @property
    def dataset_manifest(self):
        return self.data.manifest(include_ids=True)

    @property
    def validation_data(self):
        return self.data.splits["validation"].batches(self.config.batch_size)

    @property
    def data_preparation(self):
        return self.data.preparation

    def __init__(
        self, config: CausalLMRecipeConfig, *, model=None, tokenizer=None, layer_factory=None,
        base_model=None, _expected_dataset=None
    ):
        descriptor = (
            config.layer_factory if config.layer_factory is not None and config.layer_factory.reference == layer_factory
            else LayerFactoryConfig(layer_factory)
        ) if isinstance(layer_factory, str) else layer_factory
        if config.layer_factory is not None and descriptor is not None and descriptor != config.layer_factory:
            raise ValueError("choose the configured layer factory or one explicit Python factory, not both")
        descriptor = config.layer_factory if descriptor is None else descriptor
        if isinstance(descriptor, LayerFactoryConfig):
            config = replace(config, layer_factory=descriptor)
        layer_factory = _resolve_layer_factory(descriptor)
        self.config = config
        self._evaluation_kind = "builtin"
        self._resume_evaluation_kind = None
        self.tokenizer = tokenizer or _load_tokenizer(
            config.model.id,
            revision=config.model.revision,
            local_files_only=config.model.local_files_only,
        )
        self.data = _prepare_chat_dataset(config, self.tokenizer)
        if _expected_dataset is not None and self.data.manifest() != _expected_dataset:
            raise ValueError("recovery dataset/tokenizer differs from the checkpoint before host loading")
        self.model = model if model is not None else _load_model(config.model, **({"source_override": base_model} if base_model is not None else {}))
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
        if config.layer_factory is not None:
            self.task_contract["layer_factory"] = config.layer_factory.to_dict()
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
                "recipe_evaluation": asdict(config.evaluation),
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
        self.train_data = _training_batches(self.data, config)

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
            "data_preparation": dict(self.data.preparation),
            "batching_plan": self.train_data.plan(),
            "validation_policy": self.config.run.to_dict()["validation"],
            "evaluation": asdict(self.config.evaluation),
            "evaluation_plan": generation_evaluation_plan(self.data.splits["validation"].examples, self.config.evaluation, self.config.generation),
            "backbone_frozen": True,
            "adapter_compute_dtype": "float32",
            "deployment": {
                "base_weights_included": False,
                "base_revision_pinned": True,
                "tokenizer_included": True,
                "backend": "torch-transformers",
                "adapter_factory": self.task_contract["adapter_factory"],
                **({"layer_factory": self.task_contract["layer_factory"]} if "layer_factory" in self.task_contract else {}),
                "state_scope": "read-only-adapters"
                if self.task_contract["adapter_factory"] == "builtin-formula"
                else "caller-managed",
            },
        }

    def check(self):
        return {
            **self.session.check(collate_chat(next(_chat_example_batches(self.data.splits["train"].examples, self.config.batch_size, self.config.pad_to_multiple_of, max_tokens_per_batch=self.config.max_tokens_per_batch)), pad_token_id=self.tokenizer.pad_token_id, pad_to_multiple_of=self.config.pad_to_multiple_of)),
            "task_recipe_sha256": self.task_fingerprint,
        }

    def validate_run(self, run, validation_fn=None):
        kind = "builtin" if validation_fn is None else "caller-supplied"
        select_generation_examples(self.data.splits["validation"].examples, self.config.evaluation)
        if run.validation is not None and kind == "builtin":
            if self._resume_evaluation_kind == "caller-supplied" or (self.session._validation is not None and self._evaluation_kind == "caller-supplied"):
                raise ValueError("continuation of caller-supplied validation requires validation_fn")
            monitor = run.validation.monitor
            known = {"loss", "perplexity", "teacher_forced_token_accuracy", "generated_samples", "generation_exact_match", "generation_score", "generation_numeric_exact_match", "generation_truncated_samples"}
            if monitor not in known:
                raise ValueError(f"unknown builtin causal-LM validation metric {monitor!r}; supply validation_fn explicitly")
            if monitor.startswith("generation_") or monitor == "generated_samples":
                if self.config.evaluation.generate_samples == 0:
                    raise ValueError("generation validation metrics require evaluation.generate_samples > 0")
                if monitor == "generation_numeric_exact_match" and self.config.evaluation.comparison != "numeric":
                    raise ValueError("generation_numeric_exact_match requires evaluation.comparison=numeric")
        self._evaluation_kind = kind
        return {"kind": kind, "profile": asdict(self.config.evaluation)}

    def fit(
        self,
        *,
        steps=None,
        run=None,
        resume_from_checkpoint=None,
        reporter=None,
        stop_requested=None,
        validation_fn=None,
        on_checkpoint=None,
    ):
        result = fit_task_recipe(
            self,
            steps=steps,
            run=run,
            resume_from_checkpoint=resume_from_checkpoint,
            reporter=reporter,
            stop_requested=stop_requested,
            validation_fn=validation_fn,
            on_checkpoint=on_checkpoint,
        )
        result["data_cursor"] = self.session._data_state if self.session._data_state is not None else self.train_data.state_dict()
        write_json(Path(result["run_directory"]) / "task_result.json", result)
        return result

    def resume_run(
        self,
        source_run,
        *,
        run=None,
        output_dir=None,
        max_seconds=None,
        refresh_completed=False,
        reporter=None,
        stop_requested=None,
        validation_fn=None,
        on_checkpoint=None,
    ):
        """Finish a prior SFT task without repeating completed optimizer updates."""
        from .recipe_recovery import resume_task_recipe

        return resume_task_recipe(
            self,
            source_run,
            run=run,
            output_dir=output_dir,
            max_seconds=max_seconds,
            refresh_completed=refresh_completed,
            reporter=reporter,
            stop_requested=stop_requested,
            validation_fn=validation_fn,
            on_checkpoint=on_checkpoint,
        )

    def write_run_artifacts(self, directory):
        self.tokenizer.save_pretrained(directory / "tokenizer")
        write_json(directory / "plan.json", self.plan())
        if self.data.source_manifest is not None:
            write_json(directory / "data_source_manifest.json", self.data.source_manifest)

    def evaluate(self, split="validation", *, output=None, generate_samples=None, checkpoint=None):
        if split not in self.data.splits or not len(self.data.splits[split]):
            raise ValueError(f"no SFT records for split {split!r}")
        context = self.session.evaluation_context() if checkpoint is None else self.session.checkpoint_context(checkpoint)
        with context:
            report, rows = _evaluate_chat(
                self.model,
                self.tokenizer,
                self.data.splits[split],
                self.config.batch_size,
                dict(self.config.template_kwargs),
                self.config.generation,
                generate_samples,
                evaluation=self.config.evaluation,
                state_context=self.session.evaluation_context,
                max_tokens_per_batch=self.config.max_tokens_per_batch,
            )
        report.update(
            split=split,
            task_recipe_sha256=self.task_fingerprint,
            dataset_sha256=self.data.splits[split].fingerprint,
            **({"source_manifest_sha256": self.data.source_manifest["sha256"]} if self.data.source_manifest is not None else {}),
        )
        _save_evaluation(output, report, rows)
        return report

    def export(self, directory, *, selection=None):
        directory = Path(directory)
        self.tokenizer.save_pretrained(directory / "tokenizer")
        saved = self.model.arti.save_pretrained(
            directory,
            base_model=self.config.model.id,
            revision=self.config.model.revision,
            metadata={"task_recipe": self.task_contract, "evaluation": asdict(self.config.evaluation), **_batching_metadata(self.config), **({"selection": dict(selection)} if selection is not None else {})},
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
                "evaluation": asdict(self.config.evaluation),
                **_batching_metadata(self.config),
                **({"selection": dict(selection)} if selection is not None else {}),
            },
        )
        return saved.artifact_path

    def load_model(self, path):
        self._load_checkpoint(path, load_checkpoint=False)

    def _save_checkpoint(self, path, **kwargs):
        if kwargs.get("training_state") is not None:
            kwargs["training_state"] = {**dict(kwargs["training_state"]), "recipe_evaluation": asdict(self.config.evaluation), "recipe_evaluator_kind": self._evaluation_kind}
        return self.model.arti.save(path, metadata={"task_recipe": self.task_contract, **_batching_metadata(self.config)}, **kwargs)

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
            if state.get("validation") is not None and state.get("recipe_evaluator_kind", "builtin") == "builtin":
                profile = state.get("recipe_evaluation", asdict(CausalLMEvaluationConfig()))
                if profile != asdict(self.config.evaluation):
                    raise ValueError("resume evaluation profile differs from its validation history")
        result = self.model.arti.load(
            path,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location or _device(self.model),
            load_checkpoint=load_checkpoint,
        )
        if load_checkpoint:
            self._resume_evaluation_kind = state.get("recipe_evaluator_kind", "builtin")
        return result


def _artifact_path(path):
    path = Path(path)
    return path / DEFAULT_ARTIFACT if path.is_dir() else path


class CausalLMPredictor:
    @property
    def serving_state_scope(self):
        return "read-only" if self.contract.get("adapter_factory") == "builtin-formula" and not self._caller_factory else "caller-managed"

    def validate_records(self, records):
        for row in records:
            normalized_messages(row.get(self.fields.messages), completed=False)

    def __init__(self, model, tokenizer, contract, *, evaluation=None, max_tokens_per_batch=None):
        self.model, self.tokenizer, self.contract = model.eval(), tokenizer, dict(contract)
        self._caller_factory = False
        self.fields = ChatFields(**contract["fields"])
        self.generation = CausalLMGenerationConfig(**contract["generation"])
        self.evaluation = evaluation or CausalLMEvaluationConfig()
        self.max_tokens_per_batch = max_tokens_per_batch
        self.model_source = None
        self._evaluation_binding = TrainingBinding.from_attachment(model.arti) if getattr(model, "arti", None) is not None else TrainingBinding(model, parameters=(), state_module=torch.nn.Module())

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
        manifest = load(artifact, load_checkpoint=False).manifest
        metadata = manifest["architecture"]["config"]
        declaration = metadata.get("task_recipe", {})
        evaluation = CausalLMEvaluationConfig(**metadata.get("evaluation", {}))
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
                + (f" (required: {declaration['layer_factory']['reference']})" if declaration.get("layer_factory") else "")
            )
        layer_factory = _resolve_layer_factory(layer_factory, expected=declaration.get("layer_factory"))
        if tokenizer is None:
            tokenizer = _load_tokenizer(artifact.parent / "tokenizer", local_files_only=True)
        if tokenizer_fingerprint(tokenizer) != declaration["tokenizer_sha256"]:
            raise ValueError("deployment tokenizer differs from the trained tokenizer")
        source = CausalLMModelConfig(**declaration["base_model"])
        if local_files_only is not None:
            source = replace(source, local_files_only=local_files_only)
        if model is None and base_model is None:
            from .recipe_deployment import local_deployment_base

            base_model = local_deployment_base(artifact, declaration)
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
        result = cls(model, tokenizer, declaration, evaluation=evaluation, max_tokens_per_batch=metadata.get("batching", {}).get("max_tokens_per_batch"))
        result._caller_factory = layer_factory is not None
        result.model_source = {
            "manifest_sha256": json_fingerprint(manifest),
            "identity_scope": "serialized-source-at-load",
        }
        return result

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

    def evaluate_records(self, records, *, batch_size=8, output=None, generate_samples=None):
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
            evaluation=self.evaluation,
            state_context=self._evaluation_binding.evaluation_context,
            max_tokens_per_batch=self.max_tokens_per_batch,
        )
        report.update(
            split="evaluation",
            task_recipe_sha256=self.contract["sha256"],
            dataset_sha256=split.fingerprint,
            **({"model_source": dict(self.model_source)} if self.model_source is not None else {}),
        )
        _save_evaluation(output, report, rows)
        return report

    def evaluate_recipe(self, config, *, split="validation", output=None, generate_samples=None):
        """Evaluate explicit data with an already loaded model; no trainer state."""
        config = _evaluation_recipe_config(config, self.contract)
        data, profile = _prepare_deployment_evaluation(config, self.tokenizer, self.contract, split, generate_samples)
        return self._evaluate_prepared_recipe(config, data, profile, split, output)

    def _evaluate_prepared_recipe(self, config, data, profile, split, output):
        report, rows = _evaluate_chat(
            self.model, self.tokenizer, data.splits[split], config.batch_size,
            self.contract["template_kwargs"], config.generation, None,
            evaluation=profile, state_context=self._evaluation_binding.evaluation_context,
            max_tokens_per_batch=config.max_tokens_per_batch,
        )
        report.update(
            ok=True, kind="recipe-evaluate", split=split,
            task_recipe_sha256=self.contract["sha256"],
            dataset_sha256=data.splits[split].fingerprint,
            dataset=data.manifest(include_ids=True),
            data_preparation=dict(data.preparation),
            data_config={"fields": asdict(config.fields), "max_length": config.max_length, "overlength": config.overlength, "pad_to_multiple_of": config.pad_to_multiple_of},
            execution="read-only-deployment",
            optimizer_created=False, training_session_created=False, training_state_loaded=False,
            **({"model_source": dict(self.model_source)} if self.model_source is not None else {}),
            **({"source_manifest_sha256": data.source_manifest["sha256"]} if data.source_manifest is not None else {}),
        )
        _save_evaluation(output, report, rows)
        return report


def _evaluation_recipe_config(config, declaration):
    config = config if isinstance(config, CausalLMRecipeConfig) else CausalLMRecipeConfig.read(config)
    # A post-training recipe chooses data and scoring, not a different network.
    source = {key: value for key, value in asdict(config.model).items() if key != "local_files_only"}
    if source != declaration["base_model"]:
        raise ValueError("evaluation model settings differ from the deployed base model")
    if dict(config.template_kwargs) != declaration["template_kwargs"]:
        raise ValueError("evaluation template_kwargs differ from the deployed chat contract")
    if config.layer_factory is not None and config.layer_factory.to_dict() != declaration.get("layer_factory"):
        raise ValueError("evaluation layer factory differs from the deployed program")
    return config


def _prepare_deployment_evaluation(config, tokenizer, declaration, split, generate_samples):
    if not isinstance(split, str) or split not in {"train", "validation", "test"}:
        raise ValueError("evaluation split must be train, validation or test")
    if tokenizer_fingerprint(tokenizer) != declaration["tokenizer_sha256"]:
        raise ValueError("deployment tokenizer differs from the trained tokenizer")
    data = _prepare_chat_dataset(config, tokenizer, split=split)
    profile = config.evaluation if generate_samples is None else replace(config.evaluation, generate_samples=generate_samples)
    generation_evaluation_plan(data.splits[split].examples, profile, config.generation)
    return data, profile


def evaluate_causal_lm_deployment(
    path, config, *, split="validation", output=None, generate_samples=None,
    device="cpu", local_files_only=None, model=None, tokenizer=None,
    layer_factory=None, base_model=None,
):
    """Prepare selected data before loading the asset, then evaluate read-only."""
    from .recipe_deployment import _declaration

    artifact, _manifest, declaration = _declaration(path, require_inference=False)
    config = _evaluation_recipe_config(config, declaration)
    if isinstance(layer_factory, str) and config.layer_factory is not None and layer_factory == config.layer_factory.reference:
        layer_factory = config.layer_factory
    if config.layer_factory is not None and layer_factory is not None and layer_factory != config.layer_factory:
        raise ValueError("choose the configured layer factory or one explicit Python factory, not both")
    layer_factory = config.layer_factory if layer_factory is None else layer_factory
    if declaration["adapter_factory"] == "caller-supplied" and layer_factory is None:
        raise ValueError("this asset requires its caller-supplied layer_factory before loading the host")
    layer_factory = _resolve_layer_factory(layer_factory, expected=declaration.get("layer_factory"))
    tokenizer = tokenizer if tokenizer is not None else _load_tokenizer(artifact.parent / "tokenizer", local_files_only=True)
    data, profile = _prepare_deployment_evaluation(config, tokenizer, declaration, split, generate_samples)
    from .attachment_training import _rng_state, _restore_rng

    rng = _rng_state()
    try:
        predictor = CausalLMPredictor.from_pretrained(
            artifact, device=device, local_files_only=local_files_only, model=model,
            tokenizer=tokenizer, layer_factory=layer_factory, base_model=base_model,
        )
        return predictor._evaluate_prepared_recipe(config, data, profile, split, output)
    finally:
        _restore_rng(rng)


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
    model, tokenizer, split, batch_size, template_kwargs, generation, generate_samples,
    *, evaluation=None, state_context=None, max_tokens_per_batch=None,
):
    _positive_int(batch_size, "evaluation.batch_size")
    _validate_token_budget(split.examples, split.pad_to_multiple_of, max_tokens_per_batch)
    profile = evaluation or CausalLMEvaluationConfig()
    if generate_samples is not None:
        profile = replace(profile, generate_samples=generate_samples)
    selected = select_generation_examples(split.examples, profile)
    snapshot = state_context or nullcontext
    modes = [(module, module.training) for module in model.modules()]
    totals = torch.zeros(2, device=_device(model))
    tokens = 0
    rows = []
    max_padded_tokens = 0
    try:
        model.eval()
        with snapshot(), torch.no_grad():
            for examples in _chat_example_batches(split.examples, batch_size, split.pad_to_multiple_of, max_tokens_per_batch=max_tokens_per_batch):
                batch = collate_chat(
                    examples,
                    pad_token_id=split.pad_token_id,
                    pad_to_multiple_of=split.pad_to_multiple_of,
                )
                output = _forward_batch(model, batch)
                max_padded_tokens = max(max_padded_tokens, batch["input_ids"].numel())
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
                        "input_sha256": json_fingerprint(list(example.messages[:-1])),
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
                "evaluation": asdict(profile),
                "generation_config": {**asdict(generation), "do_sample": False, "use_cache": True},
                "input_config": {"tokenizer_sha256": tokenizer_fingerprint(tokenizer), "template_kwargs": dict(template_kwargs)},
                "batching_config": {"batch_size_cap": batch_size, "max_tokens_per_batch": max_tokens_per_batch, "max_padded_tokens": max_padded_tokens},
            }
        if selected:
            with snapshot():
                completions = _generate_chat(
                    model, tokenizer, [list(example.messages[:-1]) for example in selected],
                    template_kwargs, generation,
                )
            by_id = {row["id"]: row for row in rows}
            generated_rows = []
            for example, result in zip(selected, completions, strict=True):
                row = by_id[example.id]
                row["generation"] = result
                generated_rows.append(row)
            scored = score_causal_lm_predictions(generated_rows, comparison=profile.comparison)
            for row in scored["predictions"]:
                by_id[row["id"]].update(row)
            report.update(scored["metrics"], generation_sample_ids=[item.id for item in selected])
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
    "prepare_causal_lm_data",
    "evaluate_causal_lm_deployment",
]
