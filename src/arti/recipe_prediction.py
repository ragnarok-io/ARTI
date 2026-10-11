"""Resumable local task inference with committed batch outputs."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Mapping

from .language_recipe import CausalLMGenerationConfig, CausalLMPredictor
from .recipe_chat import ChatFields, normalized_messages
from .recipe_data import _positive_int, json_fingerprint, read_jsonl
from .recipe_files import PredictionFilesConfig, _stage_prediction_files, _verify_prediction_files
from .recipe_episode import (
    EpisodeRunConfig,
    finished_episode_results,
    inspect_episode_work,
    run_episode_batch,
    verify_episode_results,
)
from .recipe_deployment import _declaration, local_task_resources
from .recipe_workflow import write_json, write_jsonl
from .serialization import _atomic_json, _file_sha256, _load_json, _replace_atomic_file


def _validate_limits(config, counts):
    if not isinstance(config.output_dir, (str, Path)) or not str(config.output_dir).strip():
        raise ValueError("prediction output_dir is required")
    object.__setattr__(config, "output_dir", Path(config.output_dir).resolve())
    for name in counts:
        if getattr(config, name) is not None:
            _positive_int(getattr(config, name), f"prediction.{name}")
    if config.max_seconds is not None and (
        isinstance(config.max_seconds, bool)
        or not isinstance(config.max_seconds, (int, float))
        or not math.isfinite(config.max_seconds)
        or config.max_seconds < 0
    ):
        raise ValueError("prediction.max_seconds must be finite and nonnegative")


@dataclass(frozen=True)
class CausalLMPredictionRunConfig:
    output_dir: str | Path
    batch_size: int | None = None
    max_new_tokens: int | None = None
    max_seconds: float | None = None

    def __post_init__(self):
        _validate_limits(self, ("batch_size", "max_new_tokens"))


@dataclass(frozen=True)
class PredictionRunConfig:
    output_dir: str | Path
    batch_size: int | None = None
    predict_kwargs: Mapping[str, Any] = field(default_factory=dict)
    id_field: str | None = None
    output_id_field: str | None = None
    max_seconds: float | None = None
    files: PredictionFilesConfig | None = None
    episodes: EpisodeRunConfig | None = None

    def __post_init__(self):
        _validate_limits(self, ("batch_size",))
        if self.files is not None and not isinstance(self.files, PredictionFilesConfig):
            raise ValueError("prediction files must use PredictionFilesConfig")
        if self.episodes is not None:
            if not isinstance(self.episodes, EpisodeRunConfig):
                raise ValueError("prediction episodes must use EpisodeRunConfig")
            if self.files is not None or self.predict_kwargs:
                raise ValueError(
                    "episode callbacks use explicit observation/state contracts, "
                    "not file producers or prediction options"
                )
        if not isinstance(self.predict_kwargs, Mapping):
            raise ValueError("prediction.predict_kwargs must be a JSON object")
        if self.files is not None and "output_dir" in self.predict_kwargs:
            raise ValueError("file producer output_dir is supplied by the prediction run")
        object.__setattr__(
            self,
            "predict_kwargs",
            json.loads(json.dumps(dict(self.predict_kwargs), allow_nan=False)),
        )
        for name in ("id_field", "output_id_field"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"prediction.{name} must be a nonempty string")


def _model_declaration(path, *, legacy=False):
    if legacy:
        return _declaration(path, require_inference=True)
    from .attachment_hub import DEFAULT_ARTIFACT
    from .task_recipe import _manifest

    path = Path(path)
    if path.is_dir() and (path / DEFAULT_ARTIFACT).exists():
        path = path / DEFAULT_ARTIFACT
    artifact, declaration, manifest = _manifest(path)
    if declaration.get("format_version") != 1 or declaration.get("format") not in {
        "arti.task-recipe",
        "arti.classification-recipe",
        "arti.causal-lm-recipe",
    }:
        raise ValueError("prediction requires an exported task recipe asset")
    if declaration.get("sha256") != json_fingerprint(
        {key: value for key, value in declaration.items() if key != "sha256"}
    ):
        raise ValueError("task declaration fingerprint differs")
    if manifest.get("checkpoint") is not None:
        raise ValueError(
            "export an inference asset before prediction; training checkpoints are separate"
        )
    local_task_resources(artifact, manifest)
    return artifact, manifest, declaration


def _batch_size(plan):
    return plan["batch_size"] if "batch_size" in plan else plan["generation"]["batch_size"]


def _predict_kwargs(plan):
    if "predict_kwargs" in plan:
        return json.loads(json.dumps(plan["predict_kwargs"]))
    return {name: plan["generation"][name] for name in ("batch_size", "max_new_tokens")}


def _output_id(plan):
    return plan.get("output_id_field", "id")


def _episode_counter_attribution():
    return {
        "policy_batch_calls": "policy callback dispatches; not neural forward counts",
        "completed_environment_step_callbacks": "library-observed returned step callbacks",
        "executed_environment_calls": "application response calls; underlying consumption not independently observed",
        "local_replay_step_calls": "library-observed returned local replay callbacks",
        "replay_environment_calls": "application replay response calls; separately attributed",
    }


def _input_batches(path, size):
    records = iter(read_jsonl(path))
    while batch := list(islice(records, size)):
        yield batch


def _prepare_prediction(path, data_path, run):
    legacy = isinstance(run, CausalLMPredictionRunConfig)
    artifact, manifest, declaration = _model_declaration(path, legacy=legacy)
    if data_path is None:
        raise ValueError("a new prediction run requires a data path")
    target = run.output_dir
    if target.exists():
        raise FileExistsError("prediction run already exists; use resume=True")
    language = declaration["format"] == "arti.causal-lm-recipe"
    if not legacy and run.files is not None and declaration["format"] != "arti.task-recipe":
        raise ValueError("file outputs require an explicit generic task producer")
    if not legacy and run.episodes is not None and declaration["format"] != "arti.task-recipe":
        raise ValueError("episodes require an explicit generic TaskPredictor/TrainingBinding")
    if language:
        fields = ChatFields(**declaration["fields"])
        generation = CausalLMGenerationConfig(
            **{
                **declaration["generation"],
                **({"batch_size": run.batch_size} if legacy and run.batch_size is not None else {}),
                **(
                    {"max_new_tokens": run.max_new_tokens}
                    if legacy and run.max_new_tokens is not None
                    else {}
                ),
                **(dict(run.predict_kwargs) if not legacy else {}),
            }
        )
        defaults = {name: getattr(generation, name) for name in ("batch_size", "max_new_tokens")}
    else:
        defaults = dict(run.predict_kwargs)
    id_field = (
        declaration.get("fields", {}).get("id", "id")
        if legacy or run.id_field is None
        else run.id_field
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        digest, seen, count = hashlib.sha256(), set(), 0
        with Path(data_path).open("rb") as source, (staging / "input.jsonl").open("wb") as snapshot:
            for line_number, line in enumerate(source, 1):
                digest.update(line)
                snapshot.write(line)
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("prediction record must be an object")
                    identifier = row.get(id_field)
                    if not isinstance(identifier, str) or not identifier or identifier in seen:
                        raise ValueError("prediction requires unique nonempty string IDs")
                    seen.add(identifier)
                    if not legacy and run.episodes is not None:
                        run.episodes.trial(row)
                    if language:
                        normalized_messages(row.get(fields.messages), completed=False)
                    count += 1
                except ValueError as error:
                    raise ValueError(f"prediction input line {line_number}: {error}") from error
        if not count:
            raise ValueError("prediction input must not be empty")
        plan = {
            "format": "arti.causal-lm-prediction-run",
            "format_version": 1,
            "model_manifest_sha256": json_fingerprint(manifest),
            "task_recipe_sha256": declaration["sha256"],
            "input": {"file": "input.jsonl", "sha256": digest.hexdigest(), "records": count},
            "state_scope": "owned-read-only-per-batch",
        }
        if legacy:
            plan.update(fields=declaration["fields"], generation={**defaults, "do_sample": False})
        else:
            plan.update(
                format="arti.prediction-run",
                task=declaration["task"],
                fields={"id": id_field},
                output_id_field=run.output_id_field or "id",
                batch_size=run.batch_size or 128,
                predict_kwargs=defaults,
            )
            if run.files is not None:
                if run.files.field == plan["output_id_field"]:
                    raise ValueError("file output field cannot replace the record ID")
                plan["files"] = run.files.to_dict()
            if run.episodes is not None:
                if run.episodes.field == plan["output_id_field"]:
                    raise ValueError("episode output field cannot replace the record ID")
                plan["episodes"] = run.episodes.to_dict()
                plan["state_scope"] = "independent-application-trials;explicit-native-model-lifetime"
        _resources, deployment_sha256 = local_task_resources(artifact, manifest)
        if deployment_sha256 is not None:
            plan["deployment_sha256"] = deployment_sha256
        plan["sha256"] = json_fingerprint(plan)
        write_json(staging / "plan.json", plan)
        (staging / "parts").mkdir()
        _atomic_json(staging / "run.json", {"status": "prepared", "attempts": []})
        _replace_atomic_file(staging, target)
    return artifact, plan, declaration


def _read_plan(directory):
    plan = _load_json(directory / "plan.json")
    if (
        plan.get("format") not in {"arti.causal-lm-prediction-run", "arti.prediction-run"}
        or plan.get("format_version") != 1
    ):
        raise ValueError("not a supported task prediction run")
    if plan.get("sha256") != json_fingerprint(
        {key: value for key, value in plan.items() if key != "sha256"}
    ):
        raise ValueError("prediction plan fingerprint differs")
    if _file_sha256(directory / "input.jsonl") != plan["input"]["sha256"]:
        raise ValueError("prediction input snapshot differs from the plan")
    return plan


def _committed_progress(directory, plan):
    size = _batch_size(plan)
    paths = sorted((directory / "parts").iterdir())
    id_field = plan["fields"]["id"]
    count, batches, tokens, seconds = 0, 0, 0, 0.0
    merged = hashlib.sha256()
    file_progress = {"output_files": 0, "output_bytes": 0, "file_preparation_seconds": 0.0}
    episode_progress = {
        "policy_batch_calls": 0,
        "completed_environment_step_callbacks": 0,
        "executed_environment_calls": 0,
        "replay_environment_calls": 0,
        "local_replay_step_calls": 0,
    }
    episode_timing_complete = True
    episode_dispatch_scopes = set()
    for records, part in zip(_input_batches(directory / "input.jsonl", size), paths):
        if part.name != f"{batches:08d}" or not part.is_dir():
            raise ValueError("prediction batches are not a contiguous committed prefix")
        receipt = _load_json(part / "receipt.json")
        if (
            receipt.get("plan_sha256") != plan["sha256"]
            or receipt.get("start") != count
            or receipt.get("records") != len(records)
            or receipt.get("input_sha256") != json_fingerprint(records)
        ):
            raise ValueError("prediction batch does not match its pinned input/plan")
        digest, expected, predictions = hashlib.sha256(), iter(records), []
        with (part / "predictions.jsonl").open("rb") as stream:
            for line in stream:
                digest.update(line)
                merged.update(line)
                if not line.strip():
                    continue
                record, row = next(expected, None), json.loads(line)
                if "files" in plan or "episodes" in plan:
                    predictions.append(row)
                if (
                    record is None
                    or not isinstance(row, dict)
                    or row.get(_output_id(plan)) != record[id_field]
                ):
                    raise ValueError("prediction batch output IDs/count differ")
        if (
            next(expected, None) is not None
            or receipt.get("predictions_sha256") != digest.hexdigest()
        ):
            raise ValueError("prediction batch output count/fingerprint differs")
        count += len(records)
        if "files" in plan:
            for key, value in _verify_prediction_files(
                directory, part, predictions, plan, receipt
            ).items():
                file_progress[key] += value
        if "episodes" in plan:
            verify_episode_results(predictions, records, EpisodeRunConfig.from_dict(plan["episodes"]))
            for key in episode_progress:
                episode_progress[key] += receipt["execution"][key]
            episode_timing_complete &= receipt["execution"].get("prediction_timing") is None
            episode_dispatch_scopes.add(receipt["execution"].get("policy_dispatch_scope", "unrecorded"))
        batches += 1
        tokens += receipt.get("generated_tokens", 0)
        seconds += receipt["prediction_seconds"]
    if batches != len(paths) or count > plan["input"]["records"]:
        raise ValueError("prediction batches exceed the pinned input")
    return {
        "committed_records": count,
        "committed_batches": batches,
        **({"generated_tokens": tokens} if "generation" in plan else {}),
        "prediction_seconds": seconds,
        "result_sha256": merged.hexdigest(),
        **(file_progress if "files" in plan else {}),
        **(episode_progress if "episodes" in plan else {}),
        **({"prediction_timing_complete": episode_timing_complete,
            "episode_counter_attribution": _episode_counter_attribution(),
            "episode_policy_dispatch_scopes": sorted(episode_dispatch_scopes)} if "episodes" in plan else {}),
    }


def inspect_prediction(directory):
    """Inspect committed outputs without loading a model or decoding tensors."""
    directory = Path(directory).resolve()
    plan = _read_plan(directory)
    progress = _committed_progress(directory, plan)
    complete = progress["committed_records"] == plan["input"]["records"]
    result = directory / "predictions.jsonl"
    ready = complete and result.exists() and _file_sha256(result) == progress["result_sha256"]
    report = _load_json(directory / "run.json")
    episode_work = (
        {} if complete else inspect_episode_work(directory, plan, progress["committed_batches"])
    )
    blocked = episode_work.get("episode_recovery", {}).get("blockers")
    finished = episode_work.get("episode_recovery", {}).get("finished_part_ready")
    return {
        "ok": True,
        "kind": "prediction-run-inspection",
        "directory": str(directory),
        "plan_sha256": plan["sha256"],
        "model_manifest_sha256": plan["model_manifest_sha256"],
        **({"deployment_sha256": plan["deployment_sha256"]} if "deployment_sha256" in plan else {}),
        "source_status": report.get("status"),
        "attempts": len(report.get("attempts", [])),
        "total_records": plan["input"]["records"],
        **progress,
        "remaining_records": plan["input"]["records"] - progress["committed_records"],
        "action": "already_completed" if ready else "finalize" if complete
        else "recovery_not_supported" if blocked
        else "publish_episode_part" if finished else "resume_prediction",
        "predictions": str(result) if ready else None,
        "loads_model_weights": False,
        **episode_work,
    }


def _commit_batch(
    directory,
    plan,
    index,
    start,
    records,
    predictions,
    seconds,
    *,
    input_receipt=None,
    output_dir=None,
    execution=None,
):
    input_receipt = input_receipt or {
        "ids": [row[plan["fields"]["id"]] for row in records],
        "records": len(records),
        "sha256": json_fingerprint(records),
    }
    ids = input_receipt["ids"]
    if (
        not isinstance(predictions, list)
        or len(predictions) != input_receipt["records"]
        or any(not isinstance(row, dict) for row in predictions)
        or [row.get(_output_id(plan)) for row in predictions] != ids
    ):
        raise ValueError("prediction outputs must match input count and ordered IDs")
    if "episodes" in plan:
        verify_episode_results(predictions, records, EpisodeRunConfig.from_dict(plan["episodes"]))
    with tempfile.TemporaryDirectory(prefix=".batch-", dir=directory) as temporary:
        staging = Path(temporary) / "batch"
        staging.mkdir()
        file_receipt = {}
        if "files" in plan:
            if output_dir is None:
                raise ValueError("file predictions require a staged producer directory")
            predictions, file_receipt = _stage_prediction_files(
                predictions,
                Path(output_dir).resolve(),
                staging,
                f"parts/{index:08d}",
                PredictionFilesConfig.from_dict(plan["files"]),
                records,
            )
        write_jsonl(staging / "predictions.jsonl", predictions)
        receipt = {
            "plan_sha256": plan["sha256"],
            "start": start,
            "records": input_receipt["records"],
            "input_sha256": input_receipt["sha256"],
            "predictions_sha256": _file_sha256(staging / "predictions.jsonl"),
            **(
                {"generated_tokens": sum(row["generated_tokens"] for row in predictions)}
                if "generation" in plan
                else {}
            ),
            "prediction_seconds": seconds,
            **file_receipt,
            **({"execution": execution} if execution is not None else {}),
        }
        write_json(staging / "receipt.json", receipt)
        _replace_atomic_file(staging, directory / "parts" / f"{index:08d}")
    return receipt


def _merge_predictions(directory, batches):
    with tempfile.TemporaryDirectory(prefix=".merge-", dir=directory) as temporary:
        output = Path(temporary) / "predictions.jsonl"
        with output.open("wb") as target:
            for index in range(batches):
                with (directory / "parts" / f"{index:08d}" / "predictions.jsonl").open(
                    "rb"
                ) as source:
                    shutil.copyfileobj(source, target, 1024 * 1024)
        _replace_atomic_file(output, directory / "predictions.jsonl")


def run_prediction(
    path,
    data_path,
    run: PredictionRunConfig | CausalLMPredictionRunConfig,
    *,
    resume=False,
    device="cpu",
    local_files_only=None,
    base_model=None,
    layer_factory=None,
    task_factory=None,
    predictor=None,
    stop_requested=None,
    on_batch=None,
    on_episode_event=None,
):
    """Load once and commit independent batches; resume only uncommitted work."""
    if not isinstance(run, (PredictionRunConfig, CausalLMPredictionRunConfig)):
        raise ValueError("run must be PredictionRunConfig or CausalLMPredictionRunConfig")
    legacy = isinstance(run, CausalLMPredictionRunConfig)
    directory = run.output_dir
    artifact = declaration = None
    if resume:
        plan = _read_plan(directory)
        if path is not None:
            artifact, manifest, declaration = _model_declaration(path, legacy=legacy)
            if plan["model_manifest_sha256"] != json_fingerprint(manifest):
                raise ValueError("prediction model source differs from the pinned run")
            _resources, deployment_sha256 = local_task_resources(artifact, manifest)
            if deployment_sha256 != plan.get("deployment_sha256"):
                raise ValueError("prediction deployment differs from the pinned run")
        if legacy:
            for name in ("batch_size", "max_new_tokens"):
                if getattr(run, name) is not None and getattr(run, name) != plan.get(
                    "generation", {}
                ).get(name):
                    raise ValueError(f"prediction {name} differs from the pinned run")
        else:
            if run.batch_size is not None and run.batch_size != _batch_size(plan):
                raise ValueError("prediction batch_size differs from the pinned run")
            if run.files is not None and run.files.to_dict() != plan.get("files"):
                raise ValueError("prediction file producer/validator differs from the pinned run")
            if run.episodes is not None and run.episodes.to_dict() != plan.get("episodes"):
                raise ValueError("episode policy/environment/evaluator differs from the pinned run")
            for name, value in run.predict_kwargs.items():
                if name not in _predict_kwargs(plan) or value != _predict_kwargs(plan)[name]:
                    raise ValueError(f"prediction option {name} differs from the pinned run")
            for name, value in (
                ("id_field", plan["fields"]["id"]),
                ("output_id_field", _output_id(plan)),
            ):
                if getattr(run, name) is not None and getattr(run, name) != value:
                    raise ValueError(f"prediction {name} differs from the pinned run")
        if data_path is not None and _file_sha256(Path(data_path)) != plan["input"]["sha256"]:
            raise ValueError("prediction data differs from the pinned run")
    else:
        if path is None:
            raise ValueError("a new prediction run requires a model asset")
        artifact, plan, declaration = _prepare_prediction(path, data_path, run)
    progress = inspect_prediction(directory)
    if progress["action"] == "publish_episode_part":
        # A terminal application result is already durable work. Adopt it into
        # the same atomic part without reconstructing any producer/model state.
        for index, records in enumerate(_input_batches(directory / "input.jsonl", _batch_size(plan))):
            if index < progress["committed_batches"]:
                continue
            finished = finished_episode_results(directory, plan, index, records)
            if finished is None:
                break
            predictions, counts = finished
            _commit_batch(
                directory, plan, index, progress["committed_records"], records, predictions, 0.0,
                execution={
                    "attempt": None, "phase": "publication-only", **counts,
                    "prediction_timing": "unrecorded-before-part-publication",
                    "counter_attribution": _episode_counter_attribution(),
                    "policy_dispatch_scope": EpisodeRunConfig.from_dict(plan["episodes"]).dispatch_scope,
                },
            )
            progress = inspect_prediction(directory)
    if progress["action"] == "recovery_not_supported":
        return {
            **progress,
            "ok": False,
            "kind": "prediction-run",
            "status": "not_run",
            "newly_predicted_records": 0,
            "reused_records": progress["committed_records"],
            "reason": "unsupported_episode_recovery",
        }
    if progress["action"] == "resume_prediction" and artifact is None:
        raise ValueError("unfinished prediction requires the pinned model asset")
    record = _load_json(directory / "run.json")
    if progress["action"] == "already_completed":
        if record.get("status") != "completed":
            record.update(
                status="completed",
                committed_records=progress["committed_records"],
                committed_batches=progress["committed_batches"],
            )
            _atomic_json(directory / "run.json", record)
        return {
            **progress,
            "kind": "prediction-run",
            "status": "completed",
            "newly_predicted_records": 0,
            "reused_records": progress["committed_records"],
        }
    attempt = {
        "number": len(record["attempts"]) + 1,
        "status": "running",
        "newly_predicted_records": 0,
        "max_seconds": run.max_seconds,
    }
    record["attempts"].append(attempt)
    reused = progress["committed_records"]
    started, work_started, requested_rows = time.monotonic(), None, 0

    def save(status, **values):
        attempt.update(status=status, elapsed_seconds=time.monotonic() - started, **values)
        record.update(
            status=status,
            committed_records=progress["committed_records"],
            committed_batches=progress["committed_batches"],
        )
        _atomic_json(directory / "run.json", record)

    def report(status, **values):
        return {
            "ok": True,
            "kind": "prediction-run",
            "directory": str(directory),
            "status": status,
            "plan_sha256": plan["sha256"],
            "model_manifest_sha256": plan["model_manifest_sha256"],
            "total_records": plan["input"]["records"],
            **{
                key: progress[key]
                for key in (
                    "committed_records",
                    "committed_batches",
                    "generated_tokens",
                    "prediction_seconds",
                    "output_files",
                    "output_bytes",
                    "file_preparation_seconds",
                    "policy_batch_calls",
                    "completed_environment_step_callbacks",
                    "executed_environment_calls",
                    "replay_environment_calls",
                    "local_replay_step_calls",
                    "prediction_timing_complete",
                    "episode_counter_attribution",
                    "episode_policy_dispatch_scopes",
                )
                if key in progress
            },
            "newly_predicted_records": attempt["newly_predicted_records"],
            "reused_records": reused,
            "budget_scope": "episode-step-starts-after-model-loading"
            if "episodes" in plan else "new-batch-starts-after-model-loading",
            "predictions": str(directory / "predictions.jsonl") if status == "completed" else None,
            **values,
        }

    save("running")
    try:
        for index, records in enumerate(
            _input_batches(directory / "input.jsonl", _batch_size(plan))
        ):
            if index < progress["committed_batches"]:
                continue
            requested = stop_requested is not None and stop_requested()
            exhausted = run.max_seconds is not None and (
                run.max_seconds == 0
                or (work_started is not None and time.monotonic() - work_started >= run.max_seconds)
            )
            if requested or exhausted:
                reason = "requested_stop" if requested else "time_budget"
                save("stopped", stop_reason=reason)
                return report("stopped", stop_reason=reason)
            if work_started is None:
                if predictor is None:
                    from .attachment_training import _rng_state, _restore_rng

                    rng = _rng_state()
                    try:
                        if legacy:
                            predictor = CausalLMPredictor.from_pretrained(
                                artifact,
                                device=device,
                                local_files_only=local_files_only,
                                base_model=base_model,
                                layer_factory=layer_factory,
                            )
                        else:
                            from .recipes import load_recipe_predictor

                            predictor = load_recipe_predictor(
                                artifact,
                                device=device,
                                task_factory=task_factory,
                                base_model=base_model,
                                layer_factory=layer_factory,
                            )
                    finally:
                        _restore_rng(rng)
                source_digest = getattr(predictor, "model_source_sha256", None)
                if source_digest is None and getattr(predictor, "model_source", None) is not None:
                    source_digest = predictor.model_source["manifest_sha256"]
                if (
                    predictor.contract["sha256"] != plan["task_recipe_sha256"]
                    or source_digest != plan["model_manifest_sha256"]
                    or getattr(predictor, "deployment_sha256", None)
                    != plan.get("deployment_sha256")
                ):
                    raise ValueError("loaded predictor differs from the pinned model source")
                work_started = time.monotonic()
            requested_rows = len(records)
            input_receipt = {
                "ids": [row[plan["fields"]["id"]] for row in records],
                "records": requested_rows,
                "sha256": json_fingerprint(records),
            }
            input_records = (
                json.loads(json.dumps(records)) if "files" in plan or "episodes" in plan else records
            )
            batch_started = time.monotonic()
            binding = getattr(predictor, "_evaluation_binding", None) or getattr(
                predictor, "binding", None
            )
            if binding is None:
                from .training_binding import TrainingBinding

                binding = TrainingBinding(predictor.model)
            with (
                tempfile.TemporaryDirectory(prefix=".producer-", dir=directory)
                if "files" in plan
                else nullcontext(None)
            ) as temporary:
                output_dir = Path(temporary) if "files" in plan else None
                episode_counts = {}
                if "episodes" in plan:

                    def episode_stop():
                        if stop_requested is not None and stop_requested():
                            return "requested_stop"
                        if (
                            run.max_seconds is not None
                            and time.monotonic() - work_started >= run.max_seconds
                        ):
                            return "host_time_budget"
                        return False
                    predictions, episode_counts = run_episode_batch(
                        directory, plan, index, input_records, predictor,
                        attempt=attempt["number"],
                        stop_requested=episode_stop,
                        on_event=on_episode_event,
                    )
                else:
                    with binding.evaluation_context():
                        predictions = predictor.predict_records(
                            records,
                            **_predict_kwargs(plan),
                            **({"output_dir": output_dir} if output_dir is not None else {}),
                        )
                seconds = time.monotonic() - batch_started
                # A committed batch is authoritative even if the progress write fails next.
                receipt = _commit_batch(
                    directory,
                    plan,
                    index,
                    progress["committed_records"],
                    input_records,
                    predictions,
                    seconds,
                    input_receipt=input_receipt,
                    output_dir=output_dir,
                    execution={
                        "attempt": attempt["number"],
                        "phase": "initial" if attempt["number"] == 1 else "recovered",
                        **episode_counts,
                        **({"counter_attribution": _episode_counter_attribution()} if "episodes" in plan else {}),
                        **({"policy_dispatch_scope": EpisodeRunConfig.from_dict(plan["episodes"]).dispatch_scope} if "episodes" in plan else {}),
                    },
                )
            progress["committed_records"] += receipt["records"]
            progress["committed_batches"] += 1
            if "generated_tokens" in progress:
                progress["generated_tokens"] += receipt["generated_tokens"]
            progress["prediction_seconds"] += seconds
            if "files" in plan:
                progress["output_files"] += len(receipt["files"])
                progress["output_bytes"] += sum(file["bytes"] for file in receipt["files"])
                progress["file_preparation_seconds"] += receipt["file_preparation_seconds"]
            if "episodes" in plan:
                for key, value in episode_counts.items():
                    progress[key] += value
                scope = receipt["execution"]["policy_dispatch_scope"]
                progress["episode_policy_dispatch_scopes"] = sorted(set(progress["episode_policy_dispatch_scopes"]) | {scope})
            attempt["newly_predicted_records"] += receipt["records"]
            requested_rows = 0
            save("running")
            if on_batch is not None:
                on_batch(report("running"), receipt)
        _merge_predictions(directory, progress["committed_batches"])
        save("completed")
        return report("completed")
    except BaseException as error:
        try:
            save(
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error_type=type(error).__name__,
                uncommitted_requested_records=requested_rows,
            )
        except OSError as progress_error:
            raise error from progress_error
        raise


__all__ = [
    "PredictionRunConfig",
    "run_prediction",
    "inspect_prediction",
    "CausalLMPredictionRunConfig",
    "run_causal_lm_prediction",
    "inspect_causal_lm_prediction",
]


def run_causal_lm_prediction(path, data_path, run, **kwargs):
    if not isinstance(run, CausalLMPredictionRunConfig):
        raise ValueError("run must be CausalLMPredictionRunConfig")
    return run_prediction(path, data_path, run, **kwargs)


def inspect_causal_lm_prediction(directory):
    return inspect_prediction(directory)
