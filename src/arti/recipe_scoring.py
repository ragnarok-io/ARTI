"""Task-owned offline scoring of committed prediction outputs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

from .python_factory import PythonFactoryConfig
from .recipe_data import json_fingerprint, read_jsonl
from .recipe_prediction import _output_id, _read_plan, inspect_prediction
from .serialization import _atomic_json


@dataclass(frozen=True)
class PredictionScoringConfig:
    scorer: PythonFactoryConfig | str
    scorer_id: str
    target_id_field: str | None = None

    def __post_init__(self):
        if isinstance(self.scorer, str):
            object.__setattr__(self, "scorer", PythonFactoryConfig(self.scorer))
        if not isinstance(self.scorer, PythonFactoryConfig):
            raise ValueError("scoring.scorer must be an installed Python callable")
        if not isinstance(self.scorer_id, str) or not self.scorer_id.strip():
            raise ValueError("scoring.scorer_id must declare the scoring rule/version")
        if self.target_id_field is not None and (
            not isinstance(self.target_id_field, str) or not self.target_id_field
        ):
            raise ValueError("scoring.target_id_field must be a nonempty string")

    def to_dict(self):
        return {
            "scorer": self.scorer.to_dict(),
            "scorer_id": self.scorer_id,
            "target_id_field": self.target_id_field,
        }


def _indexed(rows, field, name):
    result = {}
    for row in rows:
        identifier = row.get(field) if isinstance(row, dict) else None
        if not isinstance(identifier, str) or not identifier or identifier in result:
            raise ValueError(f"{name} requires distinct nonempty string IDs in {field!r}")
        result[identifier] = row
    return result


def _read_run(directory, allow_partial):
    directory = Path(directory).resolve()
    progress = inspect_prediction(directory)
    plan = _read_plan(directory)
    if progress["remaining_records"] and not allow_partial:
        raise ValueError("prediction run is unfinished; use allow_partial=True explicitly")
    if not progress["committed_records"]:
        raise ValueError("scoring requires at least one committed prediction")
    inputs = _indexed(read_jsonl(directory / "input.jsonl"), plan["fields"]["id"], "inputs")
    predictions = _indexed(
        (
            row
            for index in range(progress["committed_batches"])
            for row in read_jsonl(directory / "parts" / f"{index:08d}" / "predictions.jsonl")
        ),
        _output_id(plan),
        "predictions",
    )
    source = {
        "directory": str(directory),
        "plan_sha256": plan["sha256"],
        "model_manifest_sha256": plan["model_manifest_sha256"],
        **({"deployment_sha256": plan["deployment_sha256"]} if "deployment_sha256" in plan else {}),
        "task_recipe_sha256": plan["task_recipe_sha256"],
        "input_sha256": plan["input"]["sha256"],
        "predictions_sha256": progress["result_sha256"],
        "total_records": progress["total_records"],
        "committed_records": progress["committed_records"],
        "uncommitted_records": progress["remaining_records"],
        "conditions": {
            key: plan[key]
            for key in (
                "task",
                "fields",
                "output_id_field",
                "generation",
                "predict_kwargs",
                "batch_size",
                "episodes",
            )
            if key in plan
        },
    }
    return directory, plan, inputs, predictions, source


def _targets(path, config, plan, inputs, identifiers):
    field = config.target_id_field or plan["fields"]["id"]
    rows = list(read_jsonl(path)) if path is not None else list(inputs.values())
    targets = _indexed(rows, field, "targets")
    missing = sorted(set(identifiers) - targets.keys())
    if missing:
        raise ValueError(f"targets are missing prediction IDs: {missing[:5]!r}")
    selected = [targets[identifier] for identifier in identifiers]
    return targets, {
        "path": str(Path(path).resolve()) if path is not None else None,
        "origin": "external-targets" if path is not None else "pinned-input-records",
        "id_field": field,
        "records": len(rows),
        "records_sha256": json_fingerprint([targets[key] for key in sorted(targets)]),
        "scored_records_sha256": json_fingerprint(selected),
        "unused_ids": sorted(targets.keys() - set(identifiers)),
    }


def _score(identifiers, inputs, predictions, targets, config):
    pairs = [
        {
            "id": identifier,
            "input": inputs[identifier],
            "target": targets[identifier],
            "prediction": predictions[identifier],
        }
        for identifier in identifiers
    ]
    return _score_pairs(pairs, config)


def _score_pairs(pairs, config, **context):
    # Score code may preprocess its arguments; saved evidence remains the original values.
    result = config.scorer.resolve()(deepcopy(pairs), **context)
    return _score_result(pairs, result)


def _score_result(pairs, result):
    identifiers = [pair["id"] for pair in pairs]
    if not isinstance(result, Mapping) or not isinstance(result.get("metrics"), Mapping):
        raise ValueError("scorer must return metrics and per-ID scores")
    try:
        metrics = json.loads(json.dumps(dict(result["metrics"]), allow_nan=False))
        scores = json.loads(json.dumps(result["scores"], allow_nan=False))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("scorer metrics/scores must be finite JSON values") from error
    if not isinstance(scores, list):
        raise ValueError("scorer scores must be a list of per-ID records")
    indexed = _indexed(scores, "id", "scorer scores")
    if set(indexed) != set(identifiers):
        raise ValueError("scorer scores must cover exactly the paired IDs")
    return metrics, [
        {**pair, "score": {key: value for key, value in indexed[pair["id"]].items() if key != "id"}}
        for pair in pairs
    ]


def _publish(result, output, directories, targets):
    if output is not None:
        path = Path(output).resolve()
        protected = {
            directory / name
            for directory in directories
            for name in ("plan.json", "run.json", "input.jsonl", "predictions.jsonl")
        }
        if (
            path in protected
            or any(path.is_relative_to(directory / "parts") for directory in directories)
            or (targets is not None and path == Path(targets).resolve())
        ):
            raise ValueError("scoring output must not overwrite its source run or targets")
    result["sha256"] = json_fingerprint(result)
    if output is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, result)
    return result


def score_prediction_run(directory, scoring, *, targets=None, allow_partial=False, output=None):
    """Score saved task results; neither a model nor a task factory is needed."""
    if not isinstance(scoring, PredictionScoringConfig):
        raise ValueError("scoring must be PredictionScoringConfig")
    directory, plan, inputs, predictions, source = _read_run(directory, allow_partial)
    identifiers = list(predictions)
    target_rows, target_source = _targets(targets, scoring, plan, inputs, identifiers)
    metrics, scores = _score(identifiers, inputs, predictions, target_rows, scoring)
    return _publish(
        {
            "ok": True,
            "kind": "prediction-scores",
            "format": "arti.prediction-scores",
            "format_version": 1,
            "scoring": scoring.to_dict(),
            "source": source,
            "targets": target_source,
            "samples": len(identifiers),
            "coverage": len(identifiers) / source["total_records"],
            "allow_partial": bool(allow_partial),
            "metrics": metrics,
            "scores": scores,
        },
        output,
        [directory],
        targets,
    )


def compare_prediction_runs(
    candidate, baseline, scoring, *, targets=None, pairing="exact", allow_partial=False, output=None
):
    """Rescore two committed arms on identical inputs and one common target set."""
    if not isinstance(scoring, PredictionScoringConfig):
        raise ValueError("scoring must be PredictionScoringConfig")
    if pairing not in {"exact", "intersection"}:
        raise ValueError("pairing must be exact or intersection")
    arms = {
        name: _read_run(path, allow_partial)
        for name, path in (("candidate", candidate), ("baseline", baseline))
    }
    current, previous = set(arms["candidate"][3]), set(arms["baseline"][3])
    if pairing == "exact" and current != previous:
        raise ValueError(
            "prediction IDs differ; use explicit intersection pairing for partial coverage"
        )
    identifiers = sorted(current & previous)
    if not identifiers:
        raise ValueError("comparison requires at least one shared prediction ID")
    for identifier in identifiers:
        if json_fingerprint(arms["candidate"][2][identifier]) != json_fingerprint(
            arms["baseline"][2][identifier]
        ):
            raise ValueError(f"prediction inputs differ for {identifier!r}")
    target_rows, target_source = _targets(
        targets, scoring, arms["candidate"][1], arms["candidate"][2], identifiers
    )
    metrics, scores = {}, {}
    for name, (_, _, inputs, predictions, _) in arms.items():
        metrics[name], scores[name] = _score(identifiers, inputs, predictions, target_rows, scoring)
    numeric = {
        key
        for key in metrics["candidate"].keys() & metrics["baseline"].keys()
        if all(
            type(metrics[name][key]) in {int, float} and math.isfinite(metrics[name][key])
            for name in arms
        )
    }
    return _publish(
        {
            "ok": True,
            "kind": "prediction-comparison",
            "format": "arti.prediction-comparison",
            "format_version": 1,
            "scoring": scoring.to_dict(),
            "sources": {name: arm[4] for name, arm in arms.items()},
            "targets": target_source,
            "pairing": pairing,
            "allow_partial": bool(allow_partial),
            "paired_samples": len(identifiers),
            "coverage": {
                name: {
                    "committed_records": len(arm[3]),
                    "total_records": arm[4]["total_records"],
                    "paired_fraction": len(identifiers) / len(arm[3]),
                    "total_fraction": len(identifiers) / arm[4]["total_records"],
                    "excluded_ids": sorted(arm[3].keys() - set(identifiers)),
                    "uncommitted_records": arm[4]["uncommitted_records"],
                }
                for name, arm in arms.items()
            },
            "metrics": metrics,
            "delta": {
                key: metrics["candidate"][key] - metrics["baseline"][key] for key in sorted(numeric)
            },
            "delta_semantics": "candidate-minus-baseline; metric direction is task-owned",
            "scores": scores,
        },
        output,
        [arm[0] for arm in arms.values()],
        targets,
    )


__all__ = ["PredictionScoringConfig", "score_prediction_run", "compare_prediction_runs"]
