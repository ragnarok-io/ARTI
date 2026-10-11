"""Reference-aware task evaluation over existing committed result evidence."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, asdict
import json
import math
from pathlib import Path

from .python_factory import PythonFactoryConfig
from .recipe_assets import DatasetSnapshot
from .recipe_data import json_fingerprint, read_jsonl
from .recipe_prediction import _batch_size, _output_id
from .recipe_scoring import (
    PredictionScoringConfig,
    _indexed,
    _read_run,
    _score_pairs,
    _score_result,
)
from .recipe_validation import _check, _criteria, _publish as _publish_report
from .recipe_workflow import read_recipe_config
from .serialization import _load_json


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _name(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class EvaluationTrialFields:
    initial_state: str | None = None
    seed: str | None = None
    trial: str | None = None
    phase: str | None = None

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is not None:
                _name(value, f"trial {name} field")

    def extract(self, row):
        result = {}
        for name, key in asdict(self).items():
            if key is not None:
                if key not in row or row[key] is None:
                    raise ValueError(f"declared trial {name} is missing")
                result[name] = row[key]
        if "seed" in result and (type(result["seed"]) is not int or result["seed"] < 0):
            raise ValueError("trial seed must be a nonnegative integer")
        return _json(result)


@dataclass(frozen=True)
class EvaluationViewConfig:
    name: str
    scoring: PredictionScoringConfig | None = None
    requires: str = "reference"
    reference_field: str | None = None
    availability_field: str | None = None
    reference_role: str | None = None
    files: bool = False
    external_protocol: str | None = None
    criteria: Mapping = field(default_factory=dict)

    def __post_init__(self):
        _name(self.name, "evaluation view name")
        if self.requires not in {"reference", "output", "external"}:
            raise ValueError("view requires must be reference, output or external")
        if self.requires == "external":
            _name(self.external_protocol, "external protocol/revision")
            if self.scoring is not None or self.files:
                raise ValueError("external views import evidence, not local scorer code")
        elif (
            not isinstance(self.scoring, PredictionScoringConfig)
            or self.external_protocol is not None
        ):
            raise ValueError("local views require PredictionScoringConfig, not external protocol")
        for name in ("reference_field", "availability_field", "reference_role"):
            if getattr(self, name) is not None:
                _name(getattr(self, name), name)
                if self.requires != "reference":
                    raise ValueError("reference requirements belong to reference views")
        if type(self.files) is not bool:
            raise ValueError("view files must be a boolean")
        object.__setattr__(self, "criteria", _criteria(self.criteria))

    def to_dict(self):
        return {
            "name": self.name,
            "scoring": None if self.scoring is None else self.scoring.to_dict(),
            "requires": self.requires,
            "reference_field": self.reference_field,
            "availability_field": self.availability_field,
            "reference_role": self.reference_role,
            "files": self.files,
            "external_protocol": self.external_protocol,
            "criteria": dict(self.criteria),
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.get("scoring") is not None:
            scoring = dict(value["scoring"])
            if isinstance(scoring.get("scorer"), Mapping):
                scoring["scorer"] = PythonFactoryConfig(**scoring["scorer"])
            value["scoring"] = PredictionScoringConfig(**scoring)
        return cls(**value)


@dataclass(frozen=True)
class EvaluationSuiteConfig:
    views: tuple[EvaluationViewConfig, ...]
    aggregator: PythonFactoryConfig | str | None = None
    aggregation_id: str | None = None
    criteria: Mapping = field(default_factory=dict)
    trials: EvaluationTrialFields = field(default_factory=EvaluationTrialFields)

    def __post_init__(self):
        object.__setattr__(self, "views", tuple(self.views))
        if not self.views or any(not isinstance(view, EvaluationViewConfig) for view in self.views):
            raise ValueError("a suite requires EvaluationViewConfig views")
        if len({view.name for view in self.views}) != len(self.views):
            raise ValueError("suite view names must be distinct")
        if isinstance(self.aggregator, str):
            object.__setattr__(self, "aggregator", PythonFactoryConfig(self.aggregator))
        if self.aggregator is not None:
            if not isinstance(self.aggregator, PythonFactoryConfig):
                raise ValueError("aggregator must be an installed PythonFactoryConfig")
            _name(self.aggregation_id, "aggregation_id")
        elif self.aggregation_id is not None or self.criteria:
            raise ValueError("composite criteria/identity require an explicit aggregator")
        if not isinstance(self.trials, EvaluationTrialFields):
            raise ValueError("trials must use EvaluationTrialFields")
        object.__setattr__(self, "criteria", _criteria(self.criteria))

    def to_dict(self):
        return {
            "format": "arti.evaluation-suite-config",
            "views": [view.to_dict() for view in self.views],
            "aggregator": None if self.aggregator is None else self.aggregator.to_dict(),
            "aggregation_id": self.aggregation_id,
            "criteria": dict(self.criteria),
            "trials": asdict(self.trials),
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.pop("format", "arti.evaluation-suite-config") != "arti.evaluation-suite-config":
            raise ValueError("not an evaluation suite configuration")
        value["views"] = tuple(EvaluationViewConfig.from_dict(view) for view in value["views"])
        if isinstance(value.get("aggregator"), Mapping):
            value["aggregator"] = PythonFactoryConfig(**value["aggregator"])
        value["trials"] = EvaluationTrialFields(**value.get("trials", {}))
        return cls(**value)

    @classmethod
    def read(cls, path):
        return cls.from_dict(read_recipe_config(path))


def _references(path):
    if path is None:
        return {"rows": [], "source": None, "snapshot": None, "directory": None}
    snapshot = (
        DatasetSnapshot(path.directory)
        if isinstance(path, DatasetSnapshot)
        else DatasetSnapshot(path)
        if Path(path).is_dir()
        else None
    )
    rows = list(snapshot.records()) if snapshot is not None else list(read_jsonl(path))
    return {
        "rows": rows,
        "snapshot": snapshot,
        "directory": snapshot.directory if snapshot is not None else Path(path).resolve().parent,
        "source": {
            "kind": "dataset-snapshot" if snapshot is not None else "records",
            "sha256": snapshot.manifest()["sha256"]
            if snapshot is not None
            else json_fingerprint(rows),
            "records": len(rows),
            "path": str(snapshot.directory if snapshot is not None else Path(path).resolve()),
            **({"manifest": snapshot.manifest()} if snapshot is not None else {}),
        },
    }


def _population(directory, allow_partial):
    root, plan, inputs, predictions, source = _read_run(directory, allow_partial)
    execution = {}
    for index in range(math.ceil(len(predictions) / _batch_size(plan))):
        part = root / "parts" / f"{index:08d}"
        receipt = _load_json(part / "receipt.json")
        context = receipt.get("execution", {"phase": "unrecorded", "attempt": None})
        for row in read_jsonl(part / "predictions.jsonl"):
            identifier = row[_output_id(plan)]
            if "episodes" in plan:
                from .recipe_episode import EpisodeRunConfig, verify_episode_results

                config = EpisodeRunConfig.from_dict(plan["episodes"])
                verify_episode_results([row], [inputs[identifier]], config)
                execution[identifier] = _json(row[config.field]["execution"])
            else:
                execution[identifier] = _json(context)
    source = {
        **source,
        "kind": "prediction-run",
        "execution_sha256": json_fingerprint(execution),
        **({"files": plan["files"]} if "files" in plan else {}),
    }
    return root, plan, inputs, predictions, source, execution


def _binding(source):
    return {
        key: source[key]
        for key in (
            "plan_sha256",
            "model_manifest_sha256",
            "task_recipe_sha256",
            "input_sha256",
            "predictions_sha256",
            "execution_sha256",
        )
    }


def _checked_external(path, source, view):
    report = _load_json(Path(path))
    if (
        report.get("format") != "arti.external-evaluation-evidence"
        or report.get("format_version") != 1
        or report.get("sha256")
        != json_fingerprint({key: value for key, value in report.items() if key != "sha256"})
        or report.get("source") != _binding(source)
        or report.get("protocol") != view.external_protocol
        or report.get("view") != view.name
    ):
        raise ValueError("external evidence source, view or protocol differs")
    return report


def _target(view, identifier, indexed, references):
    target = indexed.get(identifier)
    if target is None:
        return None, "missing_reference", None
    if view.availability_field is not None:
        available = target.get(view.availability_field)
        if available is not None and type(available) is not bool:
            raise ValueError("reference availability must be a boolean or absent")
        if available is not True:
            return None, "reference_unavailable", None
    if view.reference_field is not None and target.get(view.reference_field) is None:
        return None, "missing_reference_field", None
    assets = None
    if view.reference_role is not None:
        snapshot = references["snapshot"]
        if snapshot is None:
            return None, "reference_assets_not_declared", None
        key = target[snapshot.manifest()["fields"]["id"]]
        bindings = snapshot.references(key, role=view.reference_role)
        if not bindings or not all(binding["available"] for binding in bindings):
            return None, "reference_assets_unavailable", None
        assets = {
            binding["asset_id"]: snapshot.manifest()["assets"][binding["asset_id"]]
            for binding in bindings
        }
    return target, None, assets


def _view_result(view, population, suite, references, external, identifiers):
    root, plan, inputs, predictions, source, execution = population
    result = {
        "config": view.to_dict(),
        "origin": "external" if view.requires == "external" else "local",
        "status": "unavailable",
        "metrics": {},
        "scores": [],
        "outcomes": [],
        "checks": [],
        "aggregate_available": False,
    }
    pairs, reasons, error = [], {}, None
    try:
        targets = (
            _indexed(
                references["rows"],
                view.scoring.target_id_field or plan["fields"]["id"],
                "references",
            )
            if view.requires == "reference"
            else {}
        )
        if view.requires == "reference":
            result["reference_population"] = {
                "records": len(targets),
                "unused_ids": sorted(targets.keys() - set(identifiers)),
            }
        evidence = None
        external_scores = {}
        if view.requires == "external" and view.name in external:
            evidence = _checked_external(external[view.name], source, view)
            external_scores = _indexed(evidence["scores"], "id", "external scores")
            result["external_evidence"] = {
                "sha256": evidence["sha256"],
                "protocol": evidence["protocol"],
                "producer": evidence["producer"],
            }
        for identifier in identifiers:
            target, reason, assets = None, None, None
            if view.requires == "reference":
                target, reason, assets = _target(view, identifier, targets, references)
            elif view.requires == "external" and identifier not in external_scores:
                reason = "external_evidence_unavailable"
            if reason is not None:
                reasons[identifier] = reason
                continue
            pairs.append(
                {
                    "id": identifier,
                    "input": inputs[identifier],
                    "prediction": predictions[identifier],
                    "target": target,
                    "execution": execution[identifier],
                    "trial": suite.trials.extract(inputs[identifier]),
                    **({"reference_assets": assets} if assets is not None else {}),
                }
            )
        if pairs:
            if view.requires == "external":
                selected = [external_scores[pair["id"]] for pair in pairs]
                aggregate_available = len(selected) == len(external_scores)
                metrics, scores = _score_result(
                    pairs,
                    {
                        "metrics": evidence["metrics"] if aggregate_available else {},
                        "scores": selected,
                    },
                )
                result["unselected_external_ids"] = sorted(
                    external_scores.keys() - {pair["id"] for pair in pairs}
                )
            else:
                context = (
                    {
                        "directory": root,
                        "reference_directory": references["directory"],
                        "files_field": plan.get("files", {}).get("field"),
                    }
                    if view.files
                    else {}
                )
                metrics, scores = _score_pairs(pairs, view.scoring, **context)
                aggregate_available = True
            result.update(
                status="scored",
                metrics=metrics,
                scores=scores,
                aggregate_available=aggregate_available,
            )
    except Exception as exception:
        error = {"type": type(exception).__name__, "message": str(exception)}
        result.update(status="failed", error=error)
    indexed_scores = {row["id"]: row["score"] for row in result["scores"]}
    for identifier in identifiers:
        status = (
            "scored"
            if identifier in indexed_scores
            else "unavailable"
            if identifier in reasons
            else "failed"
            if error
            else "unavailable"
        )
        result["outcomes"].append(
            {
                "id": identifier,
                "status": status,
                "score": indexed_scores.get(identifier),
                "reason": reasons.get(identifier) or ("scorer_error" if error else None),
                "execution": execution[identifier],
                "trial": suite.trials.extract(inputs[identifier]),
            }
        )
    count = len(indexed_scores)
    result["coverage"] = {
        "scored": count,
        "selected": len(identifiers),
        "committed": len(predictions),
        "total": source["total_records"],
        "uncommitted": source["uncommitted_records"],
        "selected_fraction": count / len(identifiers),
        "total_fraction": count / source["total_records"],
        "unavailable_ids": [
            row["id"] for row in result["outcomes"] if row["status"] == "unavailable"
        ],
        "failed_ids": [row["id"] for row in result["outcomes"] if row["status"] == "failed"],
    }
    result["checks"] = _check(result["metrics"], view.criteria)
    return result


def _evaluate(population, suite, references, external, identifiers):
    _, _, _, predictions, source, _ = population
    for identifier in identifiers:
        suite.trials.extract(population[2][identifier])
    views = {
        view.name: _view_result(view, population, suite, references, external, identifiers)
        for view in suite.views
    }
    result = {
        "ok": True,
        "format": "arti.evaluation-suite",
        "format_version": 1,
        "kind": "evaluation-suite",
        "suite": suite.to_dict(),
        "source": source,
        "references": references["source"],
        "views": views,
        "samples": len(identifiers),
        "selection": {
            "ids": identifiers,
            "excluded_ids": sorted(predictions.keys() - set(identifiers)),
        },
        "metrics": {},
        "checks": [],
        "aggregation_status": "not_requested",
    }
    if suite.aggregator is not None:
        try:
            aggregated = suite.aggregator.resolve()(deepcopy(views))
            if not isinstance(aggregated, Mapping) or not isinstance(
                aggregated.get("metrics"), Mapping
            ):
                raise ValueError("suite aggregator must return a metrics JSON object")
            result.update(
                metrics=_json(dict(aggregated["metrics"])),
                aggregation_status="completed",
                aggregation_metadata=_json(
                    {key: value for key, value in aggregated.items() if key != "metrics"}
                ),
            )
        except Exception as error:
            result.update(
                aggregation_status="failed",
                aggregation_error={"type": type(error).__name__, "message": str(error)},
            )
    result["checks"] = _check(result["metrics"], suite.criteria)
    checks = result["checks"] + [check for view in views.values() for check in view["checks"]]
    passed = all(check["passed"] for check in checks) if checks else None
    failed = result["aggregation_status"] == "failed" or any(
        view["status"] == "failed" for view in views.values()
    )
    complete = all(
        view["coverage"]["scored"] == source["total_records"] and view["aggregate_available"]
        for view in views.values()
    )
    status = (
        "failed"
        if failed
        else "rejected"
        if passed is False
        else "passed"
        if passed
        else "evaluated"
    )
    result.update(
        ok=not failed and passed is not False,
        criteria_passed=passed,
        complete_evidence=complete,
        status=status if complete or failed or passed is False else f"{status}_with_gaps",
    )
    return result


def _publish(result, output, populations, references, external_paths=()):
    if output is None:
        return {**result, "sha256": json_fingerprint(result)}
    target = Path(output).resolve()
    for population in populations:
        root = population[0]
        if target in {
            root / name for name in ("plan.json", "run.json", "input.jsonl", "predictions.jsonl")
        } or target.is_relative_to(root / "parts"):
            raise ValueError("suite report must not overwrite result evidence")
    source = references["source"]
    if source is not None:
        path = Path(source["path"])
        if target == path or (references["snapshot"] is not None and target.is_relative_to(path)):
            raise ValueError("suite report must not overwrite references")
    if target in {Path(path).resolve() for path in external_paths}:
        raise ValueError("suite report must not overwrite external evidence")
    return _publish_report(target, result)


def evaluate_prediction_suite(
    directory, suite, *, references=None, external=None, allow_partial=False, output=None
):
    """Evaluate original committed results; absent references do not become labels."""
    suite = suite if isinstance(suite, EvaluationSuiteConfig) else EvaluationSuiteConfig.read(suite)
    population, references, external = (
        _population(directory, allow_partial),
        _references(references),
        dict(external or {}),
    )
    if external.keys() - {view.name for view in suite.views if view.requires == "external"}:
        raise ValueError("external evidence names a non-external suite view")
    result = _evaluate(population, suite, references, external, list(population[3]))
    return _publish(result, output, [population], references, external.values())


def evaluation_source_identity(directory, *, allow_partial=False):
    """Content-qualified source fields required in externally produced evidence."""
    return _binding(_population(directory, allow_partial)[4])


def import_evaluation_evidence(
    directory, suite, view_name, evidence, *, allow_partial=False, output=None
):
    """Seal caller-produced evidence with matching declared source and protocol."""
    suite = suite if isinstance(suite, EvaluationSuiteConfig) else EvaluationSuiteConfig.read(suite)
    view = next(
        (view for view in suite.views if view.name == view_name and view.requires == "external"),
        None,
    )
    if view is None:
        raise ValueError("import requires an external suite view")
    population = _population(directory, allow_partial)
    data = _load_json(Path(evidence)) if isinstance(evidence, (str, Path)) else _json(dict(evidence))
    if (
        data.get("source") != _binding(population[4])
        or data.get("protocol") != view.external_protocol
    ):
        raise ValueError("external evaluator must declare matching source and protocol")
    if not data.get("producer"):
        raise ValueError("external evaluator producer identity is required")
    scores = _indexed(data.get("scores", ()), "id", "external scores")
    if scores.keys() - population[3].keys():
        raise ValueError("external scores name uncommitted or unknown IDs")
    pairs = [{"id": identifier} for identifier in scores]
    _score_result(pairs, data)
    result = {
        "ok": True,
        "format": "arti.external-evaluation-evidence",
        "format_version": 1,
        "kind": "external-evaluation-evidence",
        "view": view_name,
        "protocol": view.external_protocol,
        "source": _binding(population[4]),
        "producer": data["producer"],
        "metrics": data["metrics"],
        "scores": list(scores.values()),
        "evidence": data,
        "origin": "external; source attribution is evaluator-declared, not locally reproduced",
    }
    references = _references(None)
    return _publish(
        result,
        output,
        [population],
        references,
        [evidence] if isinstance(evidence, (str, Path)) else [],
    )


def compare_prediction_suites(
    candidate,
    baseline,
    suite,
    *,
    references=None,
    external=None,
    pairing="exact",
    allow_partial=False,
    output=None,
):
    """Pair original inputs, trial initial states/seeds and each view's population."""
    suite = suite if isinstance(suite, EvaluationSuiteConfig) else EvaluationSuiteConfig.read(suite)
    if pairing not in {"exact", "intersection"}:
        raise ValueError("suite pairing must be exact or intersection")
    populations = {
        name: _population(path, allow_partial)
        for name, path in (("candidate", candidate), ("baseline", baseline))
    }
    current, previous = set(populations["candidate"][3]), set(populations["baseline"][3])
    if pairing == "exact" and current != previous:
        raise ValueError("suite populations differ; intersection pairing must be explicit")
    identifiers = sorted(current & previous)
    if not identifiers:
        raise ValueError("suite comparison requires common committed IDs")
    for identifier in identifiers:
        if json_fingerprint(populations["candidate"][2][identifier]) != json_fingerprint(
            populations["baseline"][2][identifier]
        ):
            raise ValueError(f"paired trial/input differs for {identifier!r}")
        episode_trials = []
        for population in populations.values():
            plan = population[1]
            if "episodes" not in plan:
                episode_trials.append(None)
                continue
            episode = population[3][identifier][plan["episodes"]["field"]]
            episode_trials.append(
                {key: episode[key] for key in ("initial_state", "seed", "trial") if key in episode}
            )
        if json_fingerprint(episode_trials[0]) != json_fingerprint(episode_trials[1]):
            raise ValueError(f"paired committed episode state/seed/trial differs for {identifier!r}")
    references, external = _references(references), dict(external or {})
    if external.keys() - populations.keys():
        raise ValueError("comparison external evidence requires candidate/baseline arms")
    names = {view.name for view in suite.views if view.requires == "external"}
    if any(
        not isinstance(mapping, Mapping) or mapping.keys() - names for mapping in external.values()
    ):
        raise ValueError("comparison evidence names a non-external suite view")
    reports = {
        name: _evaluate(population, suite, references, external.get(name, {}), identifiers)
        for name, population in populations.items()
    }
    deltas = {}
    for view in suite.views:
        arms = [reports[name]["views"][view.name] for name in populations]
        comparable = all(arm["aggregate_available"] for arm in arms) and (
            {row["id"] for row in arms[0]["scores"]} == {row["id"] for row in arms[1]["scores"]}
        )
        numeric = arms[0]["metrics"].keys() & arms[1]["metrics"].keys() if comparable else ()
        deltas[view.name] = {
            "comparable": comparable,
            "delta": {
                key: arms[0]["metrics"][key] - arms[1]["metrics"][key]
                for key in numeric
                if all(type(arm["metrics"][key]) in {int, float} for arm in arms)
            },
        }
    composite_comparable = all(value["comparable"] for value in deltas.values())
    result = {
        "ok": all(report["ok"] for report in reports.values()),
        "format": "arti.evaluation-suite-comparison",
        "format_version": 1,
        "kind": "evaluation-suite-comparison",
        "pairing": pairing,
        "paired_samples": len(identifiers),
        "arms": reports,
        "views": deltas,
        "composite_comparable": composite_comparable,
        "delta": {
            key: reports["candidate"]["metrics"][key] - reports["baseline"]["metrics"][key]
            for key in reports["candidate"]["metrics"].keys()
            & reports["baseline"]["metrics"].keys()
            if composite_comparable
            and all(type(reports[name]["metrics"][key]) in {int, float} for name in reports)
        },
        "delta_semantics": "candidate-minus-baseline; direction/units remain task-owned",
    }
    return _publish(
        result,
        output,
        list(populations.values()),
        references,
        [path for mapping in external.values() for path in mapping.values()],
    )


__all__ = [
    "EvaluationTrialFields",
    "EvaluationViewConfig",
    "EvaluationSuiteConfig",
    "evaluate_prediction_suite",
    "compare_prediction_suites",
    "import_evaluation_evidence",
    "evaluation_source_identity",
]
