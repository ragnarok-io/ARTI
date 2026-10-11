"""Read-only project declarations over ARTI's existing task lifecycle.

Paths bind existing inputs, configuration files and receipts. This module does
not run a task, resolve a caller's Python reference, or infer task semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import re
from types import MappingProxyType

from .recipe_workflow import read_recipe_config


CONFIG_ROLES = (
    "recipe", "snapshot", "transform", "prediction", "files", "episodes",
    "suite", "runtime", "export",
)
ARTIFACT_ROLES = (
    "training_run", "model", "dataset", "cache", "preparation_report",
    "prediction_run", "evaluation", "deployment", "result_export", "runtime_receipt",
)
SCHEMA_KINDS = ("project", *CONFIG_ROLES)


def _paths(value, name, allowed=None):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must map names to paths")
    result = {}
    for key, path in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} names must be nonempty strings")
        if allowed is not None and key not in allowed:
            raise ValueError(f"unsupported {name} role: {key}")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{name}.{key} must be a nonempty path string")
        result[key] = path
    return MappingProxyType(result)


@dataclass(frozen=True)
class TaskProjectConfig:
    """Portable path bindings; application options stay in their native configs."""

    name: str
    configs: Mapping[str, str]
    resources: Mapping[str, str] = field(default_factory=dict)
    inputs: Mapping[str, str] = field(default_factory=dict)
    artifacts: Mapping[str, str] = field(default_factory=dict)
    format: str = "arti.task-project"
    format_version: int = 1
    _project_file: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.format != "arti.task-project" or type(self.format_version) is not int or self.format_version != 1:
            raise ValueError("unsupported task project format/version")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("project name must be a nonempty string")
        for section, roles in (("configs", CONFIG_ROLES), ("artifacts", ARTIFACT_ROLES), ("resources", None), ("inputs", None)):
            object.__setattr__(self, section, _paths(getattr(self, section), section, roles))
        if "recipe" not in self.configs:
            raise ValueError("configs.recipe is required")
        if self._project_file is not None:
            object.__setattr__(self, "_project_file", Path(self._project_file).resolve())

    @property
    def project_file(self):
        return self._project_file

    @property
    def project_root(self):
        if self.project_file is None:
            raise ValueError("read a saved project before resolving relative paths")
        return self.project_file.parent

    def resolve_path(self, value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def resolved_paths(self, section):
        if section not in {"configs", "resources", "inputs", "artifacts"}:
            raise ValueError("unknown project path section")
        return {name: self.resolve_path(path) for name, path in getattr(self, section).items()}

    def resolved_recipe(self):
        """Bind explicit project resources without editing task-owned options."""
        path = self.resolved_paths("configs")["recipe"]
        from .task_recipe import TaskRecipeConfig
        config = TaskRecipeConfig.from_dict(_config_value(path))
        resources = {**dict(config.resources), **{name: str(value) for name, value in self.resolved_paths("resources").items()}}
        run = config.run
        if run.output_dir is not None:
            run = replace(run, output_dir=_local(path, run.output_dir))
        return replace(config, resources=resources, run=run)

    def to_dict(self):
        return {
            "format": self.format, "format_version": self.format_version,
            "name": self.name,
            **{section: dict(getattr(self, section)) for section in ("configs", "resources", "inputs", "artifacts")},
        }

    @classmethod
    def from_dict(cls, payload):
        value = dict(payload)
        # Editor-only annotation is not part of the execution declaration.
        value.pop("$schema", None)
        if "_project_file" in value:
            raise ValueError("_project_file is not a project declaration field")
        return cls(**value)

    @classmethod
    def read(cls, path):
        return read_task_project(path)


def read_task_project(path):
    path = Path(path).resolve()
    return replace(TaskProjectConfig.from_dict(read_recipe_config(path)), _project_file=path)


def _local(config_file, value):
    target = Path(value)
    return target.resolve() if target.is_absolute() else (config_file.parent / target).resolve()


def _object(properties, required=(), **extra):
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False, **extra}


def _string(nullable=False):
    return {"type": ["string", "null"] if nullable else "string", "minLength": 1, "pattern": r"\S"}


def _integer(minimum=1, nullable=False):
    return {"type": ["integer", "null"] if nullable else "integer", "minimum": minimum}


def _number(minimum=0, nullable=False, positive=False):
    return {"type": ["number", "null"] if nullable else "number", "exclusiveMinimum" if positive else "minimum": minimum}


def _factory(nullable=False):
    reference = {**_string(), "pattern": r"^[^\s:]+:[^\s:]+$"}
    result = {"anyOf": [reference, _object({"reference": reference, "kwargs": {"type": "object"}}, ("reference",))]}
    if nullable:
        result["anyOf"].append({"type": "null"})
    return result


def _native_schema(kind):
    """Field schemas accompany native constructors; task JSON remains open."""
    obj = {"type": "object"}
    boolean = {"type": "boolean"}
    strings = {"type": "array", "items": _string()}
    portmap = {"type": "object", "propertyNames": _string(), "additionalProperties": {"enum": ["file", "directory"]}}
    criteria = {"type": "object", "propertyNames": _string(), "additionalProperties": _object({"minimum": {"type": "number"}, "maximum": {"type": "number"}}, minProperties=1)}
    if kind == "project":
        paths = {"type": "object", "propertyNames": _string(), "additionalProperties": _string()}
        return _object({
            "$schema": _string(), "format": {"const": "arti.task-project"}, "format_version": {"const": 1, "type": "integer"},
            "name": _string(), "configs": _object({key: _string() for key in CONFIG_ROLES}, ("recipe",)),
            "resources": paths, "inputs": paths, "artifacts": _object({key: _string() for key in ARTIFACT_ROLES}),
        }, ("format", "format_version", "name", "configs"))
    if kind == "recipe":
        validation = _object({"monitor": _string(), "mode": {"enum": ["min", "max"]}, "selection": {"enum": ["best", "last"]}, "min_delta": _number(), "patience": _integer(nullable=True), "start_step": _integer(0), "min_evaluations": _integer()})
        training = _object({"engine": {"enum": ["torch", "transformers", "accelerate"]}, "objective": {"enum": ["tensor_alignment", "model_loss"]}, "learning_rate": _number(positive=True), "steps": _integer(0), "gradient_accumulation_steps": _integer(), "mixed_precision": {"enum": ["no", "fp16", "bf16"]}, "corruption_probability": {"type": "number", "minimum": 0, "exclusiveMaximum": 1}, "device": {"enum": ["auto", "cpu", "cuda"]}, "distributed": boolean})
        run = _object({"output_dir": _string(True), "checkpoint_steps": _integer(), "checkpoint_seconds": _number(positive=True), "log_steps": _integer(), "max_seconds": _number(nullable=True, positive=True), "evaluation_steps": _integer(nullable=True), "metadata": obj, "validation": {"anyOf": [validation, {"type": "null"}]}})
        return _object({"format": {"const": "arti.task-recipe-config"}, "task": _string(), "factory": _factory(), "options": obj, "resources": obj, "training": training, "run": run}, ("task", "factory"))
    if kind == "snapshot":
        ref = _object({"asset_id": _string(), "role": _string(), "required": boolean}, ("asset_id",))
        return _object({"format": {"const": "arti.dataset-snapshot-config"}, "records": _string(), "assets": {"type": "object", "propertyNames": _string(), "additionalProperties": _string(True)}, "bindings": {"type": "object", "propertyNames": _string(), "additionalProperties": {"type": "array", "items": ref}}, "asset_metadata": {"type": "object", "propertyNames": _string(), "additionalProperties": obj}, "id_field": _string(), "split_field": _string(True), "group_field": _string(True), "source_revision": _string(True), "metadata": obj}, ("records",))
    if kind == "transform":
        return _object({"format": {"const": "arti.asset-transform-config"}, "factory": _factory(), "revision": _string(), "options": obj}, ("factory", "revision"))
    if kind == "files":
        return _object({"format": {"const": "arti.prediction-files-config"}, "field": _string(), "validator": _factory(True), "revision": _string(True)})
    if kind == "episodes":
        return _object({"format": {"const": "arti.episode-run-config"}, "adapter": _factory(), "policy_id": _string(), "environment_id": _string(), "evaluator_id": _string(), "initial_state_field": _string(), "seed_field": _string(), "recovery": {"enum": ["snapshot", "replay", "none"]}, "effects": {"enum": ["local", "external"]}, "policy_state_lifetime": {"enum": ["step", "trial"]}, "environment_state_lifetime": {"const": "trial"}, "native_model_state_lifetime": {"const": "read-only-policy-step"}, "max_steps": _integer(nullable=True), "max_calls": _integer(nullable=True), "field": _string(), "trial_field": _string(True), "independent_batch": boolean}, ("adapter", "policy_id", "environment_id", "evaluator_id", "initial_state_field", "seed_field"))
    if kind == "prediction":
        return _object({"format": {"const": "arti.prediction-run-config"}, "output_dir": _string(), "batch_size": _integer(nullable=True), "predict_kwargs": obj, "id_field": _string(True), "output_id_field": _string(True), "max_seconds": _number(nullable=True), "files": {"anyOf": [{"$ref": "#/$defs/files"}, {"type": "null"}]}, "episodes": {"anyOf": [{"$ref": "#/$defs/episodes"}, {"type": "null"}]}}, ("output_dir",))
    if kind == "suite":
        scoring = _object({"scorer": _factory(), "scorer_id": _string(), "target_id_field": _string(True)}, ("scorer", "scorer_id"))
        view = _object({"name": _string(), "scoring": {"anyOf": [scoring, {"type": "null"}]}, "requires": {"enum": ["reference", "output", "external"]}, "reference_field": _string(True), "availability_field": _string(True), "reference_role": _string(True), "files": boolean, "external_protocol": _string(True), "criteria": criteria}, ("name",))
        trials = _object({key: _string(True) for key in ("initial_state", "seed", "trial", "phase")})
        return _object({"format": {"const": "arti.evaluation-suite-config"}, "views": {"type": "array", "minItems": 1, "items": view}, "aggregator": _factory(True), "aggregation_id": _string(True), "criteria": criteria, "trials": trials}, ("views",))
    if kind == "runtime":
        backend = _object({"factory": _factory(), "revision": _string()}, ("factory", "revision"))
        return _object({"format": {"const": "arti.task-runtime-profile"}, "entrypoint": _factory(), "revision": _string(), "python": _string(), "requirements": strings, "inputs": portmap, "outputs": portmap, "options": obj, "environment": {"type": "object", "additionalProperties": {"type": "string"}}, "network": {"enum": ["inherit", "disabled"]}, "limits": {"type": "object", "additionalProperties": _number(positive=True)}, "require_enforcement": strings, "allowed_dependencies": {"anyOf": [strings, {"type": "null"}]}, "container": {"anyOf": [{"type": "object", "properties": {"image": _string()}, "required": ["image"]}, {"type": "null"}]}, "backend": {"anyOf": [backend, {"type": "null"}]}}, ("entrypoint", "revision"))
    if kind == "export":
        return _object({"format": {"const": "arti.result-export-config"}, "writer": _factory(), "revision": _string(), "options": obj, "validator": _factory(True), "validation_id": _string(True), "archive": {"enum": ["directory", "zip"]}, "instructions": {"type": "string"}, "unresolved_checks": strings, "examples": strings}, ("writer", "revision"))
    raise ValueError(f"unknown task workflow schema kind: {kind}")


def project_schema(kind="project"):
    """Return a fresh Draft 2020-12 schema without optional dependencies."""
    if kind not in SCHEMA_KINDS:
        raise ValueError(f"unknown task workflow schema kind: {kind}")
    schema = _native_schema(kind)
    _schema_rules(kind, schema)
    schema.update({"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": f"urn:arti:task-workflow:{kind}:1", "title": f"ARTI task workflow {kind} configuration"})
    if kind == "prediction":
        schema["$defs"] = {key: _native_schema(key) for key in ("files", "episodes")}
        for key, value in schema["$defs"].items():
            _schema_rules(key, value)
    return deepcopy(schema)


def _schema_rules(kind, schema):
    def present(name):
        return {"required": [name], "properties": {name: {"not": {"type": "null"}}}}
    def pair(left, right):
        return [
            {"if": present(left), "then": {"required": [right], "properties": {right: {"not": {"type": "null"}}}}},
            {"if": present(right), "then": {"required": [left], "properties": {left: {"not": {"type": "null"}}}}},
        ]
    rules = []
    if kind == "files":
        rules = pair("validator", "revision")
    elif kind == "export":
        rules = pair("validator", "validation_id")
    elif kind == "episodes":
        rules = [{"if": {"properties": {"effects": {"const": "external"}}, "required": ["effects"]}, "then": {"properties": {"recovery": {"enum": ["snapshot", "none"]}}}}]
    elif kind == "prediction":
        rules = [{"if": present("episodes"), "then": {"properties": {"files": {"type": "null"}, "predict_kwargs": {"type": "object", "maxProperties": 0}}}}]
    elif kind == "recipe":
        schema["properties"]["run"]["allOf"] = [{"if": present("validation"), "then": {"required": ["evaluation_steps"], "properties": {"evaluation_steps": _integer()}}}]
    elif kind == "suite":
        rules = pair("aggregator", "aggregation_id")
        rules.append({"if": {"not": present("aggregator")}, "then": {"properties": {"criteria": {"type": "object", "maxProperties": 0}}}})
        view = schema["properties"]["views"]["items"]
        view["allOf"] = [{"if": {"required": ["requires"], "properties": {"requires": {"const": "external"}}}, "then": {"required": ["external_protocol"], "properties": {"external_protocol": _string(), "scoring": {"type": "null"}, "files": {"const": False}}}, "else": {"required": ["scoring"], "properties": {"scoring": {"not": {"type": "null"}}, "external_protocol": {"type": "null"}}}}, {"if": {"required": ["requires"], "properties": {"requires": {"enum": ["output", "external"]}}}, "then": {"properties": {name: {"type": "null"} for name in ("reference_field", "availability_field", "reference_role")}}}]
    if rules:
        schema["allOf"] = rules


def project_schema_json(kind="project"):
    return json.dumps(project_schema(kind), indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_task_project_schema(path, kind="project"):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(project_schema_json(kind), encoding="utf-8")
    return target


def check_task_project_schema(path, kind="project"):
    target = Path(path)
    return target.is_file() and target.read_text(encoding="utf-8") == project_schema_json(kind)


def packaged_project_schema(kind="project"):
    """Read the schema included in the installed package."""
    if kind not in SCHEMA_KINDS:
        raise ValueError(f"unknown task workflow schema kind: {kind}")
    from importlib.resources import files
    return json.loads(files("arti").joinpath("schemas", f"task-{kind}.schema.json").read_text(encoding="utf-8"))


def _schema_errors(value, schema, location, root=None):
    """Validate the published structural subset; native constructors own rules.

    This deliberately has no jsonschema runtime dependency. The public schemas
    can also be consumed by ordinary Draft 2020-12 validators and editors.
    """
    root = schema if root is None else root
    if "$ref" in schema:
        target = root
        for name in schema["$ref"].removeprefix("#/").split("/"):
            target = target[name]
        return _schema_errors(value, target, location, root)
    if "anyOf" in schema:
        branches = [_schema_errors(value, item, location, root) for item in schema["anyOf"]]
        if any(not branch for branch in branches):
            return []
        # Prefer locations within a matching object/array to an unhelpful union error.
        return min(branches, key=lambda branch: (len(branch), -sum(len(item[0]) for item in branch)))
    errors = []
    def error(message):
        errors.append((location, message))
    if "not" in schema and not _schema_errors(value, schema["not"], location, root):
        error("does not satisfy the declared field combination")
    for rule in schema.get("allOf", ()):
        errors.extend(_schema_errors(value, rule, location, root))
    if "if" in schema:
        branch = "then" if not _schema_errors(value, schema["if"], location, root) else "else"
        if branch in schema:
            errors.extend(_schema_errors(value, schema[branch], location, root))
    kinds = schema.get("type")
    if kinds:
        kinds = [kinds] if isinstance(kinds, str) else kinds
        tests = {"object": isinstance(value, Mapping), "array": isinstance(value, list), "string": isinstance(value, str), "integer": type(value) is int, "number": type(value) in {int, float}, "boolean": type(value) is bool, "null": value is None}
        if not any(tests.get(kind, False) for kind in kinds):
            return [(location, f"must have type {' or '.join(kinds)}")]
    if "const" in schema and value != schema["const"]:
        error(f"must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        error(f"must be one of {schema['enum']!r}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            error("must be a nonempty string")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            error("must contain a non-whitespace character")
    if type(value) in {int, float}:
        import math
        if type(value) is float and not math.isfinite(value):
            error("must be finite")
        for keyword, predicate in (("minimum", lambda x, y: x < y), ("exclusiveMinimum", lambda x, y: x <= y), ("exclusiveMaximum", lambda x, y: x >= y)):
            if keyword in schema and predicate(value, schema[keyword]):
                error(f"must satisfy {keyword}={schema[keyword]}")
    if isinstance(value, Mapping):
        for name in schema.get("required", ()):
            if name not in value:
                errors.append((f"{location}.{name}", "is required"))
        if len(value) < schema.get("minProperties", 0):
            error("must have at least one property")
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            error(f"must have at most {schema['maxProperties']} properties")
        for name, child in value.items():
            childloc = f"{location}.{name}"
            if "propertyNames" in schema:
                errors.extend(_schema_errors(name, schema["propertyNames"], childloc, root))
            properties = schema.get("properties", {})
            if name in properties:
                errors.extend(_schema_errors(child, properties[name], childloc, root))
            elif schema.get("additionalProperties") is False:
                errors.append((childloc, "is not a supported field"))
            elif isinstance(schema.get("additionalProperties"), dict):
                errors.extend(_schema_errors(child, schema["additionalProperties"], childloc, root))
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            error("must contain at least one item")
        if "items" in schema:
            for index, child in enumerate(value):
                errors.extend(_schema_errors(child, schema["items"], f"{location}[{index}]", root))
    return errors


def _config_value(path):
    value = dict(read_recipe_config(path))
    value.pop("$schema", None)
    return value


def _factory_errors(role, value):
    from .python_factory import PythonFactoryConfig
    fields = {
        "recipe": ("factory",), "transform": ("factory",), "files": ("validator",),
        "episodes": ("adapter",), "runtime": ("entrypoint",), "suite": ("aggregator",),
        "export": ("writer", "validator"),
    }.get(role, ())
    candidates = [(f"configs.{role}.{name}", value.get(name)) for name in fields]
    errors = []
    if role == "runtime" and value.get("backend") is not None:
        candidates.append(("configs.runtime.backend.factory", value["backend"].get("factory")))
    if role == "suite":
        for index, view in enumerate(value.get("views", ())):
            if view.get("scoring") is not None:
                candidates.append((f"configs.suite.views[{index}].scoring.scorer", view["scoring"].get("scorer")))
    if role == "prediction":
        for nested in ("files", "episodes"):
            if value.get(nested) is not None:
                for location, error in _factory_errors(nested, value[nested]):
                    errors.append((location.replace(f"configs.{nested}", f"configs.prediction.{nested}"), error))
    for location, declaration in candidates:
        if declaration is None:
            continue
        try:
            PythonFactoryConfig(**declaration) if isinstance(declaration, Mapping) else PythonFactoryConfig(declaration)
        except (TypeError, ValueError) as error:
            errors.append((location + (".reference" if isinstance(declaration, Mapping) else ""), str(error)))
    return errors


def _files(value):
    from .recipe_files import PredictionFilesConfig
    value = dict(value)
    value.pop("format", None)
    return PredictionFilesConfig.from_dict(value)


def _native_config(role, path, value, configs):
    """Construct existing declarations, never their referenced Python code."""
    if role == "recipe":
        from .task_recipe import TaskRecipeConfig
        return TaskRecipeConfig.from_dict(value)
    if role == "snapshot":
        from .recipe_assets import DatasetSnapshotConfig
        value = dict(value)
        value.pop("format", None)
        value["records"] = _local(path, value["records"])
        value["assets"] = {name: None if source is None else _local(path, source) for name, source in value.get("assets", {}).items()}
        return DatasetSnapshotConfig(**value)
    if role == "transform":
        from .recipe_assets import AssetTransformConfig
        from .python_factory import PythonFactoryConfig
        value = dict(value)
        value.pop("format", None)
        if isinstance(value.get("factory"), Mapping):
            value["factory"] = PythonFactoryConfig(**value["factory"])
        return AssetTransformConfig(**value)
    if role == "files":
        return _files(value)
    if role == "episodes":
        from .recipe_episode import EpisodeRunConfig
        return EpisodeRunConfig.from_dict(value)
    if role == "prediction":
        from .recipe_prediction import PredictionRunConfig
        from .recipe_episode import EpisodeRunConfig
        value = dict(value)
        value.pop("format", None)
        value["output_dir"] = _local(path, value["output_dir"])
        if value.get("files") is not None:
            value["files"] = _files(value["files"])
        else:
            value["files"] = configs.get("files")
        if value.get("episodes") is not None:
            value["episodes"] = EpisodeRunConfig.from_dict(value["episodes"])
        else:
            value["episodes"] = configs.get("episodes")
        return PredictionRunConfig(**value)
    if role == "suite":
        from .recipe_suite import EvaluationSuiteConfig
        return EvaluationSuiteConfig.from_dict(value)
    if role == "runtime":
        from .recipe_runtime import TaskRuntimeProfile
        return TaskRuntimeProfile.from_dict(value)
    if role == "export":
        from .recipe_export import ResultExportConfig
        return ResultExportConfig.from_dict(value)
    raise ValueError(f"unsupported config role: {role}")


def check_task_project(path):
    """Check declarations, named paths and connections without executing code.

    Existing output absence means not_created. Structural compatibility does
    not establish model quality, artifact integrity or host enforcement.
    """
    path = Path(path).resolve()
    report = {
        "format": "arti.task-project-check", "format_version": 1, "schema_version": 1,
        "ok": False, "project": {"file": str(path), "root": str(path.parent), "name": None},
        "configs": {}, "bindings": {"resources": {}, "inputs": {}}, "artifacts": {}, "issues": [],
        "execution": {"user_code_imported": False, "model_constructed": False, "host_probed": False},
        "scope": "static declarations and explicit path availability; no runtime qualification",
        "path_resolution": {"project_bindings": "project-file-relative", "native_config_paths": "config-file-relative", "application_options": "unchanged"},
    }
    def issue(code, location, message, severity="error"):
        report["issues"].append({"severity": severity, "code": code, "field": location, "message": str(message)})
    try:
        raw = read_recipe_config(path)
    except (OSError, ValueError, TypeError) as error:
        issue("project_unreadable", "project", error)
        return report
    errors = _schema_errors(raw, project_schema(), "project")
    for location, message in errors:
        issue("invalid_field", location, message)
    if errors:
        return report
    project = replace(TaskProjectConfig.from_dict(raw), _project_file=path)
    report["project"]["name"] = project.name
    for section in ("resources", "inputs", "artifacts"):
        for name, target in project.resolved_paths(section).items():
            present = target.exists()
            item = {"path": str(target), "status": "present" if present else "not_created" if section == "artifacts" else "missing", "kind": "directory" if target.is_dir() else "file" if target.is_file() else None}
            if section == "artifacts":
                report[section][name] = item
            else:
                report["bindings"][section][name] = item
                if not present:
                    issue("missing_binding", f"project.{section}.{name}", f"required path is unavailable: {target}")
    values, native, config_paths = {}, {}, project.resolved_paths("configs")
    # Separate file/episode declarations must be read before the prediction config.
    for role in (key for key in CONFIG_ROLES if key != "prediction"):
        if role not in config_paths:
            continue
        target = config_paths[role]
        report["configs"][role] = {"path": str(target), "status": "invalid"}
        try:
            value = _config_value(target)
            errors = _schema_errors(value, project_schema(role), f"configs.{role}")
            for location, message in errors:
                issue("invalid_field", location, message)
            if errors:
                continue
            errors = _factory_errors(role, value)
            for location, message in errors:
                issue("invalid_factory", location, message)
            if errors:
                continue
            values[role] = value
            native[role] = _native_config(role, target, value, native)
            report["configs"][role]["status"] = "valid"
        except (OSError, ValueError, TypeError, KeyError) as error:
            issue("invalid_config", f"configs.{role}", error)
    if "prediction" in config_paths:
        target = config_paths["prediction"]
        report["configs"]["prediction"] = {"path": str(target), "status": "invalid"}
        try:
            value = _config_value(target)
            errors = _schema_errors(value, project_schema("prediction"), "configs.prediction")
            for location, message in errors:
                issue("invalid_field", location, message)
            if not errors:
                errors = _factory_errors("prediction", value)
                for location, message in errors:
                    issue("invalid_factory", location, message)
            if not errors:
                values["prediction"] = value
                native["prediction"] = _native_config("prediction", target, value, native)
                report["configs"]["prediction"]["status"] = "valid"
        except (OSError, ValueError, TypeError, KeyError) as error:
            issue("invalid_config", "configs.prediction", error)
    for role in ("files", "episodes"):
        inline = values.get("prediction", {}).get(role)
        if inline is not None and role in native and "prediction" in native:
            attached = getattr(native["prediction"], role)
            if attached.to_dict() != native[role].to_dict():
                issue("conflicting_config", f"configs.prediction.{role}", f"inline declaration differs from configs.{role}")
    snapshot = native.get("snapshot")
    if snapshot is not None:
        if not snapshot.records.is_file():
            issue("missing_snapshot_records", "configs.snapshot.records", f"records are unavailable: {snapshot.records}")
        for identifier, references in snapshot.bindings.items():
            for reference in references:
                if reference.asset_id not in snapshot.assets:
                    issue("unknown_asset", f"configs.snapshot.bindings.{identifier}", f"asset is undeclared: {reference.asset_id}")
                elif reference.required and snapshot.assets[reference.asset_id] is None:
                    issue("missing_required_asset", f"configs.snapshot.assets.{reference.asset_id}", "required asset has no path")
        for name, target in snapshot.assets.items():
            if target is not None and not target.is_file():
                issue("missing_snapshot_asset", f"configs.snapshot.assets.{name}", f"asset is unavailable: {target}; declare null for an unavailable optional asset")
    prediction = native.get("prediction")
    artifacts = project.resolved_paths("artifacts")
    if prediction is not None and "prediction_run" in artifacts and prediction.output_dir != artifacts["prediction_run"]:
        issue("conflicting_path", "configs.prediction.output_dir", "prediction output_dir differs from artifacts.prediction_run")
    recipe = native.get("recipe")
    if recipe is not None and recipe.run.output_dir is not None and "training_run" in artifacts and _local(config_paths["recipe"], recipe.run.output_dir) != artifacts["training_run"]:
        issue("selected_run", "project.artifacts.training_run", "selected run differs from the initial recipe output; recovery may select another run", "info")
    runtime = native.get("runtime")
    if runtime is not None:
        bound_names = {}
        for section in ("inputs", "resources", "artifacts"):
            for name, target in project.resolved_paths(section).items():
                if name not in runtime.inputs:
                    continue
                if name in bound_names and target != bound_names[name]:
                    issue("conflicting_binding", f"project.{section}.{name}", "a named runtime input has differing project path bindings")
                bound_names[name] = target
        bound = {**project.resolved_paths("inputs"), **project.resolved_paths("resources"), **artifacts}
        for name, kind in runtime.inputs.items():
            if name not in bound:
                issue("unbound_runtime_input", f"configs.runtime.inputs.{name}", "named runtime input needs an explicit project input, resource or artifact binding")
            elif bound[name].exists() and ((kind == "file" and not bound[name].is_file()) or (kind == "directory" and not bound[name].is_dir())):
                issue("runtime_input_kind", f"configs.runtime.inputs.{name}", f"named input requires a {kind}")
    report["ok"] = not any(item["severity"] == "error" for item in report["issues"])
    return report


__all__ = [
    "TaskProjectConfig", "read_task_project", "check_task_project",
    "project_schema", "project_schema_json", "write_task_project_schema", "check_task_project_schema",
    "packaged_project_schema", "CONFIG_ROLES", "ARTIFACT_ROLES", "SCHEMA_KINDS",
]
