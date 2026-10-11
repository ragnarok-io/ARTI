# Task Project Configuration

A task project declares where an application's existing lifecycle configuration,
inputs, resources and output receipts live. It does not schedule or execute them.
The application still owns its model, objective, data loading, collation,
preprocessing, prediction and evaluation callbacks.

```json
{
  "format": "arti.task-project",
  "format_version": 1,
  "name": "my-task",
  "configs": {
    "recipe": "configs/recipe.json",
    "snapshot": "configs/snapshot.json",
    "prediction": "configs/prediction.json",
    "files": "configs/files.json",
    "suite": "configs/suite.json",
    "runtime": "configs/runtime.json",
    "export": "configs/export.json"
  },
  "resources": {"source": "data/input.bin"},
  "inputs": {"requests": "data/requests.jsonl"},
  "artifacts": {
    "training_run": "runs/train",
    "model": "artifacts/model",
    "dataset": "artifacts/dataset",
    "cache": "artifacts/cache",
    "preparation_report": "reports/preparation.json",
    "prediction_run": "runs/prediction",
    "evaluation": "reports/evaluation.json",
    "deployment": "artifacts/deployment",
    "result_export": "artifacts/results.zip",
    "runtime_receipt": "reports/runtime.json"
  }
}
```

`configs.recipe` is required and references a generic `TaskRecipeConfig`.
Other supported roles are `snapshot`, `transform`, `prediction`, `files`,
`episodes`, `suite`, `runtime` and `export`. Each uses its existing lifecycle
configuration class. Project and config files can be JSON or TOML.
Application-defined input and resource names remain open; config and artifact
roles are explicit so a typo does not silently become an unused lifecycle stage.

Paths in the project declaration are relative to the project file. Native path
fields inside a referenced config are relative to that config file. For example,
`configs/prediction.json` uses `"output_dir": "../runs/prediction"` to select the
project's `runs/prediction` journal. `snapshot.records`, snapshot asset paths and
`recipe.run.output_dir` follow the same config-file rule. Absolute paths are
accepted but keep their original location when the project moves.

The resolver does not change the working directory or rewrite strings inside
application-owned `options`. `resolved_recipe()` binds only explicitly named
project resources into `TaskRecipeConfig.resources`; an explicit project binding
overrides the recipe resource of the same name. Other native resource values and
all task options remain unchanged.

```python
from arti.task_project import check_task_project, read_task_project

project = read_task_project("project.json")
paths = project.resolved_paths("artifacts")
recipe_config = project.resolved_recipe()
report = check_task_project("project.json")
assert report["ok"], report["issues"]
```

## Static Check Scope

The checker validates field types, supported versions, installed-reference
syntax, shared native config rules and explicit path availability. It also checks
connections between declarations: an episode cannot share a prediction run with
a staged file producer; duplicate inline and referenced declarations must agree;
snapshot references must name declared assets; runtime input ports need explicit
bindings and existing bindings must have the declared file/directory kind.

Existing prediction configuration and `artifacts.prediction_run` must identify
the same journal. `artifacts.training_run` may intentionally select a recovered
run with a different path from the initial recipe output; that is an informational
selection receipt, not an error.

Input and resource paths must already exist. Future output artifacts are reported
as `not_created`, without failing a fresh project. A missing optional snapshot
asset is declared as `null` with `required: false`, rather than a nonexistent
path. The checker does not create artifacts, decode asset contents, verify
record-level dataset semantics, import caller factories, instantiate a model,
probe the host, run a backend or install dependencies. `ok: true` means the
declarations passed static checks; it does not establish runtime, hardware or
task-quality qualification.

The report uses `arti.task-project-check` version 1, separate config/binding/
artifact statuses and an `issues` array. Each issue has `severity`, `code`,
`field` and `message`. `execution` records that user code import, model
construction and host probing did not occur. `path_resolution` states the two
path origins and that application options were left unchanged.

## JSON Schemas

The schemas use JSON Schema Draft 2020-12. They declare concrete lifecycle
fields, types, enums and supported cross-field combinations while keeping
application JSON options and caller-defined metric names open. Native
configuration constructors remain the authority for semantic rules such as
Python reference syntax, requirement parsing and distinct evaluation view names.
The checker needs no optional schema-validator dependency.

The installed package includes `arti/schemas/task-{kind}.schema.json`. Read them
through `packaged_project_schema(kind)` or generate the same schema through
`project_schema(kind)`. Schema generation and freshness checks are deterministic.

```python
from arti.task_project import (
    check_task_project_schema,
    packaged_project_schema,
    write_task_project_schema,
)

schema = packaged_project_schema("project")
write_task_project_schema("task-project.schema.json", "project")
assert check_task_project_schema("task-project.schema.json", "project")
```

Available kinds and reference files:

- [Project](task-project.schema.json)
- [Task recipe](task-recipe.schema.json)
- [Dataset snapshot](task-snapshot.schema.json)
- [Asset transform](task-transform.schema.json)
- [Prediction run](task-prediction.schema.json)
- [Named prediction files](task-files.schema.json)
- [Interactive episodes](task-episodes.schema.json)
- [Evaluation suite](task-suite.schema.json)
- [Runtime profile](task-runtime.schema.json)
- [Result export](task-export.schema.json)

Prediction and file configs may include the optional format markers
`arti.prediction-run-config` and `arti.prediction-files-config`. The project
checker removes those markers before constructing the existing declaration
classes. It supports both inline prediction `files`/`episodes` and separate
`configs.files`/`configs.episodes`; declaring different values in both places is
an error. Format markers and config references do not load caller code.
