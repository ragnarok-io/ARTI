# A Complete Task From An Installed Package

This guide runs an application-owned file task using the installed ARTI package.
You do not need an ARTI source checkout or its `examples` directory. The generated
project owns the model, objective, preprocessing, dataset format, scorer, runtime
entrypoint and result writer. ARTI's existing training runs, prediction journal,
evaluation and delivery services own the lifecycle.

The tutorial uses small generated vectors on CPU. It downloads no dataset or
model, starts no provider, installs no dependency and submits no results. Its
acceptance demonstrates recovery and delivery, not task quality or hardware
qualification. Interactive episodes can use the same project declaration; the
current scaffold deliberately demonstrates a complete file task.

## Start With The Installed Wheel

Use a Python 3.10 or later environment with PyTorch and the wheel's other declared
dependencies already provisioned. Select the reviewed `arti-fit` wheel that
includes the task project commands. An offline installation of that local
artifact is:

```sh
python -m pip install --no-index --no-deps /path/to/arti_fit-3.3.0-py3-none-any.whl
python -c "import arti, torch; print(arti.__file__); print(torch.__version__)"
python -m arti doctor --allow-cpu-torch
```

On PowerShell, use the actual Windows wheel path, for example
`C:\artifacts\arti_fit-3.3.0-py3-none-any.whl`. `--no-deps --no-index` requires an
already compatible environment; it does not fetch or repair dependencies. The
printed `arti.__file__` should point to the selected installed package. Run the
remaining commands outside any ARTI checkout so a source tree cannot shadow it.

## Generate And Check Your Project

The following commands work in PowerShell and a POSIX shell:

```sh
python -m arti task init my-task --name "my task" --template files
cd my-task
python -m arti task check project.json
python -m arti task status project.json
python -m arti task explain project.json
```

Generation requires a fresh destination and refuses to overwrite an existing
project. It copies package-resource templates and creates an immutable local
snapshot of the generated records/assets. It does not execute your task factory
or train a model. The project contains:

| Path | Ownership and purpose |
| --- | --- |
| `project.json` | Named paths to existing lifecycle configurations, inputs, resources and selected artifacts |
| `configs/` | Native recipe, snapshot, transform, prediction, files, suite, runtime and export configurations |
| `code/task_app.py` | Your model, binary codec, native tensor collation/loss, scorers, runtime callback and result writer |
| `workflow.py` | Explicit application commands that call existing ARTI APIs |
| `TUTORIAL.md` | This complete installed-package tutorial, copied from package resources |
| `data/source/` | Generated original records and binary vectors |
| `data/snapshot/` | Immutable content-addressed snapshot; this remains usable without the original source files |
| `data/cache/` | Persistent derived assets keyed by input bytes and transform identity |
| `runs/`, `artifacts/` | Native run evidence, inference package, prediction parts, evaluation and export |

`task check` checks declarations, path availability and cross-configuration
compatibility without importing application callbacks. Missing future artifacts
are expected before their stage has run. A successful check does not establish
that an application callback executes correctly or that a host enforces your
limits. Status/explain also inspect existing evidence without running the task.

For editor validation, export the installed schemas:

```sh
python -m arti schema task-workflow generate --kind project --output project.schema.json
python -m arti schema task-workflow generate --kind recipe --output recipe.schema.json
```

Associate those files with `project.json` and `configs/recipe.json` in your
editor's JSON settings. Other supported kinds are `snapshot`, `transform`,
`prediction`, `files`, `episodes`, `suite`, `runtime` and `export`. Configurations
remain the existing lifecycle types; schema validation introduces no new trainer
or scheduling engine.

## Prepare Once And Observe Actual Reuse

```sh
python workflow.py prepare
python workflow.py prepare
```

The first report shows `produced: 6`; the second shows `produced: 0` and
`reused: 6`. Twelve records refer to six unique generated vectors. The transform
is application-owned and its factory, revision and options identify the cache.
Reuse preserves the committed cache files. Changing decoding/preprocessing means
changing its declaration and revision, rather than reusing an old identity.

## Train, Interrupt And Recover

```sh
python workflow.py train --stop-after 2
python -m arti task status project.json
python -m arti task explain project.json
python workflow.py recover
```

`--stop-after 2` deliberately interrupts after the second optimizer checkpoint.
The interrupted native run remains in `runs/training`; recovery creates
`runs/recovered` and performs the remaining two optimizer updates. The current
task owns its mean-square-error objective, batches, preprocessing buffers and
trainable ARTI/native parameters. Recovery restores the existing checkpoint,
optimizer, RNG and progress instead of starting a second training path.

The driver updates only the project's explicit selected `training_run` and
`model` paths. The source run, its checkpoint and its receipts remain available.
The native validation policy selects the model; last and selected checkpoints
are different concepts. No task-quality threshold is inferred from completing
this small tutorial.

For uninterrupted training, run `python workflow.py train` in a newly generated
project. Existing native runs refuse an unrelated restart into the same run
directory. `recover` delegates to the shared recovery planner, including its
already-completed behavior.

## Commit File Outputs And Resume Only The Remainder

Package the selected model/resources/code before package-qualified prediction:

```sh
python workflow.py package
python workflow.py predict --stop-after 1
python -m arti task status project.json
python workflow.py predict --resume
python workflow.py predict --resume
```

Packaging serializes the selected inference closure. It executes no inference.
The first prediction command commits one two-record part, including each JSON
result, named binary file and receipt, then deliberately stops. The first retry
reports two reused and two newly predicted records. The completed retry reports
zero newly predicted records and preserves existing parts/files.

The source input IDs remain `record-8` through `record-11`. `code/task_app.py`
produces a native scalar and staged binary output; its validator checks that the
file value equals the structured prediction. The current prediction journal
publishes the whole closure atomically. A filename without a committed receipt
is not a completed result.

The generated driver always predicts through the selected deployment so the
result identity includes its exact resources/code/package. An existing
model-only plan cannot silently become evidence for a different package on
resume or export.

## Evaluate The Original Population

```sh
python workflow.py evaluate
python -m arti task explain project.json
```

The suite reads committed outputs and original references. Its reference view
keeps per-record error; its file view keeps per-record file validity. Each view
covers all four original requested IDs, including initial and recovered parts.
The scorer does not load a model or repeat prediction. Reference availability,
coverage, execution completion and application acceptance criteria remain
separate. The template declares no model-quality acceptance threshold.

Add your own views, aggregation and criteria in `configs/suite.json`. Keep an
unavailable reference view unavailable; do not invent labels or replace missing
coverage with a favorable aggregate.

## Run The Portable Package In A Cold Process

```sh
python workflow.py runtime
python workflow.py runtime --resume
```

The runtime profile selects the trusted application callback already included
in `artifacts/deployment`. Its first run starts a fresh local Python worker with
named inputs and outputs, generates its own package-qualified prediction journal
in `artifacts/cold-prediction` and validates those outputs. The second command
explicitly resumes that journal and reports zero new predictions. Runtime
receipts preserve which evidence came from the current or a prior execution.

The worker requires ARTI and the declared dependencies to be already installed.
It does not install wheels or fetch packages. `network: disabled` sets offline
hints; it does not enforce socket isolation. The built-in local worker's
qualified enforcement is its direct-worker wait deadline when declared. Memory,
network and process-tree isolation require a separately selected and tested host
implementation; required unsupported enforcement returns `not_run`. A runtime
receipt does not convert those limits into qualified hardware/container claims.

## Export Existing Results

```sh
python workflow.py export
python -m arti task status project.json
python -c "import arti; print(arti.inspect_result_export('artifacts/results.zip'))"
```

The application writer formats already committed records. The ZIP preserves
their exact expected-ID/file closure, original structured/binary outputs,
selected deployment, provenance, evaluation/runtime evidence, instructions and
unresolved checks. It also attaches original requests/references, suite config,
and both prediction journals. Export does not regenerate results, train, run
tools or submit the archive anywhere. Inspection verifies the closure without
importing writer or model code.

## Move To A Fresh Offline Consumer

Copy `artifacts/results.zip` to an unrelated directory on a compatible host with
the same selected installed ARTI candidate and dependencies. The following
consumer uses only the archive and the installed package; it can run after the
original task directory and original application/data sources are unavailable.
Create `consume.py` next to that archive:

```python
import json
from pathlib import Path
import zipfile

import arti

root = Path("extracted").resolve()
with zipfile.ZipFile("results.zip") as archive:
    archive.extractall(root)
checked = arti.inspect_result_export(root)
manifest = json.loads((root / ".arti-export/manifest.json").read_text())
attached = manifest["attachments"]

def attachment(name):
    return root / attached[name]["path"]

report = arti.run_task_runtime(
    root / ".arti-export/deployment",
    inputs={
        "requests": attachment("inputs"),
        "references": attachment("references"),
        "suite": attachment("suite"),
    },
    outputs={
        "results": attachment("cold_journal"),
        "evaluation": Path("consumer-evaluation.json"),
    },
    options={"resume": True},
    output="consumer-runtime.json",
)
assert report["status"] == "completed", report
assert report["worker"]["result"]["prediction"]["newly_predicted_records"] == 0
print(checked["sha256"], report["status"])
```

Run `python consume.py`. The archive's completed cold journal is adopted after
relocation; the runtime rescores it without replaying completed predictions. To
perform a distinct fresh inference run with the archived selected model, use a
fresh result directory and omit `options={"resume": True}`. That new inference
is a new runtime operation, separate from formatting or inspecting the export.

## Adapt The Application

Start with `code/task_app.py` and native configs. Replace the generated source
records/assets, binary decoder, model, objective, preprocessing, collation,
prediction formats and scorers with your own application declarations. Update
the snapshot, transform and callback revisions when their behavior changes.
Keep configuration paths relative to the project/config file as declared, and
select explicit code/resources for packaging. Do not package your entire private
working directory or rely on repository `examples` being installed.

Use `task check` before execution and `task status`/`task explain` after each
stage. These commands report existing evidence and unresolved work; they never
repair resources, rerun callbacks, repeat ambiguous external actions or imply a
public release. See [Custom Task Workflows](custom-task-workflows.md),
[Task Recovery](recipe-recovery.md), [Task Evaluation Suites](evaluation-suites.md),
[Runtime Profiles](task-runtime.md) and [Result Export](result-export.md) for the
underlying interfaces and their supported boundaries.
