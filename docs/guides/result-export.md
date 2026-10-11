# Export Committed Task Results

An application supplies its own result layout and writer. The export service
reads the existing prediction journal, verifies every committed result and named
file, and publishes a fresh portable directory or ZIP archive. It never loads a
model, regenerates predictions or submits anything externally.

```python
layout = arti.ResultExportConfig(
    writer="my_application.layout:write_results",
    revision="my-layout-revision",
    options=my_layout_options,
    validator="my_application.layout:validate_results",
    validation_id="my-layout-validator-revision",
    archive="zip",
    instructions=my_execution_instructions,
    unresolved_checks=my_unverified_checks,
    examples=my_selected_example_ids,
)
receipt = arti.export_prediction_results(
    prediction_directory, layout, "outputs/handoff.zip",
    expected_ids=my_original_ids,
    evidence={"evaluation": suite_report, "runtime": runtime_receipt},
    attachments=my_explicit_manifests_and_notes,
    deployment=portable_task_package,
)
```

The expected IDs must match the complete original input population exactly.
An unfinished journal, missing output, duplicate ID or silently smaller result
population fails before publication. Omitting `expected_ids` uses the original
input snapshot. The caller chooses examples by their original logical IDs;
the service does not infer success or failure labels.

## Application Writer And Validator

The installed writer receives `ResultExportContext` and private copies of the
original input, native prediction and execution evidence for each ID:

```python
def write_results(context, items):
    # Write application-owned encodings beneath context.directory.
    return [
        {"id": item["id"], "files": my_named_relative_files(item)}
        for item in items
    ]

def validate_results(results, *, directory):
    # Inspect the declared layout using the application's schema/codec.
    return {"ok": True, "records": len(results)}
```

`context.directory` is a private staging area. `context.source_directory` is
the verified journal, and `context.options` contains the declared JSON options.
Every writer row declares a named file mapping. A value may be a relative path
or `{"file": relative_path, "metadata": application_metadata}`. The service
streams each unique file, computes its byte count and digest, and retains
aliases when several IDs share a table. Layout paths cannot overwrite the
reserved `.arti-export` provenance directory. No filename, encoding, model
schema or numerical layout is built into the library.

A supplied validator has a separate revision and must return `ok=True`.
The service then verifies file content again. Missing files, writer/validator
errors, copy failure or publication failure leave the source journal intact
and publish no partial handoff. A destination must be fresh and outside the
source journal and directory attachments. Changing layout/revision creates a
new export from the same committed native results.

## Exact Closure And Provenance

The archive contains the formatted declarations, original native outputs and
binary file closure, original input snapshot, prediction plan, per-ID execution
evidence, caller instructions and a checksummed inventory. Selected examples
retain their input/output/execution evidence. Immutable file bytes are copied
in chunks; they are not expanded into JSON. Structured population metadata is
materialized using the existing offline evaluation service, so this is not a
constant-memory or distributed export API.

Known evaluation, deployment-validation, external-evaluation and runtime
receipts must have valid content identities and describe the selected source.
Suite evidence binds its actual result/execution/population conditions; imported
external evidence retains its declared external origin. A runtime receipt binds
the packaged deployment and describes application execution, which is distinct
from a particular task outcome. Receipt bytes are pinned and checked while
copying. Arbitrary caller documents belong in `attachments`, rather than being
relabeled as verified evaluation evidence.

A supplied deployment must match the committed model, task and package identity.
Packaging reuses the current portable task closure, including selected code,
requirements, runtime/environment declarations, resources and selection lineage.
Data/transform manifests can travel in that resource closure or as explicitly
selected attachments. No trainer tree or unselected project directory is
automatically included. Without a deployment, the export records source
identities and an unresolved model/runtime-package check.

Instructions, unknown checks and application code authority remain explicit.
Bundled wheels are offline inputs and are never installed by export. Container,
provider, network/resource enforcement and task quality remain unresolved unless
their separately attributed receipts actually qualify them. Formatting success
is not runtime success or permission to send the archive elsewhere.

## Inspect And Move

```python
inspection = arti.inspect_result_export("outputs/handoff.zip")
```

Inspection verifies the whole file/directory inventory, every byte count/digest,
the original ID population and named formatted/native file declarations. It
imports neither writer nor model code. Move the complete directory or ZIP and
inspect it after deleting source scratch files. A ZIP must be extracted before
using its enclosed deployment through existing task runtime/predict APIs.

An exported native result list is an evidence representation, not a new
prediction journal. To resume or rescore using the current journal API, explicitly
attach the complete journal and required reference/suite inputs; this preserves
the same original plan and committed parts. Exporting or inspecting does not
replay that journal. A fresh cold process can invoke its selected packaged
entrypoint against those explicit bindings, with completed outputs reused.

## CLI

```bash
arti recipe export-results outputs/prediction --config layout.json --expected-ids-json ids.json --evidence-json evidence.json --attachments-json attachments.json --deployment outputs/package --output outputs/handoff.zip
arti recipe inspect-results outputs/handoff.zip
```

Layout JSON/TOML uses `format="arti.result-export-config"` and the Python fields.
Evidence/attachment mapping paths resolve relative to their mapping JSON file.
Inspection and export also accept legacy JSON-only prediction journals. See
[Durable Task File Outputs](prediction-files.md),
[Task Evaluation Suites](evaluation-suites.md) and
[Runtime Profiles](task-runtime.md) for the shared upstream contracts.

The generated neutral CPU example runs the complete file lifecycle:

```bash
python -m examples.file_lifecycle_workflow --output outputs/file-lifecycle
```

It checks physical preparation-cache reuse, exact native training recovery and
validation selection, interrupted file-result recovery, reference/output suite
coverage, cold packaged runtime reuse and archive inspection after source
deletion. Its caller-selected code and journal also support a second fresh
consumer after extraction. The fixture proves local lifecycle behavior; it
does not qualify task quality, network isolation, hardware or throughput.
