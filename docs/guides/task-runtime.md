# Runtime Profiles And Offline Handoff

Attach a caller-declared runtime profile and explicit trusted code/wheel inputs
to the existing portable task package. The model/task/resource closure remains
the same authority. A thin host process invokes application code; it does not
create another trainer, prediction journal, registry or scheduler.

```python
runtime = arti.TaskRuntimeProfile(
    "my_package.application:main", revision="my-entrypoint-revision",
    python=">=3.10", requirements=("my-application-dependency>=1",),
    inputs={"requests": "file"}, outputs={"results": "directory"},
    options={"my-task-options": "caller-owned"},
    environment={"MY_SETTING": "explicit-value"},
    network="disabled",
    limits={"process_wall_seconds": my_worker_allowance, "memory_bytes": my_memory_declaration},
    require_enforcement=("process_wall_seconds",),
)
arti.package_recipe_deployment(
    model_asset, "outputs/package",
    resources=my_inference_resources,
    runtime=runtime, code={"application": "curated-application-root"},
    wheels={"dependency": "my_application_dependency-1.0-py3-none-any.whl"},
)
```

Code roots contain explicitly selected installed-module layouts and become the
child's import path. Only supplied roots are copied, not the trainer directory,
working checkout or arbitrary private project files. File/directory inventory,
hashes and runtime profile are part of deployment identity. Repackaging inherits
them and moving the whole bundle retains content identity. Inspect/loader checks
detect missing/changed code before application import. The task's own factory
descriptor still matches its saved model contract; this is not a replacement
for that contract or permission to distribute private input resources.

Wheel inputs are copied and inspected with the maintained Packaging wheel-name
parser and distribution metadata. They are offline installation inputs, not
automatically installed/imported packages. An absent installed dependency stays
absent even when its wheel is bundled. A caller can explicitly install approved
dependencies through its existing deployment tooling; this library never runs
pip, resolves/downloads packages or builds an image by itself.

## Native Application Entrypoint

The installed callable receives `TaskRuntimeContext`:

```python
def main(context):
    return arti.run_prediction(
        context.package,
        context.inputs["requests"],
        arti.PredictionRunConfig(context.outputs["results"], **context.options["prediction"]),
        task_factory="my_package.task:build_task",
        resume=context.options.get("resume", False),
    )
```

`package`, named input/output paths, declared profile and resolved JSON options
are explicit. The application can instead call shared deployment validation,
suite evaluation or other task logic. IO names/kinds/options are caller-owned;
no codec, model, task or submission protocol is built in. Supplied bindings must
match named ports and keep input/package locations separate from output storage.
Logical binding separation is not a hostile-code filesystem sandbox.

```python
inspection = arti.inspect_task_runtime("outputs/package", python=python_executable)
receipt = arti.run_task_runtime(
    "outputs/package", python=python_executable,
    inputs={"requests": "requests.jsonl"},
    outputs={"results": "outputs/prediction"},
    output="outputs/runtime-receipt.json",
)
```

The metadata probe inspects the selected interpreter and installed distribution
versions without importing model/application code. Requirement markers and
version ranges use Packaging and the probed interpreter's platform variables.
The framework's real dependency metadata is checked too; a stdlib-only or
dependency-restricted declaration does not make the current Torch runtime work
without Torch. `allowed_dependencies` checks declared direct requirements,
not arbitrary imports or transitive code authority. Direct URL origin cannot be
verified by a version string and is reported incompatible rather than guessed.

Missing dependencies, unsupported Python, or unsupported required enforcement
return `not_run`, not model readiness or automatic installation. Compatible
metadata does not prove model quality, native ABI/operator compatibility or
startup throughput. Runtime success means the entrypoint completed; its raw
application result remains separate (it can report stopped/rejected/evaluated).

## Declaration, Observation And Enforcement

Each receipt separates:

- `declared`: profile, requirements, network policy and numerical limit values.
- `observed`: actual interpreter/platform/versions, PID, exit code and elapsed
  time; code-file content identity is recorded separately.
- `enforced`: only the selected backend's actual enforcement, with its scope.
- `unenforced`: resource declarations not enforced by this backend.

The included backend is a trusted local subprocess. It creates a fresh temporary
working directory, uses the selected interpreter, replaces inherited Python
search roots with framework plus explicitly packaged code roots, and disables
bytecode writes into the immutable package. It uses no shell command strings.
Input/output bindings can remain outside that working directory; no trainer
checkout is needed. Output journals remain durable application-owned locations.

For an observed Windows CPython virtual environment, the worker launches its
observed base interpreter directly while retaining the selected environment.
This avoids terminating only a virtual-environment redirector on timeout. The
receipt's `observed.launch` records selected and actual process executables and
the selected prefix. An unavailable observed base fails before application
launch; the runtime does not select an alternate interpreter or install one.

`process_wall_seconds` is an actual wait deadline for the direct worker after
process creation. On timeout the host kills and waits for that worker, reporting
`terminated_by_host`, not endogenous task completion. Process creation, metadata
probe and descendants are outside this scope. This is not a global job deadline,
process-tree isolation, memory/cpu/GPU cap or machine sandbox. The receipt states
the scope. Logs are redirected to files and diagnostic tails are bounded;
large logs do not become whole host-memory byte arrays.

`network="disabled"` sets supported offline library hints; it does not block
network sockets. Requiring network isolation on the local backend is incompatible.
Memory/cpu/GPU declarations are not silently enforced by environment variables.
Requiring unsupported caps similarly prevents execution. Strong isolation and
host/container enforcement belong to an actual selected host backend, not a
Python permission framework.

A `container` JSON descriptor may carry the caller's image identity and runtime
requirements as part of handoff. The local backend reports this execution
unsupported and `container_verification=not_run`; it does not pull/run/build an
image or claim its dependency/resource behavior. Host/container execution uses
the explicit selected extension below; the included local worker does not
qualify container or strong isolation behavior.
Provider/GPU/hardware checks likewise remain unrun unless separately executed.

## Caller-Selected Host Backend

`TaskRuntimeBackendConfig(factory, revision)` adds an optional trusted host
callback to `TaskRuntimeProfile.backend`. Default profiles omit this field in
serialization, preserving existing package identity and local API behavior.

```python
runtime = arti.TaskRuntimeProfile(
    "my_application.application:main", "my-entrypoint-revision",
    backend=arti.TaskRuntimeBackendConfig(
        "my_application.host:execute", "my-host-revision",
    ),
    inputs=my_input_ports, outputs=my_output_ports,
    limits=my_resource_declaration, require_enforcement=my_required_constraints,
)
```

The callback is explicitly installed caller-owned host code, separate from the
packaged child entrypoint, task model/resources and optional wheel inputs.
A bundled host module still needs an explicit host import/install choice;
the library does not automatically install it or rewrite host import paths.
Factory/revision identity travels with the profile. Code behavior and reported
target environment remain attributed to the caller.

One synchronous callback receives `TaskRuntimeBackendContext` and
`operation="inspect"` or `"run"`. Inspection reports actual availability,
target Python/marker/distribution observations, supported capability names and
any container identity/verification. Execution supplies the original package
identity, observed work, explicit application execution/status and applied
constraint values/scopes. The context exposes `inspect_local()`/`run_local()`
to delegate to the existing real local probe/worker without replacing original
package/profile identity. No registry, queue, scheduler or retry loop is added.

```python
metadata = arti.inspect_task_runtime(package)  # no backend/application import
target = arti.inspect_task_runtime(package, inspect_backend=True)
result = arti.run_task_runtime(package, inputs=my_inputs, outputs=my_outputs)
```

For a selected backend, default inspection returns `not_run` until that backend
has been explicitly inspected. `inspect_backend=True` imports trusted host code
and discloses its trust boundary; it does not import the child task or model.
Execution inspects its explicitly selected host before one run. Target dependency
compatibility is checked against the reported target environment. A failing local
controller probe remains separate unavailable local evidence and cannot falsely
disqualify a different selected host. Unselected local-worker behavior is unchanged.

Receipts expose `origin="backend-reported"`, separate local observations,
`locally_verified_execution=False`, declared/applied/unenforced limits and their
scopes. Applied numeric values use strict JSON identity; a capability name or
counter alone cannot replace an applied receipt. Core installation remains false;
the backend's installation observation can be false/true/unknown and is separately
attributed. Completion requires application-execution and observed-work evidence.
A malformed/failed callback after possible effects has an unknown execution
outcome and is never automatically repeated.

These admission/receipt checks validate the extension contract, not independent
strong-enforcement proof. The neutral `examples/runtime_backend.py` delegates
to an actual cold local worker and its direct-worker wait deadline. It has no
network/memory/process-tree/container/provider enforcement. Required unavailable
capabilities return `not_run`; container/provider/hardware qualification remains
unrun unless a separately authorized real backend is executed and measured.
External action recovery remains the episode adapter's explicit contract.

## Recovery And Evidence

Each invocation publishes an atomic derived runtime receipt with retained
content-addressed history. Interpreter/package/profile/code identities, named
bindings, resolved option digest and application response remain inspectable.
Code imports and entrypoint execution have distinct recorded phases. A killed
process is not interpreted as an application answer or evidence of quality.

Resume behavior belongs to the existing task service. For example, rerun a
prediction entrypoint with `options={"resume": True}` and the same output journal;
it loads only unfinished work or returns committed outputs. No second journal
restarts completed partitions. For ambiguous external actions, this wrapper
cannot promise replay safety or exactly-once effects; the episode adapter's
explicit recovery contract is required.

The `examples/runtime_task.py` entrypoints reuse real native prediction and
suite validation. Neutral tests delete the source/trainer tree, relocate the
package, execute file outputs in a cold process, then rescore/reuse the existing
results without model replay. They are local lifecycle evidence, not performance
or remote deployment qualification.

## CLI

```bash
arti recipe package-deployment outputs/model --resources-json resources.json --runtime-profile runtime.json --code-json code.json --wheels-json wheels.json --output outputs/package
arti recipe inspect-runtime outputs/package --python path/to/python --output runtime-inspection.json
arti recipe run-runtime outputs/package --inputs-json inputs.json --outputs-json outputs.json --options-json options.json --output runtime-receipt.json
```

Profile JSON/TOML uses `format="arti.task-runtime-profile"` and Python field
names. Code/wheel and input/output mapping paths resolve relative to their JSON
manifest. Requirements and runtime options remain explicit. The existing
package/inspect/predict/validate APIs continue without a runtime profile.

References checked 2026-10-11:
[CPython subprocess lifecycle](https://docs.python.org/3/library/subprocess.html)
defines process wait/termination and creation-time scope;
[Packaging requirement contracts](https://packaging.pypa.io/en/stable/requirements.html)
provides maintained requirement parsing/markers. Transfer these mechanisms;
do not infer network isolation or hard resource caps from a profile declaration.
