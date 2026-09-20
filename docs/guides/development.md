# Development Workflow

ARTI uses uv for environment and package management.

```bash
uv sync --extra dev --extra docs --extra bench
```

## Core Checks

```bash
uv run arti gate quick
uv run arti gate docs
uv run arti gate package
```

The quick gate includes core layer checks plus lightweight API and CLI adaptation release-chain smoke tests: plan, adapter artifact, build lock, apply report, patched state dict, deployment manifest, and deployment validation.

## Generated Docs

Source-backed reference pages can be generated through the installed CLI:

```bash
uv run arti docs generate
uv run arti docs check
uv run arti schema fit-config generate
uv run arti schema fit-config check
uv run arti schema task-graph generate
uv run arti schema task-graph check
```

The docs quality gate runs `arti docs check`, `arti schema fit-config check`, and `arti schema task-graph check` before `mkdocs build --strict`, so changes to profiles, scales, plugins, declarative config fields, or task graph artifacts must update the generated reference files in `docs/reference/`.

Run Ruff before the Python test gates:

```bash
uv run --extra dev ruff check .
```

Ruff targets Python 3.10 and currently enforces syntax, undefined-name,
import-use, and core pycodestyle correctness rules. Formatting is available as
`uv run --extra dev ruff format`, but repository-wide formatting is not a CI
gate; keep formatting changes scoped to files already being edited.

On Windows, if pytest cannot access the default user temp directory, use a workspace-local base temp:

```bash
uv run arti gate quick
```

## Package Check

The package check builds both wheel and source distribution, installs the wheel into a temporary virtual environment, and imports:

```text
arti
arti.torch
arti.jax
```

This protects the public package boundary from local editable-install assumptions.

## Backend Boundary

`arti.torch` is production-facing today. `arti.jax` provides a small optional functional backend: it reports `available` when JAX is installed and `unavailable` otherwise. Full `arti.fit()` adaptation and industrial packaging flows remain PyTorch-first.

## GPU Checks

CUDA-specific smoke tests are skipped when no CUDA device is available. GPU scaling claims require explicit CUDA evidence artifacts, not CPU-only tests.

Runtime helpers are available from both the root API and `arti.torch`:

```python
import arti

report = arti.cuda_device_report()
smoke = arti.cuda_smoke_report()
device = arti.require_cuda()
```

Start with the capability report:

```bash
uv run python -m arti.cli doctor --allow-cpu-torch --output benchmarks/results/doctor.json
uv run python benchmarks/report_backend_capabilities.py
uv run python benchmarks/verify_backend_capabilities.py
```

The report separates NVIDIA hardware visibility from PyTorch CUDA runtime availability. A machine can have an NVIDIA GPU while the current Python environment still uses a CPU-only PyTorch build.
