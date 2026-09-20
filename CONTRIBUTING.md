# Contributing To ARTI

ARTI is a PyTorch-first latent tensor dynamics package. Contributions should keep the package domain-free, tensor-first, and mechanism-verifiable.

## Local Setup

Start with the new developer guide:

```text
docs/guides/new-developer.md
```

```bash
uv sync --extra dev --extra docs --extra bench
```

On Windows, keep uv state inside the project if global cache paths are not writable:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
$env:UV_PYTHON_INSTALL_DIR='.uv-python'
uv sync --extra dev --extra docs --extra bench
```

## Required Checks

Run focused package checks before submitting a change:

```bash
uv run python scripts/quality_gate.py quick
uv run python scripts/quality_gate.py docs
uv run python scripts/quality_gate.py package
```

If Windows denies access to the default user temp pytest directory, run tests with a workspace-local base temp:

```bash
uv run python scripts/quality_gate.py quick
```

Mechanism benchmark gates are scoped. They support controlled tensor-mechanism claims, not broad downstream superiority:

```bash
uv run python scripts/quality_gate.py mechanism
```

## Backend Policy

- The root `arti` API is PyTorch-first.
- `arti.torch` is the explicit PyTorch backend namespace.
- `arti.jax` is reserved and currently reports `planned`.
- Do not add a partial JAX implementation that cannot pass JIT/grad/device smoke tests.

## GPU Evidence

CPU tests prove correctness only. GPU-readiness requires CUDA smoke checks and, for scaling claims, generated CUDA evidence packets with non-null peak memory rows.

Run when CUDA is available:

```bash
uv run python scripts/quality_gate.py cuda
```

## Documentation

Concepts, guides, and API pages live in `docs/`. API pages are generated through mkdocstrings from code under `src/arti`.

Use concise docstrings for public APIs. Avoid claiming downstream task superiority unless a preregistered benchmark gate supports it.

## Releases

Before release, update `CHANGELOG.md`, `pyproject.toml`, and `src/arti/_version.py` together. The release checklist lives in `docs/guides/release.md`.
