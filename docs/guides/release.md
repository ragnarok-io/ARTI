# Release Checklist

ARTI releases should be cut only after package, docs, mainline gate, and mechanism checks pass.

## Current Release Level

The current package has a stable public mechanism surface. Stability covers the
documented component identities and serialization contracts, not a production
service SLA or universal downstream superiority.

Supported stable claims:

- PyTorch-first latent tensor layers and residual/sequence blocks.
- Declarative `arti.fit()` / `arti.project()` adaptation flow.
- CLI build chain for plan, adapter artifact, build lock, apply report, patched state dict, deployment manifest, and task graph artifacts.
- Machine-readable fit config and task graph schemas.
- CPU test coverage and CUDA-ready runtime checks.
- Optional `arti.jax` functional backend with explicit availability status.

Do not claim yet:

- Compatibility outside the documented stable component identities.
- Production SLA or broad downstream superiority.
- Nature-level external validation.
- CUDA scaling evidence unless the CUDA gate was run on a CUDA-enabled PyTorch runtime.
- Full JAX parity for ARTI layers, virtual recall, `arti.fit()`, and adapter build flows.

## Version

Update all version sources together:

- `pyproject.toml` project version
- `src/arti/_version.py`
- `CHANGELOG.md`

## Public Snapshot Synchronization

The private repository is the development source; the public repository is a
sanitized release line with its own history. Before producing a new public
snapshot, inspect every public commit made since the previous snapshot and
backport source-owned fixes into the private repository first.

Source-owned changes include implementation fixes, compatibility shims,
tests, author identity, citation metadata, and reusable documentation. Public
release-owned changes include the `arti-fit` distribution name, Stable/LTS
version and release wording, PyPI Trusted Publisher workflow, tags, and public
installation commands. Do not overwrite release-owned fields with private
development values, and do not copy release credentials or tag workflows into
the private line.

The public snapshot must be generated only after both repositories are clean
and the backport audit is recorded in the release notes.

## Required Local Gates

```bash
uv lock
uv run --extra dev arti gate quick
uv run --extra docs arti gate docs
uv run --extra dev arti gate mainline
uv run --extra dev arti gate package
uv run --extra dev python scripts/check_release_readiness.py
```

The quick gate also runs lightweight API and CLI release-chain smoke tests covering plan, adapter artifact, build lock, apply report, patched state dict, deployment manifest, and deployment validation.

For a stable release, the quick, docs, mainline, and package gates plus
`check_release_readiness.py` are the minimum blocking checks. Mechanism, Qwen,
and CUDA gates strengthen the release notes, but their claim boundaries must be
stated explicitly.

## CI Gates

The GitHub Actions workflow runs:

- quick quality gate on Ubuntu and Windows for Python 3.10 and 3.12
- docs quality gate
- mainline quality gate covering current core mechanism evidence
- package quality gate, including release readiness
- JAX optional backend tests with `--extra jax` plus `arti doctor --require-jax-smoke`
- Web artifact export, TypeScript build, and ORT WASM parity
- mechanism quality gate plus `check_release_readiness.py --require-mechanism`

`scripts/check_release_readiness.py` also checks that `.github/workflows/ci.yml` still contains the quick, docs, mainline, package, JAX optional backend, and mechanism evidence jobs. This keeps release confidence tied to executable CI coverage instead of a checklist that can drift.

CUDA evidence is not assumed in hosted CI. Run the CUDA gate on a CUDA-enabled local or self-hosted machine before making local CUDA claims.

## Mechanism Gates

```bash
uv run --extra dev arti gate mechanism
uv run --extra dev python scripts/check_release_readiness.py --require-mechanism
```

## Qwen Dynamic Vocab Evidence

Run the Qwen gate before claiming Qwen runtime-vocab, glyph tensor, pulse, or
metadata-bridge evidence:

```bash
uv run --extra dev arti gate qwen --reuse-passing-producers
uv run --extra dev python scripts/check_release_readiness.py --require-qwen
```

The Qwen gate may load local or cached Qwen weights. Its positive claim remains
adapter-level: frozen or semi-frozen Qwen paths with ARTI-side runtime-vocab
interfaces, not full tokenizer replacement or broad open-ended dialogue parity.
Use `docs/validation/qwen-dynamic-vocab-goal.md` as the requirement-to-evidence
map when release notes mention the Qwen dynamic-vocab alpha path.

## GPU Evidence

Do not claim CUDA scaling evidence unless CUDA-specific artifacts were generated on a CUDA device:

```bash
uv run --extra dev arti gate cuda
uv run --extra dev python scripts/check_release_readiness.py --require-cuda
```

When NVIDIA hardware is visible, `benchmarks/verify_torch_cuda_runtime.py` must pass without `--allow-cpu-torch` before CUDA runtime claims are made.

## Package Artifacts

`scripts/check_package.py` must build:

- source distribution
- wheel

The wheel must contain:

- `arti/__init__.py`
- `arti/py.typed`
- `arti/torch/__init__.py`
- `arti/torch/cuda.py`
- `arti/jax/__init__.py`
- `arti/web/__init__.py`
- `arti/web/exporter.py`
- `arti/schemas/fit-config.schema.json`
- `arti/schemas/task-graph.schema.json`

## Final Stable Decision

Before cutting a stable release:

- Confirm `CHANGELOG.md` accurately describes the stable public surface.
- Confirm README and docs do not promise downstream superiority, CUDA scaling,
  or full JAX parity beyond recorded evidence.
- Attach the latest quick/docs/mainline/package gate outputs to the release notes or CI run.
- Mark CUDA as `not claimed` unless `arti gate cuda` passed on a CUDA-enabled PyTorch runtime.
- Run `scripts/check_release_readiness.py` after regenerating gate reports; use `--require-mechanism` when mechanism evidence is part of the release notes, `--require-qwen` when Qwen dynamic-vocab evidence is part of the release notes, and `--require-cuda` only when local CUDA evidence is part of the release claim. The CUDA requirement includes both the CUDA evidence packet and a passing `torch_cuda_smoke_status`.
