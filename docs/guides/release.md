# Public Release Checklist

This guide describes the public `arti-fit` distribution and its GitHub release
workflow. The Python import package is named `arti`.

## Blocking Gates

The tag-triggered release workflow calls the same `.github/workflows/ci.yml`
through GitHub Actions `workflow_call`, as used by branch pushes and pull
requests. It waits for that workflow before building or publishing. The CI
workflow runs the package test suite, builds the documentation with strict link
checks, and validates the wheel and source distribution.

The public matrix covers Windows and Ubuntu with Python 3.10/3.11/3.12.
The `quick`, `mainline` and `recipes` software gates cover public contracts,
native graphs, compilation, task continuation, evaluation and ASGI serving.
The separate native JAX job installs JAX/Flax and runs the Formula/ProgramGraph
gate and native training example. Tests do not download a pretrained model or
launch research GPU training. Optional backend/hardware limits remain explicit.

The release build then runs:

```bash
uv run --extra dev python scripts/check_release_readiness.py
uv run --extra dev python scripts/check_package.py
```

Readiness verifies that `pyproject.toml`, `src/arti/_version.py`,
`CITATION.cff`, `CHANGELOG.md`, and README install pins agree; it also checks
that the public CI and release workflows remain connected. `check_package.py`
builds and inspects the wheel and source distribution, including required
package files, sensitive-content markers, excluded local-only paths, and the
exact member inventory in `release-artifact-manifest.json`. Update that
version-bound inventory only after reviewing both built archives; it is a
release-check input and is not included in either distribution.

## Releasing

Before copying a new development snapshot, compare public source-owned fixes
against the private source and backport missing repairs. Preserve release-owned
distribution identity, version, citation, archive audits and publishing workflows.
Review only tracked approved source and examples, not the working directory's
training outputs, data, notes or caches. The website is a separate deployment.

1. Update the package version in `pyproject.toml` and `src/arti/_version.py`.
2. Add an exact `## <version>` heading to `CHANGELOG.md`, update `CITATION.cff`
   and every pinned install command in `README.md`.
3. Run the public test, documentation, readiness, and package checks locally.
4. Review the exact commit contents and the built wheel/source distribution for
   private paths, credentials, machine-local artifacts, and unsupported claims.
5. Push the reviewed commit and a matching `v<version>` tag. The release
workflow runs the same CI before building, then uses PyPI Trusted Publishing.

Do not claim CUDA scaling, complete optional-backend parity, or broad downstream
superiority unless the corresponding evidence exists for the released source.
