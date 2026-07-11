# Contributing

ARTI is a tensor-first, domain-independent neural-network library.

```bash
uv sync --extra dev
uv run --extra dev pytest
uv build
```

Keep changes scoped, document public APIs, and add focused tests for shapes,
masks, gradients, devices, and serialization where applicable.
