# Fit Config Template

ARTI fit configs describe adapter insertion mechanics, not data loading,
business schemas, labels, or task-specific rules.

Generate a starter JSON config:

```bash
uv run --extra dev arti init-config arti.json --profile latent-adapt --scale small
```

Generate an observer-phase config:

```bash
uv run --extra dev arti init-config arti.json --profile observer-phase --scale small
```

Validate it:

```bash
uv run --extra dev arti validate config arti.json
```

Check the bundled schema:

```bash
uv run --extra dev arti schema fit-config check --output docs/reference/fit-config.schema.json
```

Minimal shape:

```json
{
  "$schema": "docs/reference/fit-config.schema.json",
  "fit": {
    "profile": "latent-adapt",
    "scale": "small"
  },
  "insertion": {
    "where": ["*.out_proj"],
    "max_adapters": 4,
    "max_extra_params": "1%"
  }
}
```

Mechanism override shape:

```json
{
  "$schema": "docs/reference/fit-config.schema.json",
  "fit": {
    "profile": "observer-phase",
    "scale": "small",
    "objectives": ["preserve-output", "task-fit", "validate"]
  },
  "mechanism": {
    "coord_dim": 16,
    "coord_frame_mode": "operator_bank",
    "observer_phase": true,
    "virtual_recall": true,
    "operator_count": 8,
    "interface_slots": 16,
    "recall_slots": 8,
    "recall_steps": 1
  },
  "runtime": {
    "causal": true,
    "mask_key": "attention_mask",
    "coord_key": "phase_coord",
    "observer_coord_key": "next_token_phase",
    "frame_operators_key": "inverse_operator_bank"
  },
  "insertion": {
    "where": ["*.out_proj", "*.mlp.fc2"],
    "max_adapters": 8,
    "max_extra_params": "1%"
  }
}
```

Use it from Python:

```python
import arti

config = arti.load_fit_config("arti.json")
project = arti.project(model).configure(config).scan(sample)
project.write_plan("artifacts/arti-plan.json")
```

Use it from CLI:

```bash
arti plan my_project.models:make_model artifacts/arti-plan.json \
  --sample-shape 2,16,768 \
  --config arti.json \
  --target-modules "*.out_proj"
```

For the full Gradle-like workflow, see `docs/guides/fit-adaptation.md`.
