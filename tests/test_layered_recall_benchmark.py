from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_layered_recall_trajectory", ROOT / "benchmarks" / "verify_layered_recall_trajectory.py")
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def result() -> dict:
    return json.loads((ROOT / "benchmarks" / "results" / "layered_recall_trajectory.json").read_text(encoding="utf-8"))


def test_generated_layered_recall_evidence_passes() -> None:
    assert module.verify(result()) == []


def test_verifier_rejects_label_or_future_token_training_targets() -> None:
    payload = result()
    payload["training_contract"]["evaluation_labels_only"] = False
    assert any("evaluation-only" in failure for failure in module.verify(payload))


def test_verifier_rejects_parameter_mismatch() -> None:
    payload = result()
    row = next(row for row in payload["runs"] if row["condition"] == "single_early")
    row["parameters"] *= 2
    assert any("parameter matched" in failure for failure in module.verify(payload))


def test_verifier_rejects_claim_when_one_seed_loses_to_single_layer() -> None:
    payload = result()
    seed = payload["config"]["seeds"][0]
    layered = next(row for row in payload["runs"] if row["seed"] == seed and row["condition"] == "layered_recall" and row["corruption"] == "combined")
    layered["final_mse"] = 1.0
    assert any("does not beat every single-layer" in failure for failure in module.verify(payload))


def test_verifier_rejects_layer_order_indifference() -> None:
    payload = result()
    seed = payload["config"]["seeds"][0]
    layered = next(row for row in payload["runs"] if row["seed"] == seed and row["condition"] == "layered_recall" and row["corruption"] == "combined")
    shuffled = next(row for row in payload["runs"] if row["seed"] == seed and row["condition"] == "shuffled_layers" and row["corruption"] == "combined")
    shuffled["final_mse"] = layered["final_mse"]
    assert any("does not damage" in failure for failure in module.verify(payload))


def test_verifier_rejects_unseen_recall_leakage() -> None:
    payload = result()
    for row in payload["runs"]:
        if row["condition"] == "layered_recall":
            row["unseen_influence_norm"] = row["influence_norm"]
    assert any("unseen influence" in failure for failure in module.verify(payload))
