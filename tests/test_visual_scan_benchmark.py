from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_visual_scan_superresolution", ROOT / "benchmarks" / "verify_visual_scan_superresolution.py")
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def result() -> dict:
    return json.loads((ROOT / "benchmarks" / "results" / "visual_scan_superresolution.json").read_text(encoding="utf-8"))


def test_generated_visual_scan_result_passes() -> None:
    assert module.verify(result()) == []


def test_verifier_rejects_magic_gain_on_repeated_phase() -> None:
    payload = result()
    repeated = next(row for row in payload["summary"] if row["condition"] == "repeated_phase")
    repeated["mean_accuracy"] = 0.95
    assert any("without complementary concat" in failure for failure in module.verify(payload))


def test_verifier_rejects_high_resolution_inference_leakage() -> None:
    payload = result()
    payload["protocol_lock"]["high_resolution_target_exposed_at_inference"] = True
    assert any("unavailable at inference" in failure for failure in module.verify(payload))


def test_verifier_rejects_categorical_shift_shortcut() -> None:
    payload = result()
    unseen = next(row for row in payload["summary"] if row["condition"] == "unseen_shift")
    unseen["mean_accuracy"] = 0.30
    unseen["mean_psnr"] = 20.0

    failures = module.verify(payload)

    assert any("held-out shifts" in failure for failure in failures)
