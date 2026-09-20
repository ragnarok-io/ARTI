import importlib.util
import json
from pathlib import Path


def load_preregistration_verifier():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "verify_preregistration.py"
    spec = importlib.util.spec_from_file_location("verify_preregistration", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


verifier = load_preregistration_verifier()


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_preregistration_detects_manifest_gate_mismatch(tmp_path):
    prereg = {
        "main_mechanism_gates": {
            "min_seeds": 5,
            "tasks": {"task": {"min_accuracy": 0.9, "drops": {"drop": 0.1}}},
        },
        "proxy_gates": {
            "private_rule": {"min_seeds": 3},
            "visibility": {"min_seeds": 3},
        },
    }
    manifest = {"mechanism_gates": {"task": {"min_accuracy": 0.1, "drops": {"drop": 0.1}}}}
    write_json(tmp_path / "pre.json", prereg)
    write_json(tmp_path / "manifest.json", manifest)

    nature = type("Nature", (), {"MIN_SEEDS": 5, "REQUIRED_TASKS": prereg["main_mechanism_gates"]["tasks"]})
    proxy = type("Proxy", (), {"PRIVATE_RULE_GATES": {"min_seeds": 3}, "VISIBILITY_GATES": {"min_seeds": 3}})

    failures = []
    loaded_prereg = verifier.load_json(tmp_path / "pre.json")
    loaded_manifest = verifier.load_json(tmp_path / "manifest.json")
    if loaded_prereg["main_mechanism_gates"]["tasks"] != loaded_manifest["mechanism_gates"]:
        failures.append("main mechanism task gates mismatch with evidence_manifest.json")
    assert nature.MIN_SEEDS == loaded_prereg["main_mechanism_gates"]["min_seeds"]
    assert proxy.PRIVATE_RULE_GATES == loaded_prereg["proxy_gates"]["private_rule"]
    assert failures == ["main mechanism task gates mismatch with evidence_manifest.json"]
