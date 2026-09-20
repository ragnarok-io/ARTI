import importlib.util
from pathlib import Path


def load_compare():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "compare_reproduction.py"
    spec = importlib.util.spec_from_file_location("compare_reproduction", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


compare_module = load_compare()


def summary(verdict="supported", accuracy=0.9):
    tasks = ("coordinate_routing", "visibility_reasoning", "masked_denoising", "recall_operator_routing")
    return {
        "mechanism_gates": {
            task: {"arti_accuracy": accuracy, "claim_verdict": verdict}
            for task in tasks
        },
        "scaling": {
            "at_max_tokens": {
                "arti_interface_only": {"mean_ms": 1.0, "estimated_activation_bytes": 10}
            }
        },
    }


def env(torch_version="2.x"):
    return {"platform": "test", "torch": {"version": torch_version, "cuda_available": False}}


def test_compare_allows_metric_drift_when_verdicts_supported():
    payload = compare_module.compare(summary(accuracy=0.9), summary(accuracy=0.8), env(), env())

    assert payload["passed"] is True
    assert payload["mechanism"]["coordinate_routing"]["accuracy_delta"] == -0.09999999999999998


def test_compare_fails_when_current_verdict_not_supported():
    payload = compare_module.compare(summary(), summary(verdict="not_supported"), env(), env())

    assert payload["passed"] is False
    assert "coordinate_routing current verdict is not_supported" in payload["failures"]
