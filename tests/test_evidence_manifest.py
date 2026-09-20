import json
import importlib.util
from pathlib import Path


def load_verifier():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "verify_evidence_manifest.py"
    spec = importlib.util.spec_from_file_location("verify_evidence_manifest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


verifier = load_verifier()
verify_artifacts = verifier.verify_artifacts
verify_mechanism_gates = verifier.verify_mechanism_gates


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_verify_artifacts_detects_seed_count_and_sha_mismatch(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,target\n1,0\n", encoding="utf-8")
    result_path = tmp_path / "result.json"
    lock_path = tmp_path / "lock.json"
    write_json(result_path, {"provenance": {"seed_values": [0]}})
    write_json(lock_path, {"sha256": "not-the-real-hash"})
    manifest = {
        "artifacts": [
            {"path": "result.json", "kind": "json", "min_seeds": 3},
            {"path": "lock.json", "kind": "json", "sha256_of": "data.csv"},
        ]
    }

    failures = verify_artifacts(tmp_path, manifest)

    assert any("seed count below manifest minimum" in failure for failure in failures)
    assert any("sha256 mismatch" in failure for failure in failures)


def test_verify_mechanism_gates_detects_failed_drop(tmp_path):
    results_path = tmp_path / "benchmarks" / "results" / "nature_aligned_results.json"
    write_json(
        results_path,
        {
            "summary": [{"task": "recall_operator_routing", "model": "arti_full", "mean_accuracy": 0.95}],
            "claim_verdicts": [
                {
                    "task": "recall_operator_routing",
                    "claim_verdict": "supported",
                    "zero_recall_drop": 0.05,
                }
            ],
        },
    )
    manifest = {
        "mechanism_gates": {
            "recall_operator_routing": {
                "min_accuracy": 0.90,
                "drops": {"zero_recall_drop": 0.10},
            }
        }
    }

    failures = verify_mechanism_gates(tmp_path, manifest)

    assert failures == ["recall_operator_routing zero_recall_drop below gate: 0.050 < 0.100"]
