import json
import importlib.util
from pathlib import Path


def load_proxy_verifier():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "verify_proxy_gates.py"
    spec = importlib.util.spec_from_file_location("verify_proxy_gates", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


proxy_verifier = load_proxy_verifier()


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_private_rule_proxy_does_not_require_transformer_margin(tmp_path):
    path = tmp_path / "proxy.json"
    write_json(
        path,
        {
            "provenance": {"seed_values": [0, 1, 2]},
            "audit": {"positive_rate": 0.5},
            "summary": [],
            "paired_margins": [
                {"comparison": "arti_full - arti_no_recall", "mean_margin": 0.03, "positive_fraction": 1.0},
                {"comparison": "arti_full - transformer_recall", "mean_margin": -0.10, "positive_fraction": 0.0},
            ],
        },
    )

    failures = proxy_verifier.verify_proxy(path, proxy_verifier.PRIVATE_RULE_GATES, "private_rule_proxy")

    assert failures == []


def test_visibility_proxy_detects_missing_ablation_effect(tmp_path):
    path = tmp_path / "visibility.json"
    write_json(
        path,
        {
            "provenance": {"seed_values": [0, 1, 2]},
            "audit": {"positive_rate": 0.5},
            "summary": [
                {"model": "arti_full", "mean_no_visibility_drop": 0.05, "mean_all_visibility_drop": 0.05}
            ],
            "paired_margins": [
                {"comparison": "arti_full - arti_no_visibility", "mean_margin": 0.30, "positive_fraction": 1.0}
            ],
        },
    )

    failures = proxy_verifier.verify_proxy(path, proxy_verifier.VISIBILITY_GATES, "visibility_proxy")

    assert "visibility_proxy arti_full mean_no_visibility_drop below gate: 0.050 < 0.200" in failures
    assert "visibility_proxy arti_full mean_all_visibility_drop below gate: 0.050 < 0.200" in failures
