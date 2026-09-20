import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "verify_stateful_recall_stream.py"
SPEC = importlib.util.spec_from_file_location("stateful_recall_stream", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_stateful_recall_stream_has_selective_online_effect_and_capacity_trend():
    result = MODULE.run()
    methods = result["methods"]
    assert methods["delta_recall"]["target_mse"] < methods["frozen"]["target_mse"] * 0.01
    assert methods["delta_recall"]["target_mse"] < methods["pure_additive"]["target_mse"] * 0.01
    assert methods["delta_recall"]["noise_leakage_norm"] < 0.02
    assert methods["equal_memory_kv"]["target_mse"] <= methods["delta_recall"]["target_mse"]
    assert result["delta_recall"]["mean_recognition"] > 0.9
    assert result["delta_recall"]["max_unseen_recognition"] < 0.02
    assert result["delta_recall"]["finite"] is True
    assert [item["retained"] for item in result["capacity"]] == [2, 4, 8]
