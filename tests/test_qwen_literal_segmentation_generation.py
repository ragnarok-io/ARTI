from __future__ import annotations

import copy
import json
from pathlib import Path

from benchmarks.verify_qwen_literal_segmentation_generation import verify


ROOT = Path(__file__).resolve().parents[1]


def payload() -> dict:
    return json.loads((ROOT / "benchmarks/results/qwen_literal_segmentation_generation_results.json").read_text(encoding="utf-8"))


def test_qwen_literal_segmentation_generation_evidence_passes() -> None:
    assert verify(payload()) == []


def test_qwen_literal_segmentation_generation_rejects_prefix_only_loss() -> None:
    data = copy.deepcopy(payload())
    data["training_objective_contract"]["answer_loss_weight"] = 1.0
    assert any("shared sentence prefix" in failure for failure in verify(data))


def test_qwen_literal_segmentation_generation_rejects_incoherent_output() -> None:
    data = copy.deepcopy(payload())
    row = next(row for row in data["runs"] if row["model"] == "output_context")
    row["samples"][0]["generation"] = "five maybe"
    assert any("complete decoded sentence" in failure for failure in verify(data))
