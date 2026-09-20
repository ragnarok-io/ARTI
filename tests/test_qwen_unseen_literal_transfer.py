from __future__ import annotations

import copy
import json
from pathlib import Path

from benchmarks.verify_qwen_unseen_literal_transfer import verify


ROOT = Path(__file__).resolve().parents[1]


def payload() -> dict:
    return json.loads((ROOT / "benchmarks/results/qwen_unseen_literal_transfer_results.json").read_text(encoding="utf-8"))


def test_qwen_unseen_literal_transfer_evidence_passes() -> None:
    assert verify(payload()) == []


def test_qwen_unseen_literal_transfer_rejects_test_selected_checkpoint() -> None:
    data = copy.deepcopy(payload())
    data["training_objective_contract"]["checkpoint_selection"] = "best heldout score"
    assert any("train-only validation" in failure for failure in verify(data))


def test_qwen_unseen_literal_transfer_rejects_confusable_collapse() -> None:
    data = copy.deepcopy(payload())
    for row in data["runs"]:
        if row["model"] == "output_context":
            row["pair_confusion_rate"] = 0.5
    assert any("stem/stem+r" in failure for failure in verify(data))
