from __future__ import annotations

import json

from benchmarks.build_qwen_ttt_natural_plans import (
    build_binding_swap_plans,
    build_coverage_plans,
    build_plans,
)


def test_paired_natural_plans_share_latest_query_but_change_history() -> None:
    plans = build_plans(4, seed=17)
    assert len(plans) == 4
    assert len({plan["user_turns"][0] for plan in plans}) == 4
    assert len({plan["user_turns"][1] for plan in plans}) == 1
    assert all(plan["scenario"] == "paired-private-label" for plan in plans)


def test_paired_natural_plan_file_is_jsonl_safe() -> None:
    plan = build_plans(1, seed=23)[0]
    encoded = json.dumps(plan, ensure_ascii=False)
    decoded = json.loads(encoded)
    assert decoded["user_turns"][1] == "Return only the private label you were asked to remember."


def test_coverage_suite_separates_templates_and_label_combinations() -> None:
    plans = build_coverage_plans(128, seed=23)
    assert len(plans) == 128
    assert {plan["metadata"]["template_block"] for plan in plans} == {0, 1, 2, 3}
    train = plans[:64]
    evaluation = plans[64:]
    assert {plan["metadata"]["split"] for plan in train} == {"train"}
    assert {plan["metadata"]["split"] for plan in evaluation} == {"eval"}
    assert not {
        plan["metadata"]["label"] for plan in train
    }.intersection(plan["metadata"]["label"] for plan in evaluation)
    assert len({plan["user_turns"][1] for plan in train}) == 2
    assert len({plan["user_turns"][1] for plan in evaluation}) == 2


def test_binding_swap_plans_preserve_token_multiset_and_change_order() -> None:
    plans = build_binding_swap_plans(8, seed=31)
    assert len(plans) == 8
    grouped: dict[str, list[dict[str, object]]] = {}
    for plan in plans:
        grouped.setdefault(str(plan["pair_id"]), []).append(plan)
    assert len(grouped) == 4
    for pair in grouped.values():
        assert len(pair) == 2
        assert pair[0]["user_turns"][1] == pair[1]["user_turns"][1]
        left_tokens = sorted(str(pair[0]["user_turns"][0]).split())
        right_tokens = sorted(str(pair[1]["user_turns"][0]).split())
        assert left_tokens == right_tokens
        assert pair[0]["user_turns"][0] != pair[1]["user_turns"][0]
