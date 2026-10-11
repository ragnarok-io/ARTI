"""Declarative response evaluation, independent of model construction or loss."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re

from .recipe_data import json_fingerprint, read_jsonl
from .recipe_workflow import write_json, write_jsonl


@dataclass(frozen=True)
class CausalLMEvaluationConfig:
    generate_samples: int = 0
    sampling: str = "first"
    sample_seed: int = 17
    comparison: str = "exact_text"

    def __post_init__(self):
        if type(self.generate_samples) is not int or self.generate_samples < 0:
            raise ValueError("evaluation.generate_samples must be nonnegative")
        if self.sampling not in {"first", "seeded"}:
            raise ValueError("evaluation.sampling must be first or seeded")
        if type(self.sample_seed) is not int or self.sample_seed < 0:
            raise ValueError("evaluation.sample_seed must be nonnegative")
        if self.comparison not in {"exact_text", "numeric"}:
            raise ValueError("evaluation.comparison must be exact_text or numeric")


_NUMBER = re.compile(
    r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
)


def _number(value):
    text = value.strip()
    if _NUMBER.fullmatch(text) is None:
        return None
    try:
        result = Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None
    return result if result.is_finite() else None


def select_generation_examples(examples, config):
    if config.generate_samples == 0:
        return []
    candidates = list(examples)
    if config.sampling == "seeded":
        candidates.sort(
            key=lambda item: (
                hashlib.sha256(f"{config.sample_seed}\0{item.id}".encode()).digest(),
                item.id,
            )
        )
    selected = candidates[: config.generate_samples]
    if config.comparison == "numeric" and any(
        _number(item.messages[-1]["content"]) is None for item in selected
    ):
        raise ValueError("numeric evaluation requires whole-response finite numeric targets")
    return selected


def generation_evaluation_plan(examples, config, generation):
    selected = select_generation_examples(examples, config)
    return {
        "generated_samples_per_check": len(selected),
        "sample_ids": [item.id for item in selected],
        "max_new_tokens_per_sample": generation.max_new_tokens,
        "max_generated_tokens_per_check": len(selected) * generation.max_new_tokens,
        "cost_is_static_upper_bound": True,
    }


def score_causal_lm_predictions(records, *, comparison="exact_text"):
    """Score stored raw generation records without a model or answer extraction."""
    CausalLMEvaluationConfig(comparison=comparison)
    rows, seen = [], set()
    for record in records:
        identifier = record.get("id")
        target = record.get("target")
        generation = record.get("generation")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("scoring requires distinct nonempty prediction IDs")
        seen.add(identifier)
        if (
            not isinstance(target, str)
            or not isinstance(generation, dict)
            or not isinstance(generation.get("text"), str)
        ):
            raise ValueError("scoring requires original target and generation.text strings")
        exact = generation["text"].strip() == target.strip()
        matched = exact
        if comparison == "numeric":
            expected, observed = _number(target), _number(generation["text"])
            if expected is None:
                raise ValueError(
                    "numeric evaluation requires whole-response finite numeric targets"
                )
            matched = observed is not None and expected == observed
        rows.append({**record, "exact_match": exact, "generation_match": matched})
    if not rows:
        raise ValueError("scoring requires at least one generated response")
    metrics = {
        "generated_samples": len(rows),
        "generation_exact_match": sum(row["exact_match"] for row in rows) / len(rows),
        "generation_score": sum(row["generation_match"] for row in rows) / len(rows),
        "generation_comparison": comparison,
        "generation_truncated_samples": sum(
            row["generation"].get("finish_reason") == "length" for row in rows
        ),
    }
    if comparison == "numeric":
        metrics["generation_numeric_exact_match"] = metrics["generation_score"]
    return {
        "ok": True,
        "kind": "causal-lm-prediction-scores",
        "metrics": metrics,
        "predictions": rows,
    }


def compare_causal_lm_predictions(candidate, baseline, *, comparison="exact_text", pairing="exact"):
    """Pair saved replies by ID and rescore both arms with the same whole-reply rule."""
    if pairing not in {"exact", "intersection"}:
        raise ValueError("pairing must be exact or intersection")
    arms = {
        name: {
            row["id"]: row
            for row in score_causal_lm_predictions(records, comparison=comparison)["predictions"]
        }
        for name, records in (("candidate", candidate), ("baseline", baseline))
    }
    candidate_ids, baseline_ids = set(arms["candidate"]), set(arms["baseline"])
    if pairing == "exact" and candidate_ids != baseline_ids:
        raise ValueError(
            "generated sample IDs differ; use explicit intersection pairing for partial coverage"
        )
    identifiers = sorted(candidate_ids & baseline_ids)
    if not identifiers:
        raise ValueError("comparison requires at least one shared generated sample")
    outcomes = dict.fromkeys(("improved", "regressed", "both_correct", "both_incorrect"), 0)
    rows, matched_inputs = [], 0
    for identifier in identifiers:
        current, previous = arms["candidate"][identifier], arms["baseline"][identifier]
        if current["target"] != previous["target"]:
            raise ValueError(f"evaluation targets differ for {identifier!r}")
        inputs = current.get("input_sha256"), previous.get("input_sha256")
        verified = all(isinstance(value, str) and value for value in inputs)
        if verified and inputs[0] != inputs[1]:
            raise ValueError(f"evaluation inputs differ for {identifier!r}")
        matched_inputs += int(verified)
        good, was_good = current["generation_match"], previous["generation_match"]
        outcome = (
            "both_correct"
            if good and was_good
            else "improved"
            if good
            else "regressed"
            if was_good
            else "both_incorrect"
        )
        outcomes[outcome] += 1
        rows.append(
            {
                "id": identifier,
                "target": current["target"],
                "input_identity": "matched" if verified else "unrecorded",
                "outcome": outcome,
                "score_delta": int(good) - int(was_good),
                "candidate": current,
                "baseline": previous,
            }
        )
    metrics = {}
    for name in arms:
        paired = [arms[name][identifier] for identifier in identifiers]
        metrics[name] = score_causal_lm_predictions(paired, comparison=comparison)["metrics"]
        counts = [row["generation"].get("generated_tokens") for row in paired]
        recorded = [value for value in counts if type(value) is int and value >= 0]
        metrics[name].update(
            token_count_recorded_samples=len(recorded),
            generated_tokens=sum(recorded) if len(recorded) == len(paired) else None,
            mean_generated_tokens=sum(recorded) / len(paired)
            if len(recorded) == len(paired)
            else None,
        )
    deltas = {
        key: metrics["candidate"][key] - metrics["baseline"][key]
        for key in ("generation_score", "generation_exact_match", "generation_truncated_samples")
    }
    if all(metrics[name]["generated_tokens"] is not None for name in arms):
        deltas["generated_tokens"] = (
            metrics["candidate"]["generated_tokens"] - metrics["baseline"]["generated_tokens"]
        )
    return {
        "ok": True,
        "kind": "causal-lm-paired-comparison",
        "comparison": comparison,
        "pairing": pairing,
        "paired_samples": len(rows),
        "outcomes": outcomes,
        "coverage": {
            name: {
                "generated_samples": len(arm),
                "paired_fraction": len(rows) / len(arm),
                "excluded_ids": sorted(set(arm) - set(identifiers)),
            }
            for name, arm in arms.items()
        },
        "input_identity": {
            "matched_samples": matched_inputs,
            "unrecorded_samples": len(rows) - matched_inputs,
            "fully_verified": matched_inputs == len(rows),
        },
        "changed_reply_samples": sum(
            row["candidate"]["generation"]["text"] != row["baseline"]["generation"]["text"]
            for row in rows
        ),
        "metrics": metrics,
        "delta": deltas,
        "predictions": rows,
    }


def _evaluation_receipt(path):
    path = Path(path)
    payload = path.read_bytes()
    report = json.loads(payload)
    if report.get("objective") != "native-causal-lm-loss":
        raise ValueError("comparison requires native causal-LM evaluation reports")
    predictions = path.with_suffix(".predictions.jsonl")
    rows = list(read_jsonl(predictions))
    generated = [row for row in rows if row.get("generation") is not None]
    source = {
        "report": str(path),
        "predictions": str(predictions),
        "report_sha256": hashlib.sha256(payload).hexdigest(),
        "prediction_records_sha256": json_fingerprint(rows),
        "report_metadata": report,
    }
    return report, generated, source


def compare_causal_lm_evaluations(
    candidate, baseline, *, comparison=None, pairing="exact", output=None
):
    """Compare existing report/JSONL receipts without loading or running a model."""
    current, candidate_rows, candidate_source = _evaluation_receipt(candidate)
    previous, baseline_rows, baseline_source = _evaluation_receipt(baseline)
    declared = {
        report.get("evaluation", {}).get("comparison") for report in (current, previous)
    } - {None}
    if comparison is None:
        if len(declared) > 1:
            raise ValueError(
                "evaluation scoring profiles differ; choose a shared comparison explicitly"
            )
        comparison = next(iter(declared), "exact_text")
    result = compare_causal_lm_predictions(
        candidate_rows, baseline_rows, comparison=comparison, pairing=pairing
    )
    result["sources"] = {"candidate": candidate_source, "baseline": baseline_source}
    conditions = {}
    for key in ("generation_config", "input_config", "evaluation", "batching_config", "model_source"):
        values = current.get(key), previous.get(key)
        conditions[key] = {
            "candidate": values[0],
            "baseline": values[1],
            "status": "unrecorded"
            if None in values
            else "matched"
            if values[0] == values[1]
            else "different",
        }
    result["conditions"] = conditions
    # Token loss is comparable only for the same encoded data and denominator.
    loss_keys = ("dataset_sha256", "samples", "supervised_tokens", "objective", "weighting")
    comparable = all(
        current.get(key) is not None and current.get(key) == previous.get(key) for key in loss_keys
    )
    loss_delta = {}
    if comparable:
        for key in ("loss", "perplexity", "teacher_forced_token_accuracy"):
            values = current.get(key), previous.get(key)
            if all(type(value) in {int, float} and math.isfinite(value) for value in values):
                loss_delta[key] = values[0] - values[1]
    result["teacher_forced_comparison"] = {
        "scope": "full_evaluation_dataset",
        "comparable": comparable,
        "delta": loss_delta,
        "reason": "same encoded data, objective and denominator"
        if comparable
        else "encoded data, objective or denominator differs or is unrecorded",
    }
    if output is not None:
        report_path = Path(output)
        pairs_path = report_path.with_suffix(".pairs.jsonl")
        inputs = {
            Path(source[key]).resolve()
            for source in result["sources"].values()
            for key in ("report", "predictions")
        }
        if report_path.resolve() in inputs or pairs_path.resolve() in inputs:
            raise ValueError("comparison output must not overwrite its input receipts")
        result["output"] = {"report": str(report_path), "pairs": str(pairs_path)}
        write_jsonl(pairs_path, result["predictions"])
        write_json(
            report_path, {key: value for key, value in result.items() if key != "predictions"}
        )
    return result


__all__ = [
    "CausalLMEvaluationConfig",
    "score_causal_lm_predictions",
    "compare_causal_lm_predictions",
    "compare_causal_lm_evaluations",
]
